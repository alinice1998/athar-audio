"""
Unified DSP Pipeline Orchestrator for Quranic Audio Style & Reverb Cloner
Coordinates the 5 stages and provides real-time progress callbacks:
1. Preprocessing & Raw Audio Cleaning
2. Neural & DSP De-Echo / De-Reverb Separation (with MD5 Caching)
3. Dry-to-Dry Tone & Dynamic EQ Matching
4. Reverb Footprint Extraction & Parametric IR Synthesis
5. Spatial Convolution, Sidechain Ducking & Limiter Mastering
"""
import os
import time
from typing import Callable, Optional
import soundfile as sf
import numpy as np

from backend.dsp.preprocessor import Preprocessor
from backend.dsp.separator import AudioSeparatorEngine
from backend.dsp.tone_matcher import ToneMatcher
from backend.dsp.reverb_cloner import ReverbCloner
from backend.dsp.mixer import StudioMixer


class QuranicAudioPipeline:
    def __init__(self, sample_rate: int = 44100, storage_dir: str = "storage"):
        self.sr = sample_rate
        self.storage_dir = storage_dir
        self.cache_dir = os.path.join(storage_dir, "cache")
        self.outputs_dir = os.path.join(storage_dir, "outputs")
        
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(self.outputs_dir, exist_ok=True)

        # Initialize DSP modules
        self.preprocessor = Preprocessor(target_sr=sample_rate)
        self.separator = AudioSeparatorEngine(cache_dir=self.cache_dir)
        self.tone_matcher = ToneMatcher(sample_rate=sample_rate)
        self.reverb_cloner = ReverbCloner(sample_rate=sample_rate)
        self.mixer = StudioMixer(sample_rate=sample_rate)

    def process(
        self,
        ref_file_path: str,
        target_file_path: str,
        clean_target: bool = True,
        eq_intensity: float = 0.8,
        eq_engine: str = "spectral",
        reverb_mode: str = "parametric",
        decay_scale: float = 1.0,
        reverb_mix: Optional[float] = None,
        enable_ducking: bool = True,
        ducking_amount_db: float = 2.5,
        session_id: str = None,
        progress_callback: Optional[Callable[[int, str, float], None]] = None
    ) -> dict:
        """
        Executes complete 5-stage DSP pipeline with callbacks and audio generation.
        """
        start_time = time.time()
        if session_id is None:
            session_id = f"session_{int(time.time() * 1000)}"

        session_out_dir = os.path.join(self.outputs_dir, session_id)
        os.makedirs(session_out_dir, exist_ok=True)

        def report(stage_idx: int, stage_name: str, pct: float):
            if progress_callback:
                progress_callback(stage_idx, stage_name, pct)

        # -------------------------------------------------------------
        # Stage 1: Preprocessing & Cleaning
        # -------------------------------------------------------------
        report(1, "تجهيز وتوحيد جودة الملفات الصوتية وتنقية التسجيل الخام", 15.0)
        ref_audio, _ = self.preprocessor.load_audio(
            ref_file_path, clean=False, is_dry_user=False
        )
        target_audio, _ = self.preprocessor.load_audio(
            target_file_path, clean=clean_target, is_dry_user=True
        )

        target_clean_path = os.path.join(session_out_dir, "target_dry_clean.wav")
        self.preprocessor.save_audio(target_audio, target_clean_path)

        # -------------------------------------------------------------
        # Stage 2: Neural Separation with MD5 Caching
        # -------------------------------------------------------------
        report(2, "الفصل الذكي لصدى التلاوة المرجعية عن الصوت الجاف", 35.0)
        ref_dry, ref_reverb, cache_folder, was_cached = self.separator.separate(
            ref_audio=ref_audio,
            sr=self.sr,
            ref_file_path=ref_file_path
        )
        cached_dry_path = os.path.join(cache_folder, "ref_dry.wav")
        cached_reverb_path = os.path.join(cache_folder, "ref_reverb.wav")

        # -------------------------------------------------------------
        # Stage 3: Dry-to-Dry EQ & Dynamics Matching
        # -------------------------------------------------------------
        report(3, "مطابقة البصمة الترددية والديناميكية بين الصوت الجاف والجاف", 60.0)
        matched_target, spectrum_info = self.tone_matcher.match(
            target_audio=target_audio,
            ref_dry_audio=ref_dry,
            eq_intensity=eq_intensity,
            engine=eq_engine
        )
        target_eq_path = os.path.join(session_out_dir, "target_eq_matched.wav")
        self.preprocessor.save_audio(matched_target, target_eq_path)

        # -------------------------------------------------------------
        # Stage 4: Reverb Extraction & Synthesis
        # -------------------------------------------------------------
        report(4, "استخلاص الفضاء المكاني وتوليد البصمة الصوتية للصدى (IR)", 80.0)
        ir_path = os.path.join(session_out_dir, "extracted_ir.wav")
        ir_audio, ir_meta = self.reverb_cloner.extract_ir(
            ref_dry=ref_dry,
            ref_reverb=ref_reverb,
            mode=reverb_mode,
            decay_scale=decay_scale,
            output_ir_path=ir_path
        )

        # Determine wet/dry mix: use user setting or detected from reference
        if reverb_mix is None:
            # Automatic detection: map linear ratio to mix slider (typically 0.25 to 0.50)
            det_ratio = ir_meta.get("wet_dry_ratio_linear", 0.35)
            effective_reverb_mix = float(np.clip(det_ratio * 0.75, 0.15, 0.70))
        else:
            effective_reverb_mix = float(np.clip(reverb_mix, 0.0, 1.0))

        # -------------------------------------------------------------
        # Stage 5: Mixing, Ducking & Mastering
        # -------------------------------------------------------------
        report(5, "الدمج المكاني، الخفض التلقائي أثناء النطق والماسترينغ النهائي", 95.0)
        final_master_path = os.path.join(session_out_dir, "quranic_master.wav")
        mastered_audio, mix_stats = self.mixer.mix_and_master(
            dry_audio=matched_target,
            impulse_response=ir_audio,
            reverb_mix=effective_reverb_mix,
            enable_ducking=enable_ducking,
            ducking_amount_db=ducking_amount_db,
            output_file_path=final_master_path
        )

        elapsed = round(time.time() - start_time, 2)
        report(5, "اكتملت المعالجة بنجاح!", 100.0)

        return {
            "success": True,
            "session_id": session_id,
            "elapsed_seconds": elapsed,
            "was_reference_cached": was_cached,
            "files": {
                "final_master": final_master_path,
                "extracted_ir": ir_path,
                "ref_dry": cached_dry_path,
                "ref_reverb": cached_reverb_path,
                "target_dry_clean": target_clean_path,
                "target_eq_matched": target_eq_path
            },
            "parameters": {
                "eq_intensity": eq_intensity,
                "eq_engine": eq_engine,
                "reverb_mode": reverb_mode,
                "decay_scale": decay_scale,
                "reverb_mix": effective_reverb_mix,
                "enable_ducking": enable_ducking,
                "ducking_amount_db": ducking_amount_db,
                "clean_target": clean_target
            },
            "acoustic_analysis": ir_meta,
            "spectrum_data": spectrum_info,
            "mastering_stats": mix_stats
        }
