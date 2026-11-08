"""
Comprehensive Test Suite for Quranic Audio Style & Reverb Cloner
Generates synthetic audio signals and verifies all 5 DSP stages and FastAPI endpoints.
"""
import os
import sys
import shutil
import unittest
import numpy as np
import soundfile as sf

# Add workspace root to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.dsp.preprocessor import Preprocessor
from backend.dsp.separator import AudioSeparatorEngine
from backend.dsp.tone_matcher import ToneMatcher
from backend.dsp.reverb_cloner import ReverbCloner
from backend.dsp.mixer import StudioMixer
from backend.dsp.pipeline import QuranicAudioPipeline


def generate_synthetic_recitation(duration: float = 3.0, sr: int = 44100, add_reverb: bool = False) -> np.ndarray:
    """Generates synthetic Quranic vowel formant harmonic signal with realistic pauses."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # Fundamental pitch ~140 Hz (male reciter) with vibrato
    f0 = 140.0 + 3.0 * np.sin(2 * np.pi * 5.0 * t)
    phase = 2 * np.pi * np.cumsum(f0) / sr

    # Harmonics (formants simulating Arabic vowels 'Aa' / 'Oo')
    harmonics = [1.0, 0.7, 0.5, 0.35, 0.2, 0.15, 0.08, 0.05]
    voice = np.zeros_like(t)
    for i, amp in enumerate(harmonics, start=1):
        voice += amp * np.sin(i * phase)

    # Modulation envelope with pauses (simulating recitation stops)
    envelope = np.clip(np.sin(2 * np.pi * 0.7 * t), 0.0, 1.0) ** 1.5
    voice = voice * envelope

    # Stereo base
    stereo = np.stack([voice, voice], axis=0).astype(np.float32)

    if add_reverb:
        # Synthesize simple reverberant room impulse
        ir_t = np.linspace(0, 1.5, int(sr * 1.5))
        ir = np.exp(-ir_t * 3.5) * np.random.normal(0, 0.5, len(ir_t))
        rev_l = np.convolve(stereo[0], ir, mode='same')
        rev_r = np.convolve(stereo[1], np.roll(ir, 100), mode='same')
        reverberant = stereo + 0.45 * np.stack([rev_l, rev_r], axis=0)
        return reverberant.astype(np.float32)

    return stereo.astype(np.float32)


class TestQuranicDSPPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_dir = "storage/test_tmp"
        os.makedirs(cls.test_dir, exist_ok=True)
        cls.sr = 44100

        # Generate audio files
        cls.ref_path = os.path.join(cls.test_dir, "test_ref.wav")
        cls.target_path = os.path.join(cls.test_dir, "test_target.wav")

        ref_audio = generate_synthetic_recitation(duration=2.5, sr=cls.sr, add_reverb=True)
        target_audio = generate_synthetic_recitation(duration=2.5, sr=cls.sr, add_reverb=False)

        sf.write(cls.ref_path, ref_audio.T, cls.sr)
        sf.write(cls.target_path, target_audio.T, cls.sr)

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.test_dir):
            shutil.rmtree(cls.test_dir, ignore_errors=True)

    def test_01_preprocessor(self):
        """Test Stage 1: Audio loading, stereo conversion, highpass, and normalization."""
        prep = Preprocessor(target_sr=self.sr)
        audio, sr = prep.load_audio(self.target_path, clean=True, is_dry_user=True)
        self.assertEqual(sr, self.sr)
        self.assertEqual(audio.shape[0], 2)
        peak_linear = np.max(np.abs(audio))
        self.assertAlmostEqual(peak_linear, 10 ** (-3.0 / 20.0), delta=0.05)

    def test_02_separator(self):
        """Test Stage 2: Separation and caching."""
        cache_dir = os.path.join(self.test_dir, "cache")
        sep = AudioSeparatorEngine(cache_dir=cache_dir)
        ref_audio, _ = sf.read(self.ref_path, dtype='float32')
        ref_audio = ref_audio.T

        # First run: should compute
        dry, rev, cdir, was_cached = sep.separate(ref_audio, self.sr, self.ref_path)
        self.assertFalse(was_cached)
        self.assertEqual(dry.shape, ref_audio.shape)
        self.assertEqual(rev.shape, ref_audio.shape)

        # Second run: must hit cache
        dry2, rev2, cdir2, was_cached2 = sep.separate(ref_audio, self.sr, self.ref_path)
        self.assertTrue(was_cached2)
        np.testing.assert_array_almost_equal(dry, dry2)

    def test_03_tone_matcher(self):
        """Test Stage 3: Smooth spectral EQ matching and spectrum JSON generation."""
        matcher = ToneMatcher(sample_rate=self.sr)
        target = generate_synthetic_recitation(duration=2.0, sr=self.sr, add_reverb=False)
        ref_dry = generate_synthetic_recitation(duration=2.0, sr=self.sr, add_reverb=False)

        matched, spec_data = matcher.match(target, ref_dry, eq_intensity=0.8, engine="spectral")
        self.assertEqual(matched.shape, target.shape)
        self.assertIn("bands", spec_data)
        self.assertIn("ref_dry", spec_data)
        self.assertIn("target_matched", spec_data)
        self.assertEqual(len(spec_data["bands"]), len(spec_data["ref_dry"]))

    def test_04_reverb_cloner(self):
        """Test Stage 4: Parametric and Wiener IR extraction."""
        cloner = ReverbCloner(sample_rate=self.sr)
        dry = generate_synthetic_recitation(duration=2.0, sr=self.sr, add_reverb=False)
        reverb = generate_synthetic_recitation(duration=2.0, sr=self.sr, add_reverb=True) - dry

        # Parametric Mode
        ir_param, meta_p = cloner.extract_ir(dry, reverb, mode="parametric", decay_scale=1.2)
        self.assertEqual(ir_param.shape[0], 2)
        self.assertGreater(meta_p["rt60_seconds"], 0.2)
        self.assertIn("predelay_ms", meta_p)

        # Wiener Mode
        ir_wiener, meta_w = cloner.extract_ir(dry, reverb, mode="wiener", decay_scale=1.0)
        self.assertEqual(ir_wiener.shape[0], 2)

    def test_05_mixer(self):
        """Test Stage 5: Convolution, Sidechain Ducking, Limiter."""
        mixer = StudioMixer(sample_rate=self.sr)
        dry = generate_synthetic_recitation(duration=2.0, sr=self.sr)
        cloner = ReverbCloner(sample_rate=self.sr)
        ir, _ = cloner.extract_ir(dry, dry * 0.3, mode="parametric")

        mastered, stats = mixer.mix_and_master(
            dry_audio=dry,
            impulse_response=ir,
            reverb_mix=0.4,
            enable_ducking=True
        )
        self.assertEqual(mastered.shape, dry.shape)
        self.assertLessEqual(stats["peak_db"], -0.95)  # Limiter at -1.0 dBFS

    def test_06_complete_pipeline(self):
        """Test End-to-End Pipeline execution."""
        pipeline = QuranicAudioPipeline(sample_rate=self.sr, storage_dir=self.test_dir)
        progress_events = []

        def on_prog(idx, name, pct):
            progress_events.append((idx, pct))

        result = pipeline.process(
            ref_file_path=self.ref_path,
            target_file_path=self.target_path,
            clean_target=True,
            eq_intensity=0.75,
            progress_callback=on_prog
        )

        self.assertTrue(result["success"])
        self.assertTrue(os.path.exists(result["files"]["final_master"]))
        self.assertTrue(os.path.exists(result["files"]["extracted_ir"]))
        self.assertGreater(len(progress_events), 4)
        self.assertEqual(progress_events[-1][1], 100.0)

    def test_07_fastapi_endpoints(self):
        """Test FastAPI API status and cache check endpoints."""
        from fastapi.testclient import TestClient
        from backend.main import app

        client = TestClient(app)
        
        # Test status endpoint
        res = client.get("/api/status")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["status"], "online")

        # Test index endpoint
        res_idx = client.get("/")
        self.assertEqual(res_idx.status_code, 200)


if __name__ == "__main__":
    unittest.main()
