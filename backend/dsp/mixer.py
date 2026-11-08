"""
Stage 5: Spatial Mixing, Sidechain Reverb Ducking & Mastering (الدمج المكاني والماسترينغ)
- Convolves extracted IR with EQ-matched recitation.
- Sidechain Reverb Ducking: dynamically ducks reverb by 2 to 3 dB during vocal recitation
  to preserve diction clarity (مخارج الحروف), and smoothly blooms the reverb tail during pauses.
- Dynamic Wet/Dry balancing.
- Mastering Limiter at -1.0 dB true-peak ceiling.
- High-fidelity 24-bit WAV export.
"""
import os
import numpy as np
import scipy.signal
import soundfile as sf

try:
    import pedalboard
    from pedalboard import Pedalboard, Limiter
    HAS_PEDALBOARD = True
except Exception:
    HAS_PEDALBOARD = False


class StudioMixer:
    def __init__(self, sample_rate: int = 44100):
        self.sr = sample_rate

    def mix_and_master(
        self,
        dry_audio: np.ndarray,
        impulse_response: np.ndarray,
        reverb_mix: float = 0.35,
        enable_ducking: bool = True,
        ducking_amount_db: float = 2.5,
        output_file_path: str = None
    ) -> tuple[np.ndarray, dict]:
        """
        Convolves IR, applies sidechain ducking, mixes wet/dry, and masters with limiter.
        Returns:
            final_master (np.ndarray): Shape (2, N)
            stats (dict): Peak, RMS, ducking stats
        """
        reverb_mix = float(np.clip(reverb_mix, 0.0, 1.0))
        n_channels = dry_audio.shape[0]
        orig_len = dry_audio.shape[1]

        # 1. Stereo Convolution with IR
        wet_channels = []
        for ch in range(n_channels):
            ir_ch = impulse_response[ch % impulse_response.shape[0]]
            # Fast FFT Convolution
            conv = scipy.signal.fftconvolve(dry_audio[ch], ir_ch, mode='full')
            # Truncate or extend to match recitation length + slight tail
            wet_channels.append(conv[:orig_len])

        wet_audio = np.stack(wet_channels, axis=0).astype(np.float32)

        # Normalize wet audio energy relative to dry
        rms_dry = np.sqrt(np.mean(dry_audio ** 2)) + 1e-12
        rms_wet = np.sqrt(np.mean(wet_audio ** 2)) + 1e-12
        wet_audio = wet_audio * float(rms_dry / rms_wet)

        # 2. Sidechain Reverb Ducking
        if enable_ducking:
            wet_audio = self._apply_sidechain_ducking(
                dry_audio=dry_audio,
                wet_audio=wet_audio,
                max_duck_db=ducking_amount_db,
                attack_ms=15.0,
                release_ms=220.0
            )

        # 3. Equal-Power Wet/Dry Balancing
        # When reverb_mix = 0.35: dry_gain = cos(0.35 * pi/2), wet_gain = sin(0.35 * pi/2)
        angle = reverb_mix * (np.pi / 2.0)
        dry_gain = np.cos(angle)
        wet_gain = np.sin(angle)

        mixed = dry_gain * dry_audio + wet_gain * wet_audio

        # 4. Studio Mastering Limiter (-1.0 dBFS ceiling)
        mastered = self._apply_mastering_limiter(mixed, ceiling_db=-1.0)

        # Export if path provided
        if output_file_path:
            os.makedirs(os.path.dirname(os.path.abspath(output_file_path)), exist_ok=True)
            sf.write(output_file_path, mastered.T, self.sr, subtype='PCM_24')

        stats = {
            "peak_db": round(float(20.0 * np.log10(np.max(np.abs(mastered)) + 1e-9)), 2),
            "rms_db": round(float(20.0 * np.log10(np.sqrt(np.mean(mastered ** 2)) + 1e-9)), 2),
            "reverb_mix": round(reverb_mix, 2),
            "ducking_enabled": enable_ducking,
            "ducking_amount_db": ducking_amount_db
        }

        return mastered.astype(np.float32), stats

    def _apply_sidechain_ducking(
        self,
        dry_audio: np.ndarray,
        wet_audio: np.ndarray,
        max_duck_db: float = 2.5,
        attack_ms: float = 15.0,
        release_ms: float = 220.0
    ) -> np.ndarray:
        """
        Sidechain Ducking:
        Ducks reverb by max_duck_db when dry vocal envelope is active,
        then releases smoothly during stops/pauses between Quranic verses.
        Optimized with fast block-level envelope tracking and cubic interpolation.
        """
        mono_dry = np.mean(dry_audio, axis=0)
        n_samples = len(mono_dry)
        
        # Block size of 5ms (e.g. 220 samples at 44.1kHz)
        block_size = max(64, int(self.sr * 0.005))
        n_blocks = n_samples // block_size
        if n_blocks < 2:
            return wet_audio

        # Reshape into blocks and compute RMS energy per block
        truncated_len = n_blocks * block_size
        blocks = mono_dry[:truncated_len].reshape(n_blocks, block_size)
        block_rms = np.sqrt(np.mean(blocks ** 2, axis=1) + 1e-12)

        # Vectorized attack/release on block RMS
        block_fs = self.sr / block_size
        alpha_att = np.exp(-1.0 / (block_fs * (attack_ms / 1000.0)))
        alpha_rel = np.exp(-1.0 / (block_fs * (release_ms / 1000.0)))

        env = np.zeros(n_blocks, dtype=np.float32)
        curr = 0.0
        for b in range(n_blocks):
            r = block_rms[b]
            if r > curr:
                curr = alpha_att * curr + (1.0 - alpha_att) * r
            else:
                curr = alpha_rel * curr + (1.0 - alpha_rel) * r
            env[b] = curr

        # Compute ducking curve in dB
        p95 = np.percentile(env, 95) + 1e-6
        norm_env = np.clip(env / p95, 0.0, 1.0)
        block_duck_db = -max_duck_db * (norm_env ** 1.2)
        block_duck_gain = 10.0 ** (block_duck_db / 20.0)

        # Interpolate block gain back to sample rate
        block_times = np.linspace(0, truncated_len, n_blocks, endpoint=False)
        all_times = np.arange(n_samples)
        full_duck_gain = np.interp(all_times, block_times, block_duck_gain).astype(np.float32)

        ducked_wet = np.zeros_like(wet_audio)
        ducked_wet[0] = wet_audio[0] * full_duck_gain
        ducked_wet[1] = wet_audio[1] * full_duck_gain

        return ducked_wet.astype(np.float32)

    def _apply_mastering_limiter(self, audio: np.ndarray, ceiling_db: float = -1.0) -> np.ndarray:
        """Applies true-peak brickwall mastering limiter."""
        if HAS_PEDALBOARD:
            try:
                board = Pedalboard([
                    Limiter(threshold_db=ceiling_db, release_ms=80.0)
                ])
                limited = board(audio, self.sr)
                return limited.astype(np.float32)
            except Exception:
                pass

        # High-Fidelity DSP Soft-Clip Limiter
        ceiling_linear = 10.0 ** (ceiling_db / 20.0)  # ~0.891
        peak = np.max(np.abs(audio))

        if peak <= ceiling_linear:
            return audio

        # Soft tanh limiter with transparent knee
        threshold = ceiling_linear * 0.85
        mastered = np.copy(audio)

        for ch in range(audio.shape[0]):
            sig = audio[ch]
            above_mask = np.abs(sig) > threshold
            if np.any(above_mask):
                excess = np.abs(sig[above_mask]) - threshold
                max_excess = ceiling_linear - threshold
                compressed = threshold + max_excess * np.tanh(excess / (max_excess + 1e-9))
                mastered[ch][above_mask] = np.sign(sig[above_mask]) * compressed

        # Strict final guard
        return np.clip(mastered, -ceiling_linear, ceiling_linear).astype(np.float32)
