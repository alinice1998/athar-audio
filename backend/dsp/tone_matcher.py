"""
Stage 3: Dry-to-Dry Spectral Tone & Dynamic Matcher (مطابقة اللون الترددي والديناميكية)
- Matches EQ exclusively between target_audio and ref_dry.wav (never with reverberant reference).
- Provides two switchable engines:
    1. Smooth Spectral EQ (1/3-octave smoothed Welch PSD, linear-phase FIR, smooth compression).
    2. Matchering Engine.
- Adjustable eq_intensity slider (0.0 to 1.0).
- Generates JSON frequency curve data for Chart.js visualization.
"""
import numpy as np
import scipy.signal

try:
    import matchering as mg
    HAS_MATCHERING = True
except Exception:
    HAS_MATCHERING = False

try:
    import pedalboard
    from pedalboard import Pedalboard, Compressor
    HAS_PEDALBOARD = True
except Exception:
    HAS_PEDALBOARD = False


class ToneMatcher:
    def __init__(self, sample_rate: int = 44100):
        self.sr = sample_rate

    def match(
        self,
        target_audio: np.ndarray,
        ref_dry_audio: np.ndarray,
        eq_intensity: float = 0.8,
        engine: str = "spectral"  # "spectral" or "matchering"
    ) -> tuple[np.ndarray, dict]:
        """
        Matches tonal character of target_audio to ref_dry_audio.
        Returns:
            matched_audio (np.ndarray): Shape (2, N)
            spectral_data (dict): Spectrum data for Chart.js plotting
        """
        eq_intensity = float(np.clip(eq_intensity, 0.0, 1.0))

        if engine.lower() == "matchering" and HAS_MATCHERING:
            try:
                matched = self._matchering_process(target_audio, ref_dry_audio)
            except Exception:
                matched = self._spectral_eq_process(target_audio, ref_dry_audio)
        else:
            matched = self._spectral_eq_process(target_audio, ref_dry_audio)

        # Blend original target and matched audio based on eq_intensity
        blended = (1.0 - eq_intensity) * target_audio + eq_intensity * matched

        # Apply smooth vocal compression (Ratio 3:1, Attack 20ms, Release 150ms)
        compressed = self._apply_vocal_compression(blended)

        # Calculate spectral response curves for plotting
        spectrum_info = self.calculate_spectrum_comparison(
            ref_dry=ref_dry_audio,
            target_dry=target_audio,
            target_matched=compressed
        )

        return compressed.astype(np.float32), spectrum_info

    def _spectral_eq_process(self, target: np.ndarray, ref_dry: np.ndarray) -> np.ndarray:
        """
        Smooth Spectral EQ Engine:
        - Computes Welch PSD on active vocal frames (> -45 dBFS).
        - Computes frequency transfer function H(f).
        - Smooths curve on 1/3-octave bands.
        - Clamps gain between -12dB and +12dB.
        - Applies via linear-phase FIR filter with fftconvolve.
        """
        nperseg = 4096
        # Downmix to mono for robust vocal spectral profiling
        target_mono = np.mean(target, axis=0)
        ref_mono = np.mean(ref_dry, axis=0)

        # Filter out silence below -45 dBFS
        silence_thresh = 10.0 ** (-45.0 / 20.0)
        t_active = target_mono[np.abs(target_mono) > silence_thresh]
        r_active = ref_mono[np.abs(ref_mono) > silence_thresh]

        if len(t_active) < nperseg:
            t_active = target_mono
        if len(r_active) < nperseg:
            r_active = ref_mono

        freqs, psd_target = scipy.signal.welch(t_active, fs=self.sr, nperseg=nperseg)
        _, psd_ref = scipy.signal.welch(r_active, fs=self.sr, nperseg=nperseg)

        # Avoid zero division
        eps = 1e-10
        raw_gain = np.sqrt((psd_ref + eps) / (psd_target + eps))

        # Convert to dB for 1/3-octave smoothing
        gain_db = 20.0 * np.log10(raw_gain + eps)

        # Smooth across fractional octave bands (Gaussian filter in log-frequency space)
        smoothed_gain_db = self._fractional_octave_smooth(freqs, gain_db, fraction=3.0)

        # Hard clamp between -12 dB and +12 dB to prevent metallic resonance
        clamped_gain_db = np.clip(smoothed_gain_db, -12.0, 12.0)
        linear_gain = 10.0 ** (clamped_gain_db / 20.0)

        # Design linear-phase FIR filter
        # Ensure freqs start at 0 and end at Nyquist (sr/2)
        norm_freqs = freqs / (self.sr / 2.0)
        norm_freqs[0] = 0.0
        norm_freqs[-1] = 1.0

        num_taps = 1025  # Odd number for Type I linear-phase FIR
        fir_coeff = scipy.signal.firwin2(num_taps, norm_freqs, linear_gain, window='hann')

        # Apply FIR filter to each channel via FFT convolution
        out_channels = []
        for ch in range(target.shape[0]):
            filtered = scipy.signal.fftconvolve(target[ch], fir_coeff, mode='same')
            out_channels.append(filtered)

        return np.stack(out_channels, axis=0).astype(np.float32)

    def _fractional_octave_smooth(self, freqs: np.ndarray, values_db: np.ndarray, fraction: float = 3.0) -> np.ndarray:
        """Smooths spectrum values using 1/N-octave moving window."""
        smoothed = np.copy(values_db)
        log_f = np.log2(np.maximum(freqs, 10.0))
        octave_width = 1.0 / fraction

        for i, f in enumerate(freqs):
            if f < 30.0:
                continue
            lower_bound = log_f[i] - octave_width / 2.0
            upper_bound = log_f[i] + octave_width / 2.0
            indices = np.where((log_f >= lower_bound) & (log_f <= upper_bound))[0]
            if len(indices) > 0:
                smoothed[i] = np.mean(values_db[indices])

        # Smooth extreme low-end (sub-bass < 80Hz) to prevent boominess
        sub_80_idx = np.where(freqs < 80.0)[0]
        if len(sub_80_idx) > 0:
            smoothed[sub_80_idx] = np.clip(smoothed[sub_80_idx], -6.0, 3.0)

        return smoothed

    def _apply_vocal_compression(self, audio: np.ndarray) -> np.ndarray:
        """Applies gentle vocal compression (Ratio 3:1, Attack 20ms, Release 150ms)."""
        if HAS_PEDALBOARD:
            try:
                board = Pedalboard([
                    Compressor(threshold_db=-18.0, ratio=3.0, attack_ms=20.0, release_ms=150.0)
                ])
                compressed = board(audio, self.sr)
                return compressed.astype(np.float32)
            except Exception:
                pass

        # SciPy soft-knee RMS compressor fallback
        return self._dsp_compressor(audio, threshold_db=-18.0, ratio=3.0, attack_ms=20.0, release_ms=150.0)

    def _dsp_compressor(
        self,
        audio: np.ndarray,
        threshold_db: float = -18.0,
        ratio: float = 3.0,
        attack_ms: float = 20.0,
        release_ms: float = 150.0
    ) -> np.ndarray:
        """Pure DSP soft RMS vocal compressor with block-level tracking."""
        mono = np.mean(audio, axis=0)
        n_samples = len(mono)
        block_size = max(64, int(self.sr * 0.005))
        n_blocks = n_samples // block_size
        if n_blocks < 2:
            return audio

        truncated_len = n_blocks * block_size
        blocks = mono[:truncated_len].reshape(n_blocks, block_size)
        block_rms = np.sqrt(np.mean(blocks ** 2, axis=1) + 1e-12)

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

        env_db = 20.0 * np.log10(env + 1e-9)
        overshoot = np.maximum(0.0, env_db - threshold_db)
        compressed_db = threshold_db + overshoot / ratio
        reduction_db = compressed_db - env_db
        block_gain = 10.0 ** (np.minimum(0.0, reduction_db) / 20.0)

        block_times = np.linspace(0, truncated_len, n_blocks, endpoint=False)
        all_times = np.arange(n_samples)
        full_gain = np.interp(all_times, block_times, block_gain).astype(np.float32)

        out = np.zeros_like(audio)
        out[0] = audio[0] * full_gain
        out[1] = audio[1] * full_gain
        return out.astype(np.float32)

    def _matchering_process(self, target: np.ndarray, ref_dry: np.ndarray) -> np.ndarray:
        """Applies matchering library processing."""
        import tempfile
        import os
        import soundfile as sf
        with tempfile.TemporaryDirectory() as tmpdir:
            t_path = os.path.join(tmpdir, "target.wav")
            r_path = os.path.join(tmpdir, "ref.wav")
            out_path = os.path.join(tmpdir, "out.wav")
            sf.write(t_path, target.T, self.sr)
            sf.write(r_path, ref_dry.T, self.sr)

            mg.process(
                target=t_path,
                reference=r_path,
                results=[mg.pcm24(out_path)]
            )
            data, _ = sf.read(out_path, dtype='float32')
            if data.ndim == 1:
                return np.stack([data, data], axis=0)
            return data.T if data.shape[1] == 2 else data

    def calculate_spectrum_comparison(
        self,
        ref_dry: np.ndarray,
        target_dry: np.ndarray,
        target_matched: np.ndarray
    ) -> dict:
        """Calculates 32 standard frequency band levels in dB for visual chart plotting."""
        bands = [
            40, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 800,
            1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000,
            10000, 12500, 16000
        ]
        
        freqs, p_ref = scipy.signal.welch(np.mean(ref_dry, axis=0), fs=self.sr, nperseg=4096)
        _, p_dry = scipy.signal.welch(np.mean(target_dry, axis=0), fs=self.sr, nperseg=4096)
        _, p_match = scipy.signal.welch(np.mean(target_matched, axis=0), fs=self.sr, nperseg=4096)

        def sample_at_bands(p_array):
            levels = []
            for b in bands:
                idx = np.argmin(np.abs(freqs - b))
                db_val = float(10.0 * np.log10(p_array[idx] + 1e-12))
                levels.append(round(db_val, 1))
            return levels

        return {
            "bands": [f"{b}Hz" if b < 1000 else f"{b//1000}kHz" for b in bands],
            "band_numbers": bands,
            "ref_dry": sample_at_bands(p_ref),
            "target_dry": sample_at_bands(p_dry),
            "target_matched": sample_at_bands(p_match)
        }
