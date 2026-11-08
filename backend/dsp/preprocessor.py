"""
Stage 1: Preprocessing and Raw Audio Cleaning (تجهيز وتنقية الصوت الخام)
- Unified sample rate to 44.1kHz or 48kHz (32-bit float).
- Mono to Stereo conversion.
- Peak normalization to -3.0 dBFS.
- Optional cleaning for raw user recitation (80Hz Highpass + NoiseGate).
"""
import os
import numpy as np
import soundfile as sf
import scipy.signal

try:
    import pedalboard
    from pedalboard import Pedalboard, HighpassFilter, NoiseGate
    HAS_PEDALBOARD = True
except Exception:
    HAS_PEDALBOARD = False


class Preprocessor:
    def __init__(self, target_sr: int = 44100, target_peak_db: float = -3.0):
        self.target_sr = target_sr
        self.target_peak_db = target_peak_db
        self.target_peak_linear = 10.0 ** (target_peak_db / 20.0)  # ~0.70794

    def load_audio(self, file_path_or_bytes, clean: bool = False, is_dry_user: bool = False) -> tuple[np.ndarray, int]:
        """
        Loads audio from path, converts to 32-bit float stereo (2, N) at target_sr.
        Optionally applies Highpass and NoiseGate if clean=True.
        """
        data, sr = sf.read(file_path_or_bytes, dtype='float32', always_2d=False)
        
        # Format shape to (channels, samples)
        if data.ndim == 1:
            # Mono to Stereo
            audio = np.stack([data, data], axis=0)
        elif data.ndim == 2:
            if data.shape[1] == 2 and data.shape[0] != 2:
                # Shape (N, 2) -> (2, N)
                audio = data.T
            elif data.shape[0] == 2:
                audio = data
            elif data.shape[1] == 1:
                mono = data[:, 0]
                audio = np.stack([mono, mono], axis=0)
            elif data.shape[0] == 1:
                mono = data[0, :]
                audio = np.stack([mono, mono], axis=0)
            else:
                # Downmix or take first 2 channels
                audio = data[:2, :]
        else:
            raise ValueError(f"Unsupported audio dimension: {data.ndim}")

        # Ensure float32
        audio = audio.astype(np.float32)

        # Resample if needed
        if sr != self.target_sr:
            audio = self.resample_audio(audio, sr, self.target_sr)
            sr = self.target_sr

        # Clean dry user audio if requested
        if clean and is_dry_user:
            audio = self.clean_audio(audio, sr)

        # Peak Normalize to -3.0 dBFS
        audio = self.normalize_peak(audio, self.target_peak_db)

        return audio, sr

    def resample_audio(self, audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """High-quality polyphase resampling for stereo audio (2, N)"""
        gcd = np.gcd(orig_sr, target_sr)
        up = target_sr // gcd
        down = orig_sr // gcd
        resampled_left = scipy.signal.resample_poly(audio[0], up, down).astype(np.float32)
        resampled_right = scipy.signal.resample_poly(audio[1], up, down).astype(np.float32)
        return np.stack([resampled_left, resampled_right], axis=0)

    def normalize_peak(self, audio: np.ndarray, target_db: float = -3.0) -> np.ndarray:
        """Normalizes audio to target_db peak."""
        current_peak = np.max(np.abs(audio))
        if current_peak <= 1e-9:
            return audio
        target_amp = 10.0 ** (target_db / 20.0)
        gain = target_amp / current_peak
        return audio * gain

    def clean_audio(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Cleans dry vocal recording:
        - 80Hz Highpass filter to eliminate low rumble/mic bumps.
        - Mild NoiseGate to silence breath & ambient floor between recitation phrases.
        """
        if HAS_PEDALBOARD:
            try:
                board = Pedalboard([
                    HighpassFilter(cutoff_frequency_hz=80.0),
                    NoiseGate(threshold_db=-46.0, ratio=3.0, attack_ms=10.0, release_ms=100.0)
                ])
                # pedalboard expects shape (channels, samples)
                cleaned = board(audio, sr)
                return cleaned.astype(np.float32)
            except Exception:
                pass

        # Robust DSP Fallback if pedalboard is not available
        return self._scipy_clean(audio, sr)

    def _scipy_clean(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """DSP fallback using SciPy Butterworth filter & soft noise gating."""
        # 4th order highpass Butterworth at 80 Hz
        sos = scipy.signal.butter(4, 80.0, 'hp', fs=sr, output='sos')
        filtered = np.zeros_like(audio)
        filtered[0] = scipy.signal.sosfilt(sos, audio[0])
        filtered[1] = scipy.signal.sosfilt(sos, audio[1])

        # Soft Noise Gate
        gate_threshold_linear = 10.0 ** (-46.0 / 20.0)  # ~0.005
        # Compute RMS envelope with 20ms window
        window_len = int(sr * 0.02)
        if window_len > 0:
            env_kernel = np.ones(window_len) / window_len
            mono_power = (filtered[0]**2 + filtered[1]**2) / 2.0
            env = np.sqrt(np.convolve(mono_power, env_kernel, mode='same') + 1e-12)
            # Smooth gain curve
            gain = np.clip((env - gate_threshold_linear * 0.5) / (gate_threshold_linear * 0.5 + 1e-9), 0.05, 1.0)
            # Smooth gain changes
            smooth_kernel = np.ones(int(sr * 0.015)) / int(sr * 0.015)
            gain = np.convolve(gain, smooth_kernel, mode='same')
            filtered[0] *= gain
            filtered[1] *= gain

        return filtered.astype(np.float32)

    def save_audio(self, audio: np.ndarray, file_path: str, sr: int = None, bit_depth: str = "PCM_24"):
        """Saves stereo audio (2, N) to WAV file with 24-bit or 32-bit depth."""
        if sr is None:
            sr = self.target_sr
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        # Transpose (2, N) to (N, 2) for soundfile
        audio_out = np.clip(audio.T, -1.0, 1.0)
        subtype = bit_depth if bit_depth in ['PCM_24', 'PCM_16', 'FLOAT'] else 'PCM_24'
        sf.write(file_path, audio_out, sr, subtype=subtype)
        return file_path

    @staticmethod
    def get_audio_stats(audio: np.ndarray, sr: int) -> dict:
        """Returns acoustic metrics of the audio array."""
        peak = float(np.max(np.abs(audio)))
        peak_db = 20.0 * np.log10(peak + 1e-9)
        rms = float(np.sqrt(np.mean(audio ** 2)))
        rms_db = 20.0 * np.log10(rms + 1e-9)
        duration = float(audio.shape[1] / sr) if audio.ndim > 1 else 0.0
        return {
            "peak": peak,
            "peak_db": round(peak_db, 2),
            "rms": rms,
            "rms_db": round(rms_db, 2),
            "duration_sec": round(duration, 2),
            "channels": int(audio.shape[0]),
            "sample_rate": sr
        }
