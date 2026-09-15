"""Dependency-light loading of 35D music features for CLI queries."""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np


def crop_or_pad(features: np.ndarray, frames: int = 120, start_frame: int = 0) -> np.ndarray:
    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2 or features.shape[1] != 35:
        raise ValueError(f"music features must have shape (T,35), got {features.shape}")
    start = max(0, int(start_frame))
    window = features[start : start + int(frames)]
    if window.shape[0] == 0:
        return np.zeros((frames, 35), dtype=np.float32)
    if window.shape[0] < frames:
        pad = np.repeat(window[-1:, :], frames - window.shape[0], axis=0)
        window = np.concatenate([window, pad], axis=0)
    return window.astype(np.float32, copy=False)


def _decode_wave(path: Path, start_sec: float, duration_sec: float) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        sample_width = handle.getsampwidth()
        sample_rate = handle.getframerate()
        frames = handle.readframes(handle.getnframes())
    if sample_width == 1:
        signal = np.frombuffer(frames, dtype=np.uint8).astype(np.float32)
        signal = (signal - 128.0) / 128.0
    elif sample_width == 2:
        signal = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    elif sample_width == 4:
        signal = np.frombuffer(frames, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"unsupported WAV sample width {sample_width} in {path}")
    if channels > 1:
        signal = signal.reshape(-1, channels).mean(axis=1)
    begin = max(0, int(float(start_sec) * sample_rate))
    length = max(1, int(float(duration_sec) * sample_rate))
    signal = signal[begin : begin + length]
    if signal.size == 0:
        signal = np.zeros(length, dtype=np.float32)
    if signal.size < length:
        signal = np.pad(signal, (0, length - signal.size))
    return signal.astype(np.float32), int(sample_rate)


def _resample(signal: np.ndarray, source_rate: int, target_rate: int = 22050) -> np.ndarray:
    if source_rate == target_rate:
        return signal
    target_length = max(1, int(round(signal.shape[0] * target_rate / source_rate)))
    old_x = np.linspace(0.0, 1.0, num=signal.shape[0], endpoint=False)
    new_x = np.linspace(0.0, 1.0, num=target_length, endpoint=False)
    return np.interp(new_x, old_x, signal).astype(np.float32)


def wav_to_features(
    path: str | Path,
    start_sec: float = 0.0,
    duration_sec: float = 4.0,
    frames: int = 120,
) -> np.ndarray:
    """Make a finite 35D/frame fallback feature matrix from a WAV segment.

    FineDance native music_npy input remains the preferred exact-feature path.
    WAV support is a self-contained operational path for Lite users and is
    not claimed as a benchmark-equivalent feature extractor.
    """

    signal, sample_rate = _decode_wave(Path(path), start_sec, duration_sec)
    signal = _resample(signal, sample_rate)
    n_fft = 1024
    window = np.hanning(n_fft).astype(np.float32)
    total = signal.shape[0]
    centers = np.linspace(0, max(0, total - 1), num=frames).astype(np.int64)
    output = np.zeros((frames, 35), dtype=np.float32)
    previous = None
    for index, center in enumerate(centers):
        left = max(0, int(center) - n_fft // 2)
        chunk = signal[left : left + n_fft]
        if chunk.shape[0] < n_fft:
            chunk = np.pad(chunk, (0, n_fft - chunk.shape[0]))
        chunk = chunk * window
        spectrum = np.abs(np.fft.rfft(chunk)).astype(np.float32)
        power = np.square(spectrum) + 1.0e-8
        freqs = np.linspace(0.0, 11025.0, num=power.shape[0], dtype=np.float32)
        total_power = float(power.sum())
        rms = float(np.sqrt(np.mean(np.square(chunk)) + 1.0e-8))
        zcr = float(np.mean(np.abs(np.diff(np.signbit(chunk).astype(np.int8)))))
        centroid = float((power * freqs).sum() / total_power)
        bandwidth = float(np.sqrt((power * np.square(freqs - centroid)).sum() / total_power))
        cumulative = np.cumsum(power)
        rolloff_idx = int(np.searchsorted(cumulative, 0.85 * total_power))
        rolloff = float(freqs[min(rolloff_idx, freqs.shape[0] - 1)])
        extras = [rms, zcr, centroid / 11025.0, bandwidth / 11025.0, rolloff / 11025.0]
        bands = np.array_split(power[1:], 30)
        band_values = [float(np.log1p(np.mean(band))) for band in bands]
        current = np.asarray(extras + band_values, dtype=np.float32)
        if previous is not None:
            current[4] = float(np.mean(np.abs(spectrum - previous)))
        previous = spectrum
        output[index] = current
    output[~np.isfinite(output)] = 0.0
    return output


def load_music(path: str | Path, start_frame: int = 0, start_sec: float = 0.0) -> np.ndarray:
    path = Path(path)
    if path.suffix.lower() == ".npy":
        array = np.load(path, allow_pickle=False)
        return crop_or_pad(array, frames=120, start_frame=start_frame)
    if path.suffix.lower() == ".wav":
        return wav_to_features(path, start_sec=start_sec, frames=120)
    raise ValueError(f"music input must be .npy or .wav, got {path}")
