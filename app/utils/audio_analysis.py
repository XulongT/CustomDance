"""Audio signal features and deterministic timing; no local intent semantics."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

import librosa
import numpy as np

from app.schemas.analysis import (
    LocalAudioFeatures,
    TimedFeatureCurve,
)

@dataclass(frozen=True)
class TimingSlot:
    slot_id: str
    start_sec: float
    duration_sec: float = 4.0


@dataclass(frozen=True)
class TimingSegment:
    segment_id: str
    start_sec: float
    end_sec: float


@dataclass(frozen=True)
class TimingPlan:
    duration_sec: float
    segments: list[TimingSegment]
    slots: list[TimingSlot]


SUPPORTED_AUDIO_SUFFIXES = {".wav"}
SLOT_DURATION_SEC = 4.0
_MAX_START_SPACING_SEC = SLOT_DURATION_SEC + 1.0


def _finite_list(values: np.ndarray) -> list[float]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if not np.isfinite(array).all():
        raise ValueError("audio analysis produced NaN or infinite values")
    return [float(value) for value in array]


def analyze_audio_local(audio_path: Path, *, sample_rate: int = 22_050) -> LocalAudioFeatures:
    """Extract reproducible local signal features without any network request."""

    path = Path(audio_path).expanduser().resolve(strict=True)
    if path.suffix.lower() not in SUPPORTED_AUDIO_SUFFIXES:
        raise ValueError(f"unsupported audio type: {path.suffix}")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    samples, actual_rate = librosa.load(path, sr=sample_rate, mono=True)
    samples = np.asarray(samples, dtype=np.float32)
    if samples.size == 0 or not np.isfinite(samples).all():
        raise ValueError("audio is empty or contains non-finite samples")
    duration = float(len(samples) / actual_rate)

    hop_length = 512
    onset_values = librosa.onset.onset_strength(y=samples, sr=actual_rate, hop_length=hop_length)
    onset_times = librosa.frames_to_time(
        np.arange(len(onset_values)), sr=actual_rate, hop_length=hop_length
    )
    tempo_value, beat_frames = librosa.beat.beat_track(
        onset_envelope=onset_values, sr=actual_rate, hop_length=hop_length
    )
    tempo = float(np.asarray(tempo_value).reshape(-1)[0])
    beat_times = librosa.frames_to_time(beat_frames, sr=actual_rate, hop_length=hop_length)

    rms_values = librosa.feature.rms(y=samples, frame_length=2048, hop_length=hop_length)[0]
    rms_times = librosa.frames_to_time(
        np.arange(len(rms_values)), sr=actual_rate, hop_length=hop_length
    )
    chroma = librosa.feature.chroma_stft(y=samples, sr=actual_rate, hop_length=hop_length)
    chroma_mean = chroma.mean(axis=1)

    boundary_candidates: list[float] = [0.0]
    if duration > 8.0:
        window_count = int(np.floor(duration / 4.0))
        window_energy = []
        for index in range(window_count):
            start, end = index * 4.0, min((index + 1) * 4.0, duration)
            mask = (rms_times >= start) & (rms_times < end)
            window_energy.append(float(rms_values[mask].mean()) if mask.any() else 0.0)
        if len(window_energy) >= 2:
            changes = np.abs(np.diff(window_energy))
            threshold = float(np.median(changes) + np.median(np.abs(changes - np.median(changes))))
            boundary_candidates.extend(
                float((index + 1) * 4.0)
                for index, change in enumerate(changes)
                if change >= threshold and (index + 1) * 4.0 < duration
            )
    boundary_candidates.append(duration)

    return LocalAudioFeatures(
        duration_sec=duration,
        sample_rate=int(actual_rate),
        tempo_bpm=max(0.0, tempo),
        beat_times_sec=_finite_list(beat_times),
        onset=TimedFeatureCurve(
            times_sec=_finite_list(onset_times), values=_finite_list(onset_values)
        ),
        rms=TimedFeatureCurve(times_sec=_finite_list(rms_times), values=_finite_list(rms_values)),
        chroma_mean=_finite_list(chroma_mean),
        section_boundaries_sec=sorted(set(_finite_list(np.asarray(boundary_candidates)))),
    )


def _window_mean(
    times: np.ndarray,
    values: np.ndarray,
    start: float,
    end: float,
) -> float:
    mask = (times >= start) & (times < end)
    return float(values[mask].mean()) if mask.any() else 0.0


def _robust_unit_scale(values: np.ndarray) -> np.ndarray:
    """Scale one salience component without letting one outlier flatten the rest."""

    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return array
    low, high = np.quantile(array, [0.1, 0.9])
    if high - low <= 1e-12:
        return np.zeros_like(array)
    return np.clip((array - low) / (high - low), 0.0, 1.0)


def _dense_slot_grid(duration: float, *, step_sec: float = 0.5) -> np.ndarray:
    max_start = duration - SLOT_DURATION_SEC
    starts = np.arange(0.0, max_start + 1e-9, step_sec, dtype=np.float64)
    if starts.size == 0 or max_start - starts[-1] > 1e-7:
        starts = np.append(starts, max_start)
    return starts


def _balanced_slot_count(duration: float) -> int:
    """Choose enough slots to keep internal gaps bounded and cover the song."""

    max_start = duration - SLOT_DURATION_SEC
    if max_start <= 1e-9:
        return 1

    # A 250 ms tolerance prevents millisecond decode tails from adding a full
    # extra motif slot at an otherwise exact musical boundary.
    coverage_span = max(0.0, duration - _MAX_START_SPACING_SEC - 0.25)
    count = int(np.ceil(coverage_span / _MAX_START_SPACING_SEC)) + 1

    # A second non-overlapping four-second slot cannot fit in every short tail.
    # Reduce the count rather than violating the no-overlap contract.
    while count > 1 and max_start + 1e-9 < SLOT_DURATION_SEC * (count - 1):
        count -= 1
    return max(1, count)


def _select_balanced_starts(
    grid_starts: np.ndarray,
    salience: np.ndarray,
    *,
    duration: float,
) -> list[float]:
    """Select salient anchors under hard opening, overlap, and gap constraints."""

    count = _balanced_slot_count(duration)
    max_start = duration - SLOT_DURATION_SEC
    if count == 1:
        return [0.0]

    maximum_start_spacing = _MAX_START_SPACING_SEC
    planning_span = min(max_start, (count - 1) * maximum_start_spacing)
    targets = np.linspace(0.0, planning_span, count)
    candidate_indices: list[list[int]] = []
    for target_index in range(count):
        remaining = count - target_index - 1
        lower = max(
            target_index * SLOT_DURATION_SEC,
            planning_span - remaining * maximum_start_spacing,
        )
        upper = min(
            target_index * maximum_start_spacing,
            planning_span - remaining * SLOT_DURATION_SEC,
        )
        indices = np.flatnonzero(
            (grid_starts >= lower - 1e-9) & (grid_starts <= upper + 1e-9)
        ).tolist()
        candidate_indices.append(indices)

    # Dynamic programming keeps the result deterministic while avoiding a
    # greedy early peak that would make later coverage impossible.
    states: dict[int, tuple[float, tuple[int, ...]]] = {}
    for index in candidate_indices[0]:
        if abs(float(grid_starts[index])) > 1e-9:
            continue
        penalty = (
            0.2 * abs(float(grid_starts[index] - targets[0])) / max(1.0, maximum_start_spacing)
        )
        states[index] = (float(salience[index] - penalty), (index,))
    for target_index in range(1, count):
        next_states: dict[int, tuple[float, tuple[int, ...]]] = {}
        target = float(targets[target_index])
        for index in candidate_indices[target_index]:
            start = float(grid_starts[index])
            penalty = 0.2 * abs(start - target) / max(1.0, maximum_start_spacing)
            best: tuple[float, tuple[int, ...]] | None = None
            for previous_index, (score, path) in states.items():
                spacing = start - float(grid_starts[previous_index])
                if spacing < SLOT_DURATION_SEC - 1e-9 or spacing > maximum_start_spacing + 1e-9:
                    continue
                proposal = (score + float(salience[index]) - penalty, path + (index,))
                if (
                    best is None
                    or proposal[0] > best[0] + 1e-12
                    or (abs(proposal[0] - best[0]) <= 1e-12 and proposal[1] < best[1])
                ):
                    best = proposal
            if best is not None:
                next_states[index] = best
        states = next_states
        if not states:
            break

    if not states:
        raise RuntimeError("could not construct a non-overlapping slot plan")
    _, path = max(
        states.values(),
        key=lambda item: (round(item[0], 12), tuple(-value for value in item[1])),
    )
    starts = [float(grid_starts[index]) for index in path]
    spacings = np.diff(starts)
    if (
        abs(starts[0]) > 1e-9
        or np.any(spacings < SLOT_DURATION_SEC - 1e-9)
        or np.any(spacings > maximum_start_spacing + 1e-9)
    ):
        raise RuntimeError("slot selector violated timing constraints")
    return starts


def build_timing_plan(features: LocalAudioFeatures) -> TimingPlan:
    """Create a deterministic offline Analyze result from local signal features."""

    duration = features.duration_sec
    if duration < SLOT_DURATION_SEC:
        raise ValueError("audio must be at least 4 seconds to create a fixed-length slot")
    rms_times = np.asarray(features.rms.times_sec, dtype=np.float64)
    rms_values = np.asarray(features.rms.values, dtype=np.float64)
    onset_times = np.asarray(features.onset.times_sec, dtype=np.float64)
    onset_values = np.asarray(features.onset.values, dtype=np.float64)
    grid_starts = _dense_slot_grid(duration)

    rms_scores = np.asarray(
        [
            _window_mean(rms_times, rms_values, start, start + SLOT_DURATION_SEC)
            for start in grid_starts
        ]
    )
    onset_scores = np.asarray(
        [
            _window_mean(onset_times, onset_values, start, start + SLOT_DURATION_SEC)
            for start in grid_starts
        ]
    )
    change_scores = np.asarray(
        [
            abs(
                _window_mean(rms_times, rms_values, start, min(duration, start + 1.0))
                - _window_mean(rms_times, rms_values, max(0.0, start - 1.0), start)
            )
            if start > 1e-9
            else 0.0
            for start in grid_starts
        ]
    )
    internal_boundaries = np.asarray(
        [
            boundary
            for boundary in features.section_boundaries_sec
            if 0.25 < boundary < duration - 0.25
        ],
        dtype=np.float64,
    )
    boundary_scores = np.asarray(
        [
            float(np.exp(-np.min(np.abs(internal_boundaries - start)) / 1.5))
            if internal_boundaries.size
            else 0.0
            for start in grid_starts
        ]
    )
    beats = np.asarray(features.beat_times_sec, dtype=np.float64)
    beat_scores = np.asarray(
        [
            float(np.exp(-np.min(np.abs(beats - start)) / 0.15)) if beats.size else 0.0
            for start in grid_starts
        ]
    )
    salience = (
        0.20 * _robust_unit_scale(rms_scores)
        + 0.25 * _robust_unit_scale(onset_scores)
        + 0.30 * _robust_unit_scale(change_scores)
        + 0.20 * _robust_unit_scale(boundary_scores)
        + 0.05 * _robust_unit_scale(beat_scores)
    )
    starts = _select_balanced_starts(
        grid_starts,
        salience,
        duration=duration,
    )

    slots = [
        TimingSlot(slot_id=f"slot-{index + 1:02d}", start_sec=float(start))
        for index, start in enumerate(starts)
    ]
    boundaries = sorted(set([0.0, duration, *features.section_boundaries_sec]))
    segments = [
        TimingSegment(segment_id=f"segment-{index + 1:02d}", start_sec=start, end_sec=end)
        for index, (start, end) in enumerate(pairwise(boundaries))
        if end - start > 1e-6
    ]
    return TimingPlan(duration_sec=duration, segments=segments, slots=slots)
