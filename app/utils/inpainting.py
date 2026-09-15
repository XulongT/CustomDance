"""Shared inpainting backend for the Completer and Remaker roles."""

from __future__ import annotations

import threading
from copy import deepcopy
from pathlib import Path
from time import perf_counter
from typing import Any

import librosa
import numpy as np
import torch
from torch import nn

from app.config import GENRE_NAMES
from app.schemas.motion import CanonicalMotion, CompleterInternalMotion
from app.utils.foot_contact import FOOT_CHANNEL_JOINTS, contact_aware_root_lock
from app.utils.kinematics import (
    axis_angle_to_matrix,
    forward_kinematics,
    matrix_to_axis_angle,
)
from app.utils.motion_adapters import canonical_to_completer, completer_to_canonical

COMPLETER_FPS = 30
COMPLETER_MAX_FRAMES = 1200
COMPLETER_MAX_SECONDS = COMPLETER_MAX_FRAMES / COMPLETER_FPS
COMPLETER_CONTEXT_FRAMES = 120
FOOT_SKATING_RETRY_THRESHOLD_M = 0.12
FOOT_SKATING_RETRY_SEED_OFFSET = 1009
GENERATION_QUALITY_RETRY_THRESHOLD = 1.0
LEG_JOINTS = np.asarray((1, 2, 4, 5, 7, 8, 10, 11), dtype=np.int64)


def clone_canonical_motion(motion: CanonicalMotion) -> CanonicalMotion:
    """Return an independent validated copy suitable for background inference."""

    return CanonicalMotion(
        smpl_poses=motion.smpl_poses.copy(),
        smpl_trans=motion.smpl_trans.copy(),
        joint_positions=motion.joint_positions.copy(),
        source_clip_id=motion.source_clip_id,
        coordinate_system=motion.coordinate_system,
        fps=motion.fps,
        schema_version=motion.schema_version,
        metadata=deepcopy(motion.metadata),
    )


def _slice_canonical_motion(
    motion: CanonicalMotion, start_frame: int, end_frame: int
) -> CanonicalMotion:
    if not 0 <= start_frame < end_frame <= motion.frames:
        raise ValueError("motion slice must be a non-empty interval")
    return CanonicalMotion(
        smpl_poses=motion.smpl_poses[start_frame:end_frame].copy(),
        smpl_trans=motion.smpl_trans[start_frame:end_frame].copy(),
        joint_positions=motion.joint_positions[start_frame:end_frame].copy(),
        source_clip_id=f"{motion.source_clip_id}-frames-{start_frame}-{end_frame}",
        coordinate_system=motion.coordinate_system,
        fps=motion.fps,
        schema_version=motion.schema_version,
        metadata=deepcopy(motion.metadata),
    )


def _smoothstep(value: float) -> float:
    clipped = min(1.0, max(0.0, float(value)))
    return clipped * clipped * (3.0 - 2.0 * clipped)


def _scaled_rotation_step(step: np.ndarray, multiplier: float) -> np.ndarray:
    """Return ``Exp(multiplier * Log(step))`` for a batch of rotation matrices."""

    vectors = matrix_to_axis_angle(step)
    return axis_angle_to_matrix(vectors * float(multiplier))


def _interpolate_rotations(
    source: np.ndarray, target: np.ndarray, weight: float
) -> np.ndarray:
    """Geodesically interpolate batched SO(3) matrices from source to target."""

    if weight <= 0:
        return source
    if weight >= 1:
        return target
    relative = np.einsum("...ji,...jk->...ik", source, target)
    correction = _scaled_rotation_step(relative, weight)
    return np.einsum("...ij,...jk->...ik", source, correction)


def _unknown_ranges(known_mask: np.ndarray) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    start: int | None = None
    for index, known in enumerate(np.asarray(known_mask, dtype=bool)):
        if not known and start is None:
            start = index
        elif known and start is not None:
            ranges.append((start, index))
            start = None
    if start is not None:
        ranges.append((start, len(known_mask)))
    return ranges


def _robust_high_indices(
    values: np.ndarray, *, minimum: float, threshold: float = 6.0
) -> np.ndarray:
    """Return conservative high-side Hampel outliers above an absolute floor."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        return np.empty(0, dtype=np.int64)
    median = float(np.median(array))
    deviations = np.abs(array - median)
    mad = float(np.median(deviations))
    scale = 1.482602218505602 * mad
    if scale < 1e-6:
        scale = max(float(np.quantile(deviations, 0.75)), 1e-6)
    scores = (array - median) / scale
    return np.flatnonzero((array >= minimum) & (scores >= threshold))


def _group_indices(indices: np.ndarray, *, max_gap: int = 2) -> list[tuple[int, int]]:
    ordered = np.unique(np.asarray(indices, dtype=np.int64))
    if ordered.size == 0:
        return []
    groups: list[tuple[int, int]] = []
    start = previous = int(ordered[0])
    for value in ordered[1:]:
        current = int(value)
        if current - previous > max_gap:
            groups.append((start, previous))
            start = current
        previous = current
    groups.append((start, previous))
    return groups


def _hermite_vector(values: np.ndarray, left: int, right: int, frame: int) -> np.ndarray:
    """Endpoint-preserving cubic interpolation adapted from the supplied fixer."""

    if right <= left:
        return values[left].copy()
    interval = float(right - left)
    t = (frame - left) / interval
    t2 = t * t
    t3 = t2 * t
    start = values[left]
    finish = values[right]
    start_velocity = (
        values[left] - values[left - 1]
        if left > 0
        else (finish - start) / interval
    )
    finish_velocity = (
        values[right + 1] - values[right]
        if right + 1 < values.shape[0]
        else (finish - start) / interval
    )
    return (
        (2.0 * t3 - 3.0 * t2 + 1.0) * start
        + (t3 - 2.0 * t2 + t) * interval * start_velocity
        + (-2.0 * t3 + 3.0 * t2) * finish
        + (t3 - t2) * interval * finish_velocity
    )


def _boundary_targets(
    reference: CanonicalMotion,
    reference_matrices: np.ndarray,
    mask: np.ndarray,
    start: int,
    end: int,
) -> tuple[tuple[np.ndarray, np.ndarray] | None, tuple[np.ndarray, np.ndarray] | None]:
    """Return one-step root translation/rotation targets at both gap boundaries."""

    left = start - 1 if start > 0 and mask[start - 1] else None
    right = end if end < reference.frames and mask[end] else None
    left_target = None
    if left is not None:
        translation_step = (
            reference.smpl_trans[left] - reference.smpl_trans[left - 1]
            if left > 0 and mask[left - 1]
            else np.zeros(3, dtype=np.float64)
        )
        rotation_step = (
            np.einsum(
                "...ji,...jk->...ik",
                reference_matrices[left - 1],
                reference_matrices[left],
            )
            if left > 0 and mask[left - 1]
            else np.broadcast_to(np.eye(3), (24, 3, 3)).copy()
        )
        left_target = (
            reference.smpl_trans[left].astype(np.float64) + translation_step,
            np.einsum("...ij,...jk->...ik", reference_matrices[left], rotation_step),
        )
    right_target = None
    if right is not None:
        translation_step = (
            reference.smpl_trans[right + 1] - reference.smpl_trans[right]
            if right + 1 < reference.frames and mask[right + 1]
            else np.zeros(3, dtype=np.float64)
        )
        rotation_step = (
            np.einsum(
                "...ji,...jk->...ik",
                reference_matrices[right],
                reference_matrices[right + 1],
            )
            if right + 1 < reference.frames and mask[right + 1]
            else np.broadcast_to(np.eye(3), (24, 3, 3)).copy()
        )
        right_target = (
            reference.smpl_trans[right].astype(np.float64) - translation_step,
            np.einsum(
                "...ij,...jk->...ik",
                reference_matrices[right],
                _scaled_rotation_step(rotation_step, -1),
            ),
        )
    return left_target, right_target


def _align_generated_root_range(
    translations: np.ndarray,
    matrices: np.ndarray,
    start: int,
    end: int,
    left_target: tuple[np.ndarray, np.ndarray] | None,
    right_target: tuple[np.ndarray, np.ndarray] | None,
) -> None:
    """Warp root heading and position across the whole gap to both anchors.

    The correction itself follows the shortest SO(3) arc. This removes the
    common failure where an otherwise usable generated segment is globally
    facing backward and is forced through a 180-degree turn near each edge.
    """

    length = end - start
    if length <= 0 or (left_target is None and right_target is None):
        return
    if length == 1 and left_target is not None and right_target is not None:
        translations[start] = 0.5 * (left_target[0] + right_target[0])
        matrices[start, 0] = _interpolate_rotations(
            left_target[1][0], right_target[1][0], 0.5
        )
        return

    left_offset = (
        left_target[0] - translations[start] if left_target is not None else None
    )
    right_offset = (
        right_target[0] - translations[end - 1] if right_target is not None else None
    )
    left_correction = (
        left_target[1][0] @ matrices[start, 0].T if left_target is not None else None
    )
    right_correction = (
        right_target[1][0] @ matrices[end - 1, 0].T
        if right_target is not None
        else None
    )
    if left_offset is None:
        left_offset = right_offset
        left_correction = right_correction
    if right_offset is None:
        right_offset = left_offset
        right_correction = left_correction
    assert left_offset is not None and right_offset is not None
    assert left_correction is not None and right_correction is not None

    for offset, frame in enumerate(range(start, end)):
        weight = 0.0 if length == 1 else offset / (length - 1)
        translations[frame] += left_offset * (1.0 - weight) + right_offset * weight
        correction = _interpolate_rotations(
            left_correction, right_correction, weight
        )
        matrices[frame, 0] = correction @ matrices[frame, 0]


def _repair_generated_translation_spikes(
    translations: np.ndarray, start: int, end: int
) -> int:
    candidate_frames = np.arange(start + 1, end - 1, dtype=np.int64)
    if candidate_frames.size < 3:
        return 0
    midpoint = 0.5 * (
        translations[candidate_frames - 1] + translations[candidate_frames + 1]
    )
    deviations = np.linalg.norm(translations[candidate_frames] - midpoint, axis=-1)
    selected = _robust_high_indices(deviations, minimum=0.08, threshold=8.0)
    repaired: set[int] = set()
    for first, last in _group_indices(candidate_frames[selected], max_gap=2):
        repair_start = max(start + 1, first - 2)
        repair_end = min(end - 2, last + 2)
        left = repair_start - 1
        right = repair_end + 1
        source = translations.copy()
        for frame in range(repair_start, repair_end + 1):
            translations[frame] = _hermite_vector(source, left, right, frame)
            repaired.add(frame)
    return len(repaired)


def _translation_path_metrics(
    translations: np.ndarray, start: int, end: int
) -> dict[str, float]:
    left = max(0, start - 1)
    right = min(len(translations) - 1, end)
    xy = np.asarray(translations[left : right + 1, :2], dtype=np.float64)
    if xy.shape[0] < 2:
        return {
            "endpoint_distance_m": 0.0,
            "path_length_m": 0.0,
            "path_excess_m": 0.0,
            "max_step_m": 0.0,
            "max_acceleration_m": 0.0,
            "max_detour_m": 0.0,
            "max_overshoot_m": 0.0,
        }
    steps = np.diff(xy, axis=0)
    step_lengths = np.linalg.norm(steps, axis=1)
    accelerations = np.diff(steps, axis=0)
    displacement = xy[-1] - xy[0]
    direct = float(np.linalg.norm(displacement))
    path = float(np.sum(step_lengths))
    if direct > 1e-8:
        direction = displacement / direct
        projected = (xy - xy[0]) @ direction
        orthogonal = (xy - xy[0]) - projected[:, None] * direction
        detour = float(np.max(np.linalg.norm(orthogonal, axis=1), initial=0.0))
        overshoot = float(
            max(
                0.0,
                -float(np.min(projected, initial=0.0)),
                float(np.max(projected, initial=0.0)) - direct,
            )
        )
    else:
        detour = float(np.max(np.linalg.norm(xy - xy[0], axis=1), initial=0.0))
        overshoot = detour
    return {
        "endpoint_distance_m": direct,
        "path_length_m": path,
        "path_excess_m": max(0.0, path - direct),
        "max_step_m": float(np.max(step_lengths, initial=0.0)),
        "max_acceleration_m": float(
            np.max(np.linalg.norm(accelerations, axis=1), initial=0.0)
        ),
        "max_detour_m": detour,
        "max_overshoot_m": overshoot,
    }


def _bounded_xy_step(value: np.ndarray, maximum_m: float = 0.04) -> np.ndarray:
    step = np.asarray(value, dtype=np.float64)
    distance = float(np.linalg.norm(step))
    return step if distance <= maximum_m else step * (maximum_m / distance)


def _minimum_jerk_root_xy(
    translations: np.ndarray,
    reference: CanonicalMotion,
    mask: np.ndarray,
    start: int,
    end: int,
) -> np.ndarray:
    length = end - start
    source = np.asarray(translations[start:end, :2], dtype=np.float64)
    if length <= 2:
        return source.copy()
    left = start - 1 if start > 0 and mask[start - 1] else None
    right = end if end < reference.frames and mask[end] else None
    if left is not None and right is not None:
        first = np.asarray(reference.smpl_trans[left, :2], dtype=np.float64)
        last = np.asarray(reference.smpl_trans[right, :2], dtype=np.float64)
        start_velocity = (
            first - reference.smpl_trans[left - 1, :2]
            if left > 0 and mask[left - 1]
            else (last - first) / (length + 1)
        )
        finish_velocity = (
            reference.smpl_trans[right + 1, :2] - last
            if right + 1 < reference.frames and mask[right + 1]
            else (last - first) / (length + 1)
        )
        start_velocity = _bounded_xy_step(start_velocity)
        finish_velocity = _bounded_xy_step(finish_velocity)
        interval = float(length + 1)
        result = np.empty_like(source)
        offsets = range(1, length + 1)
    else:
        first = source[0]
        last = source[-1]
        start_velocity = (
            first - reference.smpl_trans[left, :2]
            if left is not None
            else (last - first) / (length - 1)
        )
        finish_velocity = (
            reference.smpl_trans[right, :2] - last
            if right is not None
            else (last - first) / (length - 1)
        )
        start_velocity = _bounded_xy_step(start_velocity)
        finish_velocity = _bounded_xy_step(finish_velocity)
        interval = float(length - 1)
        result = np.empty_like(source)
        offsets = range(length)
    for output_index, offset in enumerate(offsets):
        t = offset / interval
        t2 = t * t
        t3 = t2 * t
        result[output_index] = (
            (2.0 * t3 - 3.0 * t2 + 1.0) * first
            + (t3 - 2.0 * t2 + t) * interval * start_velocity
            + (-2.0 * t3 + 3.0 * t2) * last
            + (t3 - t2) * interval * finish_velocity
        )
    if left is None or right is None:
        result[0] = first
        result[-1] = last
    return result


def _smooth_xy_residual(values: np.ndarray) -> np.ndarray:
    residual = np.asarray(values, dtype=np.float64).copy()
    if residual.shape[0] < 5:
        return residual
    kernel = np.asarray((1.0, 2.0, 3.0, 2.0, 1.0), dtype=np.float64) / 9.0
    for _ in range(2):
        padded = np.pad(residual, ((2, 2), (0, 0)), mode="edge")
        residual = np.stack(
            [np.convolve(padded[:, axis], kernel, mode="valid") for axis in range(2)],
            axis=1,
        )
    return residual


def _stabilize_generated_translation_paths(
    motion: CanonicalMotion,
    reference: CanonicalMotion,
    known_mask: np.ndarray,
    ranges: list[tuple[int, int]],
) -> list[dict[str, Any]]:
    """Replace unsafe root detours before the final contact-aware lock."""

    mask = np.asarray(known_mask, dtype=bool)
    records: list[dict[str, Any]] = []
    changed = False
    for start, end in ranges:
        length = end - start
        if length < 3:
            continue
        before = _translation_path_metrics(motion.smpl_trans, start, end)
        path_excess_limit = max(0.15, 0.008 * length)
        reasons: list[str] = []
        if length <= 20:
            reasons.append("short_gap_endpoint_velocity_path")
        if before["max_step_m"] > 0.08:
            reasons.append("single_frame_root_step_over_8cm")
        if before["max_acceleration_m"] > 0.08:
            reasons.append("root_acceleration_over_8cm_per_frame2")
        if before["path_excess_m"] > path_excess_limit:
            reasons.append("root_path_excess")
        if before["max_detour_m"] > 0.15:
            reasons.append("root_path_outside_15cm_corridor")
        if not reasons:
            continue

        source = motion.smpl_trans[start:end, :2].astype(np.float64, copy=True)
        baseline = _minimum_jerk_root_xy(
            motion.smpl_trans, reference, mask, start, end
        )
        if length <= 20:
            candidate = baseline
            retained_residual = 0.0
            method = "short_gap_minimum_jerk_xy"
        else:
            retained_residual = 0.20 if length <= 45 else 0.35
            residual = _smooth_xy_residual(source - baseline)
            t = np.linspace(0.0, 1.0, length, dtype=np.float64)
            taper = np.sin(np.pi * t) ** 2
            residual_norm = np.linalg.norm(residual, axis=1)
            over_limit = residual_norm > 0.08
            residual[over_limit] *= (0.08 / residual_norm[over_limit])[:, None]
            candidate = baseline + retained_residual * residual * taper[:, None]
            candidate[0] = source[0]
            candidate[-1] = source[-1]
            method = "minimum_jerk_xy_with_bounded_low_frequency_residual"

        motion.smpl_trans[start:end, :2] = candidate.astype(np.float32)
        after = _translation_path_metrics(motion.smpl_trans, start, end)
        if (
            after["max_step_m"] > before["max_step_m"] + 1e-6
            or after["path_excess_m"] > before["path_excess_m"] + 1e-6
        ):
            motion.smpl_trans[start:end, :2] = source.astype(np.float32)
            after = before
            applied = False
        else:
            changed = True
            applied = True
        records.append(
            {
                "range": [start, end],
                "reasons": reasons,
                "before": before,
                "after": after,
                "applied": applied,
                "method": method,
                "retained_model_residual": retained_residual,
                "protected_known_endpoints": True,
            }
        )
    if changed:
        motion.joint_positions = forward_kinematics(
            motion.smpl_poses, motion.smpl_trans
        )
    return records


def _repair_generated_rotation_jumps(
    matrices: np.ndarray, start: int, end: int
) -> int:
    if end - start < 4:
        return 0
    repaired: set[tuple[int, int]] = set()
    for joint in range(24):
        pair_frames = np.arange(start + 1, end, dtype=np.int64)
        relative = np.einsum(
            "tji,tjk->tik",
            matrices[pair_frames - 1, joint],
            matrices[pair_frames, joint],
        )
        angles = np.linalg.norm(matrix_to_axis_angle(relative), axis=-1)
        minimum = 1.0 if joint == 0 else 1.35
        selected = _robust_high_indices(angles, minimum=minimum, threshold=6.0)
        for first, last in _group_indices(pair_frames[selected], max_gap=2):
            repair_start = max(start, first - 2)
            repair_end = min(end - 1, last + 2)
            left = repair_start - 1
            right = repair_end + 1
            if left < 0 or right >= matrices.shape[0]:
                continue
            source_left = matrices[left, joint].copy()
            source_right = matrices[right, joint].copy()
            interval = float(right - left)
            for frame in range(repair_start, repair_end + 1):
                matrices[frame, joint] = _interpolate_rotations(
                    source_left, source_right, (frame - left) / interval
                )
                repaired.add((frame, joint))
    return len(repaired)


def _root_path_metrics(matrices: np.ndarray, start: int, end: int) -> dict[str, float]:
    root = np.asarray(matrices[start:end, 0], dtype=np.float64)
    if root.shape[0] < 2:
        return {
            "total_degrees": 0.0,
            "endpoint_degrees": 0.0,
            "max_step_degrees": 0.0,
        }
    relative = np.einsum("tji,tjk->tik", root[:-1], root[1:])
    steps = np.linalg.norm(matrix_to_axis_angle(relative), axis=-1)
    endpoint = root[0].T @ root[-1]
    return {
        "total_degrees": float(np.degrees(np.sum(steps))),
        "endpoint_degrees": float(np.degrees(np.linalg.norm(matrix_to_axis_angle(endpoint)))),
        "max_step_degrees": float(np.degrees(np.max(steps))),
    }


def _stabilize_generated_root_path(
    matrices: np.ndarray, start: int, end: int
) -> dict[str, Any] | None:
    """Replace implausible long-way root turns with the shortest SO(3) path.

    Completer predicts continuous 6D rotations, but an unconstrained gap can
    still make a genuine 300-degree turn between anchors that are much closer.
    Preserve the already stitched first/last generated frames so boundary
    velocities remain exact, and only replace the unsafe interior path.
    """

    length = end - start
    if length < 3:
        return None
    before = _root_path_metrics(matrices, start, end)
    excess_path = before["total_degrees"] - before["endpoint_degrees"]
    reasons: list[str] = []
    if before["max_step_degrees"] > 30.0:
        reasons.append("single_frame_step_over_30_degrees")
    if excess_path > 120.0:
        reasons.append("path_over_shortest_arc_by_120_degrees")
    if length <= 30 and excess_path > 45.0:
        reasons.append("short_gap_long_way_turn")
    if not reasons:
        return None

    first = matrices[start, 0].copy()
    last = matrices[end - 1, 0].copy()
    for offset, frame in enumerate(range(start, end)):
        weight = _smoothstep(offset / (length - 1))
        matrices[frame, 0] = _interpolate_rotations(first, last, weight)
    return {
        "range": [start, end],
        "reasons": reasons,
        "before": before,
        "after": _root_path_metrics(matrices, start, end),
        "repair": "shortest_arc_so3_with_stitched_endpoints_preserved",
    }


def _lift_generated_foot_penetration(
    motion: CanonicalMotion,
    reference: CanonicalMotion,
    known_mask: np.ndarray,
    ranges: list[tuple[int, int]],
    *,
    context_frames: int = 60,
    edge_frames: int = 4,
) -> list[dict[str, Any]]:
    """Lift generated root Z where predicted feet pass through the known floor."""

    mask = np.asarray(known_mask, dtype=bool)
    corrections: list[dict[str, Any]] = []
    for start, end in ranges:
        context_start = max(0, start - context_frames)
        context_end = min(reference.frames, end + context_frames)
        context_indices = np.flatnonzero(mask[context_start:context_end]) + context_start
        if context_indices.size == 0:
            continue
        known_feet = reference.joint_positions[context_indices][:, [7, 8, 10, 11], 2]
        floor_z = float(np.quantile(known_feet, 0.05))
        generated_feet = motion.joint_positions[start:end, [7, 8, 10, 11], 2]
        lowest = np.min(generated_feet, axis=1)
        penetration = np.maximum(0.0, floor_z - lowest)
        weights = np.ones(end - start, dtype=np.float64)
        for offset in range(min(edge_frames, end - start)):
            weight = _smoothstep(offset / max(edge_frames, 1))
            weights[offset] = min(weights[offset], weight)
            weights[-1 - offset] = min(weights[-1 - offset], weight)
        applied = penetration * weights
        if float(np.max(applied, initial=0.0)) <= 1e-6:
            continue
        motion.smpl_trans[start:end, 2] += applied.astype(np.float32)
        corrections.append(
            {
                "range": [start, end],
                "known_floor_z": floor_z,
                "max_penetration_before_m": float(np.max(penetration)),
                "max_root_lift_m": float(np.max(applied)),
            }
        )

    if corrections:
        motion.joint_positions = forward_kinematics(motion.smpl_poses, motion.smpl_trans)
        for record in corrections:
            start, end = record["range"]
            lowest = np.min(motion.joint_positions[start:end, [7, 8, 10, 11], 2], axis=1)
            record["max_penetration_after_m"] = float(
                np.max(np.maximum(0.0, record["known_floor_z"] - lowest))
            )
    return corrections


def _final_contact_slip(
    motion: CanonicalMotion,
    contacts: np.ndarray,
    ranges: list[tuple[int, int]],
) -> float:
    payload = np.asarray(contacts, dtype=np.float64)
    if payload.shape != (motion.frames, 4) or not np.isfinite(payload).all():
        return 0.0
    feet = motion.joint_positions[:, FOOT_CHANNEL_JOINTS].astype(np.float64)
    maximum = 0.0
    for start, end in ranges:
        floor_z = float(np.quantile(feet[start:end, :, 2], 0.05))
        vertical = feet[:, :, 2]
        vertical_step = np.zeros_like(vertical)
        if motion.frames > 1:
            vertical_step[0] = np.abs(vertical[1] - vertical[0])
            vertical_step[-1] = np.abs(vertical[-1] - vertical[-2])
        if motion.frames > 2:
            vertical_step[1:-1] = 0.5 * np.abs(vertical[2:] - vertical[:-2])
        reliable = (
            (payload >= 0.5)
            & (vertical <= floor_z + 0.10)
            & (vertical_step <= 0.015)
        )
        for channel in range(4):
            indices = np.flatnonzero(reliable[start:end, channel]) + start
            for first, last in _group_indices(indices, max_gap=1):
                if last - first + 1 < 3:
                    continue
                xy = feet[first : last + 1, channel, :2]
                slip = np.linalg.norm(xy - xy[0], axis=1)
                maximum = max(maximum, float(np.max(slip, initial=0.0)))
    return maximum


def _generation_quality_metrics(
    motion: CanonicalMotion,
    ranges: list[tuple[int, int]],
    contacts: np.ndarray,
) -> dict[str, Any]:
    matrices = axis_angle_to_matrix(motion.smpl_poses.reshape(-1, 24, 3))
    root_relative_feet = (
        motion.joint_positions[:, FOOT_CHANNEL_JOINTS]
        - motion.smpl_trans[:, None, :]
    )
    records: list[dict[str, Any]] = []
    maximum_score = 0.0
    violations: set[str] = set()
    max_contact_slip = _final_contact_slip(motion, contacts, ranges)
    for start, end in ranges:
        transition_start = max(0, start - 1)
        transition_end = min(motion.frames - 1, end)
        pair_frames = np.arange(transition_start, transition_end, dtype=np.int64)
        if pair_frames.size:
            relative = np.einsum(
                "taji,tajk->taik",
                matrices[pair_frames],
                matrices[pair_frames + 1],
            )
            rotation_steps = np.linalg.norm(matrix_to_axis_angle(relative), axis=-1)
            maximum_rotation_degrees = float(np.degrees(np.max(rotation_steps)))
            maximum_leg_rotation_degrees = float(
                np.degrees(np.max(rotation_steps[:, LEG_JOINTS]))
            )
            foot_steps = np.linalg.norm(
                root_relative_feet[pair_frames + 1, :, :2]
                - root_relative_feet[pair_frames, :, :2],
                axis=-1,
            )
            maximum_root_relative_foot_step = float(np.max(foot_steps))
        else:
            maximum_rotation_degrees = 0.0
            maximum_leg_rotation_degrees = 0.0
            maximum_root_relative_foot_step = 0.0
        root = _translation_path_metrics(motion.smpl_trans, start, end)
        length = end - start
        if length <= 20:
            transitions = max(length + 1, 1)
            root_step_limit = max(
                0.04,
                2.0 * root["endpoint_distance_m"] / transitions,
            )
            root_acceleration_limit = 0.03
            path_excess_limit = max(0.03, 0.20 * root["endpoint_distance_m"])
            root_overshoot_limit = max(0.02, 0.10 * root["endpoint_distance_m"])
            average_step = root["path_length_m"] / transitions
            peak_to_mean = root["max_step_m"] / max(average_step, 0.005)
        else:
            root_step_limit = 0.08
            root_acceleration_limit = 0.08
            path_excess_limit = max(0.15, 0.008 * length)
            root_overshoot_limit = 0.15
            peak_to_mean = 0.0
        ratios = {
            "contact_slip": max_contact_slip / FOOT_SKATING_RETRY_THRESHOLD_M,
            "root_step": root["max_step_m"] / root_step_limit,
            "root_acceleration": (
                root["max_acceleration_m"] / root_acceleration_limit
            ),
            "root_path_excess": root["path_excess_m"] / path_excess_limit,
            "root_overshoot": root["max_overshoot_m"] / root_overshoot_limit,
            "root_peak_to_mean": peak_to_mean / 2.0,
            "leg_rotation": maximum_leg_rotation_degrees / 75.0,
            "all_joint_rotation": maximum_rotation_degrees / 120.0,
            "root_relative_foot_step": maximum_root_relative_foot_step / 0.25,
        }
        range_score = float(max(ratios.values(), default=0.0))
        maximum_score = max(maximum_score, range_score)
        for name, ratio in ratios.items():
            if ratio > GENERATION_QUALITY_RETRY_THRESHOLD:
                violations.add(name)
        records.append(
            {
                "range": [start, end],
                "score": range_score,
                "ratios": ratios,
                "root": root,
                "max_leg_rotation_degrees": maximum_leg_rotation_degrees,
                "max_all_joint_rotation_degrees": maximum_rotation_degrees,
                "max_root_relative_foot_step_m": maximum_root_relative_foot_step,
                "root_step_limit_m": root_step_limit,
                "root_acceleration_limit_m": root_acceleration_limit,
                "root_path_excess_limit_m": path_excess_limit,
                "root_overshoot_limit_m": root_overshoot_limit,
            }
        )
    return {
        "score": maximum_score,
        "requires_retry": maximum_score > GENERATION_QUALITY_RETRY_THRESHOLD,
        "threshold": GENERATION_QUALITY_RETRY_THRESHOLD,
        "violations": sorted(violations),
        "max_contact_slip_m": max_contact_slip,
        "ranges": records,
    }


def _selection_quality(record: dict[str, Any]) -> tuple[float, float, bool, list[str]]:
    foot_score = float(record.get("foot_skating_max_after_m", 0.0))
    quality_score = float(
        record.get(
            "generation_quality_score",
            foot_score / FOOT_SKATING_RETRY_THRESHOLD_M,
        )
    )
    requires_retry = bool(
        record.get(
            "generation_quality_requires_retry",
            foot_score > FOOT_SKATING_RETRY_THRESHOLD_M,
        )
    )
    violations = [str(item) for item in record.get("generation_quality_violations", [])]
    return quality_score, foot_score, requires_retry, violations


def stitch_generated_boundaries(
    reference: CanonicalMotion,
    generated: CanonicalMotion,
    known_mask: np.ndarray,
    *,
    blend_frames: int = 12,
) -> tuple[CanonicalMotion, list[list[int]]]:
    """Blend generated gaps into known motion with position and SO(3) anchors.

    Each internal gap uses frame ``start - 1`` and frame ``end`` as immutable
    anchors. The first/last generated frames continue the anchor's measured
    linear and angular velocity, then a smooth transition hands control back to
    the generated motion inside the selected interval. Known frames are never
    changed.
    """

    mask = np.asarray(known_mask, dtype=bool)
    if mask.shape != (reference.frames,) or generated.frames != reference.frames:
        raise ValueError("boundary stitch inputs must have matching frame counts")
    if blend_frames < 1:
        raise ValueError("blend_frames must be positive")

    result = clone_canonical_motion(generated)
    translations = result.smpl_trans.astype(np.float64, copy=True)
    matrices = axis_angle_to_matrix(result.smpl_poses.reshape(-1, 24, 3))
    reference_matrices = axis_angle_to_matrix(reference.smpl_poses.reshape(-1, 24, 3))
    ranges = _unknown_ranges(mask)
    translation_repairs = 0
    rotation_repairs = 0
    root_path_repairs: list[dict[str, Any]] = []

    for start, end in ranges:
        length = end - start
        left = start - 1 if start > 0 and mask[start - 1] else None
        right = end if end < reference.frames and mask[end] else None
        boundary_width = min(
            blend_frames,
            max(1, length // 2) if left is not None and right is not None else length,
        )
        left_target, right_target = _boundary_targets(
            reference, reference_matrices, mask, start, end
        )
        _align_generated_root_range(
            translations, matrices, start, end, left_target, right_target
        )
        translation_repairs += _repair_generated_translation_spikes(
            translations, start, end
        )
        rotation_repairs += _repair_generated_rotation_jumps(matrices, start, end)

        if left is not None:
            width = boundary_width
            if left > 0 and mask[left - 1]:
                translation_step = (
                    reference.smpl_trans[left] - reference.smpl_trans[left - 1]
                ).astype(np.float64)
                rotation_step = np.einsum(
                    "...ji,...jk->...ik",
                    reference_matrices[left - 1],
                    reference_matrices[left],
                )
            else:
                translation_step = np.zeros(3, dtype=np.float64)
                rotation_step = np.broadcast_to(np.eye(3), (24, 3, 3)).copy()
            for offset in range(width):
                weight = 1.0 if width == 1 else 1.0 - _smoothstep(offset / (width - 1))
                frame = start + offset
                target_translation = (
                    reference.smpl_trans[left].astype(np.float64)
                    + translation_step * (offset + 1)
                )
                translations[frame] = (
                    translations[frame] * (1.0 - weight) + target_translation * weight
                )
                target_rotation = np.einsum(
                    "...ij,...jk->...ik",
                    reference_matrices[left],
                    _scaled_rotation_step(rotation_step, offset + 1),
                )
                matrices[frame] = _interpolate_rotations(
                    matrices[frame], target_rotation, weight
                )

        if right is not None:
            width = boundary_width
            if right + 1 < reference.frames and mask[right + 1]:
                translation_step = (
                    reference.smpl_trans[right + 1] - reference.smpl_trans[right]
                ).astype(np.float64)
                rotation_step = np.einsum(
                    "...ji,...jk->...ik",
                    reference_matrices[right],
                    reference_matrices[right + 1],
                )
            else:
                translation_step = np.zeros(3, dtype=np.float64)
                rotation_step = np.broadcast_to(np.eye(3), (24, 3, 3)).copy()
            for offset in range(width):
                weight = 1.0 if width == 1 else 1.0 - _smoothstep(offset / (width - 1))
                frame = end - 1 - offset
                target_translation = (
                    reference.smpl_trans[right].astype(np.float64)
                    - translation_step * (offset + 1)
                )
                translations[frame] = (
                    translations[frame] * (1.0 - weight) + target_translation * weight
                )
                target_rotation = np.einsum(
                    "...ij,...jk->...ik",
                    reference_matrices[right],
                    _scaled_rotation_step(rotation_step, -(offset + 1)),
                )
                matrices[frame] = _interpolate_rotations(
                    matrices[frame], target_rotation, weight
                )

        # Boundary blending is itself a temporal transform. Re-run the SO(3)
        # outlier repair afterwards so a per-frame cross-fade cannot reintroduce
        # a knee, hip, or elbow flip that the raw-model pass already removed.
        rotation_repairs += _repair_generated_rotation_jumps(matrices, start, end)
        root_path_repair = _stabilize_generated_root_path(matrices, start, end)
        if root_path_repair is not None:
            root_path_repairs.append(root_path_repair)

    result.smpl_trans = translations.astype(np.float32)
    result.smpl_poses = matrix_to_axis_angle(matrices).reshape(-1, 72)
    result.joint_positions = forward_kinematics(result.smpl_poses, result.smpl_trans)
    ground_corrections = _lift_generated_foot_penetration(result, reference, mask, ranges)
    root_translation_repairs = _stabilize_generated_translation_paths(
        result,
        reference,
        mask,
        ranges,
    )
    foot_lock_corrections: list[dict[str, Any]] = []
    contact_payload = result.metadata.get("completer_contacts")
    if contact_payload is not None:
        generated_contacts = np.asarray(contact_payload, dtype=np.float32)
        if generated_contacts.shape != (result.frames, 4):
            raise ValueError("Completer contact payload must have shape (T,4)")
        foot_lock_corrections = contact_aware_root_lock(
            result,
            reference,
            mask,
            generated_contacts,
            ranges,
        )
    result.smpl_poses[mask] = reference.smpl_poses[mask]
    result.smpl_trans[mask] = reference.smpl_trans[mask]
    result.joint_positions[mask] = reference.joint_positions[mask]
    if not (
        np.isfinite(result.smpl_poses).all()
        and np.isfinite(result.smpl_trans).all()
        and np.isfinite(result.joint_positions).all()
    ):
        raise RuntimeError("boundary stitching produced non-finite motion")
    result.metadata = {
        **deepcopy(result.metadata),
        "boundary_stitch_diagnostics": {
            "root_endpoint_alignment": "two-sided shortest-arc SO(3) correction",
            "translation_interpolation": "endpoint-preserving cubic Hermite for detected spikes",
            "rotation_interpolation": "shortest-arc SLERP for detected jump spans",
            "translation_repaired_frames": translation_repairs,
            "rotation_repaired_frame_joints": rotation_repairs,
            "root_path_safety": "reject long-way turns in SO(3), not axis-angle space",
            "root_path_repairs": root_path_repairs,
            "root_translation_repairs": root_translation_repairs,
            "ground_corrections": ground_corrections,
            "contact_aware_foot_lock": (
                "model_contact_plus_height_vertical_and_horizontal_reliability_"
                "regularized_root_xy"
            ),
            "foot_lock_corrections": foot_lock_corrections,
        },
    }
    return result, [[start, end] for start, end in ranges]


def _fit_feature_frames(array: np.ndarray, target_frames: int, name: str) -> np.ndarray:
    value = np.asarray(array, dtype=np.float32)
    if value.ndim == 1:
        value = value[:, None]
    if value.ndim != 2 or value.shape[0] == 0:
        raise ValueError(f"{name} extraction returned an invalid array")
    if value.shape[0] >= target_frames:
        return value[:target_frames]
    missing = target_frames - value.shape[0]
    if missing > 2:
        raise ValueError(
            f"{name} extraction is {missing} frames shorter than the analyzed timeline"
        )
    return np.pad(value, ((0, missing), (0, 0)), mode="edge")


def extract_baseline_audio_features(audio_path: Path, target_frames: int) -> np.ndarray:
    """Extract Completer's baseline 35D features at exactly 30 FPS.

    Layout: onset envelope (1), MFCC (20), chroma CENS (12), onset peak
    one-hot (1), and beat one-hot (1). The implementation mirrors the public
    preprocessing contract while explicitly validating timeline length.
    """

    path = Path(audio_path).expanduser().resolve(strict=True)
    if target_frames <= 0 or target_frames > COMPLETER_MAX_FRAMES:
        raise ValueError(
            f"Completer supports 1–{COMPLETER_MAX_FRAMES} frames "
            f"(up to {COMPLETER_MAX_SECONDS:.0f} seconds)"
        )
    sample_rate = COMPLETER_FPS * 512
    signal, _ = librosa.load(path, sr=sample_rate, mono=True)
    if signal.size == 0 or not np.isfinite(signal).all():
        raise ValueError("audio is empty or contains non-finite samples")
    actual_duration = signal.size / sample_rate
    timeline_duration = target_frames / COMPLETER_FPS
    if abs(actual_duration - timeline_duration) > (1 / COMPLETER_FPS + 1e-3):
        raise ValueError(
            "audio duration and analyzed timeline disagree by more than one frame; "
            "re-run Analyze before generation"
        )

    hop = 512
    envelope = librosa.onset.onset_strength(y=signal, sr=sample_rate, hop_length=hop)
    mfcc = librosa.feature.mfcc(
        y=signal, sr=sample_rate, hop_length=hop, n_mfcc=20
    ).T
    chroma = librosa.feature.chroma_cens(
        y=signal, sr=sample_rate, hop_length=hop, n_chroma=12
    ).T

    envelope = _fit_feature_frames(envelope, target_frames, "onset envelope")[:, 0]
    mfcc = _fit_feature_frames(mfcc, target_frames, "MFCC")
    chroma = _fit_feature_frames(chroma, target_frames, "chroma")
    peak_indices = librosa.onset.onset_detect(
        onset_envelope=envelope, sr=sample_rate, hop_length=hop
    )
    peak_onehot = np.zeros(target_frames, dtype=np.float32)
    peak_onehot[np.asarray(peak_indices)[np.asarray(peak_indices) < target_frames]] = 1.0

    tempo_values = librosa.feature.tempo(
        onset_envelope=envelope, sr=sample_rate, hop_length=hop
    )
    start_bpm = float(tempo_values[0]) if len(tempo_values) else 120.0
    if not np.isfinite(start_bpm) or start_bpm <= 0:
        start_bpm = 120.0
    _, beat_indices = librosa.beat.beat_track(
        onset_envelope=envelope,
        sr=sample_rate,
        hop_length=hop,
        start_bpm=start_bpm,
        tightness=100,
    )
    beat_onehot = np.zeros(target_frames, dtype=np.float32)
    beat_indices = np.asarray(beat_indices, dtype=np.int64)
    beat_onehot[beat_indices[beat_indices < target_frames]] = 1.0

    result = np.concatenate(
        (
            envelope[:, None],
            mfcc,
            chroma,
            peak_onehot[:, None],
            beat_onehot[:, None],
        ),
        axis=1,
    ).astype(np.float32, copy=False)
    if result.shape != (target_frames, 35) or not np.isfinite(result).all():
        raise ValueError("Completer audio features failed shape/finite validation")
    return np.ascontiguousarray(result)


class InpaintingBackend:
    """One lazy CUDA inpainting model shared by complete() and remake().

    This adapter intentionally does not construct the upstream training wrapper,
    optimizer, Accelerator, W&B client, or duplicate EMA model.
    """

    def __init__(
        self,
        completer_root: Path,
        checkpoint_path: Path,
        condition_normalizer_path: Path,
        *,
        device: str = "cuda",
    ):
        self.root = Path(completer_root).expanduser().resolve(strict=True)
        self.checkpoint_path = Path(checkpoint_path).expanduser().resolve(strict=True)
        self.condition_normalizer_path = (
            Path(condition_normalizer_path).expanduser().resolve(strict=True)
        )
        if not (self.root / "inpainting.py").is_file():
            raise FileNotFoundError(f"invalid Completer root: {self.root}")
        if device != "cuda":
            raise ValueError("real Completer validation requires CUDA; CPU fallback is disabled")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; Completer will not silently fall back to CPU")
        self.device = torch.device(device)
        self._lock = threading.RLock()
        self._model: nn.Module | None = None
        self._motion_scale: torch.Tensor | None = None
        self._motion_offset: torch.Tensor | None = None
        self._condition_scale, self._condition_offset = self._load_condition_scaler()

    def _load_condition_scaler(self) -> tuple[torch.Tensor, torch.Tensor]:
        with np.load(self.condition_normalizer_path, allow_pickle=False) as archive:
            if str(archive["schema_version"].item()) != "1.0":
                raise ValueError("unsupported Completer condition normalizer schema")
            scale = np.asarray(archive["scale"], dtype=np.float32)
            offset = np.asarray(archive["offset"], dtype=np.float32)
        if scale.shape != (35,) or offset.shape != (35,):
            raise ValueError("Completer condition normalizer must be 35D")
        if not np.isfinite(scale).all() or not np.isfinite(offset).all() or np.any(scale <= 0):
            raise ValueError("Completer condition normalizer is invalid")
        return (
            torch.from_numpy(scale).to(self.device),
            torch.from_numpy(offset).to(self.device),
        )

    def _load(self) -> None:
        if self._model is not None:
            return
        from app.models import inpainting as model_module
        for module in (model_module,):
            module_path = Path(module.__file__).resolve()
            if self.root not in module_path.parents:
                raise RuntimeError(
                    f"Completer module was loaded from an unexpected path: {module_path}"
                )

        checkpoint: dict[str, Any] = torch.load(
            self.checkpoint_path, map_location="cpu", weights_only=True
        )
        # Preserve the serialized schema and layer keys of the author-provided checkpoint.
        if checkpoint.get("schema_version") != "customdance-flowerdance-inference-v1":
            raise ValueError("expected the CustomDance inference bundle, not a training checkpoint")
        motion_scale = torch.as_tensor(checkpoint["motion_scale"], dtype=torch.float32)
        motion_offset = torch.as_tensor(checkpoint["motion_offset"], dtype=torch.float32)
        if (
            motion_scale.shape != (151,)
            or motion_offset.shape != (151,)
            or not torch.isfinite(motion_scale).all()
            or not torch.isfinite(motion_offset).all()
            or torch.any(motion_scale <= 0)
        ):
            raise ValueError("Completer checkpoint motion normalizer is invalid")

        model = model_module.DanceDecoder(
            nfeats=151,
            seq_len=COMPLETER_MAX_FRAMES,
            latent_dim=512,
            ff_size=1024,
            num_layers=8,
            num_heads=8,
            dropout=0.1,
            cond_feature_dim=35,
            activation=torch.nn.functional.gelu,
        )
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model.to(self.device).eval()
        self._motion_scale = motion_scale.to(self.device)
        self._motion_offset = motion_offset.to(self.device)
        self._model = model

    @staticmethod
    def _validate_common(
        current_motion: CanonicalMotion,
        known_mask: np.ndarray,
        audio_features: np.ndarray,
        genre: int,
        seed: int,
        steps: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        frames = current_motion.frames
        if frames > COMPLETER_MAX_FRAMES:
            raise ValueError(
                f"Completer supports at most {COMPLETER_MAX_FRAMES} frames / "
                f"{COMPLETER_MAX_SECONDS:.0f} seconds; chunking is not implemented"
            )
        mask = np.asarray(known_mask, dtype=bool)
        if mask.shape != (frames,):
            raise ValueError("known/filled mask must match current motion frames")
        condition = np.ascontiguousarray(np.asarray(audio_features, dtype=np.float32))
        if condition.shape != (frames, 35) or not np.isfinite(condition).all():
            raise ValueError("audio_features must be finite with shape (T, 35)")
        if not 0 <= genre < len(GENRE_NAMES):
            raise ValueError("genre must be a Completer genre index from 0 through 15")
        if not 0 <= seed <= 2**31 - 1:
            raise ValueError("seed must be between 0 and 2147483647")
        if not 2 <= steps <= 101:
            raise ValueError("steps must be between 2 and 101")
        return mask, condition

    def audio_features(self, audio_path: Path, target_frames: int) -> np.ndarray:
        return extract_baseline_audio_features(audio_path, target_frames)

    def _inpaint(
        self,
        current_motion: CanonicalMotion,
        known_mask: np.ndarray,
        audio_features: np.ndarray,
        *,
        genre: int,
        seed: int,
        steps: int,
        operation: str,
        generated_range: tuple[int, int] | None,
    ) -> CanonicalMotion:
        inpaint_started = perf_counter()
        mask, condition = self._validate_common(
            current_motion, known_mask, audio_features, genre, seed, steps
        )
        model_load_started = perf_counter()
        self._load()
        model_load_ms = (perf_counter() - model_load_started) * 1000.0
        assert self._model is not None
        assert self._motion_scale is not None and self._motion_offset is not None

        model_started = perf_counter()
        raw_motion = canonical_to_completer(current_motion).features
        x = torch.from_numpy(raw_motion).unsqueeze(0).to(self.device)
        cond = torch.from_numpy(condition).unsqueeze(0).to(self.device)
        x = torch.clamp(
            x * self._motion_scale.view(1, 1, -1)
            + self._motion_offset.view(1, 1, -1),
            -1,
            1,
        )
        cond = torch.clamp(
            cond * self._condition_scale.view(1, 1, -1)
            + self._condition_offset.view(1, 1, -1),
            -1,
            1,
        )
        genre_tensor = torch.tensor([genre], dtype=torch.long, device=self.device)
        known = (
            torch.from_numpy(mask)
            .to(self.device)
            .view(1, current_motion.frames, 1)
            .expand(-1, -1, 151)
        )
        generator = torch.Generator(device=self.device)
        generator.manual_seed(seed)
        noise = torch.randn(
            x.shape, dtype=x.dtype, device=self.device, generator=generator
        )
        x_t = noise.clone()
        times = torch.linspace(1.0, 0.0, steps, device=self.device)

        with torch.inference_mode():
            for index in range(steps - 1):
                current_time = times[index]
                next_time = times[index + 1]
                time_batch = current_time.expand(1)
                interval_batch = (current_time - next_time).expand(1)
                velocity = self._model.infer_pred(
                    x_t, cond, genre_tensor, time_batch, interval_batch
                )
                x_t = x_t + velocity * (next_time - current_time)
                known_at_next_time = (1 - next_time) * x + next_time * noise
                # A Boolean clamp is required here. The upstream implementation
                # multiplies the mask by time, which weakens the pose constraint
                # on every step and removes it entirely at t=0.
                x_t = torch.where(known, known_at_next_time, x_t)

            raw_result = (
                (torch.clamp(x_t, -1, 1) - self._motion_offset.view(1, 1, -1))
                / self._motion_scale.view(1, 1, -1)
            )[0].detach().cpu().numpy().astype(np.float32)
        model_inference_ms = (perf_counter() - model_started) * 1000.0
        postprocess_started = perf_counter()
        internal = CompleterInternalMotion(
            features=raw_result,
            source_clip_id=f"completer-{operation}-{current_motion.source_clip_id}",
            metadata={"normalization": "checkpoint_minmax_inverse"},
        )
        output = completer_to_canonical(internal)

        # The public inpaint routine attenuates its mask to zero at t=0. The
        # adapter therefore enforces the user-visible contract in Canonical space.
        output.smpl_poses[mask] = current_motion.smpl_poses[mask]
        output.smpl_trans[mask] = current_motion.smpl_trans[mask]
        output.joint_positions[mask] = current_motion.joint_positions[mask]
        output, stitched_ranges = stitch_generated_boundaries(
            current_motion,
            output,
            mask,
        )
        stitch_diagnostics = output.metadata.get("boundary_stitch_diagnostics", {})
        completer_contacts = deepcopy(output.metadata.get("completer_contacts", []))
        quality = _generation_quality_metrics(
            output,
            [(int(start), int(end)) for start, end in stitched_ranges],
            np.asarray(completer_contacts, dtype=np.float32),
        )
        foot_lock_corrections = deepcopy(
            stitch_diagnostics.get("foot_lock_corrections", [])
        )
        protected_error = max(
            float(np.max(np.abs(output.smpl_poses[mask] - current_motion.smpl_poses[mask])))
            if mask.any()
            else 0.0,
            float(np.max(np.abs(output.smpl_trans[mask] - current_motion.smpl_trans[mask])))
            if mask.any()
            else 0.0,
            float(
                np.max(
                    np.abs(
                        output.joint_positions[mask]
                        - current_motion.joint_positions[mask]
                    )
                )
            )
            if mask.any()
            else 0.0,
        )
        if protected_error != 0.0:
            raise RuntimeError("Completer protected-frame restoration failed")
        if output.frames != current_motion.frames:
            raise RuntimeError("Completer changed the sequence frame count")
        postprocess_ms = (perf_counter() - postprocess_started) * 1000.0

        inference_record: dict[str, Any] = {
            "backend": "CustomDance shared inpainting real checkpoint",
            "operation": operation,
            "seed": seed,
            "steps": steps,
            "genre_index": genre,
            "genre_name": GENRE_NAMES[genre],
            "frames": output.frames,
            "model_max_frames": COMPLETER_MAX_FRAMES,
            "known_mask_semantics": "known=1, unknown=0",
            "known_frame_constraint": "boolean_clamp_at_every_ode_step_including_t0",
            "protected_frame_count": int(mask.sum()),
            "protected_max_abs_error": protected_error,
            "boundary_stitch": "non_overlapping_velocity_and_so3_anchor_blend",
            "boundary_blend_frames": 12,
            "root_endpoint_alignment": stitch_diagnostics.get("root_endpoint_alignment"),
            "translation_spike_repaired_frames": stitch_diagnostics.get(
                "translation_repaired_frames", 0
            ),
            "rotation_jump_repaired_frame_joints": stitch_diagnostics.get(
                "rotation_repaired_frame_joints", 0
            ),
            "root_path_repairs": deepcopy(stitch_diagnostics.get("root_path_repairs", [])),
            "root_translation_repairs": deepcopy(
                stitch_diagnostics.get("root_translation_repairs", [])
            ),
            "ground_corrections": deepcopy(stitch_diagnostics.get("ground_corrections", [])),
            "contact_channels": "left_ankle,right_ankle,left_foot,right_foot",
            "contact_aware_foot_lock": stitch_diagnostics.get(
                "contact_aware_foot_lock"
            ),
            "foot_lock_corrections": foot_lock_corrections,
            "foot_skating_max_before_m": max(
                (
                    float(record.get("max_slip_before_m", 0.0))
                    for record in foot_lock_corrections
                ),
                default=0.0,
            ),
            "foot_skating_max_after_m": quality["max_contact_slip_m"],
            "foot_skating_quality": (
                "review"
                if quality["max_contact_slip_m"] > FOOT_SKATING_RETRY_THRESHOLD_M
                else "pass"
            ),
            "generation_quality_score": quality["score"],
            "generation_quality_requires_retry": quality["requires_retry"],
            "generation_quality_violations": quality["violations"],
            "generation_quality": quality,
            "generated_ranges": stitched_ranges,
            "checkpoint_asset_id": "inpainting-stage3-checkpoint",
            "condition_normalizer_asset_id": "completer-condition-normalizer",
            "stage_timings_ms": {
                "model_load": round(model_load_ms, 3),
                "model_inference_and_transfer": round(model_inference_ms, 3),
                "canonical_postprocess": round(postprocess_ms, 3),
                "total": round((perf_counter() - inpaint_started) * 1000.0, 3),
            },
        }
        if generated_range is not None:
            inference_record["generated_start_frame"] = generated_range[0]
            inference_record["generated_end_frame"] = generated_range[1]
        history = list(current_motion.metadata.get("completer_inference", []))
        history.append(inference_record)
        output.metadata = {
            **deepcopy(current_motion.metadata),
            "completer_contacts": completer_contacts,
            "completer_inference": history,
        }
        return output

    def _inpaint_ranges(
        self,
        current_motion: CanonicalMotion,
        known_mask: np.ndarray,
        audio_features: np.ndarray,
        *,
        genre: int,
        seed: int,
        steps: int,
        operation: str,
    ) -> CanonicalMotion:
        """Inpaint each gap independently with one four-second context per side."""

        ranges_started = perf_counter()
        mask, condition = self._validate_common(
            current_motion, known_mask, audio_features, genre, seed, steps
        )
        ranges = _unknown_ranges(mask)
        if not ranges:
            raise ValueError(f"{operation.title()} requires at least one generated frame")

        result = clone_canonical_motion(current_motion)
        existing_contacts = np.asarray(
            result.metadata.get("completer_contacts", []), dtype=np.float32
        )
        if existing_contacts.shape == (result.frames, 4):
            result_contacts = existing_contacts.copy()
        else:
            result_contacts = canonical_to_completer(result).contacts.copy()
        window_records: list[dict[str, Any]] = []
        global_root_repairs: list[dict[str, Any]] = []
        global_root_translation_repairs: list[dict[str, Any]] = []
        global_ground_corrections: list[dict[str, Any]] = []
        global_foot_lock_corrections: list[dict[str, Any]] = []
        gap_seeds: list[int] = []

        for index, (start, end) in enumerate(ranges):
            previous_gap_end = ranges[index - 1][1] if index > 0 else 0
            next_gap_start = ranges[index + 1][0] if index + 1 < len(ranges) else result.frames
            window_start = max(previous_gap_end, start - COMPLETER_CONTEXT_FRAMES)
            window_end = min(next_gap_start, end + COMPLETER_CONTEXT_FRAMES)
            local_start = start - window_start
            local_end = end - window_start
            local_motion = _slice_canonical_motion(result, window_start, window_end)
            local_known = np.ones(local_motion.frames, dtype=bool)
            local_known[local_start:local_end] = False
            base_gap_seed = (seed + index) % (2**31)
            generated = self._inpaint(
                local_motion,
                local_known,
                condition[window_start:window_end],
                genre=genre,
                seed=base_gap_seed,
                steps=steps,
                operation=f"{operation}_gap",
                generated_range=(local_start, local_end),
            )
            selected_seed = base_gap_seed
            selected_record = generated.metadata.get("completer_inference", [{}])[-1]
            (
                selected_quality_score,
                selected_foot_score,
                requires_retry,
                selected_violations,
            ) = _selection_quality(selected_record)
            quality_attempts = [
                {
                    "seed": base_gap_seed,
                    "quality_score": selected_quality_score,
                    "foot_skating_score_m": selected_foot_score,
                    "violations": selected_violations,
                    "stage_timings_ms": deepcopy(
                        selected_record.get("stage_timings_ms", {})
                    ),
                }
            ]
            if requires_retry:
                retry_seed = (base_gap_seed + FOOT_SKATING_RETRY_SEED_OFFSET) % (
                    2**31
                )
                retry = self._inpaint(
                    local_motion,
                    local_known,
                    condition[window_start:window_end],
                    genre=genre,
                    seed=retry_seed,
                    steps=steps,
                    operation=f"{operation}_gap",
                    generated_range=(local_start, local_end),
                )
                retry_record = retry.metadata.get("completer_inference", [{}])[-1]
                (
                    retry_quality_score,
                    retry_foot_score,
                    _,
                    retry_violations,
                ) = _selection_quality(retry_record)
                quality_attempts.append(
                    {
                        "seed": retry_seed,
                        "quality_score": retry_quality_score,
                        "foot_skating_score_m": retry_foot_score,
                        "violations": retry_violations,
                        "stage_timings_ms": deepcopy(
                            retry_record.get("stage_timings_ms", {})
                        ),
                    }
                )
                if (retry_quality_score, retry_foot_score) < (
                    selected_quality_score,
                    selected_foot_score,
                ):
                    generated = retry
                    selected_seed = retry_seed
                    selected_quality_score = retry_quality_score
                    selected_foot_score = retry_foot_score
                    selected_violations = retry_violations
            gap_seeds.append(selected_seed)
            result.smpl_poses[start:end] = generated.smpl_poses[local_start:local_end]
            result.smpl_trans[start:end] = generated.smpl_trans[local_start:local_end]
            result.joint_positions[start:end] = generated.joint_positions[local_start:local_end]
            local_contacts = np.asarray(
                generated.metadata.get("completer_contacts", []), dtype=np.float32
            )
            if local_contacts.shape != (generated.frames, 4):
                raise RuntimeError("Completer did not preserve its four contact channels")
            result_contacts[start:end] = local_contacts[local_start:local_end]

            record = deepcopy(generated.metadata["completer_inference"][-1])
            record["quality_attempts"] = quality_attempts
            record["selected_seed"] = selected_seed
            record["selected_quality_score"] = selected_quality_score
            record["selected_foot_skating_score_m"] = selected_foot_score
            record["selected_quality_violations"] = selected_violations
            record["operation"] = f"{operation}_gap"
            record["global_range"] = [start, end]
            record["context_window"] = [window_start, window_end]
            record["generated_ranges"] = [[start, end]]
            record["generated_start_frame"] = start
            record["generated_end_frame"] = end
            for repair in record.get("root_path_repairs", []):
                repair["range"] = [
                    repair["range"][0] + window_start,
                    repair["range"][1] + window_start,
                ]
                global_root_repairs.append(deepcopy(repair))
            for repair in record.get("root_translation_repairs", []):
                repair["range"] = [
                    repair["range"][0] + window_start,
                    repair["range"][1] + window_start,
                ]
                global_root_translation_repairs.append(deepcopy(repair))
            for correction in record.get("ground_corrections", []):
                correction["range"] = [
                    correction["range"][0] + window_start,
                    correction["range"][1] + window_start,
                ]
                global_ground_corrections.append(deepcopy(correction))
            for correction in record.get("foot_lock_corrections", []):
                global_correction = deepcopy(correction)
                global_correction["range"] = [
                    global_correction["range"][0] + window_start,
                    global_correction["range"][1] + window_start,
                ]
                for contact_run in global_correction.get(
                    "accepted_contact_runs", []
                ):
                    contact_run["range"] = [
                        contact_run["range"][0] + window_start,
                        contact_run["range"][1] + window_start,
                    ]
                global_foot_lock_corrections.append(global_correction)
            window_records.append(record)

        protected_error = max(
            float(np.max(np.abs(result.smpl_poses[mask] - current_motion.smpl_poses[mask])))
            if mask.any()
            else 0.0,
            float(np.max(np.abs(result.smpl_trans[mask] - current_motion.smpl_trans[mask])))
            if mask.any()
            else 0.0,
            float(
                np.max(np.abs(result.joint_positions[mask] - current_motion.joint_positions[mask]))
            )
            if mask.any()
            else 0.0,
        )
        if protected_error != 0.0:
            raise RuntimeError("per-gap Completer changed a protected frame")
        if not (
            np.isfinite(result.smpl_poses).all()
            and np.isfinite(result.smpl_trans).all()
            and np.isfinite(result.joint_positions).all()
        ):
            raise RuntimeError("per-gap Completer produced non-finite motion")

        inference_record: dict[str, Any] = {
            "backend": "CustomDance shared inpainting real checkpoint",
            "operation": operation,
            "seed": seed,
            "gap_seeds": gap_seeds,
            "foot_skating_retry_threshold_m": FOOT_SKATING_RETRY_THRESHOLD_M,
            "foot_skating_retry_policy": (
                "one_deterministic_retry_keep_lower_composite_quality_score"
            ),
            "generation_quality_retry_threshold": GENERATION_QUALITY_RETRY_THRESHOLD,
            "steps": steps,
            "genre_index": genre,
            "genre_name": GENRE_NAMES[genre],
            "frames": result.frames,
            "model_max_frames": COMPLETER_MAX_FRAMES,
            "known_mask_semantics": "known=1, unknown=0",
            "known_frame_constraint": "boolean_clamp_at_every_ode_step_including_t0",
            "protected_frame_count": int(mask.sum()),
            "protected_max_abs_error": protected_error,
            "inference_strategy": "one_gap_per_window_with_four_second_context_per_side",
            "context_frames_per_side": COMPLETER_CONTEXT_FRAMES,
            "boundary_stitch": "non_overlapping_velocity_and_so3_anchor_blend",
            "boundary_blend_frames": 12,
            "root_endpoint_alignment": "two-sided shortest-arc SO(3) correction",
            "translation_spike_repaired_frames": sum(
                int(record["translation_spike_repaired_frames"]) for record in window_records
            ),
            "rotation_jump_repaired_frame_joints": sum(
                int(record["rotation_jump_repaired_frame_joints"]) for record in window_records
            ),
            "root_path_repairs": global_root_repairs,
            "root_translation_repairs": global_root_translation_repairs,
            "ground_corrections": global_ground_corrections,
            "contact_channels": "left_ankle,right_ankle,left_foot,right_foot",
            "contact_aware_foot_lock": (
                "model_contact_plus_height_vertical_and_horizontal_reliability_"
                "regularized_root_xy"
            ),
            "foot_lock_corrections": global_foot_lock_corrections,
            "foot_skating_max_before_m": max(
                (
                    float(record.get("max_slip_before_m", 0.0))
                    for record in global_foot_lock_corrections
                ),
                default=0.0,
            ),
            "foot_skating_max_after_m": max(
                (
                    float(record.get("selected_foot_skating_score_m", 0.0))
                    for record in window_records
                ),
                default=0.0,
            ),
            "foot_skating_quality": (
                "review"
                if any(
                    float(record.get("selected_foot_skating_score_m", 0.0))
                    > FOOT_SKATING_RETRY_THRESHOLD_M
                    for record in window_records
                )
                else "pass"
            ),
            "generation_quality_score": max(
                (
                    float(record.get("selected_quality_score", 0.0))
                    for record in window_records
                ),
                default=0.0,
            ),
            "generation_quality_requires_review": any(
                float(record.get("selected_quality_score", 0.0))
                > GENERATION_QUALITY_RETRY_THRESHOLD
                for record in window_records
            ),
            "generation_quality_violations": sorted(
                {
                    str(violation)
                    for record in window_records
                    for violation in record.get("selected_quality_violations", [])
                }
            ),
            "generated_ranges": [[start, end] for start, end in ranges],
            "gap_inferences": window_records,
            "checkpoint_asset_id": "inpainting-stage3-checkpoint",
            "condition_normalizer_asset_id": "completer-condition-normalizer",
            "attempt_count": sum(
                len(record.get("quality_attempts", [])) for record in window_records
            ),
            "base_attempt_count": len(window_records),
            "retry_attempt_count": sum(
                max(0, len(record.get("quality_attempts", [])) - 1)
                for record in window_records
            ),
            "stage_timings_ms": {
                "gap_attempts_total": round(
                    sum(
                        float(
                            attempt.get("stage_timings_ms", {}).get("total", 0.0)
                        )
                        for record in window_records
                        for attempt in record.get("quality_attempts", [])
                    ),
                    3,
                ),
                "total": round((perf_counter() - ranges_started) * 1000.0, 3),
            },
        }
        if operation == "remake" and len(ranges) == 1:
            inference_record["generated_start_frame"] = ranges[0][0]
            inference_record["generated_end_frame"] = ranges[0][1]
        history = list(current_motion.metadata.get("completer_inference", []))
        history.append(inference_record)
        result.source_clip_id = f"completer-{operation}-{current_motion.source_clip_id}"
        result.metadata = {
            **deepcopy(current_motion.metadata),
            "completer_contacts": result_contacts.astype(float).tolist(),
            "completer_inference": history,
        }
        return result

    def complete(
        self,
        current_motion: CanonicalMotion,
        filled_mask: np.ndarray,
        audio_features: np.ndarray,
        genre: int,
        seed: int,
        steps: int,
    ) -> CanonicalMotion:
        """Generate every unfilled frame and restore all filled frames exactly."""

        with self._lock:
            mask = np.asarray(filled_mask, dtype=bool)
            return self._inpaint_ranges(
                current_motion,
                mask,
                audio_features,
                genre=genre,
                seed=seed,
                steps=steps,
                operation="complete",
            )

    def remake(
        self,
        current_motion: CanonicalMotion,
        start_frame: int,
        end_frame: int,
        audio_features: np.ndarray,
        genre: int,
        seed: int,
        steps: int,
    ) -> CanonicalMotion:
        """Regenerate only ``[start_frame, end_frame)`` for the whole body."""

        if not 0 <= start_frame < end_frame <= current_motion.frames:
            raise ValueError("Remake range must be a non-empty interval inside the motion")
        known = np.ones(current_motion.frames, dtype=bool)
        known[start_frame:end_frame] = False
        with self._lock:
            return self._inpaint_ranges(
                current_motion,
                known,
                audio_features,
                genre=genre,
                seed=seed,
                steps=steps,
                operation="remake",
            )
