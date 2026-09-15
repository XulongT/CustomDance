"""Support-foot detection and contact-aware root stabilization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

from app.schemas.motion import CanonicalMotion
from app.utils.kinematics import forward_kinematics

FOOT_CHANNEL_JOINTS = np.asarray((7, 8, 10, 11), dtype=np.int64)
FOOT_CHANNEL_NAMES = (
    "left_ankle",
    "right_ankle",
    "left_foot",
    "right_foot",
)
FOOT_SIDE_JOINTS = {
    "left": np.asarray((7, 10), dtype=np.int64),
    "right": np.asarray((8, 11), dtype=np.int64),
}


@dataclass(frozen=True, slots=True)
class EndpointSupport:
    side: str
    confidence: float
    position_xy: np.ndarray


def _validated_joints(joint_positions: np.ndarray) -> np.ndarray:
    joints = np.asarray(joint_positions, dtype=np.float64)
    if joints.ndim != 3 or joints.shape[1:] != (24, 3):
        raise ValueError("joint_positions must have shape (T,24,3)")
    if joints.shape[0] == 0 or not np.isfinite(joints).all():
        raise ValueError("joint_positions must be non-empty and finite")
    return joints


def endpoint_supports(
    joint_positions: np.ndarray,
    *,
    floor_z: float,
    at_start: bool,
    window_frames: int = 4,
    maximum_height_m: float = 0.08,
    maximum_horizontal_step_m: float = 0.006,
    maximum_vertical_step_m: float = 0.006,
    minimum_confidence: float = 0.67,
) -> dict[str, EndpointSupport]:
    """Detect reliable left/right support feet at one motion endpoint.

    A side is trusted only when at least one of its ankle/toe joints stays near
    the floor and the side midpoint is almost stationary for the endpoint
    window. This deliberately rejects ordinary root travel without compensating
    leg motion.
    """

    joints = _validated_joints(joint_positions)
    if not np.isfinite(floor_z):
        raise ValueError("floor_z must be finite")
    width = min(max(int(window_frames), 2), joints.shape[0])
    sample = joints[:width] if at_start else joints[-width:]
    endpoint_index = 0 if at_start else -1
    supports: dict[str, EndpointSupport] = {}
    for side, indices in FOOT_SIDE_JOINTS.items():
        side_joints = sample[:, indices]
        side_position = np.mean(side_joints, axis=1)
        low_fraction = float(
            np.mean(np.min(side_joints[..., 2], axis=1) <= floor_z + maximum_height_m)
        )
        steps = np.diff(side_position, axis=0)
        stable = (
            np.linalg.norm(steps[:, :2], axis=1) <= maximum_horizontal_step_m
        ) & (np.abs(steps[:, 2]) <= maximum_vertical_step_m)
        stable_fraction = float(np.mean(stable)) if stable.size else 0.0
        confidence = low_fraction * stable_fraction
        if confidence >= minimum_confidence:
            supports[side] = EndpointSupport(
                side=side,
                confidence=confidence,
                position_xy=np.mean(
                    joints[endpoint_index, indices, :2], axis=0
                ).astype(np.float64),
            )
    return supports


def _true_runs(values: np.ndarray, start: int, end: int) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    run_start: int | None = None
    for frame in range(start, end):
        if bool(values[frame]) and run_start is None:
            run_start = frame
        elif not bool(values[frame]) and run_start is not None:
            runs.append((run_start, frame))
            run_start = None
    if run_start is not None:
        runs.append((run_start, end))
    return runs


def _smoothstep(value: float) -> float:
    clipped = min(1.0, max(0.0, float(value)))
    return clipped * clipped * (3.0 - 2.0 * clipped)


def _ramped_run_weights(values: np.ndarray, edge_frames: int = 3) -> np.ndarray:
    weights = np.asarray(values, dtype=np.float64).copy()
    width = min(max(int(edge_frames), 1), max(weights.size // 2, 1))
    for offset in range(width):
        ramp = _smoothstep((offset + 1) / (width + 1))
        weights[offset] *= ramp
        weights[-1 - offset] *= ramp
    return weights


def _limit_vector_steps(values: np.ndarray, maximum_step_m: float) -> np.ndarray:
    """Project a correction curve onto a conservative per-frame speed bound."""

    limited = np.asarray(values, dtype=np.float64).copy()
    if limited.shape[0] < 2:
        return limited
    limited[0] = 0.0
    limited[-1] = 0.0
    for _ in range(3):
        for frame in range(1, limited.shape[0]):
            delta = limited[frame] - limited[frame - 1]
            distance = float(np.linalg.norm(delta))
            if distance > maximum_step_m:
                limited[frame] = limited[frame - 1] + delta * (
                    maximum_step_m / distance
                )
        limited[-1] = 0.0
        for frame in range(limited.shape[0] - 2, -1, -1):
            delta = limited[frame] - limited[frame + 1]
            distance = float(np.linalg.norm(delta))
            if distance > maximum_step_m:
                limited[frame] = limited[frame + 1] + delta * (
                    maximum_step_m / distance
                )
        limited[0] = 0.0
    return limited


def _regularized_root_correction(
    targets: np.ndarray,
    weights: np.ndarray,
    start: int,
    end: int,
    *,
    maximum_step_m: float,
    contact_weight: float = 40.0,
    velocity_weight: float = 8.0,
    acceleration_weight: float = 24.0,
) -> np.ndarray:
    """Solve contact targets with velocity/acceleration regularization."""

    length = end - start
    correction = np.zeros((length, 2), dtype=np.float64)
    if length < 3 or not np.any(weights[start:end] > 0.0):
        return correction

    local_weights = np.asarray(weights[start:end], dtype=np.float64)
    diagonal = contact_weight * local_weights + 1e-8
    endpoint_weight = np.zeros(length, dtype=np.float64)
    endpoint_weight[[0, -1]] = 1e6
    system = sparse.diags(diagonal + endpoint_weight, format="csc")
    first_difference = sparse.diags(
        (-np.ones(length - 1), np.ones(length - 1)),
        (0, 1),
        shape=(length - 1, length),
        format="csc",
    )
    system = system + velocity_weight * (first_difference.T @ first_difference)
    if length > 2:
        second_difference = sparse.diags(
            (np.ones(length - 2), -2.0 * np.ones(length - 2), np.ones(length - 2)),
            (0, 1, 2),
            shape=(length - 2, length),
            format="csc",
        )
        system = system + acceleration_weight * (
            second_difference.T @ second_difference
        )
    right_hand_side = contact_weight * np.asarray(
        targets[start:end], dtype=np.float64
    )
    for axis in range(2):
        correction[:, axis] = np.asarray(
            spsolve(system, right_hand_side[:, axis]), dtype=np.float64
        )
    correction[0] = 0.0
    correction[-1] = 0.0
    return _limit_vector_steps(correction, maximum_step_m)


def _root_xy_metrics(translations: np.ndarray, start: int, end: int) -> dict[str, float]:
    xy = np.asarray(translations[start:end, :2], dtype=np.float64)
    steps = np.diff(xy, axis=0)
    step_lengths = np.linalg.norm(steps, axis=1)
    accelerations = np.diff(steps, axis=0)
    direct = float(np.linalg.norm(xy[-1] - xy[0])) if len(xy) > 1 else 0.0
    path = float(np.sum(step_lengths))
    return {
        "endpoint_distance_m": direct,
        "path_length_m": path,
        "path_excess_m": max(0.0, path - direct),
        "max_step_m": float(np.max(step_lengths, initial=0.0)),
        "max_acceleration_m": float(
            np.max(np.linalg.norm(accelerations, axis=1), initial=0.0)
        ),
    }


def contact_aware_root_lock(
    motion: CanonicalMotion,
    reference: CanonicalMotion,
    known_mask: np.ndarray,
    generated_contacts: np.ndarray,
    ranges: list[tuple[int, int]],
    *,
    minimum_contact_frames: int = 3,
    model_contact_threshold: float = 0.5,
    maximum_height_above_floor_m: float = 0.10,
    maximum_vertical_step_m: float = 0.015,
    maximum_root_relative_horizontal_step_m: float = 0.06,
    maximum_root_correction_m: float = 0.35,
    maximum_root_correction_step_m: float = 0.03,
    skating_threshold_m: float = 0.12,
) -> list[dict[str, Any]]:
    """Apply root-XY corrections that lock model-supported contact feet.

    Protected frames and the first/last generated frame stay unchanged. Contact
    constraints are combined with regularized weighted least squares, so
    simultaneous contacts do not require choosing one side and support changes
    cannot introduce an instantaneous root jump.
    """

    mask = np.asarray(known_mask, dtype=bool)
    contacts = np.asarray(generated_contacts, dtype=np.float64)
    if mask.shape != (motion.frames,) or reference.frames != motion.frames:
        raise ValueError("foot locking inputs must share the motion frame count")
    if contacts.shape != (motion.frames, 4) or not np.isfinite(contacts).all():
        raise ValueError("generated Completer contacts must be finite (T,4)")
    contacts = np.clip(contacts, 0.0, 1.0)
    if minimum_contact_frames < 3:
        raise ValueError("minimum_contact_frames must be at least 3")

    diagnostics: list[dict[str, Any]] = []
    for start, end in ranges:
        if not 0 <= start < end <= motion.frames:
            raise ValueError("generated range is outside the motion")
        context_start = max(0, start - 60)
        context_end = min(motion.frames, end + 60)
        known_indices = np.flatnonzero(mask[context_start:context_end]) + context_start
        if known_indices.size:
            floor_z = float(
                np.quantile(
                    reference.joint_positions[known_indices][:, FOOT_CHANNEL_JOINTS, 2],
                    0.05,
                )
            )
        else:
            floor_z = float(
                np.quantile(
                    motion.joint_positions[start:end, FOOT_CHANNEL_JOINTS, 2], 0.05
                )
            )

        before_positions = motion.joint_positions[:, FOOT_CHANNEL_JOINTS].astype(
            np.float64, copy=True
        )
        vertical = before_positions[..., 2]
        root_relative = before_positions - motion.smpl_trans[:, None, :]
        vertical_step = np.zeros_like(vertical)
        horizontal_relative_step = np.zeros_like(vertical)
        if motion.frames > 1:
            vertical_step[0] = np.abs(vertical[1] - vertical[0])
            vertical_step[-1] = np.abs(vertical[-1] - vertical[-2])
            adjacent_horizontal = np.linalg.norm(
                np.diff(root_relative[..., :2], axis=0), axis=-1
            )
            horizontal_relative_step[0] = adjacent_horizontal[0]
            horizontal_relative_step[-1] = adjacent_horizontal[-1]
        if motion.frames > 2:
            vertical_step[1:-1] = 0.5 * np.abs(vertical[2:] - vertical[:-2])
            horizontal_relative_step[1:-1] = np.maximum(
                adjacent_horizontal[:-1], adjacent_horizontal[1:]
            )
        contact_mask = (
            (contacts >= model_contact_threshold)
            & (vertical <= floor_z + maximum_height_above_floor_m)
            & (vertical_step <= maximum_vertical_step_m)
            & (
                horizontal_relative_step
                <= maximum_root_relative_horizontal_step_m
            )
        )

        target_sum = np.zeros((motion.frames, 2), dtype=np.float64)
        target_weight = np.zeros(motion.frames, dtype=np.float64)
        accepted_runs: list[dict[str, Any]] = []
        for channel, joint_index in enumerate(FOOT_CHANNEL_JOINTS):
            for run_start, run_end in _true_runs(contact_mask[:, channel], start, end):
                if run_end - run_start < minimum_contact_frames:
                    continue
                anchors: list[np.ndarray] = []
                anchor_source = "generated_contact_median"
                if run_start <= start + 1 and start > 0 and mask[start - 1]:
                    anchors.append(
                        reference.joint_positions[start - 1, joint_index, :2].astype(
                            np.float64
                        )
                    )
                    anchor_source = "left_known_context"
                if run_end >= end - 1 and end < motion.frames and mask[end]:
                    anchors.append(
                        reference.joint_positions[end, joint_index, :2].astype(np.float64)
                    )
                    anchor_source = (
                        "two_sided_known_context"
                        if len(anchors) == 2
                        else "right_known_context"
                    )
                if anchors:
                    anchor_xy = np.mean(anchors, axis=0)
                else:
                    anchor_xy = np.median(
                        before_positions[run_start:run_end, channel, :2], axis=0
                    )
                run_weights = _ramped_run_weights(
                    np.maximum(
                        contacts[run_start:run_end, channel], model_contact_threshold
                    )
                )
                desired = (
                    anchor_xy[None]
                    - before_positions[run_start:run_end, channel, :2]
                )
                target_sum[run_start:run_end] += desired * run_weights[:, None]
                target_weight[run_start:run_end] += run_weights
                before_relative = (
                    before_positions[run_start:run_end, channel, :2]
                    - before_positions[run_start, channel, :2]
                )
                accepted_runs.append(
                    {
                        "channel": FOOT_CHANNEL_NAMES[channel],
                        "joint_index": int(joint_index),
                        "range": [run_start, run_end],
                        "anchor_source": anchor_source,
                        "frames": run_end - run_start,
                        "max_root_relative_horizontal_step_m": float(
                            np.max(
                                horizontal_relative_step[run_start:run_end, channel],
                                initial=0.0,
                            )
                        ),
                        "max_slip_before_m": float(
                            np.max(np.linalg.norm(before_relative, axis=1), initial=0.0)
                        ),
                    }
                )

        if not accepted_runs:
            diagnostics.append(
                {
                    "range": [start, end],
                    "known_floor_z": floor_z,
                    "accepted_contact_runs": [],
                    "contact_frames": 0,
                    "max_slip_before_m": 0.0,
                    "max_slip_after_m": 0.0,
                    "max_root_correction_m": 0.0,
                    "max_root_correction_step_m": 0.0,
                    "rejected_horizontal_contact_frames": int(
                        np.count_nonzero(
                            (contacts[start:end] >= model_contact_threshold)
                            & (
                                horizontal_relative_step[start:end]
                                > maximum_root_relative_horizontal_step_m
                            )
                        )
                    ),
                    "quality": "not_evaluated_no_reliable_contact",
                }
            )
            continue

        correction = _regularized_root_correction(
            target_sum,
            target_weight,
            start,
            end,
            maximum_step_m=maximum_root_correction_step_m,
        )
        correction_norm = np.linalg.norm(correction, axis=1)
        over_limit = correction_norm > maximum_root_correction_m
        correction[over_limit] *= (
            maximum_root_correction_m / correction_norm[over_limit]
        )[:, None]
        correction[0] = 0.0
        correction[-1] = 0.0
        before_root_metrics = _root_xy_metrics(motion.smpl_trans, start, end)
        original_translation = motion.smpl_trans[start:end].copy()
        motion.smpl_trans[start:end, :2] += correction.astype(np.float32)
        motion.joint_positions = forward_kinematics(
            motion.smpl_poses, motion.smpl_trans
        )
        after_root_metrics = _root_xy_metrics(motion.smpl_trans, start, end)
        rollback_reasons: list[str] = []
        if (
            after_root_metrics["max_acceleration_m"]
            > max(0.10, before_root_metrics["max_acceleration_m"] + 0.06)
            and after_root_metrics["max_acceleration_m"]
            > before_root_metrics["max_acceleration_m"] * 1.5
        ):
            rollback_reasons.append("root_acceleration_worsened")
        if (
            after_root_metrics["path_excess_m"]
            > before_root_metrics["path_excess_m"] + 0.15
        ):
            rollback_reasons.append("root_path_excess_worsened")
        if rollback_reasons:
            motion.smpl_trans[start:end] = original_translation
            motion.joint_positions = forward_kinematics(
                motion.smpl_poses, motion.smpl_trans
            )
            after_root_metrics = before_root_metrics
            correction[:] = 0.0

        max_after = 0.0
        for record in accepted_runs:
            run_start, run_end = record["range"]
            joint_index = int(record["joint_index"])
            after_xy = motion.joint_positions[run_start:run_end, joint_index, :2]
            after_relative = after_xy - after_xy[0]
            record["max_slip_after_m"] = float(
                np.max(np.linalg.norm(after_relative, axis=1), initial=0.0)
            )
            max_after = max(max_after, float(record["max_slip_after_m"]))
        max_before = max(float(record["max_slip_before_m"]) for record in accepted_runs)
        diagnostics.append(
            {
                "range": [start, end],
                "known_floor_z": floor_z,
                "accepted_contact_runs": accepted_runs,
                "contact_frames": int(np.count_nonzero(target_weight[start:end])),
                "max_slip_before_m": max_before,
                "max_slip_after_m": max_after,
                "max_root_correction_m": float(
                    np.max(np.linalg.norm(correction, axis=1), initial=0.0)
                ),
                "max_root_correction_step_m": float(
                    np.max(
                        np.linalg.norm(np.diff(correction, axis=0), axis=1),
                        initial=0.0,
                    )
                ),
                "maximum_allowed_root_correction_step_m": (
                    maximum_root_correction_step_m
                ),
                "maximum_root_relative_horizontal_step_m": (
                    maximum_root_relative_horizontal_step_m
                ),
                "rejected_horizontal_contact_frames": int(
                    np.count_nonzero(
                        (contacts[start:end] >= model_contact_threshold)
                        & (
                            horizontal_relative_step[start:end]
                            > maximum_root_relative_horizontal_step_m
                        )
                    )
                ),
                "root_metrics_before": before_root_metrics,
                "root_metrics_after": after_root_metrics,
                "correction_accepted": not rollback_reasons,
                "rollback_reasons": rollback_reasons,
                "quality_threshold_m": skating_threshold_m,
                "quality": "pass" if max_after <= skating_threshold_m else "review",
                "solver": "regularized_weighted_root_xy_with_speed_limit",
                "protected_generated_endpoints": True,
            }
        )
    return diagnostics


__all__ = [
    "FOOT_CHANNEL_JOINTS",
    "FOOT_CHANNEL_NAMES",
    "FOOT_SIDE_JOINTS",
    "EndpointSupport",
    "contact_aware_root_lock",
    "endpoint_supports",
]
