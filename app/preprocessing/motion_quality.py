"""FineDance motion-quality measurements from NumPy forward kinematics.

The raw FineDance archive used by Lite stores 315D frames: 3D root
translation followed by 52 six-dimensional local rotations.  This module
does not modify raw arrays.  It decodes them to the 55-joint SMPL-X layout and
computes conservative quality evidence for a later review/manifest pass.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np


SMPLX_PARENTS = np.asarray(
    [
        -1,
        0,
        0,
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        9,
        9,
        12,
        13,
        14,
        16,
        17,
        18,
        19,
        15,
        15,
        15,
        20,
        25,
        26,
        20,
        28,
        29,
        20,
        31,
        32,
        20,
        34,
        35,
        20,
        37,
        38,
        21,
        40,
        41,
        21,
        43,
        44,
        21,
        46,
        47,
        21,
        49,
        50,
        21,
        52,
        53,
    ],
    dtype=np.int64,
)

# FineDance's SMPL-X joint names in the first 55-joint layout.
JOINTS = {
    "pelvis": 0,
    "left_knee": 4,
    "right_knee": 5,
    "spine1": 3,
    "spine2": 6,
    "spine3": 9,
    "neck": 12,
    "left_ankle": 7,
    "right_ankle": 8,
    "left_foot": 10,
    "right_foot": 11,
    "left_wrist": 20,
    "right_wrist": 21,
    "left_hand": 22,
    "right_hand": 23,
}

FOOT_JOINTS = np.asarray([7, 8, 10, 11], dtype=np.int64)
HAND_JOINTS = np.asarray([20, 21, 22, 23], dtype=np.int64)

DEFAULT_THRESHOLDS = {
    # These are review flags, not a final automatic deletion policy.
    "static_mean_joint_speed_mps": 0.15,
    "static_active_frame_ratio": 0.25,
    "leading_idle_speed_mps": 0.05,
    "leading_idle_min_seconds": 1.0,
    "leading_active_min_seconds": 0.5,
    "leading_context_frames": 8,
    "boundary_scan_seconds": 8.0,
    "boundary_idle_min_seconds": 4.0,
    "boundary_context_frames": 8,
    "floor_contact_margin_m": 0.04,
    "floor_contact_speed_mps": 0.20,
    "floor_plane_min_points": 30,
    "floor_plane_min_xz_span_m": 0.50,
    "floor_plane_warn_tilt_deg": 5.0,
    "floor_plane_max_tilt_deg": 5.0,
    "floor_plane_max_p95_error_m": 0.08,
    "low_pelvis_height_m": 0.55,
    "low_pelvis_ratio": 0.30,
    # Genre-independent standing-library rule. A short knee bend or jump does
    # not qualify: the pelvis must stay below the height gate for both a
    # substantial fraction of the four-second window and one sustained run.
    "standing_low_pelvis_height_m": 0.55,
    "standing_low_pelvis_ratio": 0.25,
    "standing_low_pelvis_min_run_seconds": 0.75,
    "pelvis_below_knees_margin_m": 0.05,
    "non_upright_tilt_deg": 55.0,
    "non_upright_ratio": 0.25,
    "hand_ground_margin_m": 0.08,
    "hand_ground_ratio": 0.10,
    "foot_floor_gap_m": 0.18,
    "foot_ground_contact_ratio": 0.45,
    "extreme_tilt_deg": 80.0,
    "extreme_tilt_ratio": 0.25,
    # Floorwork is a genre-configured exclusion, not a global low-pose rule.
    "floorwork_low_pelvis_ratio": 0.30,
    "floorwork_non_upright_ratio": 0.25,
    "floorwork_hand_ground_ratio": 0.10,
    "floorwork_foot_floor_gap_m": 0.18,
    "floorwork_foot_ground_contact_ratio": 0.45,
    "floorwork_extreme_tilt_deg": 80.0,
    "floorwork_extreme_tilt_ratio": 0.25,
    "ground_offset_robust_z": 4.0,
}


def _normalize(values: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(values, axis=-1, keepdims=True)
    return values / np.maximum(norm, 1.0e-8)


def rotation_6d_to_matrix(values: np.ndarray) -> np.ndarray:
    """Convert Zhou/PyTorch3D 6D rotations to row-basis matrices.

    FineDance stores the first two *rows* of each rotation matrix.  Stacking
    the orthonormal basis on the last axis would instead return the transpose
    (the inverse rotation), visibly reversing movements such as forward arm
    raises.
    """

    first = _normalize(values[..., :3])
    second = values[..., 3:]
    second = _normalize(second - np.sum(first * second, axis=-1, keepdims=True) * first)
    third = np.cross(first, second)
    return np.stack((first, second, third), axis=-2)


def _longest_true_run(mask: np.ndarray) -> int:
    """Return the longest contiguous true run in a one-dimensional mask."""

    values = np.asarray(mask, dtype=bool).reshape(-1)
    longest = 0
    current = 0
    for value in values:
        current = current + 1 if bool(value) else 0
        longest = max(longest, current)
    return int(longest)


def decode_motion_to_joints(raw: np.ndarray, rest_joints: np.ndarray) -> np.ndarray:
    """Decode FineDance raw 315/319D frames to ``(T,55,3)`` positions."""

    raw = np.asarray(raw, dtype=np.float32)
    rest_joints = np.asarray(rest_joints, dtype=np.float32)
    if raw.ndim != 2 or raw.shape[1] not in (315, 319):
        raise ValueError(f"expected motion shape (T,315/319), got {raw.shape}")
    if rest_joints.shape != (55, 3):
        raise ValueError(f"expected rest joints shape (55,3), got {rest_joints.shape}")
    raw = raw[:, :315]
    frames = raw.shape[0]
    rotations = rotation_6d_to_matrix(raw[:, 3:315].reshape(frames, 52, 6))
    local = np.broadcast_to(np.eye(3, dtype=np.float32), (frames, 55, 3, 3)).copy()
    local[:, :22] = rotations[:, :22]
    local[:, 25:] = rotations[:, 22:]
    root = raw[:, :3].copy()
    # Match FineDance's renderer convention: the root is lifted in Y.
    root[:, 1] += 1.3

    global_rotations = np.empty((frames, 55, 3, 3), dtype=np.float32)
    positions = np.empty((frames, 55, 3), dtype=np.float32)
    for joint in range(55):
        parent = int(SMPLX_PARENTS[joint])
        if parent < 0:
            global_rotations[:, joint] = local[:, joint]
            positions[:, joint] = root + np.einsum(
                "tij,j->ti", global_rotations[:, joint], rest_joints[joint]
            )
        else:
            global_rotations[:, joint] = np.einsum(
                "tij,tjk->tik", global_rotations[:, parent], local[:, joint]
            )
            offset = rest_joints[joint] - rest_joints[parent]
            positions[:, joint] = positions[:, parent] + np.einsum(
                "tij,j->ti", global_rotations[:, parent], offset
            )
    if not np.isfinite(positions).all():
        raise FloatingPointError("non-finite FK positions")
    return positions


def detect_leading_trim(
    positions: np.ndarray,
    fps: int = 30,
    speed_threshold_mps: float = 0.05,
    min_idle_seconds: float = 1.0,
    min_active_seconds: float = 0.5,
    context_frames: int = 8,
) -> dict[str, int | float | bool]:
    """Find a long inactive prefix followed by sustained activity.

    Only a prefix is eligible.  A stationary interval after the first
    sustained active run is never trimmed, so choreography pauses remain in
    the source sequence.
    """

    positions = np.asarray(positions, dtype=np.float32)
    velocity = np.linalg.norm(np.diff(positions, axis=0), axis=-1) * float(fps)
    per_frame_speed = velocity.mean(axis=1)
    active = per_frame_speed > float(speed_threshold_mps)
    min_idle_frames = int(round(float(min_idle_seconds) * float(fps)))
    min_active_frames = max(1, int(round(float(min_active_seconds) * float(fps))))
    active_start = None
    run = 0
    for index, is_active in enumerate(active):
        run = run + 1 if bool(is_active) else 0
        if run >= min_active_frames:
            active_start = index - min_active_frames + 1
            break
    if active_start is None or active_start < min_idle_frames:
        return {
            "leading_idle_frames": int(max(0, active_start or 0)),
            "leading_idle_seconds": float(max(0, active_start or 0) / float(fps)),
            "leading_active_start_frame": int(active_start or 0),
            "trim_start_frame": 0,
            "trim_applied": False,
        }
    trim_start = max(0, int(active_start) - int(context_frames))
    return {
        "leading_idle_frames": int(active_start),
        "leading_idle_seconds": float(active_start / float(fps)),
        "leading_active_start_frame": int(active_start),
        "trim_start_frame": int(trim_start),
        "trim_applied": bool(trim_start > 0),
    }


def _per_frame_joint_speed(positions: np.ndarray, fps: int) -> np.ndarray:
    """Return one mean joint speed for every frame, including frame zero."""

    velocity = np.linalg.norm(np.diff(positions, axis=0), axis=-1) * float(fps)
    per_frame = np.zeros((positions.shape[0],), dtype=np.float32)
    if velocity.shape[0]:
        per_frame[1:] = velocity.mean(axis=1)
        per_frame[0] = per_frame[1]
    return per_frame


def detect_boundary_trim(
    positions: np.ndarray,
    fps: int = 30,
    speed_threshold_mps: float = 0.05,
    min_idle_seconds: float = 4.0,
    scan_seconds: float = 8.0,
    context_frames: int = 8,
    min_active_seconds: float = 0.5,
) -> dict[str, int | float | bool]:
    """Trim only long inactive runs touching the beginning or end.

    The scan is boundary-only: an inactive run in the middle of a source is
    never returned as a trim.  ``scan_seconds`` is recorded as an explicit
    policy parameter for reproducibility; a boundary run may extend beyond
    that scan window, but no interior run can qualify.
    """

    positions = np.asarray(positions, dtype=np.float32)
    if positions.ndim != 3 or positions.shape[0] < 2:
        raise ValueError(f"expected positions shape (T,J,3), got {positions.shape}")
    speed = _per_frame_joint_speed(positions, fps=int(fps))
    active = speed > float(speed_threshold_mps)
    min_idle_frames = int(round(float(min_idle_seconds) * float(fps)))
    min_active_frames = max(1, int(round(float(min_active_seconds) * float(fps))))
    scan_frames = int(round(float(scan_seconds) * float(fps)))

    leading_active = None
    for index in range(0, max(0, active.shape[0] - min_active_frames + 1)):
        if bool(np.all(active[index : index + min_active_frames])):
            leading_active = index
            break
    leading_idle = int(leading_active if leading_active is not None else 0)
    leading_trim = 0
    if leading_active is not None and leading_idle >= min_idle_frames:
        leading_trim = max(0, int(leading_active) - int(context_frames))

    trailing_active_end = None
    for index in range(active.shape[0] - min_active_frames, -1, -1):
        if bool(np.all(active[index : index + min_active_frames])):
            trailing_active_end = index + min_active_frames - 1
            break
    trailing_idle = int(
        active.shape[0] - 1 - trailing_active_end
        if trailing_active_end is not None
        else active.shape[0]
    )
    trailing_trim = active.shape[0]
    if trailing_active_end is not None and trailing_idle >= min_idle_frames:
        trailing_trim = min(
            active.shape[0],
            int(active.shape[0] - trailing_idle + context_frames),
        )

    all_inactive = trailing_active_end is None
    return {
        "boundary_scan_frames": int(min(scan_frames, positions.shape[0])),
        "boundary_scan_seconds": float(min(scan_frames, positions.shape[0]) / float(fps)),
        "leading_idle_frames": leading_idle,
        "leading_idle_seconds": float(leading_idle / float(fps)),
        "trailing_idle_frames": trailing_idle,
        "trailing_idle_seconds": float(trailing_idle / float(fps)),
        "trim_start_frame": int(leading_trim),
        "trim_end_frame": int(trailing_trim),
        "trim_applied": bool(leading_trim > 0 or trailing_trim < positions.shape[0]),
        "all_inactive": bool(all_inactive),
    }


def floor_plane_metrics(
    positions: np.ndarray,
    fps: int = 30,
    thresholds: Mapping[str, float] | None = None,
) -> dict[str, float | int | bool]:
    """Fit a robust floor plane from low, slow ankle/toe contact points.

    A height offset is harmless and is handled separately.  A persistent
    plane tilt is a source-level quality failure; this function only measures
    it and never edits coordinates.
    """

    positions = np.asarray(positions, dtype=np.float32)
    if positions.ndim != 3 or positions.shape[1:] != (55, 3):
        raise ValueError(f"expected positions shape (T,55,3), got {positions.shape}")
    config = dict(DEFAULT_THRESHOLDS)
    if thresholds:
        config.update({str(k): float(v) for k, v in thresholds.items()})
    feet = positions[:, FOOT_JOINTS, :]
    foot_speed = np.zeros((feet.shape[0], feet.shape[1]), dtype=np.float32)
    if feet.shape[0] > 1:
        foot_speed[1:] = np.linalg.norm(np.diff(feet, axis=0), axis=-1) * float(fps)
        foot_speed[0] = foot_speed[1]
    foot_y = feet[:, :, 1]
    foot_min = np.min(foot_y, axis=1)
    floor_offset_y = float(np.percentile(foot_min, 5.0))
    # Use the lowest foot/toe point once per frame and retain only frames in
    # which that point is slow.  A height gate would hide a genuinely sloped
    # floor by discarding its higher end; the robust fit/error checks below
    # handle airborne or noisy lowest points.
    lowest_joint = np.argmin(foot_y, axis=1)
    frame_index = np.arange(feet.shape[0])
    lowest_points = feet[frame_index, lowest_joint]
    lowest_speed = foot_speed[frame_index, lowest_joint]
    points = lowest_points[lowest_speed <= float(config["floor_contact_speed_mps"])]
    if points.shape[0] == 0:
        return {
            "floor_offset_y": floor_offset_y,
            "floor_plane_a": 0.0,
            "floor_plane_b": 0.0,
            "floor_plane_tilt_deg": 0.0,
            "floor_plane_p95_abs_error_m": 1.0e9,
            "floor_plane_rmse_m": 1.0e9,
            "floor_plane_points": 0,
            "floor_plane_xz_span_m": 0.0,
            "floor_plane_confident": False,
        }

    design = np.column_stack((points[:, 0], points[:, 2], np.ones(points.shape[0])))
    target = points[:, 1]
    keep = np.ones(points.shape[0], dtype=bool)
    coefficients = np.zeros((3,), dtype=np.float64)
    for _ in range(4):
        if int(np.sum(keep)) < 3:
            break
        coefficients, _, _, _ = np.linalg.lstsq(design[keep], target[keep], rcond=None)
        # Avoid spurious NumPy 2.0/macOS Accelerate matvec warnings while
        # preserving the exact least-squares prediction.
        residual = target - np.einsum("ij,j->i", design, coefficients, optimize=True)
        center = float(np.median(residual[keep]))
        mad = float(np.median(np.abs(residual[keep] - center)))
        scale = max(1.4826 * mad, 0.005)
        keep = np.abs(residual - center) <= max(0.04, 3.0 * scale)

    residual = target - np.einsum("ij,j->i", design, coefficients, optimize=True)
    selected_residual = residual[keep] if np.any(keep) else residual
    xz = points[keep][:, (0, 2)] if np.any(keep) else points[:, (0, 2)]
    xz_span = float(np.linalg.norm(np.ptp(xz, axis=0))) if xz.shape[0] else 0.0
    tilt = float(np.degrees(np.arctan(np.hypot(coefficients[0], coefficients[1]))))
    p95_error = float(np.percentile(np.abs(selected_residual), 95.0))
    rmse = float(np.sqrt(np.mean(np.square(selected_residual))))
    point_count = int(selected_residual.shape[0])
    confident = bool(
        point_count >= int(config["floor_plane_min_points"])
        and xz_span >= float(config["floor_plane_min_xz_span_m"])
    )
    return {
        "floor_offset_y": floor_offset_y,
        "floor_plane_a": float(coefficients[0]),
        "floor_plane_b": float(coefficients[1]),
        "floor_plane_tilt_deg": tilt,
        "floor_plane_p95_abs_error_m": p95_error,
        "floor_plane_rmse_m": rmse,
        "floor_plane_points": point_count,
        "floor_plane_xz_span_m": xz_span,
        "floor_plane_confident": confident,
    }


def sequence_metrics(
    positions: np.ndarray,
    fps: int = 30,
    thresholds: Mapping[str, float] | None = None,
) -> dict[str, float | int | bool]:
    """Compute interpretable activity, floor, foot, and tilt statistics."""

    positions = np.asarray(positions, dtype=np.float32)
    if positions.ndim != 3 or positions.shape[1:] != (55, 3):
        raise ValueError(f"expected positions shape (T,55,3), got {positions.shape}")
    if positions.shape[0] < 2:
        raise ValueError("at least two frames are required")

    quality_thresholds = dict(DEFAULT_THRESHOLDS)
    if thresholds:
        quality_thresholds.update({str(k): float(v) for k, v in thresholds.items()})

    y = positions[:, :, 1]
    foot_y = y[:, FOOT_JOINTS]
    hand_y = y[:, HAND_JOINTS]
    floor_y = float(np.percentile(y.reshape(-1), 1.0))
    foot_floor_y = float(np.percentile(np.min(foot_y, axis=1), 5.0))
    foot_min = np.min(foot_y, axis=1)
    hand_min = np.min(hand_y, axis=1)

    torso = positions[:, JOINTS["spine3"]] - positions[:, JOINTS["pelvis"]]
    torso_norm = np.linalg.norm(torso, axis=-1)
    torso_cos = torso[:, 1] / np.maximum(torso_norm, 1.0e-8)
    torso_tilt = np.degrees(np.arccos(np.clip(torso_cos, -1.0, 1.0)))

    velocity = np.linalg.norm(np.diff(positions, axis=0), axis=-1) * float(fps)
    per_frame_joint_speed = velocity.mean(axis=1)
    foot_velocity = np.linalg.norm(np.diff(positions[:, FOOT_JOINTS], axis=0), axis=-1) * float(fps)
    root_velocity = np.linalg.norm(np.diff(positions[:, 0], axis=0), axis=-1) * float(fps)
    # For pelvis height, use the robust foot-contact floor rather than the
    # lowest point among every joint. Hands or a horizontal torso can be below
    # the feet during floorwork and would otherwise make the pelvis look
    # artificially high. The fixed neutral template makes this metric
    # comparable across FineDance sources.
    pelvis_height = y[:, JOINTS["pelvis"]] - foot_floor_y
    low_pelvis = pelvis_height <= quality_thresholds["standing_low_pelvis_height_m"]
    low_pelvis_longest_run_frames = _longest_true_run(low_pelvis)
    knee_ceiling = np.minimum(
        y[:, JOINTS["left_knee"]],
        y[:, JOINTS["right_knee"]],
    )
    pelvis_below_knees = y[:, JOINTS["pelvis"]] <= (
        knee_ceiling + quality_thresholds["pelvis_below_knees_margin_m"]
    )
    leading = detect_leading_trim(
        positions,
        fps=fps,
        speed_threshold_mps=quality_thresholds["leading_idle_speed_mps"],
        min_idle_seconds=quality_thresholds["leading_idle_min_seconds"],
        min_active_seconds=quality_thresholds["leading_active_min_seconds"],
        context_frames=int(quality_thresholds["leading_context_frames"]),
    )
    boundary = detect_boundary_trim(
        positions,
        fps=fps,
        speed_threshold_mps=quality_thresholds["leading_idle_speed_mps"],
        min_idle_seconds=quality_thresholds["boundary_idle_min_seconds"],
        scan_seconds=quality_thresholds["boundary_scan_seconds"],
        context_frames=int(quality_thresholds["boundary_context_frames"]),
        min_active_seconds=quality_thresholds["leading_active_min_seconds"],
    )
    floor = floor_plane_metrics(positions, fps=fps, thresholds=quality_thresholds)

    return {
        "frames": int(positions.shape[0]),
        **leading,
        **boundary,
        **floor,
        "floor_height_y": floor_y,
        "foot_floor_height_y": foot_floor_y,
        "foot_floor_gap_m": float(foot_floor_y - floor_y),
        "foot_ground_contact_ratio": float(np.mean(foot_min <= floor_y + 0.08)),
        "hand_ground_contact_ratio": float(np.mean(hand_min <= floor_y + 0.08)),
        "pelvis_height_median_m": float(np.median(pelvis_height)),
        "pelvis_height_p05_m": float(np.percentile(pelvis_height, 5.0)),
        "low_pelvis_ratio": float(np.mean(low_pelvis)),
        "low_pelvis_longest_run_frames": int(low_pelvis_longest_run_frames),
        "low_pelvis_longest_run_seconds": float(low_pelvis_longest_run_frames / float(fps)),
        "pelvis_below_knees_ratio": float(np.mean(pelvis_below_knees)),
        "torso_tilt_median_deg": float(np.median(torso_tilt)),
        "torso_tilt_p95_deg": float(np.percentile(torso_tilt, 95.0)),
        "torso_non_upright_ratio": float(np.mean(torso_tilt >= 55.0)),
        "torso_extreme_tilt_ratio": float(np.mean(torso_tilt >= 80.0)),
        "mean_joint_speed_mps": float(velocity.mean()),
        "p95_joint_speed_mps": float(np.percentile(velocity, 95.0)),
        "active_frame_ratio_gt_0.05": float(np.mean(per_frame_joint_speed > 0.05)),
        "mean_foot_speed_mps": float(foot_velocity.mean()),
        "p95_foot_speed_mps": float(np.percentile(foot_velocity, 95.0)),
        "mean_root_speed_mps": float(root_velocity.mean()),
        "root_y_std_m": float(np.std(positions[:, 0, 1])),
        "pose_span_m": float(np.mean(np.ptp(positions, axis=0))),
    }


def classify_quality(
    metrics: Mapping[str, float],
    thresholds: Mapping[str, float] | None = None,
    ground_offset_outlier: bool = False,
) -> dict[str, object]:
    """Attach conservative review flags to one metric row.

    High foot speed is intentionally reported but does not independently
    exclude a motion: normal breaking footwork can also move the feet quickly.
    """

    config = dict(DEFAULT_THRESHOLDS)
    if thresholds:
        config.update({str(k): float(v) for k, v in thresholds.items()})
    flags: list[str] = []
    static = (
        float(metrics["mean_joint_speed_mps"]) <= config["static_mean_joint_speed_mps"]
        and float(metrics["active_frame_ratio_gt_0.05"]) <= config["static_active_frame_ratio"]
    )
    low_body = float(metrics["low_pelvis_ratio"]) >= config["floorwork_low_pelvis_ratio"]
    sustained_low_pelvis = (
        float(metrics["low_pelvis_ratio"]) >= config["standing_low_pelvis_ratio"]
        and float(metrics["low_pelvis_longest_run_seconds"])
        >= config["standing_low_pelvis_min_run_seconds"]
    )
    tilted = float(metrics["torso_non_upright_ratio"]) >= config["floorwork_non_upright_ratio"]
    hands_down = float(metrics["hand_ground_contact_ratio"]) >= config["floorwork_hand_ground_ratio"]
    feet_separated = (
        float(metrics["foot_floor_gap_m"]) >= config["floorwork_foot_floor_gap_m"]
        and float(metrics["foot_ground_contact_ratio"]) <= config["floorwork_foot_ground_contact_ratio"]
    )
    if static:
        # Informational only: mid-sequence choreography holds are retained.
        flags.append("low_activity_info")
    if sustained_low_pelvis:
        flags.append("sustained_low_pelvis_candidate")
    if bool(metrics.get("floor_plane_confident", False)):
        if float(metrics["floor_plane_tilt_deg"]) > config["floor_plane_warn_tilt_deg"]:
            flags.append("floor_tilt_warning")
        if (
            float(metrics["floor_plane_tilt_deg"]) > config["floor_plane_max_tilt_deg"]
            or float(metrics["floor_plane_p95_abs_error_m"]) > config["floor_plane_max_p95_error_m"]
        ):
            flags.append("sloped_floor_candidate")
    if (low_body and (tilted or hands_down)) or feet_separated:
        flags.append("floorwork_candidate")
    if (
        float(metrics["torso_extreme_tilt_ratio"]) >= config["floorwork_extreme_tilt_ratio"]
        and float(metrics["torso_tilt_p95_deg"]) >= config["floorwork_extreme_tilt_deg"]
    ):
        flags.append("extreme_tilt_candidate")
    if ground_offset_outlier:
        flags.append("ground_offset_outlier")
    high_foot_motion = float(metrics["p95_foot_speed_mps"]) >= 2.5
    if high_foot_motion:
        flags.append("high_foot_motion_info")
    exclusion_flags = {
        "sustained_low_pelvis_candidate",
        "floorwork_candidate",
        "extreme_tilt_candidate",
        "sloped_floor_candidate",
        "ground_offset_outlier",
    }
    return {
        "flags": flags,
        "review_candidate": bool(set(flags) & exclusion_flags),
        "high_foot_motion_only": bool(high_foot_motion and not (set(flags) & exclusion_flags)),
    }
