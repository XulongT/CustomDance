"""Robust, translation-only floor normalization for Canonical SMPL24 clips."""

from __future__ import annotations

from typing import Any

import numpy as np

from app.schemas.motion import CanonicalMotion

# SMPL24 ankles and feet in the project's established joint order.
FOOT_JOINTS = np.asarray((7, 8, 10, 11), dtype=np.int64)

# This is the median floor of the frozen Motion Library v2 corpus. Keeping that
# corpus-level height avoids shifting FlowerDance's established root-translation
# distribution while making every source clip share one explicit floor.
DEFAULT_LIBRARY_FLOOR_Z = -0.9368451237678528
FLOOR_NORMALIZATION_CONTRACT = "canonical-smpl24-contact-floor-constant-z-v1"


def estimate_contact_floor(joint_positions: np.ndarray) -> dict[str, Any]:
    """Estimate floor height from low, slow SMPL foot-contact samples.

    The estimate is invariant to a constant vertical translation. It never
    changes pose or root dynamics; it only supplies diagnostics for a later
    constant-Z normalization.
    """

    positions = np.asarray(joint_positions, dtype=np.float64)
    if positions.ndim != 3 or positions.shape[1:] != (24, 3):
        raise ValueError("joint_positions must have shape (T, 24, 3)")
    if positions.shape[0] < 2 or not np.isfinite(positions).all():
        raise ValueError("joint_positions must contain at least two finite frames")

    feet = positions[:, FOOT_JOINTS]
    heights = feet[:, :, 2]
    frame_delta = np.diff(feet, axis=0)
    speeds = np.empty_like(heights)
    speeds[1:] = np.linalg.norm(frame_delta, axis=2)
    speeds[0] = speeds[1]

    height_limit = float(np.quantile(heights, 0.35))
    speed_limit = max(float(np.quantile(speeds, 0.40)), 1e-6)
    contact_mask = (heights <= height_limit) & (speeds <= speed_limit)
    minimum_samples = max(12, positions.shape[0] // 10)
    method = "low_height_low_velocity"
    if int(contact_mask.sum()) < minimum_samples:
        height_limit = float(np.quantile(heights, 0.20))
        contact_mask = heights <= height_limit + 1e-9
        method = "low_height_fallback"

    samples = heights[contact_mask]
    if samples.size == 0:
        raise ValueError("unable to estimate a floor from the foot samples")
    floor_z = float(np.median(samples))
    mad = float(np.median(np.abs(samples - floor_z)))
    selected_positions = feet[contact_mask]
    selected_frames = np.flatnonzero(np.any(contact_mask, axis=1))

    plane_tilt_deg: float | None = None
    plane_residual_mad: float | None = None
    xy = selected_positions[:, :2]
    if selected_positions.shape[0] >= 12:
        centered_xy = xy - np.median(xy, axis=0, keepdims=True)
        design = np.column_stack(
            (centered_xy, np.ones(selected_positions.shape[0], dtype=np.float64))
        )
        coefficients, _, rank, _ = np.linalg.lstsq(
            design, selected_positions[:, 2], rcond=None
        )
        if rank == 3 and float(np.max(np.ptp(xy, axis=0))) >= 0.05:
            predicted = design @ coefficients
            plane_tilt_deg = float(
                np.degrees(np.arctan(np.linalg.norm(coefficients[:2])))
            )
            plane_residual_mad = float(
                np.median(np.abs(selected_positions[:, 2] - predicted))
            )

    sample_score = min(1.0, samples.size / 48.0)
    coverage_score = min(1.0, selected_frames.size / max(1.0, positions.shape[0] * 0.25))
    dispersion_score = max(0.0, 1.0 - mad / 0.04)
    confidence = float(
        np.clip(0.35 * sample_score + 0.25 * coverage_score + 0.40 * dispersion_score, 0, 1)
    )
    return {
        "floor_z": floor_z,
        "method": method,
        "contact_sample_count": int(samples.size),
        "contact_frame_count": int(selected_frames.size),
        "contact_fraction": float(samples.size / heights.size),
        "contact_height_mad": mad,
        "contact_height_limit": height_limit,
        "contact_speed_limit": speed_limit,
        "plane_tilt_deg": plane_tilt_deg,
        "plane_residual_mad": plane_residual_mad,
        "confidence": confidence,
    }


def normalize_canonical_floor(
    motion: CanonicalMotion,
    *,
    target_floor_z: float = DEFAULT_LIBRARY_FLOOR_Z,
) -> tuple[CanonicalMotion, dict[str, Any]]:
    """Return a copy shifted by one constant Z offset to the library floor."""

    target = float(target_floor_z)
    if not np.isfinite(target):
        raise ValueError("target_floor_z must be finite")
    estimate = estimate_contact_floor(motion.joint_positions)
    source_floor = float(estimate["floor_z"])
    shift = target - source_floor

    translation = motion.smpl_trans.copy()
    joints = motion.joint_positions.copy()
    translation[:, 2] += shift
    joints[:, :, 2] += shift
    normalization = {
        "contract": FLOOR_NORMALIZATION_CONTRACT,
        "source_floor_z": source_floor,
        "target_floor_z": target,
        "floor_shift_z": shift,
        "translation_axes_changed": ["z"],
        "pose_changed": False,
        **{key: value for key, value in estimate.items() if key != "floor_z"},
    }
    metadata = dict(motion.metadata)
    metadata["library_floor_normalization"] = normalization
    normalized = CanonicalMotion(
        smpl_poses=motion.smpl_poses.copy(),
        smpl_trans=translation,
        joint_positions=joints,
        source_clip_id=motion.source_clip_id,
        coordinate_system=motion.coordinate_system,
        fps=motion.fps,
        schema_version=motion.schema_version,
        metadata=metadata,
    )
    return normalized, normalization


__all__ = [
    "DEFAULT_LIBRARY_FLOOR_Z",
    "FLOOR_NORMALIZATION_CONTRACT",
    "FOOT_JOINTS",
    "estimate_contact_floor",
    "normalize_canonical_floor",
]
