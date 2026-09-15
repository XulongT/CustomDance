"""Deterministic range smoothing for AKE translation and RKE joint groups."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from app.schemas.motion import CanonicalMotion
from app.utils.diagnoser import BODY_GROUPS
from app.utils.kinematics import forward_kinematics


def _hermite_value(
    values: np.ndarray,
    left: int,
    right: int,
    frame: int,
) -> np.ndarray:
    """Interpolate one frame while retaining the anchor-side velocities."""

    interval = float(right - left)
    t = (frame - left) / interval
    t2 = t * t
    t3 = t2 * t
    p0 = values[left]
    p1 = values[right]
    v0 = values[left] - values[left - 1] if left > 0 else (p1 - p0) / interval
    v1 = (
        values[right + 1] - values[right]
        if right + 1 < values.shape[0]
        else (p1 - p0) / interval
    )
    h00 = 2.0 * t3 - 3.0 * t2 + 1.0
    h10 = t3 - 2.0 * t2 + t
    h01 = -2.0 * t3 + 3.0 * t2
    h11 = t3 - t2
    return h00 * p0 + h10 * interval * v0 + h01 * p1 + h11 * interval * v1


def _smooth_joint_rotations(
    poses: np.ndarray,
    start_frame: int,
    end_frame: int,
    joint_indices: tuple[int, ...],
) -> None:
    left = start_frame - 1
    right = end_frame
    sample_times = np.arange(start_frame, end_frame, dtype=np.float64)
    key_times = np.asarray([left, right], dtype=np.float64)
    for joint in joint_indices:
        offset = joint * 3
        anchors = Rotation.from_rotvec(poses[[left, right], offset : offset + 3])
        poses[start_frame:end_frame, offset : offset + 3] = (
            Slerp(key_times, anchors)(sample_times).as_rotvec().astype(np.float32)
        )


def smooth_motion_range(
    motion: CanonicalMotion,
    start_frame: int,
    end_frame: int,
    *,
    smooth_translation: bool,
    joint_groups: list[str] | tuple[str, ...],
) -> CanonicalMotion:
    """Smooth ``[start_frame, end_frame)`` without generative inpainting.

    AKE maps to root translation and RKE selections map to the canonical,
    non-overlapping SMPL24 groups in :data:`BODY_GROUPS`. The selected range
    must retain one immutable frame on each side because those frames are the
    interpolation anchors.
    """

    if not 0 < start_frame < end_frame < motion.frames:
        raise ValueError(
            "Smooth range must leave one anchor frame before and after the selection"
        )
    groups = tuple(dict.fromkeys(joint_groups))
    unknown = sorted(set(groups) - set(BODY_GROUPS))
    if unknown:
        raise ValueError(f"unknown RKE joint group(s): {', '.join(unknown)}")
    if not smooth_translation and not groups:
        raise ValueError("Smooth requires AKE translation or at least one RKE joint group")

    poses = motion.smpl_poses.copy()
    translation = motion.smpl_trans.copy()
    if smooth_translation:
        left = start_frame - 1
        right = end_frame
        source_translation = translation.copy()
        for frame in range(start_frame, end_frame):
            translation[frame] = _hermite_value(
                source_translation,
                left,
                right,
                frame,
            )

    joint_indices = tuple(
        sorted({joint for group in groups for joint in BODY_GROUPS[group]})
    )
    if joint_indices:
        _smooth_joint_rotations(poses, start_frame, end_frame, joint_indices)

    joint_positions = forward_kinematics(poses, translation)
    history = list(motion.metadata.get("motion_smoothing", []))
    history.append(
        {
            "method": "endpoint Hermite translation + shortest-arc joint SLERP",
            "start_frame": start_frame,
            "end_frame": end_frame,
            "smoothed_root_translation": smooth_translation,
            "smoothed_joint_groups": list(groups),
            "smoothed_joint_indices": list(joint_indices),
            "range_semantics": "start inclusive, end exclusive",
            "exterior_frames_preserved": True,
            "unselected_local_rotations_preserved": True,
        }
    )
    return CanonicalMotion(
        smpl_poses=poses,
        smpl_trans=translation,
        joint_positions=joint_positions,
        source_clip_id=f"smooth-{motion.source_clip_id}",
        coordinate_system=motion.coordinate_system,
        fps=motion.fps,
        schema_version=motion.schema_version,
        metadata={**deepcopy(motion.metadata), "motion_smoothing": history},
    )


__all__ = ["smooth_motion_range"]
