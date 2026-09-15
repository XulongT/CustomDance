"""Adapters between Canonical SMPL and Completer 151D."""

from __future__ import annotations

import numpy as np

from app.schemas.motion import CanonicalMotion, CompleterInternalMotion
from app.utils.kinematics import (
    axis_angle_to_matrix,
    convert_finedance_y_up_to_canonical,
    foot_contacts,
    forward_kinematics,
    matrix_to_axis_angle,
    matrix_to_rotation_6d,
    rotation_6d_to_matrix,
)


def canonical_from_finedance(
    root_translation: np.ndarray,
    local_axis_angle: np.ndarray,
    source_clip_id: str,
    metadata: dict | None = None,
) -> CanonicalMotion:
    poses, translation = convert_finedance_y_up_to_canonical(
        local_axis_angle, root_translation
    )
    joints = forward_kinematics(poses, translation)
    return CanonicalMotion(
        smpl_poses=poses,
        smpl_trans=translation,
        joint_positions=joints,
        source_clip_id=source_clip_id,
        metadata={"source_coordinate_system": "right-handed-y-up", **(metadata or {})},
    )


def canonical_to_completer(motion: CanonicalMotion) -> CompleterInternalMotion:
    matrices = axis_angle_to_matrix(motion.smpl_poses.reshape(-1, 24, 3))
    rotation_6d = matrix_to_rotation_6d(matrices).reshape(motion.frames, 144)
    contacts = foot_contacts(motion.joint_positions)
    features = np.concatenate((contacts, motion.smpl_trans, rotation_6d), axis=-1)
    return CompleterInternalMotion(
        features=features,
        source_clip_id=motion.source_clip_id,
        metadata={"normalization": "not_applied", "coordinate_system": "right-handed-z-up"},
    )


def completer_to_canonical(
    motion: CompleterInternalMotion, metadata: dict | None = None
) -> CanonicalMotion:
    matrices = rotation_6d_to_matrix(motion.rotation_6d)
    poses = matrix_to_axis_angle(matrices).reshape(-1, 72)
    translation = motion.root_translation.astype(np.float32, copy=True)
    joints = forward_kinematics(poses, translation)
    return CanonicalMotion(
        smpl_poses=poses,
        smpl_trans=translation,
        joint_positions=joints,
        source_clip_id=motion.source_clip_id,
        metadata={"completer_contacts": motion.contacts.tolist(), **(metadata or {})},
    )
