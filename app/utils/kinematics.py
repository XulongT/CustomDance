"""SMPL24 rotations, coordinate conversion, FK, contacts, and resampling."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

SMPL24_NAMES = (
    "root", "left_hip", "right_hip", "spine1", "left_knee", "right_knee",
    "spine2", "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot",
    "neck", "left_collar", "right_collar", "head", "left_shoulder",
    "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
    "left_hand", "right_hand",
)

SMPL24_PARENTS = np.asarray(
    [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21],
    dtype=np.int64,
)

# Standard neutral SMPL24 rest offsets used by FlowerDance's public implementation.
SMPL24_OFFSETS_Y_UP = np.asarray(
    [
        [0.0, 0.0, 0.0],
        [0.05858135, -0.08228004, -0.01766408],
        [-0.06030973, -0.09051332, -0.01354254],
        [0.00443945, 0.12440352, -0.03838522],
        [0.04345142, -0.38646945, 0.00803700],
        [-0.04325663, -0.38368791, -0.00484304],
        [0.00448844, 0.13795640, 0.02682033],
        [-0.01479032, -0.42687458, -0.03742800],
        [0.01905555, -0.42004550, -0.03456167],
        [-0.00226458, 0.05603239, 0.00285505],
        [0.04105436, -0.06028581, 0.12204243],
        [-0.03483987, -0.06210566, 0.13032329],
        [-0.01339020, 0.21163553, -0.03346758],
        [0.07170245, 0.11399969, -0.01889817],
        [-0.08295366, 0.11247234, -0.02370739],
        [0.01011321, 0.08893734, 0.05040987],
        [0.12292141, 0.04520509, -0.01904600],
        [-0.11322832, 0.04685326, -0.00847207],
        [0.25533190, -0.01564902, -0.02294649],
        [-0.26012748, -0.01436928, -0.03126873],
        [0.26570925, 0.01269811, -0.00737473],
        [-0.26910836, 0.00679372, -0.00602676],
        [0.08669055, -0.01063603, -0.01559429],
        [-0.08875370, -0.00865157, -0.01010708],
    ],
    dtype=np.float32,
)

ROTATE_Y_UP_TO_Z_UP = np.asarray(
    [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]], dtype=np.float64
)


def axis_angle_to_matrix(axis_angle: np.ndarray) -> np.ndarray:
    array = np.asarray(axis_angle, dtype=np.float64)
    if array.shape[-1] != 3:
        raise ValueError("axis-angle input must end with dimension 3")
    matrices = Rotation.from_rotvec(array.reshape(-1, 3)).as_matrix()
    return matrices.reshape(array.shape[:-1] + (3, 3))


def matrix_to_axis_angle(matrix: np.ndarray) -> np.ndarray:
    array = np.asarray(matrix, dtype=np.float64)
    if array.shape[-2:] != (3, 3):
        raise ValueError("rotation matrix input must end with shape (3, 3)")
    vectors = Rotation.from_matrix(array.reshape(-1, 3, 3)).as_rotvec()
    return vectors.reshape(array.shape[:-2] + (3,)).astype(np.float32)


def matrix_to_rotation_6d(matrix: np.ndarray) -> np.ndarray:
    array = np.asarray(matrix, dtype=np.float64)
    if array.shape[-2:] != (3, 3):
        raise ValueError("rotation matrix input must end with shape (3, 3)")
    return array[..., :2, :].reshape(array.shape[:-2] + (6,)).astype(np.float32)


def rotation_6d_to_matrix(rotation_6d: np.ndarray) -> np.ndarray:
    array = np.asarray(rotation_6d, dtype=np.float64)
    if array.shape[-1] != 6:
        raise ValueError("6D rotation input must end with dimension 6")
    first = array[..., :3]
    second = array[..., 3:]
    first_norm = np.linalg.norm(first, axis=-1, keepdims=True)
    if np.any(first_norm < 1e-8):
        raise ValueError("degenerate first 6D rotation vector")
    basis1 = first / first_norm
    second_orthogonal = second - np.sum(basis1 * second, axis=-1, keepdims=True) * basis1
    second_norm = np.linalg.norm(second_orthogonal, axis=-1, keepdims=True)
    if np.any(second_norm < 1e-8):
        raise ValueError("degenerate second 6D rotation vector")
    basis2 = second_orthogonal / second_norm
    basis3 = np.cross(basis1, basis2)
    return np.stack((basis1, basis2, basis3), axis=-2)


def forward_kinematics(
    local_axis_angle: np.ndarray,
    root_translation: np.ndarray,
    offsets: np.ndarray = SMPL24_OFFSETS_Y_UP,
) -> np.ndarray:
    rotations = axis_angle_to_matrix(np.asarray(local_axis_angle).reshape(-1, 24, 3))
    translation = np.asarray(root_translation, dtype=np.float64)
    if translation.shape != (rotations.shape[0], 3):
        raise ValueError("root_translation must have shape (T, 3)")
    offsets = np.asarray(offsets, dtype=np.float64)
    if offsets.shape != (24, 3):
        raise ValueError("SMPL24 offsets must have shape (24, 3)")

    frames = rotations.shape[0]
    world_rotation = np.empty((frames, 24, 3, 3), dtype=np.float64)
    world_position = np.empty((frames, 24, 3), dtype=np.float64)
    for joint, parent in enumerate(SMPL24_PARENTS):
        if parent == -1:
            world_rotation[:, joint] = rotations[:, joint]
            world_position[:, joint] = translation
            continue
        world_position[:, joint] = (
            np.einsum("tij,j->ti", world_rotation[:, parent], offsets[joint])
            + world_position[:, parent]
        )
        world_rotation[:, joint] = np.einsum(
            "tij,tjk->tik", world_rotation[:, parent], rotations[:, joint]
        )
    if not np.isfinite(world_position).all():
        raise ValueError("forward kinematics produced non-finite positions")
    return world_position.astype(np.float32)


def convert_finedance_y_up_to_canonical(
    local_axis_angle: np.ndarray, root_translation: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    local = np.asarray(local_axis_angle, dtype=np.float64).reshape(-1, 24, 3)
    matrices = axis_angle_to_matrix(local)
    matrices[:, 0] = np.einsum("ij,tjk->tik", ROTATE_Y_UP_TO_Z_UP, matrices[:, 0])
    poses = matrix_to_axis_angle(matrices).reshape(-1, 72)
    translation = np.asarray(root_translation, dtype=np.float64) @ ROTATE_Y_UP_TO_Z_UP.T
    return poses.astype(np.float32), translation.astype(np.float32)


def foot_contacts(joint_positions: np.ndarray, threshold: float = 0.01) -> np.ndarray:
    positions = np.asarray(joint_positions, dtype=np.float32)
    if positions.ndim != 3 or positions.shape[1:] != (24, 3):
        raise ValueError("joint_positions must have shape (T, 24, 3)")
    feet = positions[:, [7, 8, 10, 11]]
    velocity = np.zeros(feet.shape[:2], dtype=np.float32)
    velocity[:-1] = np.linalg.norm(feet[1:] - feet[:-1], axis=-1)
    return (velocity < threshold).astype(np.float32)
