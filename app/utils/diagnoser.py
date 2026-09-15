"""Diagnoser: anomaly detection using absolute and relative kinetic-energy signals."""

from __future__ import annotations

from collections import OrderedDict

import numpy as np

from app.schemas.diagnostics import DiagnosticPeak, MotionDiagnostics
from app.schemas.motion import CanonicalMotion
from app.utils.kinematics import axis_angle_to_matrix, matrix_to_axis_angle

# A complete, non-overlapping partition of Canonical SMPL24 joints.
BODY_GROUPS: OrderedDict[str, tuple[int, ...]] = OrderedDict(
    (
        ("head", (12, 15)),
        ("torso", (0, 3, 6, 9, 13, 14)),
        ("left_arm", (16, 18, 20, 22)),
        ("right_arm", (17, 19, 21, 23)),
        ("left_leg", (1, 4, 7, 10)),
        ("right_leg", (2, 5, 8, 11)),
    )
)

_DISPLAY_BASELINE_WINDOW = 15
_DISPLAY_FLOOR_ROBUST_Z = 2.0
_DISPLAY_SATURATION_ROBUST_Z = 12.0


def _rolling_median(values: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """Return a centered median and MAD with edge-preserving padding."""

    if window < 3 or window % 2 == 0:
        raise ValueError("rolling median window must be odd and at least three")
    radius = window // 2
    padded = np.pad(np.asarray(values, dtype=np.float64), radius, mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, window)
    median = np.median(windows, axis=-1)
    mad = np.median(np.abs(windows - median[:, None]), axis=-1)
    return median, mad


def _anomaly_display_curve(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert raw energy into local Hampel scores and a quiet 0..1 display curve.

    Raw AKE/RKE remains available to callers. The display path works in log-energy
    space, subtracts a half-second rolling median, and suppresses everything below
    a robust z-score of two. Obvious discontinuities saturate at twelve and receive
    a small one-frame halo without reducing the actual peak.
    """

    curve = np.asarray(values, dtype=np.float64)
    if curve.ndim != 1 or not np.isfinite(curve).all() or np.any(curve < 0):
        raise ValueError("diagnostic energy curve must be finite, non-negative, and 1D")
    log_curve = np.log1p(curve)
    baseline, local_mad = _rolling_median(log_curve, _DISPLAY_BASELINE_WINDOW)
    global_median = float(np.median(log_curve))
    global_mad = float(np.median(np.abs(log_curve - global_median)))
    global_scale = 1.482602218505602 * global_mad
    scale_floor = max(global_scale * 0.15, 1e-6)
    scale = np.maximum(1.482602218505602 * local_mad, scale_floor)
    scores = np.maximum(log_curve - baseline, 0.0) / scale

    normalized = np.clip(
        (scores - _DISPLAY_FLOOR_ROBUST_Z)
        / (_DISPLAY_SATURATION_ROBUST_Z - _DISPLAY_FLOOR_ROBUST_Z),
        0.0,
        1.0,
    )
    normalized = normalized * normalized * (3.0 - 2.0 * normalized)
    halo = normalized.copy()
    if len(halo) > 1:
        halo[1:] = np.maximum(halo[1:], normalized[:-1] * 0.28)
        halo[:-1] = np.maximum(halo[:-1], normalized[1:] * 0.28)
    return scores, halo


def _robust_positive_peaks(
    values: np.ndarray,
    scores: np.ndarray,
    curve: str,
    fps: int,
    threshold: float,
) -> list[DiagnosticPeak]:
    curve_values = np.asarray(values, dtype=np.float64)
    candidates = np.flatnonzero(scores >= threshold)
    peaks: list[DiagnosticPeak] = []
    for run in np.split(candidates, np.flatnonzero(np.diff(candidates) > 1) + 1):
        if len(run) == 0:
            continue
        index = int(max(run, key=lambda item: (scores[item], curve_values[item], -item)))
        peaks.append(
            DiagnosticPeak(
                curve=curve,
                frame_index=int(index),
                time_sec=float(index / fps),
                value=float(curve_values[index]),
                robust_z=float(scores[index]),
            )
        )
    return peaks


def diagnose_motion(
    motion: CanonicalMotion, *, peak_threshold_robust_z: float = 6.0
) -> MotionDiagnostics:
    """Compute AKE from world joint velocity and RKE from local SO(3) velocity.

    For frame ``p`` and joint ``j``:

    ``v[p,j] = (o[p,j] - o[p-1,j]) / dt``

    ``omega[p,j] = Log(R[p-1,j]^T R[p,j]) / dt``

    AKE is the mean squared translational speed over the torso group. Each RKE
    curve is the mean squared angular speed over its corresponding body group.
    Frame zero is defined as zero because no preceding sample exists.
    """

    if not np.isfinite(peak_threshold_robust_z) or peak_threshold_robust_z <= 0:
        raise ValueError("peak threshold must be finite and positive")
    frames = motion.frames
    dt = 1.0 / motion.fps
    torso = np.asarray(BODY_GROUPS["torso"], dtype=np.int64)

    velocity = np.zeros_like(motion.joint_positions, dtype=np.float64)
    if frames > 1:
        velocity[1:] = np.diff(motion.joint_positions.astype(np.float64), axis=0) / dt
    squared_speed = np.sum(velocity * velocity, axis=-1)
    ake = np.mean(squared_speed[:, torso], axis=1)

    matrices = axis_angle_to_matrix(motion.smpl_poses.reshape(frames, 24, 3))
    angular_velocity = np.zeros((frames, 24, 3), dtype=np.float64)
    if frames > 1:
        relative = np.einsum(
            "tjki,tjkl->tjil", matrices[:-1], matrices[1:]
        )
        angular_velocity[1:] = matrix_to_axis_angle(relative).astype(np.float64) / dt
    squared_angular_speed = np.sum(angular_velocity * angular_velocity, axis=-1)
    rke = {
        name: np.mean(squared_angular_speed[:, np.asarray(indices)], axis=1)
        for name, indices in BODY_GROUPS.items()
    }
    if not np.isfinite(ake).all() or any(not np.isfinite(curve).all() for curve in rke.values()):
        raise ValueError("AKE/RKE computation produced non-finite values")

    ake_scores, ake_anomaly = _anomaly_display_curve(ake)
    rke_scored = {name: _anomaly_display_curve(curve) for name, curve in rke.items()}
    peaks = _robust_positive_peaks(
        ake, ake_scores, "ake", motion.fps, peak_threshold_robust_z
    )
    for name, curve in rke.items():
        peaks.extend(
            _robust_positive_peaks(
                curve,
                rke_scored[name][0],
                f"rke.{name}",
                motion.fps,
                peak_threshold_robust_z,
            )
        )
    peaks.sort(key=lambda item: (item.frame_index, item.curve))
    return MotionDiagnostics(
        frame_count=frames,
        dt=dt,
        coordinate_system=motion.coordinate_system.value,
        position_convention=(
            "Canonical SMPL24 world joint positions; finite differences in source "
            "coordinate units per second"
        ),
        rotation_convention=(
            "local active right-handed SMPL axis-angle rotations in radians; "
            "delta=R_prev^T@R_current; principal SO(3) logarithm"
        ),
        group_mapping={name: list(indices) for name, indices in BODY_GROUPS.items()},
        ake=ake.astype(float).tolist(),
        rke={name: curve.astype(float).tolist() for name, curve in rke.items()},
        ake_anomaly=ake_anomaly.astype(float).tolist(),
        rke_anomaly={
            name: display.astype(float).tolist()
            for name, (_, display) in rke_scored.items()
        },
        peaks=peaks,
        peak_threshold_robust_z=peak_threshold_robust_z,
        display_floor_robust_z=_DISPLAY_FLOOR_ROBUST_Z,
        display_saturation_robust_z=_DISPLAY_SATURATION_ROBUST_Z,
        display_filter="log-energy + 15-frame rolling Hampel baseline + smoothstep emphasis",
    )
