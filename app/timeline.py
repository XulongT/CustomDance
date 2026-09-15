"""Slot state machine and exact four-second Fill In editing."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np

from app.schemas.analysis import AnalyzeResult
from app.schemas.motion import CanonicalMotion
from app.schemas.timeline import SlotStatus, TimelineSlot, TimelineState
from app.utils.foot_contact import endpoint_supports
from app.utils.kinematics import (
    axis_angle_to_matrix,
    forward_kinematics,
    matrix_to_axis_angle,
)
from app.utils.motion_grounding import estimate_contact_floor

_MAX_HEADING_EXTRAPOLATION = np.deg2rad(20.0)
_MAX_GAP_HEADING_EXTRAPOLATION = np.deg2rad(45.0)
_MAX_HORIZONTAL_SPEED_PER_FRAME = 0.05
_MAX_GAP_EXTRAPOLATION_FRAMES = 30
_GAP_VELOCITY_DECAY = 0.85
_MAX_GAP_ROOT_DISPLACEMENT = 0.50
_MAX_HARD_SUPPORT_RELATIVE_MISMATCH_M = 0.12
_MAX_DUAL_SUPPORT_TARGET_SPREAD_M = 0.10
_TOUCHING_SEAM_HALF_WINDOW_FRAMES = 12
_TOUCHING_ROOT_XY_JUMP_M = 0.06
_TOUCHING_ROOT_Z_JUMP_M = 0.08
_TOUCHING_FOOT_JUMP_M = 0.12
_TOUCHING_LEG_ROTATION_DEGREES = 45.0
_LEG_JOINTS = np.asarray((1, 2, 4, 5, 7, 8, 10, 11), dtype=np.int64)


@dataclass(frozen=True, slots=True)
class CompletePreparation:
    """Idempotently rebuilt filled motion plus the effective generation mask."""

    editor: TimelineEditor
    known_mask: np.ndarray
    diagnostics: dict[str, Any]


def _wrap_angle(value: float) -> float:
    return float((value + np.pi) % (2.0 * np.pi) - np.pi)


def _root_heading(matrix: np.ndarray) -> float:
    """Return the character heading projected onto the canonical XY floor."""

    rotation = np.asarray(matrix, dtype=np.float64)
    forward = rotation[:2, 2]
    norm = float(np.linalg.norm(forward))
    if norm < 1e-6:
        # In the neutral SMPL basis local +X points right and local +Z points
        # forward. The fallback preserves that quarter-turn relationship.
        right = rotation[:2, 0]
        if float(np.linalg.norm(right)) < 1e-6:
            raise ValueError("root rotation has no stable horizontal heading")
        return _wrap_angle(float(np.arctan2(right[1], right[0])) - np.pi / 2.0)
    return float(np.arctan2(forward[1], forward[0]))


def _yaw_matrix(angle: float) -> np.ndarray:
    cosine = float(np.cos(angle))
    sine = float(np.sin(angle))
    return np.asarray(
        ((cosine, -sine, 0.0), (sine, cosine, 0.0), (0.0, 0.0, 1.0)),
        dtype=np.float64,
    )


def _decayed_gap_displacement(step: np.ndarray, horizon: int) -> tuple[np.ndarray, float]:
    """Return a decayed, distance-capped root displacement for an empty gap."""

    frames = min(max(int(horizon), 1), _MAX_GAP_EXTRAPOLATION_FRAMES)
    multiplier = float(
        np.sum(_GAP_VELOCITY_DECAY ** np.arange(frames, dtype=np.float64))
    )
    displacement = np.asarray(step, dtype=np.float64) * multiplier
    distance = float(np.linalg.norm(displacement))
    if distance > _MAX_GAP_ROOT_DISPLACEMENT:
        displacement *= _MAX_GAP_ROOT_DISPLACEMENT / distance
    return displacement, multiplier


def _continuity_alignment(
    motion: CanonicalMotion,
    filled_mask: np.ndarray,
    previous: int,
    alignment_gap_frames: int,
) -> tuple[float, np.ndarray, str, float, float]:
    """Predict the next clip root without treating a contact label as a hard anchor."""

    previous_matrix = axis_angle_to_matrix(motion.smpl_poses[previous, :3])
    previous_heading = _root_heading(previous_matrix)
    if previous > 0 and filled_mask[previous - 1]:
        previous_previous_matrix = axis_angle_to_matrix(
            motion.smpl_poses[previous - 1, :3]
        )
        previous_previous_heading = _root_heading(previous_previous_matrix)
        heading_step = np.clip(
            _wrap_angle(previous_heading - previous_previous_heading),
            -_MAX_HEADING_EXTRAPOLATION,
            _MAX_HEADING_EXTRAPOLATION,
        )
        horizontal_step = (
            motion.smpl_trans[previous, :2]
            - motion.smpl_trans[previous - 1, :2]
        ).astype(np.float64)
        speed = float(np.linalg.norm(horizontal_step))
        if speed > _MAX_HORIZONTAL_SPEED_PER_FRAME:
            horizontal_step *= _MAX_HORIZONTAL_SPEED_PER_FRAME / speed
        if alignment_gap_frames == 0:
            target_heading = previous_heading + float(heading_step)
            target_xy = motion.smpl_trans[previous, :2] + horizontal_step
            return (
                target_heading,
                target_xy,
                "previous_filled_velocity_extrapolation",
                1.0,
                float(np.linalg.norm(horizontal_step)),
            )
        horizon = min(
            alignment_gap_frames + 1,
            _MAX_GAP_EXTRAPOLATION_FRAMES,
        )
        displacement, multiplier = _decayed_gap_displacement(horizontal_step, horizon)
        heading_offset = np.clip(
            float(heading_step) * multiplier,
            -_MAX_GAP_HEADING_EXTRAPOLATION,
            _MAX_GAP_HEADING_EXTRAPOLATION,
        )
        return (
            previous_heading + float(heading_offset),
            motion.smpl_trans[previous, :2] + displacement,
            "previous_filled_gap_decayed_velocity_extrapolation",
            multiplier,
            float(np.linalg.norm(displacement)),
        )
    return (
        previous_heading,
        motion.smpl_trans[previous, :2].copy(),
        "previous_filled_frame" if alignment_gap_frames == 0 else "previous_filled_gap_frame",
        0.0,
        0.0,
    )


def estimate_motion_floor_z(joint_positions: np.ndarray) -> float:
    """Estimate the canonical floor using the shared library contact contract."""

    return float(estimate_contact_floor(joint_positions)["floor_z"])


def _declared_library_floor_z(clip: CanonicalMotion, actual_floor_z: float) -> float | None:
    payload = clip.metadata.get("library_floor_normalization")
    if not isinstance(payload, dict):
        return None
    value = payload.get("target_floor_z")
    if not isinstance(value, (int, float)) or not np.isfinite(value):
        return None
    declared = float(value)
    # A v2.1 clip is already normalized. Trust the declaration only after a
    # small read-time consistency check; otherwise use the legacy fallback.
    if abs(actual_floor_z - declared) > 0.01:
        return None
    return declared


def ranges_overlap(start_a: float, end_a: float, start_b: float, end_b: float) -> bool:
    """Return strict overlap; touching endpoints do not conflict."""

    return max(start_a, start_b) < min(end_a, end_b)


def empty_timeline_motion(duration_sec: float) -> CanonicalMotion:
    frames = round(duration_sec * 30)
    if frames <= 0 or abs(frames / 30 - duration_sec) > 1 / 30 + 1e-9:
        raise ValueError("timeline duration must map to a positive 30 FPS frame count")
    poses = np.zeros((frames, 72), dtype=np.float32)
    # Canonical motion is Z-up while the neutral SMPL rest offsets are Y-up.
    # FineDance clips carry this +90° X basis conversion in their root pose;
    # the editable placeholder must do the same or it lies flat on the floor.
    poses[:, 0] = np.pi / 2
    translation = np.zeros((frames, 3), dtype=np.float32)
    return CanonicalMotion(
        smpl_poses=poses,
        smpl_trans=translation,
        joint_positions=forward_kinematics(poses, translation),
        source_clip_id="timeline-empty",
        metadata={"kind": "editable_timeline", "filled_ranges": []},
    )


def _connection_metrics(
    motion: CanonicalMotion,
    previous_assignment: dict[str, Any],
    following_assignment: dict[str, Any],
) -> dict[str, Any]:
    left = int(previous_assignment["end_frame"]) - 1
    right = int(following_assignment["start_frame"])
    if not 0 <= left < right < motion.frames:
        raise ValueError("slot assignments must be ordered, non-overlapping frame ranges")
    feet = np.asarray((7, 8, 10, 11), dtype=np.int64)
    root_xy_jump = float(
        np.linalg.norm(motion.smpl_trans[right, :2] - motion.smpl_trans[left, :2])
    )
    root_z_jump = abs(float(motion.smpl_trans[right, 2] - motion.smpl_trans[left, 2]))
    foot_jumps = np.linalg.norm(
        motion.joint_positions[right, feet, :2]
        - motion.joint_positions[left, feet, :2],
        axis=1,
    )
    matrices = axis_angle_to_matrix(
        motion.smpl_poses[[left, right]].reshape(-1, 24, 3)
    )
    relative = np.einsum("aji,ajk->aik", matrices[0], matrices[1])
    rotation_degrees = np.degrees(
        np.linalg.norm(matrix_to_axis_angle(relative), axis=-1)
    )
    gap_frames = right - left - 1
    return {
        "slot_pair": [
            str(previous_assignment["slot_id"]),
            str(following_assignment["slot_id"]),
        ],
        "known_endpoint_frames": [left, right],
        "gap_frames": gap_frames,
        "root_xy_jump_m": root_xy_jump,
        "root_z_jump_m": root_z_jump,
        "maximum_foot_xy_jump_m": float(np.max(foot_jumps, initial=0.0)),
        "foot_xy_jump_m": foot_jumps.astype(float).tolist(),
        "maximum_leg_rotation_degrees": float(
            np.max(rotation_degrees[_LEG_JOINTS], initial=0.0)
        ),
        "following_alignment": str(following_assignment.get("root_alignment", "")),
        "support_constraints_compatible": bool(
            following_assignment.get("support_constraints_compatible", False)
        ),
        "common_support_sides": list(
            following_assignment.get("common_support_sides", [])
        ),
        "support_target_spread_m": float(
            following_assignment.get("support_target_spread_m", 0.0)
        ),
        "maximum_support_relative_mismatch_m": float(
            following_assignment.get("maximum_support_relative_mismatch_m", 0.0)
        ),
    }


class TimelineEditor:
    def __init__(
        self,
        state: TimelineState,
        motion: CanonicalMotion,
        filled_mask: np.ndarray | None = None,
        assignments: list[dict] | None = None,
    ):
        self.state = state
        self.motion = motion
        self.filled_mask = np.asarray(
            filled_mask if filled_mask is not None else np.zeros(motion.frames, dtype=bool),
            dtype=bool,
        )
        if self.filled_mask.shape != (motion.frames,):
            raise ValueError("filled_mask must match timeline frames")
        self.assignments = list(assignments or [])

    @classmethod
    def from_analysis(cls, analysis: AnalyzeResult) -> TimelineEditor:
        slots = [
            TimelineSlot(
                slot_id=suggestion.slot_id,
                start_sec=suggestion.start_sec,
                duration_sec=suggestion.duration_sec,
                music_description=suggestion.music_description,
                cue=suggestion.cue,
            )
            for suggestion in analysis.slots
        ]
        if len({slot.slot_id for slot in slots}) != len(slots):
            raise ValueError("Analyze result contains duplicate slot IDs")
        return cls(TimelineState(slots=slots), empty_timeline_motion(analysis.duration_sec))

    def _slot(self, slot_id: str) -> TimelineSlot:
        for slot in self.state.slots:
            if slot.slot_id == slot_id:
                return slot
        raise KeyError(f"unknown slot_id: {slot_id}")

    def focus(self, slot_id: str) -> TimelineState:
        selected = self._slot(slot_id)
        if selected.status is SlotStatus.INVALID:
            raise ValueError("invalid slots cannot be focused or accepted")
        if selected.status is SlotStatus.CANDIDATE:
            selected.status = SlotStatus.ACCEPTED
            for slot in self.state.slots:
                if (
                    slot.slot_id != selected.slot_id
                    and slot.status is SlotStatus.CANDIDATE
                    and ranges_overlap(
                        selected.start_sec,
                        selected.end_sec,
                        slot.start_sec,
                        slot.end_sec,
                    )
                ):
                    slot.status = SlotStatus.INVALID
        self.state.current_focused_slot_id = selected.slot_id
        return self.state

    def fill(self, clip: CanonicalMotion, *, retrieval_score: float) -> TimelineState:
        focused_id = self.state.current_focused_slot_id
        if focused_id is None:
            raise ValueError("Fill In requires a current focused slot")
        slot = self._slot(focused_id)
        if slot.status is not SlotStatus.ACCEPTED:
            raise ValueError("Fill In requires the focused slot to be accepted and unfilled")
        if clip.frames != 120 or abs(clip.duration_sec - 4.0) > 1e-9:
            raise ValueError("Fill In motion must be exactly 4 seconds / 120 frames")
        start_frame = round(slot.start_sec * self.motion.fps)
        end_frame = start_frame + 120
        if end_frame > self.motion.frames:
            raise ValueError("slot exceeds timeline motion range")

        measured_source_floor_z = estimate_motion_floor_z(clip.joint_positions)
        declared_source_floor_z = _declared_library_floor_z(
            clip, measured_source_floor_z
        )
        source_floor_z = (
            declared_source_floor_z
            if declared_source_floor_z is not None
            else measured_source_floor_z
        )
        stored_floor_z = self.motion.metadata.get("timeline_floor_z")
        if isinstance(stored_floor_z, (int, float)) and np.isfinite(stored_floor_z):
            target_floor_z = float(stored_floor_z)
        elif self.filled_mask.any():
            target_floor_z = estimate_motion_floor_z(
                self.motion.joint_positions[self.filled_mask]
            )
        else:
            # The first accepted clip establishes the timeline floor without
            # changing its vertical motion. Later clips are normalized to it.
            target_floor_z = source_floor_z
        vertical_shift = target_floor_z - source_floor_z

        clip_root_matrices = axis_angle_to_matrix(clip.smpl_poses[:, :3])
        source_heading = _root_heading(clip_root_matrices[0])
        previous_filled = np.flatnonzero(self.filled_mask[:start_frame])
        alignment_gap_frames: int | None = None
        common_support_sides: tuple[str, ...] = ()
        hard_support_sides: tuple[str, ...] = ()
        previous_supports = {}
        source_supports = {}
        support_alignment: list[dict] = []
        support_target_spread_m = 0.0
        maximum_support_relative_mismatch_m = 0.0
        support_constraints_compatible = False
        root_extrapolation_displacement_m = 0.0
        root_extrapolation_multiplier = 0.0
        if previous_filled.size:
            previous = int(previous_filled[-1])
            alignment_gap_frames = start_frame - previous - 1
            previous_matrix = axis_angle_to_matrix(self.motion.smpl_poses[previous, :3])
            previous_heading = _root_heading(previous_matrix)
            contiguous_start = previous
            while contiguous_start > 0 and self.filled_mask[contiguous_start - 1]:
                contiguous_start -= 1
            previous_supports = endpoint_supports(
                self.motion.joint_positions[contiguous_start : previous + 1],
                floor_z=target_floor_z,
                at_start=False,
            )
            source_supports = endpoint_supports(
                clip.joint_positions,
                floor_z=source_floor_z,
                at_start=True,
            )
            common_support_sides = tuple(
                side
                for side in ("left", "right")
                if side in previous_supports and side in source_supports
            )
            (
                continuity_heading,
                continuity_xy,
                continuity_anchor,
                root_extrapolation_multiplier,
                root_extrapolation_displacement_m,
            ) = _continuity_alignment(
                self.motion,
                self.filled_mask,
                previous,
                alignment_gap_frames,
            )

            support_rotation = _yaw_matrix(
                _wrap_angle(previous_heading - source_heading)
            )
            compatibility_targets: list[np.ndarray] = []
            relative_mismatches: list[float] = []
            for side in common_support_sides:
                source_relative = (
                    source_supports[side].position_xy - clip.smpl_trans[0, :2]
                )
                root_target = previous_supports[side].position_xy - (
                    support_rotation[:2, :2] @ source_relative
                )
                compatibility_targets.append(root_target)
                relative_mismatches.append(
                    float(
                        np.linalg.norm(
                            root_target - self.motion.smpl_trans[previous, :2]
                        )
                    )
                )
            if len(compatibility_targets) == 2:
                support_target_spread_m = float(
                    np.linalg.norm(compatibility_targets[0] - compatibility_targets[1])
                )
            maximum_support_relative_mismatch_m = max(relative_mismatches, default=0.0)
            support_constraints_compatible = bool(common_support_sides) and (
                maximum_support_relative_mismatch_m
                <= _MAX_HARD_SUPPORT_RELATIVE_MISMATCH_M
                and support_target_spread_m <= _MAX_DUAL_SUPPORT_TARGET_SPREAD_M
            )
            if support_constraints_compatible:
                hard_support_sides = common_support_sides
                target_heading = previous_heading
                weights = np.asarray(
                    [
                        previous_supports[side].confidence
                        * source_supports[side].confidence
                        for side in common_support_sides
                    ],
                    dtype=np.float64,
                )
                target_xy = np.average(
                    np.stack(compatibility_targets), axis=0, weights=weights
                )
                alignment_anchor = (
                    "dual_support_weighted_least_squares"
                    if len(common_support_sides) == 2
                    else f"{common_support_sides[0]}_support_foot"
                )
                root_extrapolation_multiplier = 0.0
                root_extrapolation_displacement_m = 0.0
            else:
                target_heading = continuity_heading
                target_xy = continuity_xy
                alignment_anchor = (
                    "root_continuity_after_incompatible_support"
                    if common_support_sides
                    else continuity_anchor
                )
        else:
            target_heading = source_heading
            target_xy = np.zeros(2, dtype=np.float32)
            alignment_anchor = "world_origin"

        # A full SO(3) alignment rotates pitch and roll into the whole clip,
        # which tilts the character's floor plane. Only heading is a valid
        # world-space placement transform for a Z-up timeline.
        heading_delta = _wrap_angle(target_heading - source_heading)
        rotation_delta = _yaw_matrix(heading_delta)
        if common_support_sides:
            for side in common_support_sides:
                previous_support = previous_supports[side]
                source_support = source_supports[side]
                source_relative = (
                    source_support.position_xy - clip.smpl_trans[0, :2]
                )
                rotated_relative = rotation_delta[:2, :2] @ source_relative
                root_target = previous_support.position_xy - rotated_relative
                weight = previous_support.confidence * source_support.confidence
                predicted_foot = (
                    rotated_relative
                    + target_xy
                )
                support_alignment.append(
                    {
                        "side": side,
                        "weight": float(weight),
                        "root_target_xy": root_target.astype(float).tolist(),
                        "anchor_residual_m": float(
                            np.linalg.norm(
                                predicted_foot - previous_supports[side].position_xy
                            )
                        ),
                        "root_relative_mismatch_m": float(
                            np.linalg.norm(
                                root_target - self.motion.smpl_trans[previous, :2]
                            )
                        ),
                        "hard_constraint": side in hard_support_sides,
                    }
                )
        aligned_root_matrices = np.einsum("ij,tjk->tik", rotation_delta, clip_root_matrices)
        aligned_poses = clip.smpl_poses.copy()
        aligned_poses[:, :3] = matrix_to_axis_angle(aligned_root_matrices)

        relative_xy = clip.smpl_trans[:, :2] - clip.smpl_trans[0, :2]
        aligned_translation = np.empty_like(clip.smpl_trans)
        aligned_translation[:, :2] = (
            relative_xy.astype(np.float64) @ rotation_delta[:2, :2].T
            + np.asarray(target_xy, dtype=np.float64)
        ).astype(np.float32)
        aligned_translation[:, 2] = clip.smpl_trans[:, 2] + vertical_shift
        aligned_joints = forward_kinematics(aligned_poses, aligned_translation)
        translation_delta = aligned_translation[0] - clip.smpl_trans[0]

        self.motion.smpl_poses[start_frame:end_frame] = aligned_poses
        self.motion.smpl_trans[start_frame:end_frame] = aligned_translation
        self.motion.joint_positions[start_frame:end_frame] = aligned_joints
        self.filled_mask[start_frame:end_frame] = True
        slot.status = SlotStatus.FILLED
        slot.filled_clip_id = clip.source_clip_id
        slot.retrieval_score = float(retrieval_score)
        assignment = {
            "slot_id": slot.slot_id,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "source_clip_id": clip.source_clip_id,
            "retrieval_score": float(retrieval_score),
            "root_alignment": alignment_anchor,
            "alignment_gap_frames": alignment_gap_frames,
            "root_alignment_axes": "world_z_yaw_only",
            "support_alignment": support_alignment,
            "common_support_sides": list(common_support_sides),
            "hard_support_sides": list(hard_support_sides),
            "support_constraints_compatible": support_constraints_compatible,
            "maximum_support_relative_mismatch_m": (
                maximum_support_relative_mismatch_m
            ),
            "support_target_spread_m": support_target_spread_m,
            "maximum_hard_support_relative_mismatch_m": (
                _MAX_HARD_SUPPORT_RELATIVE_MISMATCH_M
            ),
            "maximum_dual_support_target_spread_m": (
                _MAX_DUAL_SUPPORT_TARGET_SPREAD_M
            ),
            "root_velocity_extrapolation_suppressed": bool(hard_support_sides),
            "root_extrapolation_velocity_decay": (
                _GAP_VELOCITY_DECAY
                if alignment_gap_frames
                and not hard_support_sides
                and root_extrapolation_multiplier > 0.0
                else None
            ),
            "root_extrapolation_multiplier": root_extrapolation_multiplier,
            "root_extrapolation_displacement_m": root_extrapolation_displacement_m,
            "root_extrapolation_max_displacement_m": _MAX_GAP_ROOT_DISPLACEMENT,
            "translation_delta": translation_delta.astype(float).tolist(),
            "root_rotation_delta_axis_angle": matrix_to_axis_angle(rotation_delta)
            .astype(float)
            .tolist(),
            "source_floor_z": source_floor_z,
            "measured_source_floor_z": measured_source_floor_z,
            "library_floor_declaration_valid": declared_source_floor_z is not None,
            "timeline_floor_z": target_floor_z,
            "floor_vertical_shift": vertical_shift,
        }
        self.assignments.append(assignment)
        self.motion.metadata = {
            **deepcopy(self.motion.metadata),
            "timeline_floor_z": target_floor_z,
            "floor_alignment": "robust_smpl_feet_constant_z_shift",
            "root_alignment_axes": "world_z_yaw_only",
            "filled_ranges": [
                [item["start_frame"], item["end_frame"]] for item in self.assignments
            ],
            "slot_assignments": deepcopy(self.assignments),
        }
        return self.state

    def prepare_complete(
        self,
        source_clips: dict[str, CanonicalMotion],
        *,
        seam_half_window_frames: int = _TOUCHING_SEAM_HALF_WINDOW_FRAMES,
    ) -> CompletePreparation:
        """Rebuild placements from source phrases and expose unsafe touching seams.

        Rebuilding from immutable Motion Library clips makes the pass idempotent:
        repeated Complete calls never accumulate a second placement transform.
        Slot interiors stay protected. Only a small edge window is released when
        two touching selected clips cannot be joined by a plausible rigid placement.
        """

        if seam_half_window_frames < 1 or seam_half_window_frames > 24:
            raise ValueError("seam_half_window_frames must be between 1 and 24")
        state = TimelineState.model_validate(self.state.model_dump(mode="json"))
        original_focus = state.current_focused_slot_id
        rebuilt = TimelineEditor(
            state,
            empty_timeline_motion(self.motion.duration_sec),
        )
        ordered_assignments = sorted(
            deepcopy(self.assignments), key=lambda item: int(item["start_frame"])
        )
        for assignment in ordered_assignments:
            slot_id = str(assignment["slot_id"])
            source_clip_id = str(assignment["source_clip_id"])
            try:
                clip = source_clips[source_clip_id]
            except KeyError as exc:
                raise KeyError(
                    f"missing source clip for Complete preparation: {source_clip_id}"
                ) from exc
            slot = rebuilt._slot(slot_id)
            if slot.status is SlotStatus.INVALID:
                raise ValueError(f"filled slot became invalid before Complete: {slot_id}")
            slot.status = SlotStatus.ACCEPTED
            slot.filled_clip_id = None
            slot.retrieval_score = None
            rebuilt.state.current_focused_slot_id = slot_id
            rebuilt.fill(clip, retrieval_score=float(assignment["retrieval_score"]))
        rebuilt.state.current_focused_slot_id = original_focus

        known_mask = rebuilt.filled_mask.copy()
        connection_records: list[dict[str, Any]] = []
        released_ranges: list[list[int]] = []
        for previous, following in zip(
            rebuilt.assignments, rebuilt.assignments[1:], strict=False
        ):
            metrics = _connection_metrics(rebuilt.motion, previous, following)
            release_reasons: list[str] = []
            if metrics["gap_frames"] == 0:
                if metrics["root_xy_jump_m"] > _TOUCHING_ROOT_XY_JUMP_M:
                    release_reasons.append("root_xy_jump")
                if metrics["root_z_jump_m"] > _TOUCHING_ROOT_Z_JUMP_M:
                    release_reasons.append("root_height_jump")
                if metrics["maximum_foot_xy_jump_m"] > _TOUCHING_FOOT_JUMP_M:
                    release_reasons.append("foot_position_jump")
                if (
                    metrics["maximum_leg_rotation_degrees"]
                    > _TOUCHING_LEG_ROTATION_DEGREES
                ):
                    release_reasons.append("leg_rotation_jump")
                if (
                    metrics["common_support_sides"]
                    and not metrics["support_constraints_compatible"]
                ):
                    release_reasons.append("incompatible_support_geometry")
            if release_reasons:
                boundary = int(following["start_frame"])
                seam_start = max(
                    int(previous["start_frame"]),
                    boundary - seam_half_window_frames,
                )
                seam_end = min(
                    int(following["end_frame"]),
                    boundary + seam_half_window_frames,
                )
                known_mask[seam_start:seam_end] = False
                released_ranges.append([seam_start, seam_end])
                metrics["complete_action"] = "virtual_seam_inpainting"
                metrics["released_range"] = [seam_start, seam_end]
                metrics["release_reasons"] = release_reasons
            else:
                metrics["complete_action"] = (
                    "existing_gap_inpainting"
                    if metrics["gap_frames"] > 0
                    else "preserve_compatible_touching_seam"
                )
                metrics["released_range"] = None
                metrics["release_reasons"] = []
            connection_records.append(metrics)

        diagnostics = {
            "schema_version": "pre-complete-continuity-v1",
            "placement_rebuilt_from_source_phrases": True,
            "idempotent_rebuild": True,
            "hard_support_relative_mismatch_limit_m": (
                _MAX_HARD_SUPPORT_RELATIVE_MISMATCH_M
            ),
            "dual_support_target_spread_limit_m": (
                _MAX_DUAL_SUPPORT_TARGET_SPREAD_M
            ),
            "touching_seam_half_window_frames": seam_half_window_frames,
            "connections": connection_records,
            "virtual_seam_ranges": released_ranges,
            "selected_frame_count": int(rebuilt.filled_mask.sum()),
            "effective_protected_frame_count": int(known_mask.sum()),
            "released_selected_frame_count": int(
                np.count_nonzero(rebuilt.filled_mask & ~known_mask)
            ),
        }
        rebuilt.motion.metadata = {
            **deepcopy(rebuilt.motion.metadata),
            "pre_complete_continuity": deepcopy(diagnostics),
        }
        return CompletePreparation(rebuilt, known_mask, diagnostics)

    def snapshot(self) -> dict:
        return {
            "timeline": self.state.model_dump(mode="json"),
            "motion_frames": self.motion.frames,
            "filled_frame_count": int(self.filled_mask.sum()),
            "assignments": deepcopy(self.assignments),
        }
