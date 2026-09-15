"""Application-owned PKL motion export."""

from __future__ import annotations

import os
import pickle
import tempfile
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from app.schemas.motion import CanonicalMotion

EXPORT_SCHEMA_VERSION = "1.0"


@dataclass(frozen=True, slots=True)
class ExportArtifact:
    path: Path
    validation: dict[str, Any]


class ExportService:
    def __init__(self, session_root: Path):
        self.session_root = Path(session_root).expanduser().resolve(strict=True)
        self.output_root = (self.session_root / "exports").resolve()
        if self.session_root not in self.output_root.parents:
            raise ValueError("export directory escaped the generated session root")
        self.output_root.mkdir(parents=True, exist_ok=True)

    def _target(self, filename: str) -> Path:
        if Path(filename).name != filename:
            raise ValueError("export filename must not contain a path")
        target = (self.output_root / filename).resolve()
        if self.output_root not in target.parents:
            raise ValueError("export target escaped the generated export directory")
        return target

    def export_pkl(
        self,
        motion: CanonicalMotion,
        metadata: dict[str, Any],
        *,
        revision: int,
    ) -> ExportArtifact:
        target = self._target(f"customdance-motion-r{revision}.pkl")
        payload = {
            "schema_version": EXPORT_SCHEMA_VERSION,
            "fps": motion.fps,
            "smpl_poses": motion.smpl_poses.astype(np.float32, copy=True),
            "smpl_trans": motion.smpl_trans.astype(np.float32, copy=True),
            "joint_positions": motion.joint_positions.astype(np.float32, copy=True),
            "coordinate_system": motion.coordinate_system.value,
            "source_clip_id": motion.source_clip_id,
            "music_metadata": deepcopy(metadata.get("music_metadata", {})),
            "slot_assignments": deepcopy(metadata.get("slot_assignments", [])),
            "retrieval_metadata": deepcopy(metadata.get("retrieval_metadata", [])),
            "completer_inference": deepcopy(
                motion.metadata.get("completer_inference", [])
            ),
            "motion_metadata": deepcopy(motion.metadata),
        }
        temporary_name: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb", dir=self.output_root, prefix=".motion-", suffix=".tmp", delete=False
            ) as handle:
                temporary_name = handle.name
                pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, target)
            temporary_name = None
        finally:
            if temporary_name is not None:
                Path(temporary_name).unlink(missing_ok=True)
        _, reloaded_motion = load_generated_pkl(target, self.session_root)
        validation = {
            "schema_version": EXPORT_SCHEMA_VERSION,
            "fps": reloaded_motion.fps,
            "frames": reloaded_motion.frames,
            "finite": True,
            "shape_valid": True,
            "trusted_reload": True,
        }
        return ExportArtifact(target, validation)


def load_generated_pkl(
    path: Path, trusted_session_root: Path
) -> tuple[dict[str, Any], CanonicalMotion]:
    """Reload only an application-generated PKL beneath a trusted session root."""

    root = Path(trusted_session_root).expanduser().resolve(strict=True)
    source = Path(path).expanduser().resolve(strict=True)
    if root not in source.parents or source.parent.name != "exports" or source.suffix != ".pkl":
        raise ValueError("PKL reload is restricted to application-generated session exports")
    with source.open("rb") as handle:
        payload = pickle.load(handle)
    required = {
        "schema_version",
        "fps",
        "smpl_poses",
        "smpl_trans",
        "joint_positions",
        "coordinate_system",
        "source_clip_id",
        "music_metadata",
        "slot_assignments",
        "retrieval_metadata",
        "completer_inference",
    }
    if not isinstance(payload, dict) or required - payload.keys():
        raise ValueError("generated PKL is missing required export fields")
    if payload["schema_version"] != EXPORT_SCHEMA_VERSION:
        raise ValueError("unsupported generated PKL schema")
    motion = CanonicalMotion(
        smpl_poses=payload["smpl_poses"],
        smpl_trans=payload["smpl_trans"],
        joint_positions=payload["joint_positions"],
        source_clip_id=str(payload["source_clip_id"]),
        coordinate_system=str(payload["coordinate_system"]),
        fps=int(payload["fps"]),
        metadata=deepcopy(payload.get("motion_metadata", {})),
    )
    return payload, motion
