"""Versioned motion representations used by CustomDance."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

CANONICAL_SCHEMA_VERSION = "1.0"
COMPLETER_SCHEMA_VERSION = "1.0"


class CoordinateSystem(str, Enum):
    RIGHT_HANDED_Z_UP = "right-handed-z-up"
    RIGHT_HANDED_Y_UP = "right-handed-y-up"


def _finite_float32(name: str, value: np.ndarray) -> np.ndarray:
    array = np.ascontiguousarray(np.asarray(value, dtype=np.float32))
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or infinite values")
    return array


@dataclass(slots=True)
class CanonicalMotion:
    """Canonical SMPL24 motion for editing, diagnosis, preview, and export."""

    smpl_poses: np.ndarray
    smpl_trans: np.ndarray
    joint_positions: np.ndarray
    source_clip_id: str
    coordinate_system: CoordinateSystem = CoordinateSystem.RIGHT_HANDED_Z_UP
    fps: int = 30
    schema_version: str = CANONICAL_SCHEMA_VERSION
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.smpl_poses = _finite_float32("smpl_poses", self.smpl_poses)
        self.smpl_trans = _finite_float32("smpl_trans", self.smpl_trans)
        self.joint_positions = _finite_float32("joint_positions", self.joint_positions)
        self.coordinate_system = CoordinateSystem(self.coordinate_system)
        if self.schema_version != CANONICAL_SCHEMA_VERSION:
            raise ValueError(f"unsupported canonical schema: {self.schema_version}")
        if self.fps != 30:
            raise ValueError("Canonical SMPL Motion must use 30 FPS")
        if self.smpl_poses.ndim != 2 or self.smpl_poses.shape[1] != 72:
            raise ValueError("smpl_poses must have shape (T, 72)")
        frames = self.smpl_poses.shape[0]
        if self.smpl_trans.shape != (frames, 3):
            raise ValueError("smpl_trans must have shape (T, 3)")
        if self.joint_positions.shape != (frames, 24, 3):
            raise ValueError("joint_positions must have shape (T, 24, 3)")
        if not self.source_clip_id:
            raise ValueError("source_clip_id must be non-empty")

    @property
    def frames(self) -> int:
        return int(self.smpl_poses.shape[0])

    @property
    def duration_sec(self) -> float:
        return self.frames / self.fps

    def metadata_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "fps": self.fps,
            "coordinate_system": self.coordinate_system.value,
            "source_clip_id": self.source_clip_id,
            "metadata": self.metadata,
        }

    def save_npz(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            smpl_poses=self.smpl_poses,
            smpl_trans=self.smpl_trans,
            joint_positions=self.joint_positions,
            metadata_json=np.asarray(json.dumps(self.metadata_dict(), sort_keys=True)),
        )

    @classmethod
    def load_npz(cls, path: Path) -> CanonicalMotion:
        with np.load(Path(path), allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata_json"].item()))
            return cls(
                smpl_poses=archive["smpl_poses"],
                smpl_trans=archive["smpl_trans"],
                joint_positions=archive["joint_positions"],
                source_clip_id=metadata["source_clip_id"],
                coordinate_system=metadata["coordinate_system"],
                fps=metadata["fps"],
                schema_version=metadata["schema_version"],
                metadata=metadata.get("metadata", {}),
            )


@dataclass(slots=True)
class CompleterInternalMotion:
    """Completer's normalized-independent 151D motion representation."""

    features: np.ndarray
    source_clip_id: str
    fps: int = 30
    schema_version: str = COMPLETER_SCHEMA_VERSION
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.features = _finite_float32("features", self.features)
        if self.schema_version != COMPLETER_SCHEMA_VERSION:
            raise ValueError(f"unsupported Completer schema: {self.schema_version}")
        if self.fps != 30:
            raise ValueError("Completer internal motion must use 30 FPS")
        if self.features.ndim != 2 or self.features.shape[1] != 151:
            raise ValueError("Completer features must have shape (T, 151)")

    @property
    def contacts(self) -> np.ndarray:
        return self.features[:, :4]

    @property
    def root_translation(self) -> np.ndarray:
        return self.features[:, 4:7]

    @property
    def rotation_6d(self) -> np.ndarray:
        return self.features[:, 7:].reshape(-1, 24, 6)
