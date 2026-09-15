"""Build a materialized train-only CustomDance Motion Library release."""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from collections import defaultdict
from collections.abc import Callable, Mapping
from itertools import pairwise
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

from app.schemas.motion import CanonicalMotion
from app.utils.kinematics import matrix_to_axis_angle, rotation_6d_to_matrix
from app.utils.motion_adapters import canonical_from_finedance
from app.utils.motion_grounding import (
    DEFAULT_LIBRARY_FLOOR_Z,
    FLOOR_NORMALIZATION_CONTRACT,
    normalize_canonical_floor,
)

MOTION_LIBRARY_V3_STANDING_SCHEMA = "customdance-motion-library-v3-standing"
DEPLOYMENT_INDEX_SCHEMA = "cd-mtr-lite-customdance-motion-library-index-v1"
_PHRASE_ID = re.compile(r"^FD_[A-Za-z0-9.-]+_[0-9]{6}_[0-9]{6}$")


def _read_json_lines(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise TypeError(f"metadata line {line_number} is not a JSON object")
            rows.append(row)
    return rows


def _floor_offset(row: Mapping[str, Any]) -> float:
    metrics = row.get("quality_metrics")
    if not isinstance(metrics, Mapping) or "floor_offset_y" not in metrics:
        raise ValueError(f"{row.get('phrase_id', '<unknown>')}: missing floor_offset_y")
    value = float(metrics["floor_offset_y"])
    if not np.isfinite(value):
        raise ValueError(f"{row.get('phrase_id', '<unknown>')}: non-finite floor_offset_y")
    return value


def expected_phrase_id(row: Mapping[str, Any]) -> str:
    source_id = str(row["source_id"])
    start = int(row["raw_start_frame"])
    end = int(row["raw_end_frame"])
    return f"FD_{source_id}_{start:06d}_{end:06d}"


def resolve_raw_motion_path(raw_root: Path, indexed_path: str) -> tuple[Path, str]:
    """Relocate a trusted index path under the configured raw FineDance root."""

    root = Path(raw_root).expanduser().resolve(strict=True)
    normalized = str(indexed_path).replace("\\", "/")
    marker = "/motion/"
    if normalized.startswith("source://"):
        suffix = normalized.removeprefix("source://")
        if not re.fullmatch(r"[A-Za-z0-9.-]+\.npy", suffix) or ".." in suffix:
            raise ValueError("unsafe source identifier in index")
    elif marker not in normalized:
        raise ValueError(f"indexed motion path has no motion component: {indexed_path!r}")
    else:
        suffix = normalized.rsplit(marker, 1)[1]
    relative = PurePosixPath("motion") / PurePosixPath(suffix)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("indexed motion path contains an unsafe suffix")
    path = (root / Path(*relative.parts)).resolve(strict=True)
    if root not in path.parents or not path.is_file():
        raise ValueError("indexed motion path escapes the configured raw root")
    return path, relative.as_posix()


def canonical_from_raw_window(
    raw_motion: np.ndarray,
    row: Mapping[str, Any],
    *,
    raw_motion_relative_path: str,
) -> CanonicalMotion:
    """Convert one indexed inclusive raw FineDance interval to Canonical SMPL24."""

    source = np.asarray(raw_motion)
    if source.ndim != 2 or source.shape[1] != 315:
        raise ValueError(f"raw FineDance motion must have shape (T,315), got {source.shape}")
    start = int(row["raw_start_frame"])
    end = int(row["raw_end_frame"])
    if start < 0 or end < start or end >= source.shape[0] or end - start + 1 != 120:
        raise ValueError(
            f"{row.get('phrase_id', '<unknown>')}: raw interval must be an in-bounds "
            "inclusive 120-frame window"
        )
    phrase_id = str(row["phrase_id"])
    if phrase_id != expected_phrase_id(row) or not _PHRASE_ID.fullmatch(phrase_id):
        raise ValueError(f"unstable or unsafe phrase_id: {phrase_id!r}")
    window = np.asarray(source[start : end + 1], dtype=np.float32)
    if window.shape != (120, 315) or not np.isfinite(window).all():
        raise ValueError(f"{phrase_id}: raw window is not finite (120,315)")

    # FineDance stores translation followed by 52 row-vector 6D rotations.
    # The established FlowerDance/CustomDance contract is the first 24 joints in
    # source order. Do not introduce a semantic SMPL-X remapping here.
    rotation_6d = window[:, 3:].reshape(120, 52, 6)[:, :24]
    rotation_matrices = rotation_6d_to_matrix(rotation_6d)
    local_axis_angle = matrix_to_axis_angle(rotation_matrices).reshape(120, 72)

    # Preserve the raw translation exactly apart from the release-provided
    # floor offset. In particular, the historical +1.3 adjustment is forbidden.
    translation = window[:, :3].copy()
    floor_offset_y = _floor_offset(row)
    translation[:, 1] -= floor_offset_y
    metadata = {
        "motion_library_version": 3,
        "phrase_id": phrase_id,
        "source_id": str(row["source_id"]),
        "source_split": str(row["source_split"]),
        "genre": str(row.get("genre", "")),
        "source_song": str(row.get("source_song", "")),
        "window_id": str(row.get("window_id", "")),
        "raw_motion_path": raw_motion_relative_path,
        "raw_start_frame": start,
        "raw_end_frame": end,
        "floor_offset_y": floor_offset_y,
        "conversion_contract": (
            "finedance-315d-first-smpl24-pytorch3d-row6d-"
            "grounded-y-to-canonical-z-v2"
        ),
    }
    metadata["retrieval_release"] = "cleaned_standing_500"
    return canonical_from_finedance(translation, local_axis_angle, phrase_id, metadata)


class MotionLibraryBuilder:
    """Materialize a deployment index into validated CanonicalMotion NPZ files."""

    def __init__(
        self,
        *,
        raw_root: Path,
        deployment_index_root: Path,
        output_root: Path,
        preparation: dict[str, Any] | None = None,
        target_floor_z: float = DEFAULT_LIBRARY_FLOOR_Z,
    ) -> None:
        self.raw_root = Path(raw_root).expanduser().resolve(strict=True)
        self.deployment_index_root = Path(deployment_index_root).expanduser().resolve(strict=True)
        self.output_root = Path(output_root).expanduser().resolve()
        self.preparation = dict(preparation or {})
        self.target_floor_z = float(target_floor_z)
        if not np.isfinite(self.target_floor_z):
            raise ValueError("target_floor_z must be finite")

    def _load_index(self) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        required = (
            self.deployment_index_root / "index.json",
            self.deployment_index_root / "metadata.jsonl",
            self.deployment_index_root / "embeddings.npy",
            self.deployment_index_root / "summary.json",
        )
        for path in required:
            if not path.is_file():
                raise FileNotFoundError(f"missing deployment index file: {path}")
        manifest = json.loads(required[0].read_text(encoding="utf-8"))
        if manifest.get("schema_version") != DEPLOYMENT_INDEX_SCHEMA:
            raise ValueError("unsupported deployment index schema")
        rows = _read_json_lines(required[1])
        embeddings = np.load(required[2], mmap_mode="r", allow_pickle=False)
        if embeddings.shape != (len(rows), 128) or embeddings.dtype != np.float32:
            raise ValueError("deployment embeddings must be float32 with shape (rows,128)")
        if not np.isfinite(embeddings).all():
            raise ValueError("deployment embeddings contain NaN or infinite values")
        if int(manifest.get("rows", -1)) != len(rows):
            raise ValueError("deployment index manifest/metadata length mismatch")
        summary = json.loads(required[3].read_text(encoding="utf-8"))
        if summary.get("status") != "READY":
            raise ValueError("deployment index summary is not READY")
        if int(summary.get("selected_rows", -1)) != len(rows):
            raise ValueError("deployment index summary/metadata length mismatch")
        return manifest, rows

    @staticmethod
    def _validate_rows(
        rows: list[dict[str, Any]], *, expected_rows: int, expected_sources: int
    ) -> None:
        if len(rows) != expected_rows:
            raise ValueError(f"deployment index has {len(rows)} rows, expected {expected_rows}")
        phrases: set[str] = set()
        intervals: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for row in rows:
            phrase_id = str(row.get("phrase_id", ""))
            if phrase_id != expected_phrase_id(row) or phrase_id in phrases:
                raise ValueError(f"duplicate or unstable phrase_id: {phrase_id!r}")
            if str(row.get("source_split", "")).lower() != "train":
                raise ValueError(f"non-train deployment phrase: {phrase_id}")
            start = int(row["raw_start_frame"])
            end = int(row["raw_end_frame"])
            if end - start + 1 != 120:
                raise ValueError(f"non-120-frame deployment phrase: {phrase_id}")
            _floor_offset(row)
            phrases.add(phrase_id)
            intervals[str(row["source_id"])].append((start, end))
        if len(intervals) != expected_sources:
            raise ValueError(
                f"deployment index has {len(intervals)} sources, expected {expected_sources}"
            )
        for source_id, values in intervals.items():
            ordered = sorted(values)
            for previous, current in pairwise(ordered):
                if current[0] <= previous[1]:
                    raise ValueError(
                        f"deployment intervals overlap for source {source_id}: "
                        f"{previous} and {current}"
                    )

    @staticmethod
    def _validate_staging(staging: Path, records: list[dict[str, Any]]) -> None:
        canonical_root = (staging / "canonical").resolve(strict=True)
        actual = {path.stem for path in canonical_root.glob("*.npz")}
        expected = {str(record["phrase_id"]) for record in records}
        if actual != expected or len(actual) != len(records):
            raise ValueError("materialized NPZ set does not match manifest phrase IDs")
        for record in records:
            path = (staging / str(record["canonical_clip_path"])).resolve(strict=True)
            if canonical_root not in path.parents:
                raise ValueError("canonical clip path escapes staging root")
            motion = CanonicalMotion.load_npz(path)
            if motion.source_clip_id != record["phrase_id"] or motion.frames != 120:
                raise ValueError(f"invalid canonical asset: {record['phrase_id']}")

    def build(
        self,
        *,
        expected_rows: int = 4982,
        expected_sources: int = 161,
        progress: Callable[[int, int], None] | None = None,
    ) -> dict[str, Any]:
        output = self.output_root
        if output.exists():
            if not output.is_dir() or any(output.iterdir()):
                raise FileExistsError(f"output directory is not empty: {output}")
            output.rmdir()
        output.parent.mkdir(parents=True, exist_ok=True)
        index_manifest, rows = self._load_index()
        self._validate_rows(rows, expected_rows=expected_rows, expected_sources=expected_sources)

        staging = Path(
            tempfile.mkdtemp(prefix=f".{output.name}.building-", dir=str(output.parent))
        ).resolve()
        records: list[dict[str, Any]] = []
        cache: dict[Path, np.ndarray] = {}
        try:
            canonical_dir = staging / "canonical"
            canonical_dir.mkdir()
            for number, row in enumerate(rows, start=1):
                raw_path, raw_relative = resolve_raw_motion_path(
                    self.raw_root, str(row["motion_path"])
                )
                if raw_path.name != f"{row['source_id']}.npy":
                    raise ValueError("index motion path does not match its source_id")
                raw_motion = cache.get(raw_path)
                if raw_motion is None:
                    raw_motion = np.load(raw_path, mmap_mode="r", allow_pickle=False)
                    if raw_motion.ndim != 2 or raw_motion.shape[1] != 315:
                        raise ValueError(f"{raw_relative}: expected raw shape (T,315)")
                    cache[raw_path] = raw_motion
                raw_canonical = canonical_from_raw_window(
                    raw_motion,
                    row,
                    raw_motion_relative_path=raw_relative,
                )
                motion, floor_normalization = normalize_canonical_floor(
                    raw_canonical,
                    target_floor_z=self.target_floor_z,
                )
                phrase_id = str(row["phrase_id"])
                relative_path = Path("canonical") / f"{phrase_id}.npz"
                destination = staging / relative_path
                if destination.exists():
                    raise FileExistsError(f"duplicate canonical destination: {destination.name}")
                motion.save_npz(destination)
                records.append(
                    {
                        "clip_id": phrase_id,
                        "phrase_id": phrase_id,
                        "source_id": str(row["source_id"]),
                        "source_split": "train",
                        "genre": str(row.get("genre", "")),
                        "source_song": str(row.get("source_song", "")),
                        "window_id": str(row.get("window_id", "")),
                        "raw_motion_path": raw_relative,
                        "raw_start_frame": int(row["raw_start_frame"]),
                        "raw_end_frame": int(row["raw_end_frame"]),
                        "floor_offset_y": _floor_offset(row),
                        "source_floor_z": floor_normalization["source_floor_z"],
                        "target_floor_z": floor_normalization["target_floor_z"],
                        "floor_shift_z": floor_normalization["floor_shift_z"],
                        "floor_contact_confidence": floor_normalization["confidence"],
                        "floor_plane_tilt_deg": floor_normalization["plane_tilt_deg"],
                        "floor_normalization_contract": FLOOR_NORMALIZATION_CONTRACT,
                        "duration": 4.0,
                        "fps": 30,
                        "frames": 120,
                        "canonical_clip_path": relative_path.as_posix(),
                    }
                )
                if progress is not None and (number % 100 == 0 or number == len(rows)):
                    progress(number, len(rows))

            manifest = {
                "schema_version": MOTION_LIBRARY_V3_STANDING_SCHEMA,
                "status": "READY",
                "clip_count": len(records),
                "canonical_fps": 30,
                "clip_frames": 120,
                "clip_duration_sec": 4.0,
                "source_split_policy": "train_only",
                "window_policy": "fixed_non_overlapping_grid",
                "provenance": {
                    "deployment_index_schema": str(index_manifest["schema_version"]),
                    "preparation": self.preparation,
                    "raw_root_layout": "finedance/{motion,label_json,music_npy,music_wav}",
                },
                "conversion_contract": {
                    "raw_shape": "(T,315) = root_translation_3 + 52xrotation_6d",
                    "root_translation_offset": "none; never add +1.3",
                    "rotation_6d_convention": (
                        "FineDance/PyTorch3D first two rows, Gram-Schmidt, "
                        "stacked as rows"
                    ),
                    "smpl24_selection": "first 24 rotations in source order",
                    "grounding": "translation_y -= quality_metrics.floor_offset_y",
                    "library_floor_normalization": (
                        "after canonical FK, estimate low/slow SMPL foot contacts and "
                        "apply one constant root/joint Z shift"
                    ),
                    "library_target_floor_z": self.target_floor_z,
                    "library_floor_contract": FLOOR_NORMALIZATION_CONTRACT,
                    "coordinates": "FineDance right-handed Y-up to canonical right-handed Z-up",
                    "joints": "CustomDance SMPL24 forward kinematics",
                },
                "clips": records,
            }
            manifest_path = staging / "manifest.json"
            manifest_path.write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
                + "\n",
                encoding="utf-8",
            )
            self._validate_staging(staging, records)
            os.replace(staging, output)
            return manifest
        except Exception:
            if staging.exists() and staging.parent == output.parent and staging.name.startswith(
                f".{output.name}.building-"
            ):
                shutil.rmtree(staging)
            raise


__all__ = [
    "DEPLOYMENT_INDEX_SCHEMA",
    "MOTION_LIBRARY_V3_STANDING_SCHEMA",
    "MotionLibraryBuilder",
    "canonical_from_raw_window",
    "expected_phrase_id",
    "resolve_raw_motion_path",
]
