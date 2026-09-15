"""Read-only access to a materialized CustomDance Motion Library release."""

from __future__ import annotations

import json
import threading
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
from typing import Any

from app.schemas.library import SUPPORTED_MOTION_LIBRARY_SCHEMAS
from app.schemas.motion import CanonicalMotion


class MotionLibrary:
    def __init__(self, root: Path, *, cache_size: int = 32):
        self.root = Path(root).expanduser().resolve(strict=True)
        with (self.root / "manifest.json").open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        records = manifest.get("clips")
        if (
            manifest.get("schema_version") not in SUPPORTED_MOTION_LIBRARY_SCHEMAS
            or manifest.get("status") != "READY"
            or not isinstance(records, list)
        ):
            raise ValueError("invalid or incomplete Motion Library manifest")
        if int(manifest.get("clip_count", -1)) != len(records):
            raise ValueError("Motion Library manifest clip count does not match records")
        self.manifest = manifest
        if cache_size < 1:
            raise ValueError("Motion Library cache_size must be positive")
        self._cache_size = int(cache_size)
        self._cache: OrderedDict[str, CanonicalMotion] = OrderedDict()
        self._cache_hits = 0
        self._cache_misses = 0
        self._lock = threading.RLock()
        self.records: dict[str, dict[str, Any]] = {}
        for record in records:
            phrase_id = str(record.get("phrase_id", ""))
            if (
                not phrase_id
                or phrase_id != str(record.get("clip_id", ""))
                or phrase_id in self.records
                or str(record.get("source_split", "")) != "train"
                or int(record.get("frames", -1)) != 120
                or int(record.get("fps", -1)) != 30
            ):
                raise ValueError(
                    "Motion Library records must be unique 120-frame train phrases"
                )
            self.records[phrase_id] = record

    def _resolve(self, relative_path: str) -> Path:
        path = (self.root / relative_path).resolve(strict=True)
        if self.root not in path.parents:
            raise ValueError("motion-library path escapes configured root")
        return path

    def get_record(self, phrase_id: str) -> dict[str, Any]:
        try:
            return self.records[phrase_id]
        except KeyError as exc:
            raise KeyError(f"unknown Motion Library phrase_id: {phrase_id}") from exc

    def load_canonical(self, phrase_id: str) -> CanonicalMotion:
        with self._lock:
            cached = self._cache.get(phrase_id)
            if cached is not None:
                self._cache.move_to_end(phrase_id)
                self._cache_hits += 1
                return self._clone(cached)

            record = self.get_record(phrase_id)
            motion = CanonicalMotion.load_npz(self._resolve(record["canonical_clip_path"]))
            if motion.source_clip_id != phrase_id:
                raise ValueError("canonical asset source_clip_id does not match manifest")
            if motion.frames != 120:
                raise ValueError("motion-library canonical clip is not exactly 120 frames")
            self._cache[phrase_id] = motion
            self._cache.move_to_end(phrase_id)
            while len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
            self._cache_misses += 1
            return self._clone(motion)

    @staticmethod
    def _clone(motion: CanonicalMotion) -> CanonicalMotion:
        """Return an isolated copy so timeline edits cannot mutate cached assets."""

        return CanonicalMotion(
            smpl_poses=motion.smpl_poses.copy(),
            smpl_trans=motion.smpl_trans.copy(),
            joint_positions=motion.joint_positions.copy(),
            source_clip_id=motion.source_clip_id,
            coordinate_system=motion.coordinate_system,
            fps=motion.fps,
            schema_version=motion.schema_version,
            metadata=deepcopy(motion.metadata),
        )

    def status(self) -> dict[str, Any]:
        return {
            "schema_version": self.manifest["schema_version"],
            "status": self.manifest["status"],
            "clip_count": len(self.records),
            "source_count": len(
                {str(record["source_id"]) for record in self.records.values()}
            ),
            "source_splits": sorted(
                {str(record["source_split"]) for record in self.records.values()}
            ),
            "cache": {
                "capacity": self._cache_size,
                "entries": len(self._cache),
                "hits": self._cache_hits,
                "misses": self._cache_misses,
            },
        }
