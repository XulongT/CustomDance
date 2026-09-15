"""Music/text retrieval over the paired Motion Library index."""

from __future__ import annotations

import importlib
import json
import os
import random
import re
import threading
import wave
from collections import Counter, OrderedDict, defaultdict
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np

from app.utils.retriever_profile import RetrievalProfile

ALLOWED_TOP_K = {5, 10, 15}
BACKEND_NAME = "CustomDance Music+Text Retriever"
DEFAULT_SEED = 20260813
DEPLOYMENT_INDEX_SCHEMA = "cd-mtr-lite-customdance-motion-library-index-v1"
_PHRASE_ID = re.compile(r"^FD_[A-Za-z0-9.-]+_[0-9]{6}_[0-9]{6}$")


class RetrievalQueryError(ValueError):
    """A user-correctable retrieval query error."""


@dataclass(frozen=True, slots=True)
class MusicTextCandidate:
    phrase_id: str
    motion_id: str
    window_id: str
    genre: str
    score: float
    source_song: str
    start_frame: int
    end_frame: int
    trimmed_start_frame: int
    trimmed_end_frame: int
    motion_path: str
    indexed_motion_path: str
    floor_offset_y: float
    source_split: str

    def __post_init__(self) -> None:
        if not _PHRASE_ID.fullmatch(self.phrase_id):
            raise ValueError(f"invalid deployment phrase_id: {self.phrase_id!r}")
        if self.source_split != "train":
            raise ValueError("deployment candidates must have source_split=train")
        if self.end_frame - self.start_frame + 1 != 120:
            raise ValueError("deployment candidates must cover exactly 120 raw frames")

    @property
    def clip_id(self) -> str:
        """Compatibility alias: public clip IDs are the stable phrase IDs."""

        return self.phrase_id

    def public_metadata(self) -> dict[str, Any]:
        return {
            "phrase_id": self.phrase_id,
            "motion_id": self.motion_id,
            "window_id": self.window_id,
            "genre": self.genre,
            "source_song": self.source_song,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "trimmed_start_frame": self.trimmed_start_frame,
            "trimmed_end_frame": self.trimmed_end_frame,
            "motion_path": self.motion_path,
            "indexed_motion_path": self.indexed_motion_path,
            "source_split": self.source_split,
        }


@dataclass(frozen=True, slots=True)
class MusicTextRetrievalResult:
    query: dict[str, Any]
    warning: str | None
    candidate_count: int
    returned: int
    results: tuple[MusicTextCandidate, ...]


def _checkpoint_floor_offset(row: dict[str, Any]) -> float:
    metrics = row.get("quality_metrics")
    value = metrics.get("floor_offset_y", 0.0) if isinstance(metrics, dict) else 0.0
    value = float(value or 0.0)
    if not np.isfinite(value):
        raise ValueError(f"{row.get('phrase_id', '<unknown>')}: non-finite floor offset")
    return value


def _expected_phrase_id(row: dict[str, Any]) -> str:
    start = int(row.get("raw_start_frame", row["start_frame"]))
    end = int(row.get("raw_end_frame", row["end_frame"]))
    return f"FD_{row['source_id']}_{start:06d}_{end:06d}"


class MusicTextRetriever:
    """Lazy, thread-safe adapter around the cleaned deployment release."""

    def __init__(
        self,
        root: Path,
        checkpoint: Path,
        index_root: Path,
        *,
        exclusions_path: Path | None = None,
        device: str = "cuda",
        profile_path: Path | None = None,
    ) -> None:
        self.root = Path(root).expanduser().resolve(strict=True)
        self.checkpoint = Path(checkpoint).expanduser().resolve(strict=True)
        self.index_root = Path(index_root).expanduser().resolve(strict=True)
        self.exclusions_path = (
            None
            if exclusions_path is None
            else Path(exclusions_path).expanduser().resolve(strict=True)
        )
        for path, label in ((self.checkpoint, "checkpoint"), (self.index_root, "index")):
            if path != self.root and self.root not in path.parents:
                raise ValueError(f"retriever {label} must stay inside {self.root}: {path}")
        if device not in {"cpu", "cuda"}:
            raise ValueError("music-text retriever device must be cpu or cuda")
        self.device = device
        self.profile = RetrievalProfile() if profile_path is None else RetrievalProfile.load(profile_path)
        self._lock = threading.RLock()
        self._model: Any | None = None
        self._checkpoint_payload: dict[str, Any] | None = None
        self._index_manifest: dict[str, Any] | None = None
        self._embeddings: np.ndarray | None = None
        self._metadata: list[dict[str, Any]] | None = None
        self._load_music: Any | None = None
        self._text_to_vector: Any | None = None
        self._validate_genre: Any | None = None
        self._deployment_index_rows = 0
        self._excluded_phrase_ids: frozenset[str] = frozenset()
        self._music_embedding_cache: OrderedDict[
            tuple[str, int, int, int, float], np.ndarray
        ] = OrderedDict()
        self._music_embedding_cache_capacity = 128
        self._music_embedding_cache_hits = 0
        self._music_embedding_cache_misses = 0

    def _load_exclusions(self, metadata: list[dict[str, Any]]) -> frozenset[str]:
        if self.exclusions_path is None:
            return frozenset()
        payload = json.loads(self.exclusions_path.read_text(encoding="utf-8"))
        records = payload.get("excluded_phrases")
        if payload.get("schema_version") != "1.0" or not isinstance(records, list):
            raise ValueError("invalid retrieval exclusions file")
        values: list[str] = []
        for record in records:
            if not isinstance(record, dict) or not str(record.get("reason", "")).strip():
                raise ValueError("every retrieval exclusion requires a documented reason")
            phrase_id = str(record.get("phrase_id", ""))
            if not _PHRASE_ID.fullmatch(phrase_id):
                raise ValueError(f"invalid excluded phrase_id: {phrase_id!r}")
            values.append(phrase_id)
        excluded = frozenset(values)
        if len(excluded) != len(values):
            raise ValueError("retrieval exclusions contain duplicate phrase IDs")
        available = {str(row["phrase_id"]) for row in metadata}
        unknown = sorted(excluded - available)
        if unknown:
            raise ValueError(f"retrieval exclusions are absent from the deployment index: {unknown}")
        return excluded

    def _inference_module(self, name: str) -> Any:
        # Asset roots contain data only. Inference code is part of this package.
        leaf = name.rsplit(".", 1)[-1]
        if leaf not in {"index", "audio", "text"}:
            raise ValueError(f"unsupported retriever module: {leaf}")
        return importlib.import_module(f"app.utils.retriever_{leaf}")

    @staticmethod
    def _validate_release(
        index_manifest: dict[str, Any],
        summary: dict[str, Any],
        embeddings: np.ndarray,
        metadata: list[dict[str, Any]],
        checkpoint: dict[str, Any],
        profile: RetrievalProfile | None = None,
    ) -> None:
        profile = profile or RetrievalProfile()
        if index_manifest.get("schema_version") != DEPLOYMENT_INDEX_SCHEMA:
            raise ValueError("retrieval index is not a supported deployment-only index")
        if int(index_manifest.get("rows", -1)) != profile.index_rows:
            raise ValueError(
                f"retrieval index has {index_manifest.get('rows')} rows; "
                f"expected {profile.index_rows}"
            )
        if embeddings.shape != (profile.index_rows, profile.embedding_dim) or embeddings.dtype != np.float32:
            raise ValueError(
                f"deployment embeddings must be float32 ({profile.index_rows},{profile.embedding_dim})"
            )
        if not np.isfinite(embeddings).all():
            raise ValueError("music-text index contains NaN or infinite values")
        if len(metadata) != profile.index_rows:
            raise ValueError("deployment metadata length does not match expected row count")
        if summary.get("status") != "READY":
            raise ValueError("deployment index summary is not READY")
        if int(summary.get("selected_rows", -1)) != profile.index_rows:
            raise ValueError("deployment summary row count does not match the release")
        if int(summary.get("selected_sources", -1)) != profile.source_count:
            raise ValueError("deployment summary source count does not match the release")
        if index_manifest.get("splits") != ["train"]:
            raise ValueError("deployment index must advertise only the train split")
        library_contract = index_manifest.get("motion_library")
        if not isinstance(library_contract, dict):
            raise TypeError("deployment index is missing its motion-library contract")
        expected_contract = {
            "selected_rows": profile.index_rows,
            "selected_sources": profile.source_count,
            "window_frames": 120,
            "hop_frames": 120,
            "overlap_frames": 0,
        }
        for key, expected in expected_contract.items():
            if int(library_contract.get(key, -1)) != expected:
                raise ValueError(
                    f"deployment motion-library contract has {key}="
                    f"{library_contract.get(key)!r}; expected {expected}"
                )
        if library_contract.get("source_split_policy") != "train_only":
            raise ValueError("deployment index must use the train_only split policy")
        phrase_ids: set[str] = set()
        sources: set[str] = set()
        intervals: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for row in metadata:
            phrase_id = str(row.get("phrase_id", ""))
            if phrase_id != _expected_phrase_id(row) or not _PHRASE_ID.fullmatch(phrase_id):
                raise ValueError(f"unstable deployment phrase_id: {phrase_id!r}")
            if phrase_id in phrase_ids:
                raise ValueError(f"duplicate deployment phrase_id: {phrase_id}")
            if str(row.get("source_split", "")).lower() != "train":
                raise ValueError(f"non-train row in deployment index: {phrase_id}")
            start = int(row.get("raw_start_frame", row["start_frame"]))
            end = int(row.get("raw_end_frame", row["end_frame"]))
            if end - start + 1 != 120 or int(row.get("num_frames", -1)) != 120:
                raise ValueError(f"non-120-frame deployment phrase: {phrase_id}")
            metrics = row.get("quality_metrics")
            if not isinstance(metrics, dict):
                raise TypeError(f"missing quality metrics for deployment phrase: {phrase_id}")
            low_ratio = float(metrics.get("low_pelvis_ratio", 0.0))
            low_run = float(metrics.get("low_pelvis_longest_run_seconds", 0.0))
            if low_ratio >= 0.25 and low_run >= 0.75:
                raise ValueError(f"standing gate violation in deployment phrase: {phrase_id}")
            phrase_ids.add(phrase_id)
            source_id = str(row["source_id"])
            sources.add(source_id)
            intervals[source_id].append((start, end))
        if len(sources) != profile.source_count:
            raise ValueError(
                f"deployment index has {len(sources)} sources; expected {profile.source_count}"
            )
        for source_id, spans in intervals.items():
            for previous, current in pairwise(sorted(spans)):
                if current[0] <= previous[1]:
                    raise ValueError(
                        f"overlapping deployment phrases for source {source_id}: "
                        f"{previous} and {current}"
                    )
        config = checkpoint.get("config", {})
        model_config = checkpoint.get("model_config", {})
        if profile.checkpoint_epoch is not None and int(checkpoint.get("epoch", -1)) != profile.checkpoint_epoch:
            raise ValueError("retrieval checkpoint epoch does not match its release profile")
        if int(model_config.get("latent_dim", -1)) != profile.embedding_dim:
            raise ValueError("checkpoint latent dimension does not match deployment index")
        text_weight = float(index_manifest.get("text_weight", -1))
        music_weight = float(index_manifest.get("music_weight", -1))
        if text_weight != profile.text_weight or music_weight != profile.music_weight:
            raise ValueError("deployment fusion does not match its release profile")
        if float(config.get("text_weight", -1)) != text_weight or float(
            config.get("music_weight", -1)
        ) != music_weight:
            raise ValueError("checkpoint and deployment index fusion settings differ")

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch

        if self.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("music-text retriever requested CUDA, but CUDA is unavailable")
        index_module = self._inference_module("index")
        audio_module = self._inference_module("audio")
        text_module = self._inference_module("text")
        index_manifest, embeddings, metadata = index_module.read_index(self.index_root)
        summary_path = (self.index_root / "summary.json").resolve(strict=True)
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        declared_checkpoint = str(index_manifest.get("checkpoint", "")).strip()
        if not declared_checkpoint:
            raise ValueError("deployment index does not declare its paired checkpoint")
        paired_checkpoint = (self.index_root / declared_checkpoint).resolve(strict=True)
        if paired_checkpoint != self.checkpoint:
            raise ValueError(
                "configured checkpoint does not match the deployment index checkpoint"
            )
        embeddings = np.asarray(embeddings)
        checkpoint = index_module.load_checkpoint(self.checkpoint, device=self.device)
        self._validate_release(index_manifest, summary, embeddings, metadata, checkpoint, self.profile)
        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        self._deployment_index_rows = len(metadata)
        self._excluded_phrase_ids = self._load_exclusions(metadata)
        active_indices = [
            index
            for index, row in enumerate(metadata)
            if str(row["phrase_id"]) not in self._excluded_phrase_ids
        ]
        embeddings = np.ascontiguousarray(embeddings[active_indices], dtype=np.float32)
        metadata = [metadata[index] for index in active_indices]
        if len(metadata) != self._deployment_index_rows - len(self._excluded_phrase_ids):
            raise ValueError("retrieval exclusion accounting mismatch")
        model = index_module.model_from_checkpoint(checkpoint, device=self.device)
        self._checkpoint_payload = checkpoint
        self._index_manifest = index_manifest
        self._embeddings = embeddings
        self._metadata = metadata
        self._model = model
        self._load_music = audio_module.load_music
        self._text_to_vector = text_module.text_to_vector
        self._validate_genre = text_module.known_genre

    @staticmethod
    def _set_seed(seed: int) -> None:
        import torch

        random.seed(int(seed))
        np.random.seed(int(seed))
        torch.manual_seed(int(seed))

    @staticmethod
    def _convert_audio_to_pcm16_wav(music_path: Path) -> Path:
        import librosa
        import soundfile as sf

        output = music_path.parent / "retrieval_audio_pcm16.wav"
        if output.is_file():
            return output
        signal, sample_rate = librosa.load(str(music_path), sr=22050, mono=True)
        signal = np.asarray(signal, dtype=np.float32)
        if signal.ndim != 1 or signal.size == 0 or not np.isfinite(signal).all():
            raise RetrievalQueryError("uploaded music could not be decoded for retrieval")
        temporary = output.with_suffix(".tmp.wav")
        sf.write(temporary, signal, int(sample_rate), format="WAV", subtype="PCM_16")
        os.replace(temporary, output)
        return output

    def _music_features(
        self, music_path: Path, *, start_frame: int, start_sec: float
    ) -> np.ndarray:
        path = Path(music_path).expanduser().resolve(strict=True)
        if path.suffix.lower() == ".npy":
            return self._load_music(path, start_frame=start_frame, start_sec=start_sec)
        try:
            return self._load_music(path, start_frame=start_frame, start_sec=start_sec)
        except (ValueError, OSError, wave.Error):
            converted = self._convert_audio_to_pcm16_wav(path)
            return self._load_music(converted, start_frame=0, start_sec=start_sec)

    @staticmethod
    def _music_cache_key(
        music_path: Path, *, start_frame: int, start_sec: float
    ) -> tuple[str, int, int, int, float]:
        path = Path(music_path).expanduser().resolve(strict=True)
        stat = path.stat()
        return (
            str(path),
            int(stat.st_size),
            int(stat.st_mtime_ns),
            int(start_frame),
            round(float(start_sec), 6),
        )

    def retrieve_candidates(
        self,
        music_path: Path,
        text: str,
        *,
        top_k: int = 5,
        explicit_genre: str,
        speed: str,
        energy: str,
        start_frame: int = 0,
        start_sec: float = 0.0,
        seed: int = DEFAULT_SEED,
    ) -> MusicTextRetrievalResult:
        if int(top_k) not in ALLOWED_TOP_K:
            raise RetrievalQueryError(f"top_k must be one of {sorted(ALLOWED_TOP_K)}")
        if not str(text).strip():
            raise RetrievalQueryError("retrieval text must not be empty")
        with self._lock:
            self._load()
            assert self._checkpoint_payload is not None
            assert self._index_manifest is not None
            assert self._embeddings is not None
            assert self._metadata is not None
            assert self._model is not None
            self._set_seed(seed)
            allowed = sorted({str(row["genre"]) for row in self._metadata})
            try:
                genre = self._validate_genre(explicit_genre, allowed=allowed)
            except ValueError as exc:
                raise RetrievalQueryError(str(exc)) from exc
            query_genres = (genre,)
            candidate_indices = [
                index
                for index, row in enumerate(self._metadata)
                if str(row.get("genre")) in query_genres
            ]

            import torch

            music_cache_key = self._music_cache_key(
                Path(music_path),
                start_frame=start_frame,
                start_sec=start_sec,
            )
            cached_music_z = self._music_embedding_cache.get(music_cache_key)
            music_embedding_cache_hit = cached_music_z is not None
            if cached_music_z is None:
                music = self._music_features(
                    Path(music_path), start_frame=start_frame, start_sec=start_sec
                )
                feature_stats = self._checkpoint_payload.get("feature_stats", {})
                if "music_mean" in feature_stats and "music_std" in feature_stats:
                    mean = np.asarray(feature_stats["music_mean"], dtype=np.float32)
                    std = np.asarray(feature_stats["music_std"], dtype=np.float32)
                    music = (music - mean) / np.maximum(std, 1.0e-8)
                if not np.isfinite(music).all():
                    raise ValueError("music query contains NaN or infinite values")
                music_tensor = torch.from_numpy(music[None]).to(self.device)
                with torch.inference_mode():
                    music_z = self._model.encode_music(music_tensor).detach().cpu().numpy()[0]
                music_z = np.ascontiguousarray(music_z, dtype=np.float32)
                if not np.isfinite(music_z).all():
                    raise ValueError("retriever produced a non-finite music query embedding")
                self._music_embedding_cache[music_cache_key] = music_z
                self._music_embedding_cache.move_to_end(music_cache_key)
                while len(self._music_embedding_cache) > self._music_embedding_cache_capacity:
                    self._music_embedding_cache.popitem(last=False)
                self._music_embedding_cache_misses += 1
            else:
                self._music_embedding_cache.move_to_end(music_cache_key)
                self._music_embedding_cache_hits += 1
                music_z = cached_music_z

            text_vectors = np.stack(
                [
                    np.asarray(self._text_to_vector(str(text), genre=item, speed=speed, energy=energy), dtype=np.float32)
                    for item in query_genres
                ]
            )
            text_tensor = torch.from_numpy(text_vectors).to(self.device)
            with torch.inference_mode():
                text_z = self._model.encode_text(text_tensor).detach().cpu().numpy()
            if not np.isfinite(music_z).all() or not np.isfinite(text_z).all():
                raise ValueError("retriever produced a non-finite query embedding")
            text_z_by_genre = dict(zip(query_genres, text_z, strict=True))

            selected: list[MusicTextCandidate] = []
            if candidate_indices:
                candidate_embeddings = self._embeddings[candidate_indices]
                candidate_text_embeddings = np.stack(
                    [
                        text_z_by_genre[str(self._metadata[index]["genre"])]
                        for index in candidate_indices
                    ]
                ).astype(np.float32, copy=False)
                text_scores = np.einsum(
                    "ij,ij->i", candidate_embeddings, candidate_text_embeddings, optimize=True
                )
                music_scores = np.einsum(
                    "ij,j->i", candidate_embeddings, music_z.astype(np.float32), optimize=True
                )
                text_weight = float(self._index_manifest["text_weight"])
                music_weight = float(self._index_manifest["music_weight"])
                scores = text_weight * text_scores + music_weight * music_scores
                order = sorted(
                    range(len(candidate_indices)),
                    key=lambda local: (
                        -float(scores[local]),
                        str(self._metadata[candidate_indices[local]]["motion_id"]),
                        int(self._metadata[candidate_indices[local]]["start_frame"]),
                    ),
                )
                for local in order[: int(top_k)]:
                    row = self._metadata[candidate_indices[local]]
                    raw_start = int(row.get("raw_start_frame", row["start_frame"]))
                    raw_end = int(row.get("raw_end_frame", row["end_frame"]))
                    selected.append(
                        MusicTextCandidate(
                            phrase_id=str(row["phrase_id"]),
                            motion_id=str(row["motion_id"]),
                            window_id=str(row["window_id"]),
                            genre=str(row["genre"]),
                            score=float(scores[local]),
                            source_song=str(row["source_song"]),
                            start_frame=raw_start,
                            end_frame=raw_end,
                            trimmed_start_frame=int(row["start_frame"]),
                            trimmed_end_frame=int(row["end_frame"]),
                            motion_path=str(row["motion_path"]),
                            indexed_motion_path=str(row["motion_path"]),
                            floor_offset_y=_checkpoint_floor_offset(row),
                            source_split="train",
                        )
                    )

            warning = None
            if len(candidate_indices) < int(top_k):
                warning = (
                    f"the selected genre pool has only {len(candidate_indices)} usable windows; "
                    "returning all available"
                )
            return MusicTextRetrievalResult(
                query={
                    "music": str(Path(music_path)),
                    "text": str(text),
                    "genre": genre,
                    "genres": list(query_genres),
                    "genre_strategy": "hard" if genre is not None else "all",
                    "top_k": int(top_k),
                    "seed": int(seed),
                    "start_frame": int(start_frame),
                    "start_sec": float(start_sec),
                    "music_embedding_cache_hit": music_embedding_cache_hit,
                    "fusion": {
                        "text": float(self._index_manifest["text_weight"]),
                        "music": float(self._index_manifest["music_weight"]),
                    },
                },
                warning=warning,
                candidate_count=len(candidate_indices),
                returned=len(selected),
                results=tuple(selected),
            )

    def status(self) -> dict[str, Any]:
        with self._lock:
            self._load()
            assert self._metadata is not None
            return {
                "backend": BACKEND_NAME,
                "device": self.device,
                "index_rows": len(self._metadata),
                "deployment_index_rows": self._deployment_index_rows,
                "excluded_rows": len(self._excluded_phrase_ids),
                "excluded_phrase_ids": sorted(self._excluded_phrase_ids),
                "index_sources": len({str(row["source_id"]) for row in self._metadata}),
                "source_splits": dict(Counter(str(row["source_split"]) for row in self._metadata)),
                "checkpoint": self.checkpoint.name,
                "music_embedding_cache": {
                    "capacity": self._music_embedding_cache_capacity,
                    "entries": len(self._music_embedding_cache),
                    "hits": self._music_embedding_cache_hits,
                    "misses": self._music_embedding_cache_misses,
                },
            }


__all__ = [
    "BACKEND_NAME",
    "DEPLOYMENT_INDEX_SCHEMA",
    "MusicTextCandidate",
    "MusicTextRetrievalResult",
    "MusicTextRetriever",
    "RetrievalQueryError",
]
