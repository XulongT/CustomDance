"""In-memory local project sessions with project-contained media paths."""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from app.schemas.analysis import AnalyzeResult, LocalAudioFeatures
from app.timeline import TimelineEditor


@dataclass(slots=True)
class ProjectSession:
    session_id: str
    root: Path
    audio_path: Path
    original_filename: str
    audio_size_bytes: int = 0
    global_intent: str = ""
    local_features: LocalAudioFeatures | None = None
    analysis: AnalyzeResult | None = None
    editor: TimelineEditor | None = None
    retrieval_results: dict[str, float] = field(default_factory=dict)
    retrieval_candidates: dict[str, Any] = field(default_factory=dict)
    completer_audio_features: np.ndarray | None = None
    motion_revision: int = 0
    lock: threading.RLock = field(default_factory=threading.RLock)

    def reset_from_analysis(
        self,
        analysis: AnalyzeResult,
        features: LocalAudioFeatures,
        *,
        global_intent: str,
    ) -> None:
        self.global_intent = global_intent
        self.local_features = features
        self.analysis = analysis
        self.editor = TimelineEditor.from_analysis(analysis)
        self.retrieval_results.clear()
        self.retrieval_candidates.clear()
        self.completer_audio_features = None
        self.motion_revision += 1


class SessionStore:
    def __init__(self, root: Path):
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._sessions: dict[str, ProjectSession] = {}
        self._lock = threading.RLock()

    def create(self, original_filename: str, suffix: str) -> ProjectSession:
        session_id = uuid.uuid4().hex
        session_root = (self.root / session_id).resolve()
        if self.root not in session_root.parents:
            raise RuntimeError("generated session path escaped the configured root")
        session_root.mkdir(parents=True, exist_ok=False)
        session = ProjectSession(
            session_id=session_id,
            root=session_root,
            audio_path=session_root / f"music{suffix}",
            original_filename=original_filename,
        )
        with self._lock:
            self._sessions[session_id] = session
        return session

    def get(self, session_id: str) -> ProjectSession:
        with self._lock:
            try:
                return self._sessions[session_id]
            except KeyError as exc:
                raise KeyError(f"unknown or expired session: {session_id}") from exc
