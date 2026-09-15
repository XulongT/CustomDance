"""Lazy real-model services shared by the local FastAPI process."""

from __future__ import annotations

import importlib
import threading

from app.config import Settings
from app.utils.inpainting import InpaintingBackend
from app.utils.motion_library import MotionLibrary
from app.utils.retriever import MusicTextRetriever


class RuntimeServices:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._lock = threading.RLock()
        self._library: MotionLibrary | None = None
        self._music_text_retriever: MusicTextRetriever | None = None
        self._inpainting: InpaintingBackend | None = None

    def motion_library(self) -> MotionLibrary:
        with self._lock:
            if self._library is None:
                root = Settings.resolve_path(self.settings.motion_library_root)
                assert root is not None
                self._library = MotionLibrary(root)
            return self._library

    def retriever(self) -> MusicTextRetriever:
        with self._lock:
            if self._music_text_retriever is None:
                if self.settings.retriever_factory:
                    self._music_text_retriever = _component(
                        self.settings.retriever_factory, self.settings
                    )
                    return self._music_text_retriever
                paths = self.settings.asset_paths()
                root = paths["music_text_retriever_root"]
                checkpoint = paths["music_text_retriever_checkpoint"]
                index = paths["music_text_retriever_index"]
                exclusions = paths["retrieval_exclusions"]
                assert root is not None
                assert checkpoint is not None
                assert index is not None
                assert exclusions is not None
                self._music_text_retriever = MusicTextRetriever(
                    root,
                    checkpoint,
                    index,
                    exclusions_path=exclusions,
                    device=self.settings.music_text_retriever_device,
                    profile_path=paths["retrieval_profile"],
                )
            return self._music_text_retriever

    def completer(self) -> InpaintingBackend:
        with self._lock:
            if self._inpainting is None:
                if self.settings.completer_factory:
                    self._inpainting = _component(self.settings.completer_factory, self.settings)
                    return self._inpainting
                root = Settings.resolve_path(self.settings.completer_root)
                checkpoint = Settings.resolve_path(self.settings.completer_checkpoint)
                condition_normalizer = Settings.resolve_path(
                    self.settings.completer_condition_normalizer
                )
                assert root is not None and checkpoint is not None
                assert condition_normalizer is not None
                self._inpainting = InpaintingBackend(
                    root,
                    checkpoint,
                    condition_normalizer,
                    device="cuda",
                )
            return self._inpainting

    def remaker(self) -> InpaintingBackend:
        """Remaker uses the same loaded inpainting model and lock as Completer."""
        return self.completer()


def _component(specification: str, settings: Settings):
    """Resolve an administrator-configured local module:function factory."""
    module, separator, name = specification.partition(":")
    if not separator or not module or not name or ":" in name:
        raise ValueError("component factory must be module:function")
    factory = getattr(importlib.import_module(module), name)
    if not callable(factory):
        raise TypeError("component factory must be callable")
    return factory(settings)
