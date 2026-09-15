"""Local-only extraction of a browser-ready Neutral SMPL skin template."""

from __future__ import annotations

import gc
import importlib.util
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import orjson

from app.utils.kinematics import SMPL24_NAMES, SMPL24_PARENTS

SMPL_DOWNLOAD_URL = "https://smpl.is.tue.mpg.de/"
NEUTRAL_MODEL_FILENAMES = (
    "SMPL_NEUTRAL.pkl",
    "basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl",
)


def _as_numpy(value: Any, *, dtype: np.dtype | type) -> np.ndarray:
    """Detach torch-like values without making torch a base dependency."""

    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value, dtype=dtype)


class NeutralSmplTemplateService:
    """Extract the small render subset of a user-supplied SMPL-Model PKL.

    The original licensed model is never copied, served, or returned. The
    response contains only the neutral template data required by a local
    Three.js SkinnedMesh preview.
    """

    def __init__(
        self,
        model_root: Path | None,
        *,
        model_loader: Callable[[Path], Any] | None = None,
    ) -> None:
        self.model_root = None if model_root is None else Path(model_root)
        self._model_loader = model_loader or self._load_with_smplx
        self._cached_payload: bytes | None = None
        self._lock = threading.Lock()

    def _model_path(self) -> Path | None:
        root = self.model_root
        if root is None:
            return None
        if root.is_file():
            return root if root.suffix.lower() == ".pkl" else None
        for filename in NEUTRAL_MODEL_FILENAMES:
            candidate = root / filename
            if candidate.is_file():
                return candidate
        return None

    def status(self) -> dict[str, Any]:
        path = self._model_path()
        runtime_installed = importlib.util.find_spec("smplx") is not None
        model_installed = path is not None
        if not model_installed:
            detail = (
                "Neutral SMPL PKL is not installed. Download it under the official "
                "license and place it in the configured SMPL_MODEL_ROOT."
            )
        elif not runtime_installed:
            detail = "The required smplx runtime is not installed."
        else:
            detail = "User-supplied Neutral SMPL is ready for local preview."
        return {
            "available": model_installed and runtime_installed,
            "model_installed": model_installed,
            "runtime_installed": runtime_installed,
            "model_type": "smpl",
            "gender": "neutral",
            "filename": path.name if path else None,
            "source": "user-supplied-local",
            "download_url": SMPL_DOWNLOAD_URL,
            "redistribution": "not-included",
            "detail": detail,
        }

    @staticmethod
    def _load_with_smplx(path: Path) -> Any:
        try:
            from smplx import SMPL
        except ImportError as exc:
            raise RuntimeError(
                "SMPL preview requires the required smplx runtime; install requirements.txt"
            ) from exc
        return SMPL(
            str(path),
            create_betas=False,
            create_global_orient=False,
            create_body_pose=False,
            create_transl=False,
        )

    @staticmethod
    def _top_four_weights(weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        unordered = np.argpartition(-weights, kth=3, axis=1)[:, :4]
        values = np.take_along_axis(weights, unordered, axis=1)
        order = np.argsort(-values, axis=1)
        indices = np.take_along_axis(unordered, order, axis=1).astype(np.uint16)
        padded = np.take_along_axis(weights, indices.astype(np.int64), axis=1)
        totals = padded.sum(axis=1, keepdims=True)
        if np.any(totals <= 1e-8):
            raise ValueError("SMPL skin weights contain an unbound vertex")
        return indices, (padded / totals).astype(np.float32)

    def _build_payload(self, model: Any) -> bytes:
        vertices = np.ascontiguousarray(
            _as_numpy(model.v_template, dtype=np.float32)
        )
        faces = np.ascontiguousarray(_as_numpy(model.faces, dtype=np.uint32))
        weights = np.ascontiguousarray(
            _as_numpy(model.lbs_weights, dtype=np.float32)
        )
        regressor = np.ascontiguousarray(
            _as_numpy(model.J_regressor, dtype=np.float32)
        )
        parents = np.ascontiguousarray(
            _as_numpy(model.parents, dtype=np.int16).reshape(-1)
        )

        vertex_count = vertices.shape[0]
        if vertices.ndim != 2 or vertices.shape[1] != 3 or vertex_count < 1:
            raise ValueError("SMPL v_template must have shape (V, 3)")
        if faces.ndim != 2 or faces.shape[1] != 3 or faces.shape[0] < 1:
            raise ValueError("SMPL faces must have shape (F, 3)")
        if weights.shape != (vertex_count, 24):
            raise ValueError("SMPL skin weights must have shape (V, 24)")
        if regressor.shape != (24, vertex_count):
            raise ValueError("SMPL joint regressor must have shape (24, V)")
        if parents.shape != (24,) or not np.array_equal(parents, SMPL24_PARENTS):
            raise ValueError("SMPL joint hierarchy does not match Canonical SMPL-24")
        if int(faces.max()) >= vertex_count:
            raise ValueError("SMPL face index exceeds the template vertex count")
        if (
            not np.isfinite(vertices).all()
            or not np.isfinite(weights).all()
            or not np.isfinite(regressor).all()
        ):
            raise ValueError("SMPL template contains non-finite values")
        if np.any(weights < 0):
            raise ValueError("SMPL skin weights must be non-negative")

        joints = np.ascontiguousarray(regressor @ vertices, dtype=np.float32)
        skin_indices, skin_weights = self._top_four_weights(weights)
        payload = {
            "schema_version": "1.0",
            "model_type": "smpl",
            "gender": "neutral",
            "coordinate_system": "right-handed-y-up",
            "native_front_axis": "+Z",
            "vertices": vertices,
            "faces": faces,
            "joints": joints,
            "parents": parents,
            "joint_names": list(SMPL24_NAMES),
            "skin_indices": skin_indices,
            "skin_weights": skin_weights,
            "rest_height": float(vertices[:, 1].max() - vertices[:, 1].min()),
            "source": "user-supplied-local",
            "redistribution": "not-included",
        }
        return orjson.dumps(payload, option=orjson.OPT_SERIALIZE_NUMPY)

    def template_json(self) -> bytes:
        with self._lock:
            if self._cached_payload is not None:
                return self._cached_payload
            path = self._model_path()
            if path is None:
                raise FileNotFoundError(
                    "Neutral SMPL PKL is missing from the configured SMPL_MODEL_ROOT"
                )
            model = self._model_loader(path)
            try:
                payload = self._build_payload(model)
            finally:
                del model
                gc.collect()
            self._cached_payload = payload
            return payload


__all__ = [
    "NEUTRAL_MODEL_FILENAMES",
    "SMPL_DOWNLOAD_URL",
    "NeutralSmplTemplateService",
]
