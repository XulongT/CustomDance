#!/usr/bin/env python3
"""Launch the loopback CustomDance app with CLI > YAML > environment precedence."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import uvicorn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.api.main import create_app
from app.config import load_settings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/default.yaml")
    parser.add_argument("--host", dest="customdance_host", default=None)
    parser.add_argument("--port", dest="customdance_port", type=int, default=None)
    parser.add_argument("--max-upload-mb", dest="customdance_max_upload_mb", type=int, default=None)
    parser.add_argument("--completer-checkpoint", type=Path, default=None)
    parser.add_argument("--completer-condition-normalizer", type=Path, default=None)
    parser.add_argument("--music-text-retriever-root", type=Path, default=None)
    parser.add_argument("--music-text-retriever-checkpoint", type=Path, default=None)
    parser.add_argument("--music-text-retriever-index", type=Path, default=None)
    parser.add_argument("--music-text-retriever-device", choices=("cpu", "cuda"), default=None)
    parser.add_argument("--motion-library-root", type=Path, default=None)
    parser.add_argument("--retrieval-profile", type=Path, default=None)
    parser.add_argument("--smpl-model-root", type=Path, default=None)
    parser.add_argument("--openai-audio-model", default=None)
    parser.add_argument("--openai-structured-model", default=None)
    parser.add_argument("--openai-max-audio-mb", type=int, default=None)
    parser.add_argument("--openai-max-duration-sec", type=float, default=None)
    args = parser.parse_args()
    raw = vars(args)
    config = raw.pop("config")
    settings = load_settings(config, **raw)
    uvicorn.run(
        create_app(settings),
        host=settings.customdance_host,
        port=settings.customdance_port,
        log_level="info",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
