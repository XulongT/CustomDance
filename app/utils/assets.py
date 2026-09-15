"""Asset manifest loading and startup validation without deserializing assets."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from app.config import PROJECT_ROOT, Settings


@dataclass(slots=True)
class AssetValidation:
    asset_id: str
    required: bool
    configured_path: str
    exists: bool
    status: str
    detail: str


_ENV_PATTERN = re.compile(r"^\$\{([A-Z0-9_]+)\}(.*)$")


def _resolve_manifest_path(
    raw_path: str, settings: Settings | None = None
) -> tuple[Path | None, str | None]:
    settings = settings or Settings()
    match = _ENV_PATTERN.match(raw_path)
    if match:
        variable, suffix = match.groups()
        base = settings.asset_paths().get(variable.lower())
        if base is None:
            return None, f"asset setting {variable} is not configured"
        path = base / suffix.lstrip("/\\") if suffix else base
        return path.resolve(), None
    path = Path(raw_path).expanduser()
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve(), None


def load_asset_manifest(path: Path | None = None) -> dict[str, Any]:
    manifest_path = path or PROJECT_ROOT / "ASSET_MANIFEST.json"
    with Path(manifest_path).open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("schema_version") != "1.0" or not isinstance(manifest.get("assets"), list):
        raise ValueError("invalid asset manifest schema")
    return manifest


def validate_assets(
    path: Path | None = None, *, settings: Settings | None = None
) -> list[AssetValidation]:
    records: list[AssetValidation] = []
    for asset in load_asset_manifest(path)["assets"]:
        resolved, error = _resolve_manifest_path(asset["local_path"], settings)
        required = bool(asset["required"])
        if error:
            records.append(
                AssetValidation(
                    asset_id=asset["asset_id"],
                    required=required,
                    configured_path=asset["local_path"],
                    exists=False,
                    status="BLOCKED" if required else "OPTIONAL_MISSING",
                    detail=error,
                )
            )
            continue
        assert resolved is not None
        kind = asset.get("kind")
        exists = resolved.is_file() if kind == "file" else (
            resolved.is_dir() if kind == "directory" else resolved.exists()
        )
        records.append(
            AssetValidation(
                asset_id=asset["asset_id"],
                required=required,
                configured_path=str(resolved),
                exists=exists,
                status="PASS" if exists else ("BLOCKED" if required else "OPTIONAL_MISSING"),
                detail="path exists" if exists else "path does not exist",
            )
        )
    return records


def validation_report(
    path: Path | None = None, *, settings: Settings | None = None
) -> dict[str, Any]:
    records = validate_assets(path, settings=settings)
    return {
        "schema_version": "1.0",
        "ready": all(record.exists for record in records if record.required),
        "assets": [asdict(record) for record in records],
    }
