"""Central configuration with environment and CLI-friendly path resolution."""

from __future__ import annotations

import ipaddress
from pathlib import Path
from typing import Any, Literal, get_args

import yaml
from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# Immutable checkpoint label order; API supplies semantics.
GenreName = Literal[
    "Dai", "ShenYun", "Wei", "Korean", "Urban", "Hiphop", "Popping", "Miao",
    "HanTang", "Breaking", "Kun", "Locking", "Jazz", "Choreography", "Chinese", "DunHuang",
]
GENRE_NAMES = get_args(GenreName)


def normalize_genre(value: object) -> int | None:
    """Validate an exact label or index, never inspect intent text."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return int(value) if float(value).is_integer() and 0 <= value < len(GENRE_NAMES) else None
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.isdigit():
            return normalize_genre(int(stripped))
        return {name.casefold(): i for i, name in enumerate(GENRE_NAMES)}.get(stripped.casefold())
    return None


def genre_name(index: int | None) -> str | None:
    return GENRE_NAMES[index] if index is not None and 0 <= index < len(GENRE_NAMES) else None


class Settings(BaseSettings):
    """Runtime configuration.

    Relative asset paths are resolved against the project root. Upstream code and
    model assets are always treated as read-only by application services.
    """

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        case_sensitive=False,
        extra="ignore",
    )

    customdance_host: str = "127.0.0.1"
    customdance_port: int = Field(default=8000, ge=1024, le=65535)
    customdance_max_upload_mb: int = Field(default=20, ge=1, le=1024)

    completer_root: Path = PROJECT_ROOT / "app/models"
    completer_checkpoint: Path = Path("assets/runtime/inpainting/stage3_model.pt")
    completer_condition_normalizer: Path = Path(
        "assets/runtime/inpainting/condition_normalizer.npz"
    )
    music_text_retriever_root: Path = Path("assets/runtime/retriever")
    music_text_retriever_checkpoint: Path = Path("stage2_model.pt")
    music_text_retriever_index: Path = Path("index")
    music_text_retriever_device: str = "cuda"
    retrieval_exclusions: Path = Path("configs/retrieval_exclusions.json")
    retrieval_profile: Path = Path("configs/retriever.json")
    retriever_factory: str | None = None
    completer_factory: str | None = None
    motion_library_root: Path = Path("assets/runtime/motion_library")
    smpl_model_root: Path = Path("assets/smpl/SMPL_NEUTRAL.pkl")

    openai_api_key: SecretStr | None = None
    openai_audio_model: str = "gpt-audio-1.5"
    openai_structured_model: str = "gpt-5.6-luna"
    openai_max_audio_mb: int = Field(default=20, ge=1, le=100)
    openai_max_duration_sec: float = Field(default=34.0, ge=4.0, le=600.0)

    @field_validator("customdance_host")
    @classmethod
    def host_must_be_loopback(cls, value: str) -> str:
        if value == "localhost":
            return value
        try:
            address = ipaddress.ip_address(value)
        except ValueError as exc:
            raise ValueError("CUSTOMDANCE_HOST must be localhost or a loopback IP") from exc
        if not address.is_loopback:
            raise ValueError("backend binding is restricted to loopback addresses")
        return value

    @field_validator("music_text_retriever_device")
    @classmethod
    def retriever_device_must_be_supported(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in {"cpu", "cuda"}:
            raise ValueError("MUSIC_TEXT_RETRIEVER_DEVICE must be cpu or cuda")
        return normalized

    @staticmethod
    def resolve_path(path: Path | None) -> Path | None:
        if path is None:
            return None
        path = path.expanduser()
        return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()

    def asset_paths(self) -> dict[str, Path | None]:
        retriever_root = self.resolve_path(self.music_text_retriever_root)
        assert retriever_root is not None

        def retriever_path(path: Path) -> Path:
            value = path.expanduser()
            return value.resolve() if value.is_absolute() else (retriever_root / value).resolve()

        return {
            "completer_root": self.resolve_path(self.completer_root),
            "completer_checkpoint": self.resolve_path(self.completer_checkpoint),
            "completer_condition_normalizer": self.resolve_path(
                self.completer_condition_normalizer
            ),
            "music_text_retriever_root": retriever_root,
            "music_text_retriever_checkpoint": retriever_path(self.music_text_retriever_checkpoint),
            "music_text_retriever_index": retriever_path(self.music_text_retriever_index),
            "retrieval_exclusions": self.resolve_path(self.retrieval_exclusions),
            "retrieval_profile": self.resolve_path(self.retrieval_profile),
            "motion_library_root": self.resolve_path(self.motion_library_root),
            "smpl_model_root": self.resolve_path(self.smpl_model_root),
        }


def load_settings(config_path: Path | None = None, **cli_overrides: Any) -> Settings:
    """Load settings with CLI > YAML > environment/.env precedence."""

    configured: dict[str, Any] = {}
    if config_path is not None:
        path = Path(config_path).expanduser().resolve(strict=True)
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("configuration YAML must contain a mapping")
        server = payload.get("server", {})
        audio = payload.get("audio", {})
        if not isinstance(server, dict) or not isinstance(audio, dict):
            raise ValueError("server and audio YAML sections must be mappings")
        mapping = {
            "customdance_host": server.get("host"),
            "customdance_port": server.get("port"),
            "customdance_max_upload_mb": server.get("max_upload_mb"),
            "openai_audio_model": audio.get("audio_model"),
            "openai_structured_model": audio.get("structured_model"),
            "openai_max_audio_mb": audio.get("max_external_audio_mb"),
            "openai_max_duration_sec": audio.get("max_external_duration_sec"),
        }
        configured.update({key: value for key, value in mapping.items() if value is not None})
        assets = payload.get("assets", {})
        components = payload.get("components", {})
        if not isinstance(assets, dict) or not isinstance(components, dict):
            raise ValueError("assets and components YAML sections must be mappings")
        asset_fields = {
            "completer_checkpoint",
            "completer_condition_normalizer",
            "music_text_retriever_root",
            "music_text_retriever_checkpoint",
            "music_text_retriever_index",
            "music_text_retriever_device",
            "retrieval_exclusions",
            "retrieval_profile",
            "motion_library_root",
            "smpl_model_root",
        }
        configured.update({key: value for key, value in assets.items() if key in asset_fields})
        configured.update(
            {
                key: value
                for key, value in components.items()
                if key in {"retriever_factory", "completer_factory"}
            }
        )
    configured.update({key: value for key, value in cli_overrides.items() if value is not None})
    return Settings(**configured)
