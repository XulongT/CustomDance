"""Versioned compatibility contract for a paired retriever and motion index."""

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator


class RetrievalProfile(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "1.0"
    index_rows: int = Field(default=4982, gt=0)
    source_count: int = Field(default=161, gt=0)
    embedding_dim: int = Field(default=128, gt=0)
    checkpoint_epoch: int | None = Field(default=500, ge=0)
    text_weight: float = Field(default=0.75, ge=0, le=1)
    music_weight: float = Field(default=0.25, ge=0, le=1)

    @model_validator(mode="after")
    def validate_contract(self):
        if self.schema_version != "1.0":
            raise ValueError("unsupported retrieval profile schema")
        if abs(self.text_weight + self.music_weight - 1.0) > 1e-6:
            raise ValueError("retrieval fusion weights must sum to one")
        return self

    @classmethod
    def load(cls, path: Path):
        return cls.model_validate(json.loads(path.read_text(encoding="utf-8")))
