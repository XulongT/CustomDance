"""Timeline and retrieval API schemas."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, model_validator


class SlotStatus(str, Enum):
    CANDIDATE = "candidate"
    ACCEPTED = "accepted"
    INVALID = "invalid"
    FILLED = "filled"


class TimelineSlot(BaseModel):
    slot_id: str
    start_sec: float = Field(ge=0)
    duration_sec: float = Field(default=4.0)
    music_description: str = Field(default="", max_length=300)
    cue: str = ""
    status: SlotStatus = SlotStatus.CANDIDATE
    filled_clip_id: str | None = None
    retrieval_score: float | None = None

    @model_validator(mode="after")
    def fixed_duration(self) -> TimelineSlot:
        if abs(self.duration_sec - 4.0) > 1e-9:
            raise ValueError("timeline slots must be exactly 4 seconds")
        return self

    @property
    def end_sec(self) -> float:
        return self.start_sec + self.duration_sec


class TimelineState(BaseModel):
    slots: list[TimelineSlot] = Field(default_factory=list)
    current_focused_slot_id: str | None = None


class RetrievalItem(BaseModel):
    clip_id: str
    score: float
    genre: str | int | float | None = None
    genre_name: str | None = None
    duration: float
    preview_metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievalResponse(BaseModel):
    query: str
    model_query: dict[str, Any] = Field(default_factory=dict)
    top_k: int
    backend: str
    genre_mode: str
    genre_filter: int | None = None
    genre_name: str | None = None
    candidate_count: int
    warning: str | None = None
    items: list[RetrievalItem]
