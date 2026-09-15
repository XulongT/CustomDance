"""HTTP request/response schemas for the local application."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.schemas.analysis import AnalyzeResult
from app.schemas.timeline import RetrievalResponse, TimelineState


class UploadResponse(BaseModel):
    session_id: str
    original_filename: str
    audio_url: str
    size_bytes: int


class AnalyzeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    mode: Literal["openai"] = "openai"
    global_intent: str = Field(default="", max_length=2000)
    confirm_reset: bool = False
    consent_to_external_api: bool = False


class AnalyzeResponse(BaseModel):
    backend: str
    external_api_used: bool
    tempo_bpm: float = Field(ge=0)
    sample_rate: int = Field(gt=0)
    semantic_status: Literal["full"]
    warnings: list[str] = Field(default_factory=list)
    timings_ms: dict[str, float] = Field(default_factory=dict)
    analysis: AnalyzeResult
    timeline: TimelineState


class RetrievalRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(min_length=1, max_length=2000)
    top_k: Literal[5, 10, 15] = 15


class FillRequest(BaseModel):
    clip_id: str = Field(min_length=1, max_length=256)


class GenerationSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    seed: int = Field(default=42, ge=0, le=2**31 - 1)
    steps: int = Field(default=101, ge=2, le=101)


class CompleteRequest(GenerationSettings):
    pass


class RemakeRequest(GenerationSettings):
    start_frame: int = Field(ge=0)
    end_frame: int = Field(gt=0)


RkeJointGroup = Literal[
    "head",
    "torso",
    "left_arm",
    "right_arm",
    "left_leg",
    "right_leg",
]


class SmoothRequest(BaseModel):
    start_frame: int = Field(ge=0)
    end_frame: int = Field(gt=0)
    smooth_translation: bool = False
    joint_groups: list[RkeJointGroup] = Field(default_factory=list, max_length=6)

    @model_validator(mode="after")
    def requires_a_smoothing_target(self) -> SmoothRequest:
        if not self.smooth_translation and not self.joint_groups:
            raise ValueError("Smooth requires AKE translation or at least one RKE joint group")
        if len(set(self.joint_groups)) != len(self.joint_groups):
            raise ValueError("Smooth joint_groups must not contain duplicates")
        return self


class MotionPayload(BaseModel):
    source_clip_id: str
    fps: int
    frames: int
    duration_sec: float
    coordinate_system: str
    smpl_poses: list
    smpl_trans: list
    joint_positions: list


class SessionStateResponse(BaseModel):
    session_id: str
    original_filename: str
    audio_url: str
    analysis: AnalyzeResult | None
    timeline: TimelineState | None
    motion_frames: int | None
    filled_frame_count: int
    assignments: list[dict]


__all__ = [
    "AnalyzeRequest",
    "AnalyzeResponse",
    "CompleteRequest",
    "FillRequest",
    "MotionPayload",
    "RemakeRequest",
    "RetrievalRequest",
    "RetrievalResponse",
    "SessionStateResponse",
    "SmoothRequest",
    "UploadResponse",
]
