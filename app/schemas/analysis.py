"""Structured Analyze output protocol."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

from app.config import GenreName


class TimedFeatureCurve(BaseModel):
    times_sec: list[float]
    values: list[float]

    @model_validator(mode="after")
    def aligned_and_finite(self) -> TimedFeatureCurve:
        if len(self.times_sec) != len(self.values):
            raise ValueError("feature curve time/value lengths must match")
        return self


class LocalAudioFeatures(BaseModel):
    schema_version: Literal["1.0"] = "1.0"
    duration_sec: float = Field(gt=0)
    sample_rate: int = Field(gt=0)
    tempo_bpm: float = Field(ge=0)
    beat_times_sec: list[float]
    onset: TimedFeatureCurve
    rms: TimedFeatureCurve
    chroma_mean: list[float] = Field(min_length=12, max_length=12)
    section_boundaries_sec: list[float]


class AnalysisSegment(BaseModel):
    segment_id: str
    start_sec: float = Field(ge=0)
    end_sec: float = Field(gt=0)
    description: str
    energy: Literal["low", "medium", "high"]
    confidence: float = Field(ge=0, le=1)


class SlotSuggestion(BaseModel):
    genre: GenreName
    slot_id: str
    start_sec: float = Field(ge=0)
    duration_sec: float = Field(default=4.0)
    music_description: str = Field(default="", max_length=300)
    cue: str
    reason: str
    confidence: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def fixed_duration(self) -> SlotSuggestion:
        if abs(self.duration_sec - 4.0) > 1e-9:
            raise ValueError("candidate slots must be exactly 4 seconds")
        return self


class AnalyzeResult(BaseModel):
    genre: GenreName
    schema_version: Literal["1.0"] = "1.0"
    duration_sec: float = Field(gt=0)
    summary: str
    segments: list[AnalysisSegment]
    slots: list[SlotSuggestion]

    @model_validator(mode="after")
    def ranges_are_valid(self) -> AnalyzeResult:
        for segment in self.segments:
            if segment.start_sec >= segment.end_sec:
                raise ValueError(f"segment {segment.segment_id} has an empty range")
            if segment.end_sec > self.duration_sec + 1e-9:
                raise ValueError(f"segment {segment.segment_id} exceeds audio duration")
        for slot in self.slots:
            if slot.start_sec + slot.duration_sec > self.duration_sec + 1e-9:
                raise ValueError(f"slot {slot.slot_id} exceeds audio duration")
        return self
