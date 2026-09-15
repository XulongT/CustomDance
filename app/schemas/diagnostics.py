"""Versioned AKE/RKE diagnostic response models."""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


class DiagnosticPeak(BaseModel):
    curve: str
    frame_index: int = Field(ge=0)
    time_sec: float = Field(ge=0)
    value: float = Field(ge=0)
    robust_z: float = Field(ge=0)


class MotionDiagnostics(BaseModel):
    schema_version: str = "1.1"
    fps: int = 30
    frame_count: int = Field(gt=0)
    dt: float = Field(gt=0)
    coordinate_system: str
    position_convention: str
    rotation_convention: str
    group_mapping: dict[str, list[int]]
    ake: list[float]
    rke: dict[str, list[float]]
    ake_anomaly: list[float]
    rke_anomaly: dict[str, list[float]]
    peaks: list[DiagnosticPeak]
    peak_threshold_robust_z: float
    display_floor_robust_z: float
    display_saturation_robust_z: float
    display_filter: str

    @model_validator(mode="after")
    def validate_curve_lengths(self) -> MotionDiagnostics:
        if self.schema_version != "1.1" or self.fps != 30:
            raise ValueError("unsupported diagnostic schema or frame rate")
        if len(self.ake) != self.frame_count or len(self.ake_anomaly) != self.frame_count:
            raise ValueError("AKE curve length does not match frame_count")
        if set(self.rke) != set(self.group_mapping) or set(self.rke_anomaly) != set(
            self.group_mapping
        ):
            raise ValueError("RKE curves and body-group mapping disagree")
        if any(
            len(curve) != self.frame_count
            for curves in (self.rke, self.rke_anomaly)
            for curve in curves.values()
        ):
            raise ValueError("RKE curve length does not match frame_count")
        display_values = [*self.ake_anomaly]
        for curve in self.rke_anomaly.values():
            display_values.extend(curve)
        if any(value < 0 or value > 1 for value in display_values):
            raise ValueError("diagnostic anomaly display values must be normalized")
        return self
