"""API-only intent semantics over immutable local audio timings."""

from __future__ import annotations

import base64
import json
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

import librosa
import numpy as np
import soundfile as sf
from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAIError
from pydantic import BaseModel, Field, SecretStr

from app.schemas.analysis import (
    AnalysisSegment,
    AnalyzeResult,
    LocalAudioFeatures,
    SlotSuggestion,
)
from app.config import GenreName, GENRE_NAMES
from app.utils.audio_analysis import TimingPlan, build_timing_plan


class OpenAIAnalyzeError(Exception):
    """Sanitized external API failure safe to expose through the local UI."""

    def __init__(
        self,
        *,
        stage: str,
        model: str,
        status_code: int,
        error_code: str,
        message: str,
        request_id: str | None = None,
    ) -> None:
        self.stage = stage
        self.model = model
        self.status_code = status_code
        self.error_code = error_code
        self.message = message
        self.request_id = request_id
        super().__init__(self.public_detail)

    @property
    def public_detail(self) -> str:
        request = f" Request ID: {self.request_id}." if self.request_id else ""
        return (
            f"OpenAI {self.stage} failed for {self.model} "
            f"({self.error_code}): {self.message}.{request}"
        )


def _safe_openai_message(exc: OpenAIError) -> str:
    """Extract the provider message while redacting secrets and bounding output."""

    body = getattr(exc, "body", None)
    message: Any = None
    if isinstance(body, dict):
        error = body.get("error", body)
        if isinstance(error, dict):
            message = error.get("message")
        elif isinstance(error, str):
            message = error
    if not isinstance(message, str) or not message.strip():
        message = str(exc)
    message = re.sub(r"sk-[A-Za-z0-9_-]+", "[redacted]", message.strip())
    message = " ".join(message.split())
    return message[:500].rstrip(" .") or "The provider did not return an error description"


def _translate_openai_error(
    exc: OpenAIError,
    *,
    stage: str,
    model: str,
) -> OpenAIAnalyzeError:
    if isinstance(exc, APITimeoutError):
        status_code = 504
        error_code = "timeout"
    elif isinstance(exc, APIConnectionError):
        status_code = 503
        error_code = "connection_error"
    elif isinstance(exc, APIStatusError):
        provider_status = int(exc.status_code)
        status_code = provider_status if provider_status < 500 else 502
        error_code = {
            400: "bad_request",
            401: "authentication_error",
            403: "permission_denied",
            404: "model_not_found",
            409: "conflict",
            422: "unprocessable_request",
            429: "rate_limit_or_quota",
        }.get(provider_status, f"provider_http_{provider_status}")
    else:
        status_code = 502
        error_code = "openai_error"
    request_id = getattr(exc, "request_id", None)
    if isinstance(request_id, str):
        request_id = request_id.strip() or None
    else:
        request_id = None
    return OpenAIAnalyzeError(
        stage=stage,
        model=model,
        status_code=status_code,
        error_code=error_code,
        message=_safe_openai_message(exc),
        request_id=request_id,
    )


class SemanticSegmentLabel(BaseModel):
    """Semantic labels for one locally timed segment."""

    segment_id: str = Field(min_length=1)
    description: str = Field(min_length=1, max_length=300)
    energy: Literal["low", "medium", "high"]
    confidence: float = Field(ge=0, le=1)


class SemanticSlotLabel(BaseModel):
    genre: GenreName
    """Semantic labels for one locally generated four-second window."""

    window_id: str = Field(min_length=1)
    music_description: str = Field(
        min_length=1,
        max_length=300,
        description="Music-only description of this exact four-second window.",
    )
    cue: str = Field(min_length=1, max_length=240)
    reason: str = Field(min_length=1, max_length=300)
    confidence: float = Field(ge=0, le=1)


class SemanticChoreographyLabels(BaseModel):
    genre: GenreName
    """Structured semantic output that deliberately contains no timestamps."""

    summary: str = Field(min_length=1, max_length=600)
    segments: list[SemanticSegmentLabel] = Field(min_length=1, max_length=32)
    slots: list[SemanticSlotLabel] = Field(min_length=1, max_length=16)


class RetrievalIntent(BaseModel):
    """API-selected attributes required by the trained text encoder."""

    genre: GenreName
    speed: Literal["slow", "medium", "fast"]
    energy: Literal["calm", "energetic"]
    query: str = Field(min_length=1, max_length=2000)


@dataclass(frozen=True, slots=True)
class OpenAIAnalyzeOutcome:
    """Analysis result and provider metadata."""

    result: AnalyzeResult
    warnings: tuple[str, ...]
    semantic_status: Literal["full"]
    audio_semantics_used: bool
    structured_semantics_used: bool
    normalized_audio_chunks: int
    normalized_sample_rate: int = 24_000
    normalized_channels: int = 1
    timings_ms: dict[str, float] = field(default_factory=dict)

    @property
    def backend(self) -> str:
        return "openai-live"


@dataclass(frozen=True, slots=True)
class _AudioChunk:
    samples: np.ndarray
    label: str

    @property
    def duration_sec(self) -> float:
        return float(self.samples.size / 24_000)


class OpenAIAnalyzeService:
    """External semantic analysis that requires explicit per-request opt-in.

    ``gpt-audio-1.5`` produces a semantic narrative without Structured Outputs.
    A text model labels only IDs from a deterministic librosa timing plan. The
    application then assembles ``AnalyzeResult`` while preserving every local
    timestamp exactly. No key, audio bytes, or authorization metadata is logged
    or persisted by this service.
    """

    NORMALIZED_SAMPLE_RATE = 24_000
    PRIMARY_CHUNK_SEC = 18.0
    MIN_RETRY_CHUNK_SEC = 4.0

    def __init__(
        self,
        *,
        api_key: SecretStr | str | None,
        audio_model: str,
        structured_model: str,
        max_audio_mb: int = 20,
        max_duration_sec: float = 34.0,
        client: Any | None = None,
    ):
        if api_key is None:
            self._api_key: str | None = None
        elif isinstance(api_key, SecretStr):
            self._api_key = api_key.get_secret_value().strip() or None
        else:
            self._api_key = api_key.strip() or None
        self.audio_model = audio_model.strip()
        self.structured_model = structured_model.strip()
        self.max_audio_bytes = max_audio_mb * 1024 * 1024
        self.max_duration_sec = float(max_duration_sec)
        self._client = client
        if not self.audio_model or not self.structured_model:
            raise ValueError("both OpenAI model names must be configured")

    def _get_client(self) -> Any:
        if self._client is None:
            if not self._api_key:
                raise RuntimeError("OPENAI_API_KEY is not configured")
            from openai import OpenAI

            self._client = OpenAI(api_key=self._api_key)
        return self._client

    def _validate_audio(
        self, audio_path: Path, features: LocalAudioFeatures, *, consent: bool
    ) -> Path:
        if not consent:
            raise PermissionError(
                "explicit consent is required because the audio may be sent to an external API"
            )
        if not self._api_key and self._client is None:
            raise RuntimeError("OPENAI_API_KEY is not configured")
        path = Path(audio_path).expanduser().resolve(strict=True)
        if path.suffix.lower() != ".wav":
            raise ValueError(
                "live OpenAI Analyze currently accepts WAV only; convert explicitly first"
            )
        if path.stat().st_size > self.max_audio_bytes:
            raise ValueError(
                f"audio exceeds the configured {self.max_audio_bytes / 1024**2:.0f} MB app limit"
            )
        if features.duration_sec > self.max_duration_sec + 1e-9:
            raise ValueError(
                f"audio duration {features.duration_sec:.3f}s exceeds the configured "
                f"{self.max_duration_sec:.3f}s app limit"
            )
        return path

    @classmethod
    def _normalized_chunks(cls, path: Path) -> list[_AudioChunk]:
        samples, sample_rate = librosa.load(
            path,
            sr=cls.NORMALIZED_SAMPLE_RATE,
            mono=True,
        )
        samples = np.asarray(samples, dtype=np.float32).reshape(-1)
        if sample_rate != cls.NORMALIZED_SAMPLE_RATE:
            raise RuntimeError("audio normalization returned an unexpected sample rate")
        if samples.size == 0 or not np.isfinite(samples).all():
            raise ValueError("audio normalization produced empty or non-finite samples")
        samples = np.clip(samples, -1.0, 1.0)
        max_chunk_samples = int(cls.PRIMARY_CHUNK_SEC * cls.NORMALIZED_SAMPLE_RATE)
        chunk_count = max(1, int(np.ceil(samples.size / max_chunk_samples)))
        arrays = np.array_split(samples, chunk_count)
        return [
            _AudioChunk(samples=np.asarray(array, dtype=np.float32), label=str(index + 1))
            for index, array in enumerate(arrays)
            if array.size
        ]

    @classmethod
    def _split_chunk(cls, chunk: _AudioChunk) -> list[_AudioChunk]:
        midpoint = chunk.samples.size // 2
        minimum = int(cls.MIN_RETRY_CHUNK_SEC * cls.NORMALIZED_SAMPLE_RATE)
        if midpoint < minimum or chunk.samples.size - midpoint < minimum:
            return []
        return [
            _AudioChunk(samples=chunk.samples[:midpoint], label=f"{chunk.label}a"),
            _AudioChunk(samples=chunk.samples[midpoint:], label=f"{chunk.label}b"),
        ]

    @classmethod
    def _encode_chunk(cls, chunk: _AudioChunk) -> str:
        buffer = BytesIO()
        sf.write(
            buffer,
            chunk.samples,
            cls.NORMALIZED_SAMPLE_RATE,
            format="WAV",
            subtype="PCM_16",
        )
        return base64.b64encode(buffer.getvalue()).decode("ascii")

    @staticmethod
    def _safe_intent(global_intent: str) -> str:
        return " ".join(global_intent.strip().split())[:500] or "not provided"

    def _request_audio_chunk(
        self,
        client: Any,
        chunk: _AudioChunk,
        *,
        global_intent: str,
    ) -> str:
        try:
            audio_completion = client.chat.completions.create(
                model=self.audio_model,
                modalities=["text"],
                max_completion_tokens=500,
                store=False,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    "Describe this audio part for choreography in at most 120 "
                                    "words. Focus only on energy, rhythm, accents, transitions, "
                                    "and movement character. Do not transcribe, quote, or "
                                    "paraphrase lyrics. Do not identify the song or artist, "
                                    "reproduce copyrighted text, or provide timestamps. Describe "
                                    "vocals only as abstract texture. "
                                    f"Global movement intent: {self._safe_intent(global_intent)}."
                                ),
                            },
                            {
                                "type": "input_audio",
                                "input_audio": {
                                    "data": self._encode_chunk(chunk),
                                    "format": "wav",
                                },
                            },
                        ],
                    }
                ],
            )
        except OpenAIError as exc:
            raise _translate_openai_error(
                exc,
                stage="audio understanding",
                model=self.audio_model,
            ) from exc

        choices = getattr(audio_completion, "choices", None) or []
        if not choices:
            raise OpenAIAnalyzeError(
                stage="audio understanding",
                model=self.audio_model,
                status_code=502,
                error_code="empty_audio_output",
                message="The model returned no text choice",
            )
        choice = choices[0]
        message = getattr(choice, "message", None)
        narrative = getattr(message, "content", None)
        if isinstance(narrative, list):
            parts: list[str] = []
            for part in narrative:
                text = part.get("text") if isinstance(part, dict) else getattr(part, "text", None)
                if isinstance(text, str) and text.strip():
                    parts.append(text.strip())
            narrative = "\n".join(parts)
        if not isinstance(narrative, str) or not narrative.strip():
            refusal = bool(getattr(message, "refusal", None))
            finish_reason = getattr(choice, "finish_reason", None)
            raise OpenAIAnalyzeError(
                stage="audio understanding",
                model=self.audio_model,
                status_code=502,
                error_code="empty_audio_output",
                message=(
                    "The model returned no text understanding "
                    f"(finish_reason={finish_reason!r}, refusal={refusal})"
                ),
            )
        return narrative.strip()

    @staticmethod
    def _is_degradable(error: OpenAIAnalyzeError) -> bool:
        return error.status_code in {502, 503, 504}

    @staticmethod
    def _signal_summary(features: LocalAudioFeatures, timing_plan: TimingPlan) -> dict[str, Any]:
        rms_times = np.asarray(features.rms.times_sec, dtype=np.float64)
        rms_values = np.asarray(features.rms.values, dtype=np.float64)
        windows: list[dict[str, float]] = []
        for start in np.arange(0.0, features.duration_sec, 4.0):
            end = min(float(start + 4.0), features.duration_sec)
            mask = (rms_times >= start) & (rms_times < end)
            windows.append(
                {
                    "start_sec": float(start),
                    "end_sec": end,
                    "mean_rms": float(rms_values[mask].mean()) if mask.any() else 0.0,
                }
            )
        rms_by_start = {round(window["start_sec"], 6): window for window in windows}
        slot_candidates: list[dict[str, Any]] = []
        for slot in timing_plan.slots:
            key = round(slot.start_sec, 6)
            window = rms_by_start.get(key)
            if window is None:
                mask = (rms_times >= slot.start_sec) & (
                    rms_times < slot.start_sec + slot.duration_sec
                )
                mean_rms = float(rms_values[mask].mean()) if mask.any() else 0.0
            else:
                mean_rms = window["mean_rms"]
            slot_candidates.append(
                {
                    "window_id": slot.slot_id,
                    "start_sec": slot.start_sec,
                    "end_sec": slot.start_sec + slot.duration_sec,
                    "mean_rms": mean_rms,
                }
            )
        return {
            "duration_sec": features.duration_sec,
            "tempo_bpm": features.tempo_bpm,
            "beat_times_sec": features.beat_times_sec[:256],
            "chroma_mean": features.chroma_mean,
            "section_boundaries_sec": features.section_boundaries_sec,
            "four_second_rms_windows": windows,
            "local_segments": [
                {
                    "segment_id": segment.segment_id,
                    "start_sec": segment.start_sec,
                    "end_sec": segment.end_sec,
                }
                for segment in timing_plan.segments
            ],
            "local_slot_candidates": slot_candidates,
        }

    @staticmethod
    def _assemble_result(
        local_plan: TimingPlan,
        labels: SemanticChoreographyLabels,
    ) -> AnalyzeResult:
        """Combine model semantics with immutable locally generated times."""

        expected_segments = {segment.segment_id for segment in local_plan.segments}
        segment_labels = {label.segment_id: label for label in labels.segments}
        if len(segment_labels) != len(labels.segments):
            raise ValueError("structured model returned duplicate segment IDs")
        if set(segment_labels) != expected_segments:
            raise ValueError(
                "structured model must label every locally generated segment exactly once"
            )

        local_slots = {slot.slot_id: slot for slot in local_plan.slots}
        slot_labels = {label.window_id: label for label in labels.slots}
        if len(slot_labels) != len(labels.slots):
            raise ValueError("structured model returned duplicate slot candidate IDs")
        if set(slot_labels) != set(local_slots):
            raise ValueError("structured model must label every local slot candidate exactly once")

        segments = [
            AnalysisSegment(
                segment_id=segment.segment_id,
                start_sec=segment.start_sec,
                end_sec=segment.end_sec,
                description=segment_labels[segment.segment_id].description,
                energy=segment_labels[segment.segment_id].energy,
                confidence=segment_labels[segment.segment_id].confidence,
            )
            for segment in local_plan.segments
        ]
        slots = [
            SlotSuggestion(
                slot_id=slot.slot_id,
                start_sec=slot.start_sec,
                duration_sec=slot.duration_sec,
                music_description=slot_labels[slot.slot_id].music_description,
                genre=slot_labels[slot.slot_id].genre,
                cue=slot_labels[slot.slot_id].cue,
                reason=slot_labels[slot.slot_id].reason,
                confidence=slot_labels[slot.slot_id].confidence,
            )
            for slot in local_plan.slots
        ]
        return AnalyzeResult(
            duration_sec=local_plan.duration_sec,
            summary=labels.summary,
            genre=labels.genre,
            segments=segments,
            slots=slots,
        )

    def _parse(self, client: Any, schema: type[BaseModel], messages: list[dict], stage: str):
        """Reject provider failures, refusals and invalid results; never fabricate semantics."""
        try:
            response = client.responses.parse(
                model=self.structured_model,
                reasoning={"effort": "none"}, text={"verbosity": "low"},
                max_output_tokens=2400, store=False, input=messages, text_format=schema,
            )
        except OpenAIError as exc:
            raise _translate_openai_error(
                exc, stage=stage, model=self.structured_model,
            ) from exc
        parsed = response.output_parsed
        try:
            if parsed is None:
                raise ValueError("API returned no structured result (possibly refused or incomplete)")
            return parsed if isinstance(parsed, schema) else schema.model_validate(parsed)
        except ValueError as exc:
            raise OpenAIAnalyzeError(
                stage=stage, model=self.structured_model, status_code=502,
                error_code="invalid_structured_output",
                message="API returned no usable validated result; retry the request",
            ) from exc

    def resolve_intent(self, query: str, global_intent: str) -> RetrievalIntent:
        return self._parse(self._get_client(), RetrievalIntent, [
            {"role": "system", "content": (
                "Interpret dance intent for a trained music/text retriever. Select one supported "
                "genre, a speed (slow/medium/fast) and energy (calm/energetic). "
                "Return a concise English motion query preserving the user's request. "
                "The local request takes precedence over global intent. "
                "Use Choreography only for genuinely unspecified/general choreography. "
                "Supported genres: " + ", ".join(GENRE_NAMES)
            )},
            {"role": "user", "content": (
                f"Global intent: {global_intent}\nLocal intent: {query}"
            )},
        ], "intent interpretation")

    def _understand_chunk(self, client: Any, chunk: _AudioChunk, *, global_intent: str) -> str:
        try:
            return self._request_audio_chunk(client, chunk, global_intent=global_intent)
        except OpenAIAnalyzeError as error:
            if not self._is_degradable(error):
                raise
            parts = self._split_chunk(chunk)
            if not parts:
                raise
            # Recover only if every retried part succeeds through the external API.
            return "\n".join(
                self._request_audio_chunk(client, part, global_intent=global_intent)
                for part in parts
            )

    def analyze(
        self, audio_path: Path, local_features: LocalAudioFeatures, *,
        global_intent: str, consent_to_external_api: bool,
    ) -> OpenAIAnalyzeOutcome:
        started = perf_counter()
        path = self._validate_audio(audio_path, local_features, consent=consent_to_external_api)
        plan = build_timing_plan(local_features)
        client = self._get_client()
        chunks = self._normalized_chunks(path)
        with ThreadPoolExecutor(max_workers=min(2, len(chunks))) as pool:
            futures = [
                pool.submit(self._understand_chunk, client, chunk, global_intent=global_intent)
                for chunk in chunks
            ]
            narratives = [future.result() for future in futures]
        labels = self._parse(client, SemanticChoreographyLabels, [
            {"role": "system", "content": (
                "Label every segment_id and window_id in the supplied timing plan exactly once. "
                "Do not invent, remove or alter timestamps. For each window, music_description "
                "describes music only; cue recommends a concrete dance movement. "
                "Return a supported global dance genre and a supported genre for each window, "
                "respecting the user's intent. Use Choreography for genuinely unspecified/general "
                "choreography. Supported genres: " + ", ".join(GENRE_NAMES)
            )},
            {"role": "user", "content": (
                f"Global intent: {global_intent}\n"
                + "Timing/features: " + json.dumps(self._signal_summary(local_features, plan))
                + "\nAudio understanding: " + "\n".join(narratives)
            )},
        ], "structured planning")
        try:
            result = self._assemble_result(plan, labels)
        except ValueError as exc:
            raise OpenAIAnalyzeError(
                stage="structured planning", model=self.structured_model,
                status_code=502, error_code="invalid_timing_ids",
                message="API labels do not match the timing plan; retry the request",
            ) from exc
        return OpenAIAnalyzeOutcome(
            result=result, warnings=(), semantic_status="full",
            audio_semantics_used=True, structured_semantics_used=True,
            normalized_audio_chunks=len(chunks),
            timings_ms={"total": (perf_counter() - started) * 1000.0},
        )
