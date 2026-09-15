"""Loopback API-only choreography application with SMPL preview and PKL export."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from time import perf_counter
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from app.api.models import (
    AnalyzeRequest,
    AnalyzeResponse,
    CompleteRequest,
    FillRequest,
    MotionPayload,
    RemakeRequest,
    RetrievalRequest,
    SessionStateResponse,
    SmoothRequest,
    UploadResponse,
)
from app.config import PROJECT_ROOT, Settings
from app.schemas.diagnostics import MotionDiagnostics
from app.schemas.timeline import RetrievalItem, RetrievalResponse
from app.utils.assets import validation_report
from app.utils.audio_analysis import (
    SUPPORTED_AUDIO_SUFFIXES,
    analyze_audio_local,
)
from app.utils.diagnoser import diagnose_motion
from app.utils.exports import ExportService
from app.utils.inpainting import (
    COMPLETER_MAX_FRAMES,
    COMPLETER_MAX_SECONDS,
    clone_canonical_motion,
)
from app.config import genre_name, normalize_genre
from app.utils.motion_smoothing import smooth_motion_range
from app.utils.retriever import BACKEND_NAME, RetrievalQueryError
from app.utils.openai_analyze import OpenAIAnalyzeError, OpenAIAnalyzeService
from app.runtime import RuntimeServices
from app.sessions import ProjectSession, SessionStore
from app.utils.smpl_template import NeutralSmplTemplateService
from app.timeline import TimelineEditor

FRONTEND_ROOT = PROJECT_ROOT / "app" / "frontend"
ALLOWED_UPLOAD_CONTENT_TYPES = {
    "audio/wav",
    "audio/x-wav",
    "audio/mpeg",
    "audio/mp4",
    "audio/x-m4a",
    "audio/flac",
    "audio/ogg",
    "application/octet-stream",
}


def _http_session(store: SessionStore, session_id: str) -> ProjectSession:
    try:
        return store.get(session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


def _state_response(session: ProjectSession) -> SessionStateResponse:
    editor = session.editor
    return SessionStateResponse(
        session_id=session.session_id,
        original_filename=session.original_filename,
        audio_url=f"/api/sessions/{session.session_id}/audio",
        analysis=session.analysis,
        timeline=editor.state if editor else None,
        motion_frames=editor.motion.frames if editor else None,
        filled_frame_count=int(editor.filled_mask.sum()) if editor else 0,
        assignments=editor.assignments if editor else [],
    )


def _export_metadata(session: ProjectSession) -> dict:
    editor = session.editor
    analysis = session.analysis
    assignments = editor.assignments if editor else []
    return {
        "music_metadata": {
            "original_filename": session.original_filename,
            "size_bytes": session.audio_size_bytes,
            "duration_sec": analysis.duration_sec if analysis else None,
            "global_intent": session.global_intent,
            "analysis_schema_version": analysis.schema_version if analysis else None,
            "analysis_summary": analysis.summary if analysis else None,
        },
        "slot_assignments": assignments,
        "retrieval_metadata": [
            {
                "slot_id": assignment.get("slot_id"),
                "source_clip_id": assignment.get("source_clip_id"),
                "retrieval_score": assignment.get("retrieval_score"),
            }
            for assignment in assignments
        ],
    }


def _resolve_generation_genre(session: ProjectSession) -> tuple[int, str]:
    if session.analysis is None:
        raise ValueError("Analyze intent through the API before generation")
    value = normalize_genre(session.analysis.genre)
    if value is None:
        raise ValueError("API analysis does not contain a valid model genre")
    return value, "api-analysis"


def create_app(
    settings: Settings | None = None,
    *,
    session_root: Path | None = None,
    runtime: RuntimeServices | None = None,
    smpl_template_service: NeutralSmplTemplateService | None = None,
    analysis_service: OpenAIAnalyzeService | None = None,
) -> FastAPI:
    settings = settings or Settings()
    store = SessionStore(session_root or PROJECT_ROOT / "outputs" / "sessions")
    runtime = runtime or RuntimeServices(settings)
    app = FastAPI(
        title="CustomDance",
        version="0.1.0",
        docs_url="/api/docs",
        redoc_url=None,
    )
    intent_service = analysis_service or OpenAIAnalyzeService(
        api_key=settings.openai_api_key,
        audio_model=settings.openai_audio_model,
        structured_model=settings.openai_structured_model,
        max_audio_mb=settings.openai_max_audio_mb,
        max_duration_sec=settings.openai_max_duration_sec,
    )
    app.state.settings = settings
    app.state.sessions = store
    app.state.runtime = runtime
    app.state.smpl_template = smpl_template_service or NeutralSmplTemplateService(
        settings.asset_paths()["smpl_model_root"]
    )
    # Validate configured paths whenever the application is created. Missing
    # assets remain explicit and inspectable while the local UI can still start.
    app.state.asset_validation = validation_report(settings=settings)
    app.mount("/static", StaticFiles(directory=FRONTEND_ROOT), name="static")

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(FRONTEND_ROOT / "index.html")

    @app.get("/health", include_in_schema=False)
    @app.get("/api/health")
    async def health() -> dict:
        retrieval_status: dict = {"status": "unavailable"}
        library_status: dict = {"status": "unavailable"}
        try:
            retrieval_status = await run_in_threadpool(
                runtime.retriever().status
            )
            retrieval_status = {"status": "READY", **retrieval_status}
        except (FileNotFoundError, ImportError, RuntimeError, TypeError, ValueError) as exc:
            retrieval_status = {"status": "BLOCKED", "detail": str(exc)}
        try:
            library_status = await run_in_threadpool(runtime.motion_library().status)
        except (FileNotFoundError, KeyError, TypeError, ValueError) as exc:
            library_status = {"status": "BLOCKED", "detail": str(exc)}
        return {
            "status": "ok",
            "loopback_only": True,
            "retrieval_backend": BACKEND_NAME,
            "retrieval": retrieval_status,
            "motion_library": library_status,
            "completer_backend": "real-checkpoint-lazy",
            "remaker_backend": "shared-with-completer",
            "diagnoser_backend": "ake-rke-anomaly-signals",
            "completer_max_frames": COMPLETER_MAX_FRAMES,
            "completer_max_seconds": COMPLETER_MAX_SECONDS,
            "asset_ready": app.state.asset_validation["ready"],
            "asset_validation": app.state.asset_validation["assets"],
        }

    @app.get("/api/models/smpl/status")
    async def smpl_status() -> dict:
        return app.state.smpl_template.status()

    @app.get("/api/models/smpl/template", response_class=Response)
    async def smpl_template() -> Response:
        try:
            payload = await run_in_threadpool(app.state.smpl_template.template_json)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (ImportError, RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=503, detail=f"SMPL preview unavailable: {exc}") from exc
        return Response(
            content=payload,
            media_type="application/json",
            headers={"Cache-Control": "private, no-store"},
        )

    @app.post("/api/sessions", response_model=UploadResponse)
    async def create_session(file: Annotated[UploadFile, File()]) -> UploadResponse:
        original = Path(file.filename or "audio").name
        suffix = Path(original).suffix.lower()
        if suffix not in SUPPORTED_AUDIO_SUFFIXES:
            raise HTTPException(status_code=415, detail=f"unsupported audio suffix: {suffix}")
        if file.content_type not in ALLOWED_UPLOAD_CONTENT_TYPES:
            raise HTTPException(status_code=415, detail="unsupported audio content type")
        session = store.create(original, suffix)
        max_bytes = settings.customdance_max_upload_mb * 1024 * 1024
        size = 0
        try:
            with session.audio_path.open("xb") as handle:
                while chunk := await file.read(1024 * 1024):
                    size += len(chunk)
                    if size > max_bytes:
                        raise HTTPException(
                            status_code=413,
                            detail=f"upload exceeds {settings.customdance_max_upload_mb} MB",
                        )
                    handle.write(chunk)
        except Exception:
            session.audio_path.unlink(missing_ok=True)
            raise
        finally:
            await file.close()
        if size == 0:
            session.audio_path.unlink(missing_ok=True)
            raise HTTPException(status_code=400, detail="uploaded audio is empty")
        session.audio_size_bytes = size
        return UploadResponse(
            session_id=session.session_id,
            original_filename=original,
            audio_url=f"/api/sessions/{session.session_id}/audio",
            size_bytes=size,
        )

    @app.get("/api/sessions/{session_id}/audio", include_in_schema=False)
    async def session_audio(session_id: str) -> FileResponse:
        session = _http_session(store, session_id)
        if not session.audio_path.is_file():
            raise HTTPException(status_code=404, detail="session audio is missing")
        return FileResponse(session.audio_path)

    @app.get("/api/sessions/{session_id}", response_model=SessionStateResponse)
    async def session_state(session_id: str) -> SessionStateResponse:
        return _state_response(_http_session(store, session_id))

    @app.post("/api/sessions/{session_id}/analyze", response_model=AnalyzeResponse)
    async def analyze(session_id: str, request: AnalyzeRequest) -> AnalyzeResponse:
        session = _http_session(store, session_id)
        with session.lock:
            if session.analysis is not None and not request.confirm_reset:
                raise HTTPException(
                    status_code=409,
                    detail="re-analysis resets all slots and fills; confirm_reset is required",
                )
        analyze_started = perf_counter()
        timings_ms: dict[str, float] = {}
        try:
            local_audio_started = perf_counter()
            features = await run_in_threadpool(analyze_audio_local, session.audio_path)
            timings_ms["local_audio_features"] = (
                perf_counter() - local_audio_started
            ) * 1000.0
            outcome = await run_in_threadpool(
                intent_service.analyze, session.audio_path, features,
                global_intent=request.global_intent,
                consent_to_external_api=request.consent_to_external_api,
            )
            result = outcome.result
            backend = outcome.backend
            external_used = True
            semantic_status = outcome.semantic_status
            warnings = list(outcome.warnings)
            timings_ms.update({f"openai_{key}": value for key, value in outcome.timings_ms.items()})
        except OpenAIAnalyzeError as exc:
            raise HTTPException(status_code=exc.status_code, detail=exc.public_detail) from exc
        except (ValueError, PermissionError, RuntimeError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        timings_ms["total"] = (perf_counter() - analyze_started) * 1000.0
        with session.lock:
            session.reset_from_analysis(result, features, global_intent=request.global_intent)
            assert session.editor is not None
            return AnalyzeResponse(
                backend=backend,
                external_api_used=external_used,
                tempo_bpm=features.tempo_bpm,
                sample_rate=features.sample_rate,
                semantic_status=semantic_status,
                warnings=warnings,
                timings_ms=timings_ms,
                analysis=result,
                timeline=session.editor.state,
            )

    @app.post("/api/sessions/{session_id}/slots/{slot_id}/focus")
    async def focus_slot(session_id: str, slot_id: str) -> dict:
        session = _http_session(store, session_id)
        with session.lock:
            if session.editor is None:
                raise HTTPException(status_code=409, detail="Analyze audio before selecting slots")
            try:
                state = session.editor.focus(slot_id)
            except (KeyError, ValueError) as exc:
                raise HTTPException(status_code=409, detail=str(exc)) from exc
            session.retrieval_results.clear()
            session.retrieval_candidates.clear()
            return state.model_dump(mode="json")

    @app.post("/api/sessions/{session_id}/retrieve", response_model=RetrievalResponse)
    async def retrieve(session_id: str, request: RetrievalRequest) -> RetrievalResponse:
        session = _http_session(store, session_id)
        with session.lock:
            if session.editor is None:
                raise HTTPException(status_code=409, detail="Analyze audio before retrieval")
            focused_id = session.editor.state.current_focused_slot_id
            if focused_id is None:
                raise HTTPException(
                    status_code=409,
                    detail="Select and accept a four-second slot before retrieval",
                )
            focused_slot = next(
                slot for slot in session.editor.state.slots if slot.slot_id == focused_id
            )
            global_intent = session.global_intent.strip()

        try:
            intent = await run_in_threadpool(
                intent_service.resolve_intent, request.query, global_intent,
            )
        except OpenAIAnalyzeError as exc:
            raise HTTPException(status_code=exc.status_code, detail=exc.public_detail) from exc
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        explicit_genre = intent.genre
        model_text = intent.query

        def search():
            backend = runtime.retriever()
            return backend.retrieve_candidates(
                session.audio_path,
                model_text,
                top_k=request.top_k,
                explicit_genre=explicit_genre,
                speed=intent.speed,
                energy=intent.energy,
                start_sec=focused_slot.start_sec,
            )

        try:
            result = await run_in_threadpool(search)
        except RetrievalQueryError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except (FileNotFoundError, ImportError, RuntimeError, TypeError, ValueError) as exc:
            raise HTTPException(
                status_code=503, detail=f"music+text retrieval failed: {exc}"
            ) from exc
        items = [
            RetrievalItem(
                clip_id=candidate.clip_id,
                score=candidate.score,
                genre=candidate.genre,
                genre_name=genre_name(normalize_genre(candidate.genre)),
                duration=4.0,
                preview_metadata=candidate.public_metadata(),
            )
            for candidate in result.results
        ]
        with session.lock:
            if session.editor is None or session.editor.state.current_focused_slot_id != focused_id:
                raise HTTPException(
                    status_code=409,
                    detail="focused slot changed while retrieval was running; retry",
                )
            session.retrieval_results = {item.clip_id: item.score for item in items}
            session.retrieval_candidates = {
                candidate.phrase_id: candidate for candidate in result.results
            }
        resolved_genre = normalize_genre(result.query.get("genre"))
        return RetrievalResponse(
            query=request.query.strip(),
            model_query=result.query,
            top_k=request.top_k,
            backend=BACKEND_NAME,
            genre_mode="api",
            genre_filter=resolved_genre,
            genre_name=genre_name(resolved_genre),
            candidate_count=result.candidate_count,
            warning=result.warning,
            items=items,
        )

    @app.get(
        "/api/sessions/{session_id}/motions/{clip_id}/preview",
        response_model=MotionPayload,
    )
    async def preview_motion(session_id: str, clip_id: str) -> MotionPayload:
        session = _http_session(store, session_id)
        with session.lock:
            candidate = session.retrieval_candidates.get(clip_id)
        if candidate is None:
            raise HTTPException(
                status_code=404,
                detail="unknown or stale retrieval candidate; run retrieval again",
            )
        try:
            motion = await run_in_threadpool(
                runtime.motion_library().load_canonical, candidate.phrase_id
            )
        except (KeyError, FileNotFoundError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return MotionPayload(
            source_clip_id=motion.source_clip_id,
            fps=motion.fps,
            frames=motion.frames,
            duration_sec=motion.duration_sec,
            coordinate_system=motion.coordinate_system.value,
            smpl_poses=motion.smpl_poses.tolist(),
            smpl_trans=motion.smpl_trans.tolist(),
            joint_positions=motion.joint_positions.tolist(),
        )

    @app.get("/api/sessions/{session_id}/motion/current", response_model=MotionPayload)
    async def current_motion(session_id: str) -> MotionPayload:
        session = _http_session(store, session_id)
        if session.editor is None:
            raise HTTPException(status_code=409, detail="Analyze audio before loading motion")
        motion = session.editor.motion
        return MotionPayload(
            source_clip_id=motion.source_clip_id,
            fps=motion.fps,
            frames=motion.frames,
            duration_sec=motion.duration_sec,
            coordinate_system=motion.coordinate_system.value,
            smpl_poses=motion.smpl_poses.tolist(),
            smpl_trans=motion.smpl_trans.tolist(),
            joint_positions=motion.joint_positions.tolist(),
        )

    @app.post("/api/sessions/{session_id}/fill")
    async def fill_in(session_id: str, request: FillRequest) -> dict:
        session = _http_session(store, session_id)
        with session.lock:
            if session.editor is None:
                raise HTTPException(status_code=409, detail="Analyze audio before Fill In")
            if request.clip_id not in session.retrieval_results:
                raise HTTPException(
                    status_code=409,
                    detail="Fill In requires a clip from the latest real retrieval results",
                )
            score = session.retrieval_results[request.clip_id]
            candidate = session.retrieval_candidates.get(request.clip_id)
            if candidate is None:
                raise HTTPException(
                    status_code=409,
                    detail="retrieval candidate metadata is missing or stale; run retrieval again",
                )
        try:
            clip = await run_in_threadpool(
                runtime.motion_library().load_canonical, candidate.phrase_id
            )
            with session.lock:
                session.editor.fill(clip, retrieval_score=score)
                session.motion_revision += 1
                return session.editor.snapshot()
        except (KeyError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/api/sessions/{session_id}/complete")
    async def complete_motion(session_id: str, request: CompleteRequest) -> dict:
        session = _http_session(store, session_id)

        def generate() -> dict:
            complete_started = perf_counter()
            backend = runtime.completer()
            with session.lock:
                if session.editor is None:
                    raise ValueError("Analyze audio before Complete")
                revision = session.motion_revision
                editor_snapshot = TimelineEditor(
                    session.editor.state.model_copy(deep=True),
                    clone_canonical_motion(session.editor.motion),
                    session.editor.filled_mask.copy(),
                    deepcopy(session.editor.assignments),
                )
                source_clip_ids = sorted(
                    {
                        str(assignment["source_clip_id"])
                        for assignment in editor_snapshot.assignments
                    }
                )
                genre, genre_source = _resolve_generation_genre(session)
                features = session.completer_audio_features
                features = None if features is None else features.copy()

            source_load_started = perf_counter()
            source_clips = {
                phrase_id: runtime.motion_library().load_canonical(phrase_id)
                for phrase_id in source_clip_ids
            }
            source_load_ms = (perf_counter() - source_load_started) * 1000.0
            preparation_started = perf_counter()
            preparation = editor_snapshot.prepare_complete(source_clips)
            preparation_ms = (perf_counter() - preparation_started) * 1000.0
            current = clone_canonical_motion(preparation.editor.motion)
            effective_known_mask = preparation.known_mask.copy()

            audio_features_started = perf_counter()
            if features is None:
                features = backend.audio_features(session.audio_path, current.frames)
            audio_features_ms = (perf_counter() - audio_features_started) * 1000.0
            completer_started = perf_counter()
            result = backend.complete(
                current,
                effective_known_mask,
                features,
                genre=genre,
                seed=request.seed,
                steps=request.steps,
            )
            completer_ms = (perf_counter() - completer_started) * 1000.0
            inference = result.metadata["completer_inference"][-1]
            inference["genre_resolution"] = genre_source
            inference["pre_complete_continuity"] = deepcopy(
                preparation.diagnostics
            )
            timings = {
                "source_clip_load": round(source_load_ms, 3),
                "pre_complete_continuity": round(preparation_ms, 3),
                "completer_audio_features": round(audio_features_ms, 3),
                "completer_generation": round(completer_ms, 3),
                "total_before_commit": round(
                    (perf_counter() - complete_started) * 1000.0, 3
                ),
            }
            inference["api_stage_timings_ms"] = timings
            with session.lock:
                if session.motion_revision != revision:
                    raise RuntimeError(
                        "timeline changed while Completer was running; retry Complete"
                    )
                preparation.editor.motion = result
                session.editor = preparation.editor
                session.completer_audio_features = features
                session.motion_revision += 1
                response = session.editor.snapshot()
                response["inference"] = inference
                response["pre_complete"] = preparation.diagnostics
                response["timings_ms"] = timings
                return response

        try:
            return await run_in_threadpool(generate)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except (FileNotFoundError, ImportError, KeyError, RuntimeError) as exc:
            raise HTTPException(
                status_code=503, detail=f"real Completer Complete failed: {exc}"
            ) from exc

    @app.post("/api/sessions/{session_id}/remake")
    async def remake_motion(session_id: str, request: RemakeRequest) -> dict:
        session = _http_session(store, session_id)

        def generate() -> dict:
            backend = runtime.remaker()
            with session.lock:
                if session.editor is None:
                    raise ValueError("Analyze audio before Remake")
                revision = session.motion_revision
                current = clone_canonical_motion(session.editor.motion)
                genre, genre_source = _resolve_generation_genre(session)
                features = session.completer_audio_features
                features = None if features is None else features.copy()
            if features is None:
                features = backend.audio_features(session.audio_path, current.frames)
            result = backend.remake(
                current,
                request.start_frame,
                request.end_frame,
                features,
                genre=genre,
                seed=request.seed,
                steps=request.steps,
            )
            result.metadata["completer_inference"][-1]["genre_resolution"] = genre_source
            with session.lock:
                if session.motion_revision != revision:
                    raise RuntimeError(
                        "timeline changed while Remaker was running; retry Remake"
                    )
                assert session.editor is not None
                session.editor.motion = result
                session.completer_audio_features = features
                session.motion_revision += 1
                response = session.editor.snapshot()
                response["inference"] = result.metadata["completer_inference"][-1]
                return response

        try:
            return await run_in_threadpool(generate)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except (FileNotFoundError, ImportError, RuntimeError) as exc:
            raise HTTPException(
                status_code=503, detail=f"real Remaker failed: {exc}"
            ) from exc

    @app.post("/api/sessions/{session_id}/smooth")
    async def smooth_motion(session_id: str, request: SmoothRequest) -> dict:
        session = _http_session(store, session_id)

        def smooth() -> dict:
            with session.lock:
                if session.editor is None:
                    raise ValueError("Analyze audio before Smooth")
                revision = session.motion_revision
                current = clone_canonical_motion(session.editor.motion)
            result = smooth_motion_range(
                current,
                request.start_frame,
                request.end_frame,
                smooth_translation=request.smooth_translation,
                joint_groups=request.joint_groups,
            )
            with session.lock:
                if session.motion_revision != revision:
                    raise RuntimeError("timeline changed while smoothing was running; retry Smooth")
                assert session.editor is not None
                session.editor.motion = result
                session.motion_revision += 1
                response = session.editor.snapshot()
                response["smoothing"] = result.metadata["motion_smoothing"][-1]
                return response

        try:
            return await run_in_threadpool(smooth)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/api/sessions/{session_id}/diagnose", response_model=MotionDiagnostics)
    async def diagnose(session_id: str) -> MotionDiagnostics:
        session = _http_session(store, session_id)
        with session.lock:
            if session.editor is None:
                raise HTTPException(status_code=409, detail="Analyze audio before Diagnose")
            motion = clone_canonical_motion(session.editor.motion)
        try:
            return await run_in_threadpool(diagnose_motion, motion)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/api/sessions/{session_id}/exports/pkl", response_class=FileResponse)
    async def export_pkl(session_id: str) -> FileResponse:
        session = _http_session(store, session_id)
        with session.lock:
            if session.editor is None:
                raise HTTPException(status_code=409, detail="Analyze audio before export")
            motion = clone_canonical_motion(session.editor.motion)
            metadata = _export_metadata(session)
            revision = session.motion_revision
        service = ExportService(session.root)
        try:
            artifact = await run_in_threadpool(
                service.export_pkl, motion, metadata, revision=revision
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except (OSError, RuntimeError) as exc:
            raise HTTPException(status_code=500, detail=f"PKL export failed: {exc}") from exc
        return FileResponse(
            artifact.path,
            media_type="application/octet-stream",
            filename="customdance-motion.pkl",
            headers={
                "X-CustomDance-Frames": str(artifact.validation["frames"]),
                "X-CustomDance-Schema": artifact.validation["schema_version"],
            },
        )

    return app
