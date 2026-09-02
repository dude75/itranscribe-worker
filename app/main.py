"""HTTP API задач (ТЗ §6). Uvicorn — один процесс."""

from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from typing import Annotated

from fastapi import Depends, FastAPI, Form, HTTPException, Query, Request, UploadFile, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response

from app.audio import PayloadTooLarge, write_upload_limited
from app.auth import api_token_is_valid, require_api_token
from app.config import get_settings
from app.engines.cache import get_cache
from app.logging_setup import setup_logging
from app.prometheus_metrics import (
    CONTENT_TYPE,
    create_metrics,
    http_path_template,
    observe_http,
    observe_queue_rejected,
    render,
    set_active,
)
from app.queueing import QueueFullError, TaskRunner, TaskRunningError
from app.schemas import (
    AsrModel,
    ErrorCode,
    ErrorDetail,
    HealthResponse,
    OptionalDiarizationModel,
    PurgeResult,
    TaskListItem,
    TaskMeta,
    TaskResponse,
    TaskStatus,
    TranscriptLine,
    error_payload,
)
from app.tasks import TaskRecord
from app.version import read_version

ALLOWED_SUFFIXES = {".wav", ".mp3", ".m4a"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    setup_logging(settings)
    logging.getLogger("app").info("service start")
    metrics = create_metrics(settings)
    set_active(metrics)
    app.state.metrics = metrics
    cache = get_cache()
    import app.pipeline as pipeline
    from app.engines.stubs import StubASR, StubDiarization

    if os.environ.get("ITRANSCRIBE_STUBS", "").lower() in {"1", "true", "yes"}:
        cache.reset()
        pipeline.resolve_asr = lambda _model: StubASR()
        pipeline.resolve_diarization = lambda _model: StubDiarization()
    else:
        cache.preload(settings)
        pipeline.resolve_asr = cache.resolve_asr
        pipeline.resolve_diarization = cache.resolve_diarization
    runner = TaskRunner(settings)
    await runner.start()
    metrics.bind(settings=settings, cache=cache, runner=runner)
    app.state.runner = runner
    app.state.engines = cache
    try:
        yield
    finally:
        await runner.stop()
        set_active(None)


app = FastAPI(title="itranscribe-worcker", version=read_version(), lifespan=lifespan)


def _api_error(status_code: int, code: ErrorCode) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content=error_payload(code),
        headers={"Connection": "close"},
    )


def _http_error(status_code: int, code: ErrorCode) -> HTTPException:
    return HTTPException(status_code=status_code, detail=error_payload(code))


def _transcribe_path(request: Request) -> bool:
    path = request.scope.get("path") or request.url.path
    return str(path).rstrip("/") == "/transcribe"


@app.middleware("http")
async def gate_transcribe_upload(request: Request, call_next):
    """Auth, Content-Length и queue_full до разбора multipart на POST /transcribe."""
    if request.method != "POST" or not _transcribe_path(request):
        return await call_next(request)
    if not api_token_is_valid(request.headers.get("authorization")):
        return _api_error(status.HTTP_401_UNAUTHORIZED, ErrorCode.unauthorized)
    settings = get_settings()
    raw_cl = request.headers.get("content-length")
    if raw_cl is not None:
        try:
            size = int(raw_cl)
        except ValueError:
            size = -1
        if size < 0 or size > settings.MAX_UPLOAD_BYTES:
            return _api_error(status.HTTP_413_CONTENT_TOO_LARGE, ErrorCode.payload_too_large)
    runner = getattr(request.app.state, "runner", None)
    if runner is not None and runner.store.count_queued() >= settings.WORKER_QUEUE_SIZE:
        observe_queue_rejected()
        return _api_error(status.HTTP_503_SERVICE_UNAVAILABLE, ErrorCode.queue_full)
    return await call_next(request)


@app.middleware("http")
async def prometheus_http_middleware(request: Request, call_next):
    started = time.perf_counter()
    path = http_path_template(request)
    status_code = 500
    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    finally:
        observe_http(request.method, path, status_code, time.perf_counter() - started)


def get_runner() -> TaskRunner:
    return app.state.runner


@app.exception_handler(HTTPException)
async def http_exception_handler(_request, exc: HTTPException) -> JSONResponse:
    if isinstance(exc.detail, dict) and "error" in exc.detail:
        return JSONResponse(status_code=exc.status_code, content=exc.detail)
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(_request, exc: RequestValidationError) -> JSONResponse:
    return JSONResponse(status_code=422, content={"detail": exc.errors()})


def _record_to_response(record: TaskRecord) -> TaskResponse:
    transcript = None
    if record.transcript is not None:
        transcript = [TranscriptLine.model_validate(item) for item in record.transcript]
    error = ErrorDetail.model_validate(record.error) if record.error is not None else None
    return TaskResponse(
        status=record.status,
        meta=TaskMeta(
            timestamp=record.timestamp,
            task_id=record.task_id,
            asr_model=record.asr_model,
            diarization_model=record.diarization_model,
            asr_checkpoint=record.asr_checkpoint,
            diarization_checkpoint=record.diarization_checkpoint,
            audio_duration_sec=record.audio_duration_sec,
            asr_time_sec=record.asr_time_sec,
            diarization_time_sec=record.diarization_time_sec,
            alignment_time_sec=record.alignment_time_sec,
            total_time_sec=record.total_time_sec,
            rtf=record.rtf,
        ),
        transcript=transcript,
        error=error,
    )


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    cache = get_cache()
    return HealthResponse(
        status="ok",
        version=read_version(),
        engines=cache.status,
        device=cache.device,
    )


@app.get("/metrics")
def metrics() -> Response:
    return Response(content=render(), media_type=CONTENT_TYPE)


@app.post("/transcribe", status_code=status.HTTP_202_ACCEPTED, response_model=TaskResponse)
async def transcribe(
    file: UploadFile,
    asr_model: AsrModel = Form(AsrModel.whisper),
    diarization_model: Annotated[OptionalDiarizationModel, Form()] = None,
    _: str = Depends(require_api_token),
    runner: TaskRunner = Depends(get_runner),
) -> TaskResponse:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise _http_error(status.HTTP_400_BAD_REQUEST, ErrorCode.invalid_file)
    settings = get_settings()
    scratch_dir = Path(settings.DATA_DIR)
    scratch_dir.mkdir(parents=True, exist_ok=True)
    scratch = scratch_dir / f".upload_{uuid.uuid4()}{suffix}"
    try:
        await asyncio.to_thread(
            write_upload_limited, file.file, scratch, settings.MAX_UPLOAD_BYTES
        )
    except PayloadTooLarge as exc:
        raise _http_error(status.HTTP_413_CONTENT_TOO_LARGE, exc.code) from None
    try:
        record = await runner.submit(scratch, asr_model, diarization_model)
    except QueueFullError as exc:
        observe_queue_rejected()
        raise _http_error(status.HTTP_503_SERVICE_UNAVAILABLE, exc.code) from None
    finally:
        scratch.unlink(missing_ok=True)
    return _record_to_response(record)


@app.delete("/tasks", response_model=PurgeResult)
async def purge_tasks(
    _: str = Depends(require_api_token),
    runner: TaskRunner = Depends(get_runner),
) -> PurgeResult:
    return await runner.purge()


@app.get("/tasks", response_model=list[TaskListItem])
def list_tasks(
    status_filter: TaskStatus | None = Query(default=None, alias="status"),
    _: str = Depends(require_api_token),
    runner: TaskRunner = Depends(get_runner),
) -> list[TaskListItem]:
    records = runner.store.list_tasks(status_filter)
    return [
        TaskListItem(
            task_id=item.task_id,
            status=item.status,
            timestamp=item.timestamp,
            asr_model=item.asr_model,
            diarization_model=item.diarization_model,
        )
        for item in records
    ]


@app.get("/tasks/{task_id}", response_model=TaskResponse)
def get_task(
    task_id: str,
    _: str = Depends(require_api_token),
    runner: TaskRunner = Depends(get_runner),
) -> TaskResponse:
    record = runner.store.get(task_id)
    if record is None:
        raise _http_error(status.HTTP_404_NOT_FOUND, ErrorCode.not_found)
    return _record_to_response(record)


@app.delete("/tasks/{task_id}")
async def delete_task(
    task_id: str,
    _: str = Depends(require_api_token),
    runner: TaskRunner = Depends(get_runner),
) -> dict[str, str]:
    try:
        await runner.delete(task_id)
    except KeyError:
        raise _http_error(status.HTTP_404_NOT_FOUND, ErrorCode.not_found) from None
    except TaskRunningError as exc:
        raise _http_error(status.HTTP_409_CONFLICT, exc.code) from None
    return {"status": "ok"}
