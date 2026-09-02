"""Пайплайн задачи: ASR, при необходимости диаризация, alignment, метрики, чистка tmp."""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

from app.alignment import align, utterances_from_words
from app.audio import FfmpegTimeout, audio_duration_sec, cleanup_tmp, prepare_wav
from app.config import Settings
from app.engines.base import ASREngine, DiarizationEngine
from app.engines.stubs import StubASR, StubDiarization
from app.metrics import MetricEvent, write_metric
from app.prometheus_metrics import observe_task_finished, queue_wait_sec
from app.schemas import (
    AsrModel,
    DiarizationModel,
    ErrorCode,
    ErrorDetail,
    TaskStatus,
    TranscriptLine,
)
from app.tasks import TaskRecord, TaskStore


class TaskFailed(Exception):
    def __init__(self, code: ErrorCode, message: str | None = None) -> None:
        super().__init__(code)
        self.code = code
        self.message = message


def resolve_asr(_model: AsrModel, _slot: int = 0) -> ASREngine:
    return StubASR()


def resolve_diarization(_model: DiarizationModel, _slot: int = 0) -> DiarizationEngine:
    return StubDiarization()


def checkpoints_for(
    settings: Settings, asr: AsrModel, diar: DiarizationModel | None
) -> tuple[str, str | None]:
    asr_ckpt = settings.WHISPER_MODEL if asr is AsrModel.whisper else settings.GIGAAM_MODEL
    if diar is None:
        return asr_ckpt, None
    diar_ckpt = settings.NEMO_MODEL if diar is DiarizationModel.nemo else settings.PYANNOTE_MODEL
    return asr_ckpt, diar_ckpt


def _diarization_label(model: DiarizationModel | None) -> str | None:
    return None if model is None else model.value


def _rtf(
    duration: float | None, asr_time: float | None, diar_time: float | None
) -> float | None:
    if duration and asr_time is not None and diar_time is not None:
        return (asr_time + diar_time) / duration
    return None


def _emit_task_metrics(
    settings: Settings,
    record: TaskRecord,
    task_id: str,
    *,
    status: str,
    duration: float | None,
    asr_time: float | None,
    diar_time: float | None,
    align_time: float | None,
    total_time: float | None,
    rtf: float | None,
    error_code: ErrorCode | None = None,
) -> None:
    write_metric(
        settings.PERFORMANCE_LOG,
        MetricEvent(
            timestamp=record.timestamp,
            task_id=task_id,
            asr_model=record.asr_model.value,
            diarization_model=_diarization_label(record.diarization_model),
            asr_checkpoint=record.asr_checkpoint,
            diarization_checkpoint=record.diarization_checkpoint,
            audio_duration_sec=duration,
            asr_time_sec=asr_time,
            diarization_time_sec=diar_time,
            alignment_time_sec=align_time,
            total_time_sec=total_time,
            rtf=rtf,
            status=status,
        ),
    )
    observe_task_finished(
        asr_model=record.asr_model.value,
        diarization_model=_diarization_label(record.diarization_model),
        status=status,
        error_code=None if error_code is None else error_code.value,
        audio_duration_sec=duration,
        asr_time_sec=asr_time,
        diarization_time_sec=diar_time,
        total_time_sec=total_time,
        rtf=rtf,
        queue_wait=queue_wait_sec(record.timestamp),
    )


def run_pipeline(store: TaskStore, settings: Settings, task_id: str, slot: int = 0) -> None:
    record = store.get(task_id)
    if record is None:
        return
    if not store.mark_running(task_id):
        return
    started = time.perf_counter()
    duration: float | None = None
    asr_time: float | None = None
    diar_time: float | None = None
    align_time: float | None = None
    total_time: float | None = None
    rtf: float | None = None
    outcome: str | None = None
    error: ErrorDetail | None = None
    try:
        upload = Path(record.upload_path) if record.upload_path else None
        if upload is None or not upload.is_file():
            raise TaskFailed(ErrorCode.missing_upload)
        try:
            wav = prepare_wav(
                record.upload_path,
                task_id,
                settings.DATA_DIR,
                timeout_sec=settings.FFMPEG_TIMEOUT_SEC,
            )
        except FfmpegTimeout as exc:
            raise TaskFailed(ErrorCode.ffmpeg_timeout, str(exc)) from exc
        duration = audio_duration_sec(wav)
        if duration <= 0:
            raise TaskFailed(ErrorCode.zero_duration, "audio duration is zero")

        asr = resolve_asr(record.asr_model, slot)

        t0 = time.perf_counter()
        words = list(asr.words(str(wav)))
        asr_time = time.perf_counter() - t0

        if record.diarization_model is None:
            diar_time = 0.0
            t0 = time.perf_counter()
            utterances = utterances_from_words(words)
            align_time = time.perf_counter() - t0
        else:
            diar = resolve_diarization(record.diarization_model, slot)
            t0 = time.perf_counter()
            segments = list(diar.segments(str(wav)))
            diar_time = time.perf_counter() - t0

            t0 = time.perf_counter()
            utterances = align(words, segments)
            align_time = time.perf_counter() - t0

        total_time = time.perf_counter() - started
        rtf = (asr_time + diar_time) / duration
        transcript = [
            TranscriptLine(speaker=u.speaker, start=u.start, end=u.end, text=u.text)
            for u in utterances
        ]
        store.mark_success(
            task_id,
            audio_duration_sec=duration,
            asr_time_sec=asr_time,
            diarization_time_sec=diar_time,
            alignment_time_sec=align_time,
            total_time_sec=total_time,
            rtf=rtf,
            transcript=transcript,
        )
        outcome = "success"
    except Exception as exc:
        total_time = time.perf_counter() - started
        rtf = _rtf(duration, asr_time, diar_time)
        if isinstance(exc, TaskFailed):
            error = ErrorDetail(code=exc.code, message=exc.message)
        else:
            error = ErrorDetail(code=ErrorCode.pipeline_error, message=str(exc))
        store.mark_error(
            task_id,
            error,
            audio_duration_sec=duration,
            asr_time_sec=asr_time,
            diarization_time_sec=diar_time,
            alignment_time_sec=align_time,
            total_time_sec=total_time,
            rtf=rtf,
        )
        outcome = "error"
    finally:
        # Не трогаем tmp running/queued: после рестарта нужен upload. Чистим только финал.
        try:
            current = store.get(task_id)
        except sqlite3.Error:
            pass
        else:
            if current is None or current.status in {TaskStatus.success, TaskStatus.error}:
                cleanup_tmp(task_id, settings.DATA_DIR)
        if outcome is not None:
            _emit_task_metrics(
                settings,
                record,
                task_id,
                status=outcome,
                duration=duration,
                asr_time=asr_time,
                diar_time=diar_time,
                align_time=align_time,
                total_time=total_time,
                rtf=rtf,
                error_code=None if error is None else error.code,
            )
