"""Пайплайн задачи: ASR, при необходимости диаризация, alignment, метрики, чистка tmp."""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

from app.alignment import align, utterances_from_words
from app.audio import FfmpegTimeout, audio_duration_sec, cleanup_tmp, infer_device, prepare_wav
from app.config import Settings
from app.engines.base import ASREngine, DiarizationEngine
from app.engines.stubs import StubASR, StubDiarization
from app.metrics import MetricEvent, write_metric
from app.prometheus_metrics import observe_task_finished, queue_wait_sec
from app.schemas import AsrModel, DiarizationModel, ErrorDetail, TaskStatus, TranscriptLine
from app.tasks import TaskStore


class TaskFailed(Exception):
    def __init__(self, code: str, message: str | None = None) -> None:
        super().__init__(code)
        self.code = code
        self.message = message


def resolve_asr(_model: AsrModel) -> ASREngine:
    return StubASR()


def resolve_diarization(_model: DiarizationModel) -> DiarizationEngine:
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


def run_pipeline(store: TaskStore, settings: Settings, task_id: str) -> None:
    infer_device()
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
    try:
        upload = Path(record.upload_path) if record.upload_path else None
        if upload is None or not upload.is_file():
            raise TaskFailed("missing_upload")
        try:
            wav = prepare_wav(
                record.upload_path,
                task_id,
                settings.DATA_DIR,
                timeout_sec=settings.FFMPEG_TIMEOUT_SEC,
            )
        except FfmpegTimeout as exc:
            raise TaskFailed("ffmpeg_timeout", str(exc)) from exc
        duration = audio_duration_sec(wav)
        if duration <= 0:
            raise TaskFailed("zero_duration", "audio duration is zero")

        asr = resolve_asr(record.asr_model)

        t0 = time.perf_counter()
        words = list(asr.words(str(wav)))
        asr_time = time.perf_counter() - t0

        if record.diarization_model is None:
            diar_time = 0.0
            t0 = time.perf_counter()
            utterances = utterances_from_words(words)
            align_time = time.perf_counter() - t0
        else:
            diar = resolve_diarization(record.diarization_model)
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
        cleanup_tmp(task_id, settings.DATA_DIR)
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
                status="success",
            ),
        )
        observe_task_finished(
            asr_model=record.asr_model.value,
            diarization_model=_diarization_label(record.diarization_model),
            status="success",
            audio_duration_sec=duration,
            asr_time_sec=asr_time,
            diarization_time_sec=diar_time,
            total_time_sec=total_time,
            rtf=rtf,
            queue_wait=queue_wait_sec(record.timestamp),
        )
    except Exception as exc:
        total_time = time.perf_counter() - started
        if isinstance(exc, TaskFailed):
            error = ErrorDetail(code=exc.code, message=exc.message)
        else:
            error = ErrorDetail(code="pipeline_error", message=str(exc))
        rtf = None
        if duration and asr_time is not None and diar_time is not None:
            rtf = (asr_time + diar_time) / duration
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
        cleanup_tmp(task_id, settings.DATA_DIR)
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
                status="error",
            ),
        )
        observe_task_finished(
            asr_model=record.asr_model.value,
            diarization_model=_diarization_label(record.diarization_model),
            status="error",
            error_code=error.code,
            audio_duration_sec=duration,
            asr_time_sec=asr_time,
            diarization_time_sec=diar_time,
            total_time_sec=total_time,
            rtf=rtf,
            queue_wait=queue_wait_sec(record.timestamp),
        )
    finally:
        # Не трогаем tmp running/queued: после рестарта нужен upload. Чистим только финал.
        try:
            current = store.get(task_id)
        except sqlite3.Error:
            return
        if current is None or current.status in {TaskStatus.success, TaskStatus.error}:
            cleanup_tmp(task_id, settings.DATA_DIR)
