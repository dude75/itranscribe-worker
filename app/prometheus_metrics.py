"""Prometheus registry and scrape-time gauges. Independent of CSV `metric_event`."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Histogram,
    Info,
    generate_latest,
)
from prometheus_client.gc_collector import GCCollector
from prometheus_client.metrics_core import GaugeMetricFamily
from prometheus_client.platform_collector import PlatformCollector
from prometheus_client.process_collector import ProcessCollector
from prometheus_client.registry import Collector
from starlette.requests import Request
from starlette.routing import Match

from app.audio import tmp_root
from app.schemas import EngineStatus, TaskStatus
from app.version import read_version

if TYPE_CHECKING:
    from app.config import Settings
    from app.engines.cache import EngineCache
    from app.queueing import TaskRunner

CONTENT_TYPE = CONTENT_TYPE_LATEST
ENGINES = ("whisper", "gigaam", "nemo", "pyannote")
ENGINE_STATUSES = ("loaded", "unavailable", "disabled")
ASR_ENGINES = ("whisper", "gigaam")

STAGE_BUCKETS = (5.0, 15.0, 30.0, 60.0, 120.0, 300.0, 600.0, 1200.0, 1800.0, 3600.0)
QUEUE_WAIT_BUCKETS = (0.1, 0.5, 1.0, 2.0, 5.0, 15.0, 30.0, 60.0, 120.0, 300.0)
RTF_BUCKETS = (0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
PRELOAD_BUCKETS = (1.0, 5.0, 15.0, 30.0, 60.0, 120.0, 300.0, 600.0, 1200.0)
HTTP_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)

_active: Metrics | None = None


def diarization_label(model: str | None) -> str:
    return "none" if model is None or model == "" else model


def queue_wait_sec(timestamp: str) -> float | None:
    try:
        queued_at = datetime.fromisoformat(timestamp)
    except ValueError:
        return None
    return max((datetime.now() - queued_at).total_seconds(), 0.0)


def http_path_template(request: Request) -> str:
    for route in request.app.router.routes:
        if not hasattr(route, "matches"):
            continue
        match, _child = route.matches(request.scope)
        if match == Match.FULL:
            path = getattr(route, "path", None)
            if isinstance(path, str) and path:
                return path
    return "unknown"


@dataclass
class RuntimeState:
    settings: Settings | None = None
    cache: EngineCache | None = None
    runner: TaskRunner | None = None
    runner_started: bool = False


class RuntimeCollector(Collector):
    def __init__(self, state: RuntimeState) -> None:
        self.state = state

    def collect(self):
        yield GaugeMetricFamily("itranscribe_up", "Process is serving /metrics", value=1.0)
        yield GaugeMetricFamily(
            "itranscribe_ready",
            "Runner started and at least one ASR family is loaded",
            value=1.0 if _ready(self.state) else 0.0,
        )
        yield from _engine_metrics(self.state.cache)
        yield from _queue_metrics(self.state)
        yield from _disk_metrics(self.state.settings)


def _ready(state: RuntimeState) -> bool:
    if not state.runner_started or state.cache is None:
        return False
    return any(state.cache.status.get(name) is EngineStatus.loaded for name in ASR_ENGINES)


def _engine_metrics(cache: EngineCache | None):
    loaded = GaugeMetricFamily(
        "itranscribe_engine_loaded",
        "1 if the engine family is loaded",
        labels=["engine"],
    )
    status_g = GaugeMetricFamily(
        "itranscribe_engine_status",
        "1 for the current engine family status",
        labels=["engine", "status"],
    )
    for engine in ENGINES:
        current = cache.status.get(engine) if cache is not None else None
        loaded.add_metric([engine], 1.0 if current is EngineStatus.loaded else 0.0)
        for status_name in ENGINE_STATUSES:
            match = current is not None and current.value == status_name
            status_g.add_metric([engine, status_name], 1.0 if match else 0.0)
    yield loaded
    yield status_g


def _queue_metrics(state: RuntimeState):
    settings = state.settings
    slots = float(settings.WORKERS) if settings is not None else 0.0
    limit = float(settings.WORKER_QUEUE_SIZE) if settings is not None else 0.0
    depth = 0.0
    running = 0.0
    age = 0.0
    store = state.runner.store if state.runner is not None else None
    if store is not None:
        try:
            depth = float(store.count_queued())
            running_rows = store.list_tasks(TaskStatus.running)
            running = float(len(running_rows))
            age = _max_running_age(running_rows)
        except (sqlite3.Error, OSError):
            pass
    yield GaugeMetricFamily("itranscribe_queue_depth", "Tasks in queued", value=depth)
    yield GaugeMetricFamily("itranscribe_tasks_running", "Tasks in running", value=running)
    yield GaugeMetricFamily("itranscribe_worker_slots", "Configured WORKERS", value=slots)
    yield GaugeMetricFamily("itranscribe_queue_limit", "Configured WORKER_QUEUE_SIZE", value=limit)
    yield GaugeMetricFamily(
        "itranscribe_task_running_age_seconds",
        "Age of the oldest running task",
        value=age,
    )


def _max_running_age(records) -> float:
    now = datetime.now()
    max_age = 0.0
    for record in records:
        if not record.started_at:
            continue
        try:
            started = datetime.fromisoformat(record.started_at)
        except ValueError:
            continue
        max_age = max(max_age, (now - started).total_seconds())
    return max_age


def _disk_metrics(settings: Settings | None):
    tmp_bytes = 0
    tmp_dirs = 0
    sqlite_bytes = 0
    if settings is not None:
        tmp_bytes, tmp_dirs = _tmp_stats(Path(settings.DATA_DIR))
        sqlite_path = Path(settings.SQLITE_PATH)
        try:
            if sqlite_path.is_file():
                sqlite_bytes = sqlite_path.stat().st_size
        except OSError:
            sqlite_bytes = 0
    yield GaugeMetricFamily("itranscribe_tmp_bytes", "Bytes under DATA_DIR/tmp", value=float(tmp_bytes))
    yield GaugeMetricFamily("itranscribe_tmp_dirs", "Task directories under DATA_DIR/tmp", value=float(tmp_dirs))
    yield GaugeMetricFamily("itranscribe_sqlite_bytes", "Size of SQLITE_PATH", value=float(sqlite_bytes))


def _tmp_stats(data_dir: Path) -> tuple[int, int]:
    root = tmp_root(data_dir)
    if not root.is_dir():
        return 0, 0
    total = 0
    dirs = 0
    try:
        children = list(root.iterdir())
    except OSError:
        return 0, 0
    for child in children:
        try:
            if child.is_dir():
                dirs += 1
                for file in child.rglob("*"):
                    if file.is_file():
                        try:
                            total += file.stat().st_size
                        except OSError:
                            pass
            elif child.is_file():
                total += child.stat().st_size
        except OSError:
            continue
    return total, dirs


class Metrics:
    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled
        self.registry = CollectorRegistry()
        ProcessCollector(registry=self.registry)
        PlatformCollector(registry=self.registry)
        GCCollector(registry=self.registry)
        self.runtime = RuntimeState()
        self._info: Info | None = None
        if not enabled:
            return
        self.registry.register(RuntimeCollector(self.runtime))
        self._info = Info("itranscribe", "Process version, device, checkpoints", registry=self.registry)
        self.preload_duration = Histogram(
            "itranscribe_preload_duration_seconds",
            "Engine family preload wall time",
            ["engine"],
            buckets=PRELOAD_BUCKETS,
            registry=self.registry,
        )
        self.queue_rejected = Counter(
            "itranscribe_queue_rejected_total",
            "POST /transcribe rejected with queue_full",
            registry=self.registry,
        )
        self.tasks_submitted = Counter(
            "itranscribe_tasks_submitted_total",
            "Tasks accepted into the queue",
            ["asr_model", "diarization_model"],
            registry=self.registry,
        )
        self.tasks_completed = Counter(
            "itranscribe_tasks_completed_total",
            "Tasks that reached success or error",
            ["asr_model", "diarization_model", "status"],
            registry=self.registry,
        )
        self.task_errors = Counter(
            "itranscribe_task_errors_total",
            "Failed tasks by error code",
            ["error_code", "asr_model", "diarization_model"],
            registry=self.registry,
        )
        self.audio_seconds = Counter(
            "itranscribe_audio_seconds_processed_total",
            "Audio duration processed",
            ["asr_model", "diarization_model", "status"],
            registry=self.registry,
        )
        self.inference_seconds = Counter(
            "itranscribe_inference_seconds_total",
            "Inference wall time by stage",
            ["stage", "asr_model", "diarization_model"],
            registry=self.registry,
        )
        self.pipeline_duration = Histogram(
            "itranscribe_pipeline_duration_seconds",
            "Pipeline wall-clock time",
            ["asr_model", "diarization_model"],
            buckets=STAGE_BUCKETS,
            registry=self.registry,
        )
        self.queue_wait = Histogram(
            "itranscribe_queue_wait_seconds",
            "Time from submit to running",
            ["asr_model", "diarization_model"],
            buckets=QUEUE_WAIT_BUCKETS,
            registry=self.registry,
        )
        self.asr_duration = Histogram(
            "itranscribe_asr_duration_seconds",
            "ASR inference time",
            ["asr_model"],
            buckets=STAGE_BUCKETS,
            registry=self.registry,
        )
        self.diarization_duration = Histogram(
            "itranscribe_diarization_duration_seconds",
            "Diarization inference time",
            ["diarization_model"],
            buckets=STAGE_BUCKETS,
            registry=self.registry,
        )
        self.rtf = Histogram(
            "itranscribe_rtf",
            "Real-time factor (asr+diar)/audio_duration",
            ["asr_model", "diarization_model"],
            buckets=RTF_BUCKETS,
            registry=self.registry,
        )
        self.audio_duration = Histogram(
            "itranscribe_audio_duration_seconds",
            "Input audio duration",
            ["asr_model", "diarization_model"],
            buckets=STAGE_BUCKETS,
            registry=self.registry,
        )
        self.restore_tasks = Counter(
            "itranscribe_restore_tasks_total",
            "Unfinished tasks handled at process start",
            ["outcome"],
            registry=self.registry,
        )
        self.http_requests = Counter(
            "itranscribe_http_requests_total",
            "HTTP requests",
            ["method", "path", "code"],
            registry=self.registry,
        )
        self.http_duration = Histogram(
            "itranscribe_http_request_duration_seconds",
            "HTTP request duration",
            ["method", "path"],
            buckets=HTTP_BUCKETS,
            registry=self.registry,
        )

    def bind(self, *, settings: Settings, cache: EngineCache, runner: TaskRunner) -> None:
        self.runtime.settings = settings
        self.runtime.cache = cache
        self.runtime.runner = runner
        self.runtime.runner_started = True
        if self._info is None:
            return
        self._info.info(
            {
                "version": read_version(),
                "device": cache.device,
                "whisper_checkpoint": settings.WHISPER_MODEL,
                "gigaam_checkpoint": settings.GIGAAM_MODEL,
                "nemo_checkpoint": settings.NEMO_MODEL,
                "pyannote_checkpoint": settings.PYANNOTE_MODEL,
            }
        )


def create_metrics(settings: Settings) -> Metrics:
    return Metrics(enabled=settings.METRICS_ENABLED)


def get_active() -> Metrics | None:
    return _active


def set_active(metrics: Metrics | None) -> None:
    global _active
    _active = metrics


def render() -> bytes:
    metrics = _active
    if metrics is None:
        return b""
    return generate_latest(metrics.registry)


def observe_preload(engine: str, duration_sec: float) -> None:
    metrics = _active
    if metrics is None or not metrics.enabled:
        return
    metrics.preload_duration.labels(engine=engine).observe(duration_sec)


def observe_queue_rejected() -> None:
    metrics = _active
    if metrics is None or not metrics.enabled:
        return
    metrics.queue_rejected.inc()


def observe_submitted(asr_model: str, diarization_model: str | None) -> None:
    metrics = _active
    if metrics is None or not metrics.enabled:
        return
    metrics.tasks_submitted.labels(
        asr_model=asr_model, diarization_model=diarization_label(diarization_model)
    ).inc()


def observe_restore(outcome: str) -> None:
    metrics = _active
    if metrics is None or not metrics.enabled:
        return
    metrics.restore_tasks.labels(outcome=outcome).inc()


def observe_http(method: str, path: str, status_code: int, duration_sec: float) -> None:
    metrics = _active
    if metrics is None or not metrics.enabled:
        return
    metrics.http_requests.labels(method=method, path=path, code=str(status_code)).inc()
    metrics.http_duration.labels(method=method, path=path).observe(duration_sec)


def observe_task_finished(
    *,
    asr_model: str,
    diarization_model: str | None,
    status: str,
    error_code: str | None = None,
    audio_duration_sec: float | None = None,
    asr_time_sec: float | None = None,
    diarization_time_sec: float | None = None,
    total_time_sec: float | None = None,
    rtf: float | None = None,
    queue_wait: float | None = None,
) -> None:
    metrics = _active
    if metrics is None or not metrics.enabled:
        return
    diar = diarization_label(diarization_model)
    metrics.tasks_completed.labels(asr_model=asr_model, diarization_model=diar, status=status).inc()
    if status == "error" and error_code:
        metrics.task_errors.labels(
            error_code=error_code, asr_model=asr_model, diarization_model=diar
        ).inc()
    if audio_duration_sec is not None:
        metrics.audio_seconds.labels(
            asr_model=asr_model, diarization_model=diar, status=status
        ).inc(audio_duration_sec)
        metrics.audio_duration.labels(asr_model=asr_model, diarization_model=diar).observe(
            audio_duration_sec
        )
    if asr_time_sec is not None:
        metrics.inference_seconds.labels(
            stage="asr", asr_model=asr_model, diarization_model=diar
        ).inc(asr_time_sec)
        metrics.asr_duration.labels(asr_model=asr_model).observe(asr_time_sec)
    if diarization_model is not None and diarization_time_sec is not None:
        metrics.inference_seconds.labels(
            stage="diarization", asr_model=asr_model, diarization_model=diar
        ).inc(diarization_time_sec)
        metrics.diarization_duration.labels(diarization_model=diar).observe(diarization_time_sec)
    if total_time_sec is not None:
        metrics.pipeline_duration.labels(asr_model=asr_model, diarization_model=diar).observe(
            total_time_sec
        )
    if rtf is not None:
        metrics.rtf.labels(asr_model=asr_model, diarization_model=diar).observe(rtf)
    if queue_wait is not None:
        metrics.queue_wait.labels(asr_model=asr_model, diarization_model=diar).observe(queue_wait)
