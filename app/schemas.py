"""Схемы API: queued / running / success / error."""

from enum import Enum
from typing import Annotated, Any

from pydantic import BaseModel, BeforeValidator, Field


class AsrModel(str, Enum):
    whisper = "whisper"
    parakeet = "parakeet"


class DiarizationModel(str, Enum):
    nemo = "nemo"
    pyannote = "pyannote"


def coerce_optional_diarization(value: Any) -> Any:
    """Пустая строка / None → без диаризации; иначе строка семейства для enum."""
    if value is None:
        return None
    if isinstance(value, DiarizationModel):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        return stripped
    return value


OptionalDiarizationModel = Annotated[
    DiarizationModel | None,
    BeforeValidator(coerce_optional_diarization),
]


class TaskStatus(str, Enum):
    queued = "queued"
    running = "running"
    success = "success"
    error = "error"


class EngineStatus(str, Enum):
    loaded = "loaded"
    unavailable = "unavailable"
    disabled = "disabled"


class ErrorCode(str, Enum):
    unauthorized = "unauthorized"
    payload_too_large = "payload_too_large"
    queue_full = "queue_full"
    invalid_file = "invalid_file"
    not_found = "not_found"
    task_running = "task_running"
    missing_upload = "missing_upload"
    ffmpeg_timeout = "ffmpeg_timeout"
    task_timeout = "task_timeout"
    zero_duration = "zero_duration"
    pipeline_error = "pipeline_error"
    engine_unavailable = "engine_unavailable"
    interrupted = "interrupted"
    process_killed = "process_killed"


def error_payload(code: ErrorCode) -> dict[str, Any]:
    return {"status": "error", "error": {"code": code.value}}


class ErrorDetail(BaseModel):
    code: ErrorCode
    message: str | None = None


def coerce_tone_bool(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        stripped = value.strip().lower()
        if not stripped:
            return False
        if stripped == "true":
            return True
        if stripped == "false":
            return False
        raise ValueError("tone must be true or false")
    raise ValueError("tone must be true or false")


ToneFormField = Annotated[bool, BeforeValidator(coerce_tone_bool)]


class ProsodyFeatures(BaseModel):
    energy_mean_db: float | None = None
    energy_max_db: float | None = None
    f0_hz_median: float | None = None
    words_per_sec: float | None = None
    pause_before_sec: float | None = None


class SerResult(BaseModel):
    emotion: str
    score: float


class UtteranceTone(BaseModel):
    valence: float | None = None
    arousal: float | None = None
    emotions: dict[str, float] = Field(default_factory=dict)
    prosody: ProsodyFeatures | None = None
    ser: SerResult | None = None


class CallSummary(BaseModel):
    opening_valence: float | None = None
    closing_valence: float | None = None
    de_escalation: bool | None = None


class TranscriptLine(BaseModel):
    speaker: str | None = None
    start: float
    end: float
    text: str
    tone: UtteranceTone | None = None


class TaskMeta(BaseModel):
    timestamp: str
    task_id: str
    asr_model: AsrModel
    diarization_model: DiarizationModel | None = None
    asr_checkpoint: str | None = None
    diarization_checkpoint: str | None = None
    audio_duration_sec: float | None = None
    asr_time_sec: float | None = None
    diarization_time_sec: float | None = None
    alignment_time_sec: float | None = None
    total_time_sec: float | None = None
    rtf: float | None = None
    tone_requested: bool = False
    tone_text_model: str | None = None
    tone_ser_model: str | None = None
    tone_prosody_param: str | None = None
    tone_layers: list[str] | None = None
    tone_skipped: bool | None = None
    tone_time_sec: float | None = None
    tone_ser_time_sec: float | None = None


class TaskResponse(BaseModel):
    status: TaskStatus
    meta: TaskMeta
    transcript: list[TranscriptLine] | None = None
    call_summary: CallSummary | None = None
    error: ErrorDetail | None = None


class TaskListItem(BaseModel):
    task_id: str
    status: TaskStatus
    timestamp: str
    asr_model: AsrModel
    diarization_model: DiarizationModel | None = None


class WorkersHealth(BaseModel):
    max: int
    active: int
    available: int


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str
    engines: dict[str, EngineStatus] = Field(default_factory=dict)
    device: str = "cpu"
    workers: WorkersHealth


class PurgeResult(BaseModel):
    status: str = "ok"
    purged_queued: int
    purged_finished: int
    purged_tmp: int
    skipped_running: int
