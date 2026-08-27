"""Схемы API: queued / running / success / error."""

from enum import Enum
from typing import Annotated, Any

from pydantic import BaseModel, BeforeValidator, Field


class AsrModel(str, Enum):
    whisper = "whisper"
    gigaam = "gigaam"


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


class ErrorDetail(BaseModel):
    code: str
    message: str | None = None


class TranscriptLine(BaseModel):
    speaker: str | None = None
    start: float
    end: float
    text: str


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


class TaskResponse(BaseModel):
    status: TaskStatus
    meta: TaskMeta
    transcript: list[TranscriptLine] | None = None
    error: ErrorDetail | None = None


class TaskListItem(BaseModel):
    task_id: str
    status: TaskStatus
    timestamp: str
    asr_model: AsrModel
    diarization_model: DiarizationModel | None = None


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str
    engines: dict[str, EngineStatus] = Field(default_factory=dict)
    device: str = "cpu"


class PurgeResult(BaseModel):
    status: str = "ok"
    purged_queued: int
    purged_finished: int
    purged_tmp: int
    skipped_running: int
