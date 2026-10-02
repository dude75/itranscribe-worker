"""Настройки процесса из `.env` (ТЗ §4)."""

from functools import lru_cache
from typing import Literal

from pydantic import AliasChoices, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ASR_FAMILIES = ("whisper", "parakeet")
DIARIZATION_FAMILIES = ("nemo", "pyannote")
TONE_PRELOAD_FAMILIES = ("text", "ser")
PROSODY_PRESETS: dict[str, tuple[str, ...]] = {
    "minimal": ("energy",),
    "standard": ("energy", "f0"),
    "extended": ("energy", "f0", "tempo", "pauses"),
}
SUPPORTED_UPLOAD_SUFFIXES = (".wav", ".mp3", ".m4a", ".flac", ".ogg", ".opus", ".webm")
UPLOAD_SUFFIX_NAMES = tuple(name.lstrip(".") for name in SUPPORTED_UPLOAD_SUFFIXES)


def _normalize_family_list(
    value: object,
    allowed: tuple[str, ...],
    *,
    field: str,
) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    tokens: list[str] = []
    for raw in value.split(","):
        token = raw.strip().lower().strip("'\"")
        if token:
            tokens.append(token)
    if not tokens:
        raise ValueError(f"{field} must not be empty")
    unknown = [token for token in tokens if token not in allowed and token != "all"]
    if unknown:
        raise ValueError(f"{field} unknown family: {unknown[0]}")
    if "all" in tokens:
        if len(tokens) > 1:
            raise ValueError(f"{field} cannot mix all with other families")
        return "all"
    selected = tuple(name for name in allowed if name in tokens)
    if selected == allowed:
        return "all"
    return ",".join(selected)


def _families_from_preload(value: str, allowed: tuple[str, ...]) -> tuple[str, ...]:
    if value == "all":
        return allowed
    wanted = set(value.split(","))
    return tuple(name for name in allowed if name in wanted)


def _normalize_upload_suffix_list(value: object, *, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    tokens: list[str] = []
    for raw in value.split(","):
        token = raw.strip().lower().strip("'\"")
        if token.startswith("."):
            token = token[1:]
        if token:
            tokens.append(token)
    if not tokens:
        raise ValueError(f"{field} must not be empty")
    unknown = [token for token in tokens if token not in UPLOAD_SUFFIX_NAMES and token != "all"]
    if unknown:
        raise ValueError(f"{field} unknown suffix: {unknown[0]}")
    if "all" in tokens:
        if len(tokens) > 1:
            raise ValueError(f"{field} cannot mix all with other suffixes")
        return "all"
    selected = tuple(name for name in UPLOAD_SUFFIX_NAMES if name in tokens)
    if selected == UPLOAD_SUFFIX_NAMES:
        return "all"
    return ",".join(selected)


def _suffixes_from_allowed(value: str) -> frozenset[str]:
    if value == "all":
        return frozenset(SUPPORTED_UPLOAD_SUFFIXES)
    return frozenset(f".{name}" for name in value.split(","))


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    API_TOKEN: str = ""
    HF_TOKEN: str = ""
    HOST: str = "127.0.0.1"
    PORT: int = 8000

    DATA_DIR: str = "./data"
    MODELS_DIR: str = "./data/models"
    SQLITE_PATH: str = "./data/tasks.db"
    LOG_DIR: str = "./data/logs"
    PERFORMANCE_LOG: str = "./data/logs/performance_log.csv"
    LOG_ENABLED: bool = True
    LOG_MAX_BYTES: int = 5 * 1024 * 1024
    LOG_BACKUP_COUNT: int = 5
    PERFORMANCE_LOG_ENABLED: bool = True
    METRICS_ENABLED: bool = True

    WHISPER_MODEL: str = "large-v3-turbo"
    PARAKEET_MODEL: str = "nvidia/parakeet-tdt-0.6b-v3"
    PARAKEET_CHUNK_SEC: float = 1380.0
    PYANNOTE_MODEL: str = "pyannote/speaker-diarization-3.1"
    NEMO_MODEL: str = "nvidia/diar_streaming_sortformer_4spk-v2"

    PRELOAD_ASR: str = "all"
    PRELOAD_DIARIZATION: str = "all"
    DEVICE: Literal["auto", "cpu", "cuda"] = "auto"

    WORKERS: int = Field(default=1, validation_alias=AliasChoices("WORKERS", "WORKERS_MAX"))
    WORKER_QUEUE_SIZE: int = 4
    MAX_UPLOAD_BYTES: int = 1024 ** 3
    ALLOWED_UPLOAD_SUFFIXES: str = "all"
    TASK_TTL_SEC: int = 3600
    FFMPEG_TIMEOUT_SEC: int = 120
    TASK_TIMEOUT_SEC: int = 14400
    TASK_MAX_RESTARTS: int = 1

    TONE_TEXT_MODEL: str = ""
    TONE_SER_MODEL: str = ""
    TONE_PROSODY: str = ""
    PRELOAD_TONE: str = "none"

    @field_validator("PRELOAD_ASR", mode="before")
    @classmethod
    def _normalize_preload_asr(cls, value: object) -> object:
        return _normalize_family_list(value, ASR_FAMILIES, field="PRELOAD_ASR")

    @field_validator("PRELOAD_DIARIZATION", mode="before")
    @classmethod
    def _normalize_preload_diarization(cls, value: object) -> object:
        return _normalize_family_list(value, DIARIZATION_FAMILIES, field="PRELOAD_DIARIZATION")

    @field_validator("PRELOAD_TONE", mode="before")
    @classmethod
    def _normalize_preload_tone(cls, value: object) -> object:
        if not isinstance(value, str):
            raise ValueError("PRELOAD_TONE must be a string")
        tokens: list[str] = []
        for raw in value.split(","):
            token = raw.strip().lower().strip("'\"")
            if token:
                tokens.append(token)
        if not tokens or tokens == ["none"]:
            return "none"
        unknown = [t for t in tokens if t not in TONE_PRELOAD_FAMILIES and t != "all"]
        if unknown:
            raise ValueError(f"PRELOAD_TONE unknown family: {unknown[0]}")
        if "all" in tokens:
            if len(tokens) > 1:
                raise ValueError("PRELOAD_TONE cannot mix all with other families")
            return "all"
        selected = tuple(name for name in TONE_PRELOAD_FAMILIES if name in tokens)
        if selected == TONE_PRELOAD_FAMILIES:
            return "all"
        return ",".join(selected)

    @field_validator("TONE_TEXT_MODEL", "TONE_SER_MODEL", mode="before")
    @classmethod
    def _strip_tone_model_ids(cls, value: object) -> object:
        if value is None:
            return ""
        if isinstance(value, str):
            return value.strip()
        return value

    @field_validator("ALLOWED_UPLOAD_SUFFIXES", mode="before")
    @classmethod
    def _normalize_allowed_upload_suffixes(cls, value: object) -> object:
        return _normalize_upload_suffix_list(value, field="ALLOWED_UPLOAD_SUFFIXES")

    @field_validator("DEVICE", mode="before")
    @classmethod
    def _normalize_device(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip().lower()
        return value

    @field_validator("TASK_MAX_RESTARTS")
    @classmethod
    def _non_negative_restarts(cls, value: int) -> int:
        if value < 0:
            raise ValueError("TASK_MAX_RESTARTS must be >= 0")
        return value

    @field_validator("TASK_TIMEOUT_SEC")
    @classmethod
    def _non_negative_task_timeout(cls, value: int) -> int:
        if value < 0:
            raise ValueError("TASK_TIMEOUT_SEC must be >= 0")
        return value

    @field_validator("MAX_UPLOAD_BYTES")
    @classmethod
    def _positive_upload_limit(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("MAX_UPLOAD_BYTES must be > 0")
        return value

    @field_validator("LOG_MAX_BYTES")
    @classmethod
    def _positive_log_max_bytes(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("LOG_MAX_BYTES must be > 0")
        return value

    @field_validator("LOG_BACKUP_COUNT")
    @classmethod
    def _positive_log_backup_count(cls, value: int) -> int:
        if value < 1:
            raise ValueError("LOG_BACKUP_COUNT must be >= 1")
        return value

    @field_validator("WORKERS")
    @classmethod
    def _positive_workers(cls, value: int) -> int:
        if value < 1:
            raise ValueError("WORKERS must be >= 1")
        return value

    def asr_families_to_preload(self) -> tuple[str, ...]:
        return _families_from_preload(self.PRELOAD_ASR, ASR_FAMILIES)

    @field_validator("PARAKEET_CHUNK_SEC")
    @classmethod
    def _positive_parakeet_chunk(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("PARAKEET_CHUNK_SEC must be > 0")
        return value

    def diarization_families_to_preload(self) -> tuple[str, ...]:
        return _families_from_preload(self.PRELOAD_DIARIZATION, DIARIZATION_FAMILIES)

    def tone_families_to_preload(self) -> tuple[str, ...]:
        if self.PRELOAD_TONE == "none":
            return ()
        wanted = set(_families_from_preload(self.PRELOAD_TONE, TONE_PRELOAD_FAMILIES))
        if "text" not in wanted or not self.tone_text_configured():
            wanted.discard("text")
        if "ser" not in wanted or not self.tone_ser_configured():
            wanted.discard("ser")
        return tuple(name for name in TONE_PRELOAD_FAMILIES if name in wanted)

    def tone_text_configured(self) -> bool:
        return bool(self.TONE_TEXT_MODEL)

    def tone_prosody_configured(self) -> bool:
        return bool(self.TONE_PROSODY.strip())

    def tone_ser_configured(self) -> bool:
        return bool(self.TONE_SER_MODEL)

    def tone_any_layer_configured(self) -> bool:
        return (
            self.tone_text_configured()
            or self.tone_prosody_configured()
            or self.tone_ser_configured()
        )

    def prosody_features(self) -> frozenset[str] | None:
        raw = self.TONE_PROSODY.strip()
        if not raw:
            return None
        if raw.lower().startswith("preset:"):
            name = raw.split(":", 1)[1].strip().lower()
            if name not in PROSODY_PRESETS:
                raise ValueError(f"TONE_PROSODY unknown preset: {name}")
            return frozenset(PROSODY_PRESETS[name])
        features = frozenset(
            token.strip().lower()
            for token in raw.split(",")
            if token.strip()
        )
        allowed = frozenset({"energy", "f0", "tempo", "pauses"})
        unknown = features - allowed
        if unknown:
            raise ValueError(f"TONE_PROSODY unknown feature: {next(iter(unknown))}")
        return features

    def allowed_upload_suffixes(self) -> frozenset[str]:
        return _suffixes_from_allowed(self.ALLOWED_UPLOAD_SUFFIXES)

    @model_validator(mode="after")
    def _validate_tone_prosody(self) -> "Settings":
        if not self.tone_prosody_configured():
            return self
        try:
            features = self.prosody_features()
        except ValueError as exc:
            raise ValueError(str(exc)) from exc
        if not features:
            raise ValueError("TONE_PROSODY must not be empty when set")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
