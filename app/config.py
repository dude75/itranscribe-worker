"""Настройки процесса из `.env` (ТЗ §4)."""

from functools import lru_cache
from typing import Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PreloadAsr = Literal["whisper", "gigaam", "all"]
PreloadDiarization = Literal["nemo", "pyannote", "all"]


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
    PERFORMANCE_LOG_ENABLED: bool = True

    WHISPER_MODEL: str = "large-v3-turbo"
    GIGAAM_MODEL: str = "multilingual_large_ctc"
    PYANNOTE_MODEL: str = "pyannote/speaker-diarization-3.1"
    NEMO_MODEL: str = "nvidia/diar_streaming_sortformer_4spk-v2"

    PRELOAD_ASR: PreloadAsr = "all"
    PRELOAD_DIARIZATION: PreloadDiarization = "all"
    DEVICE: Literal["auto", "cpu", "cuda"] = "auto"

    WORKERS: int = 1
    WORKER_QUEUE_SIZE: int = 4
    TASK_TTL_SEC: int = 3600

    @field_validator("PRELOAD_ASR", "PRELOAD_DIARIZATION", "DEVICE", mode="before")
    @classmethod
    def _normalize_preload(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip().lower()
        return value

    def asr_families_to_preload(self) -> tuple[str, ...]:
        if self.PRELOAD_ASR == "all":
            return ("whisper", "gigaam")
        return (self.PRELOAD_ASR,)

    def diarization_families_to_preload(self) -> tuple[str, ...]:
        if self.PRELOAD_DIARIZATION == "all":
            return ("nemo", "pyannote")
        return (self.PRELOAD_DIARIZATION,)


@lru_cache
def get_settings() -> Settings:
    return Settings()
