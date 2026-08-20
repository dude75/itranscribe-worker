"""Кэш движков процесса. Preload при старте, веса общие для слотов."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from app.audio import infer_device
from app.config import Settings
from app.engines.asr.gigaam import GigaAMASR
from app.engines.asr.whisper import FasterWhisperASR
from app.engines.base import ASREngine, DiarizationEngine
from app.engines.diarization.nemo import NemoSortformerDiarizer
from app.engines.diarization.pyannote import PyannoteDiarizer
from app.engines.stubs import StubASR, StubDiarization
from app.pipeline import TaskFailed
from app.schemas import AsrModel, DiarizationModel, EngineStatus

log = logging.getLogger(__name__)


class EngineCache:
    def __init__(self) -> None:
        self.preloaded = False
        self.status: dict[str, EngineStatus] = {
            "whisper": EngineStatus.unavailable,
            "gigaam": EngineStatus.unavailable,
            "nemo": EngineStatus.unavailable,
            "pyannote": EngineStatus.unavailable,
        }
        self._asr: dict[AsrModel, ASREngine | None] = {}
        self._diar: dict[DiarizationModel, DiarizationEngine | None] = {}
        self.device = "cpu"

    def reset(self) -> None:
        self.preloaded = False
        self.status = {
            "whisper": EngineStatus.loaded,
            "gigaam": EngineStatus.loaded,
            "nemo": EngineStatus.loaded,
            "pyannote": EngineStatus.loaded,
        }
        self._asr.clear()
        self._diar.clear()
        self.device = "cpu"

    def preload(self, settings: Settings) -> None:
        models_dir = str(Path(settings.MODELS_DIR).resolve())
        Path(models_dir).mkdir(parents=True, exist_ok=True)
        os.environ["HF_HOME"] = models_dir
        os.environ["HF_HUB_CACHE"] = str(Path(models_dir) / "hub")
        os.environ["NEMO_CACHE_DIR"] = models_dir
        if settings.HF_TOKEN:
            os.environ["HF_TOKEN"] = settings.HF_TOKEN
            os.environ["HUGGING_FACE_HUB_TOKEN"] = settings.HF_TOKEN
        device, dtype = infer_device(settings.DEVICE)
        self.device = device
        asr_wanted = set(settings.asr_families_to_preload())
        diar_wanted = set(settings.diarization_families_to_preload())
        log.info(
            "preload asr=%s diarization=%s device=%s dtype=%s",
            settings.PRELOAD_ASR,
            settings.PRELOAD_DIARIZATION,
            device,
            dtype,
        )

        if "whisper" in asr_wanted:
            try:
                self._asr[AsrModel.whisper] = FasterWhisperASR(
                    settings.WHISPER_MODEL, models_dir, device=device, compute_type=dtype
                )
                self.status["whisper"] = EngineStatus.loaded
            except Exception as exc:
                log.warning("whisper preload failed: %s", type(exc).__name__)
                self._asr[AsrModel.whisper] = None
                self.status["whisper"] = EngineStatus.unavailable
        else:
            self._asr[AsrModel.whisper] = None
            self.status["whisper"] = EngineStatus.disabled

        if "gigaam" in asr_wanted:
            try:
                self._asr[AsrModel.gigaam] = GigaAMASR(
                    settings.GIGAAM_MODEL,
                    models_dir,
                    device=device,
                    hf_token=settings.HF_TOKEN,
                )
                self.status["gigaam"] = EngineStatus.loaded
            except Exception as exc:
                log.warning("gigaam preload failed: %s", type(exc).__name__)
                self._asr[AsrModel.gigaam] = None
                self.status["gigaam"] = EngineStatus.unavailable
        else:
            self._asr[AsrModel.gigaam] = None
            self.status["gigaam"] = EngineStatus.disabled

        if "nemo" in diar_wanted:
            try:
                self._diar[DiarizationModel.nemo] = NemoSortformerDiarizer(
                    settings.NEMO_MODEL,
                    models_dir,
                    device=device,
                    hf_token=settings.HF_TOKEN,
                )
                self.status["nemo"] = EngineStatus.loaded
            except Exception as exc:
                log.warning("nemo preload failed: %s: %s", type(exc).__name__, exc)
                self._diar[DiarizationModel.nemo] = None
                self.status["nemo"] = EngineStatus.unavailable
        else:
            self._diar[DiarizationModel.nemo] = None
            self.status["nemo"] = EngineStatus.disabled

        if "pyannote" in diar_wanted:
            try:
                self._diar[DiarizationModel.pyannote] = PyannoteDiarizer(
                    settings.PYANNOTE_MODEL,
                    models_dir,
                    hf_token=settings.HF_TOKEN,
                    device=device,
                )
                self.status["pyannote"] = EngineStatus.loaded
            except Exception as exc:
                log.warning("pyannote preload failed: %s", type(exc).__name__)
                self._diar[DiarizationModel.pyannote] = None
                self.status["pyannote"] = EngineStatus.unavailable
        else:
            self._diar[DiarizationModel.pyannote] = None
            self.status["pyannote"] = EngineStatus.disabled

        self.preloaded = True

    def resolve_asr(self, model: AsrModel) -> ASREngine:
        if not self.preloaded:
            return StubASR()
        engine = self._asr.get(model)
        if engine is None:
            raise TaskFailed("engine_unavailable", f"{model.value} unavailable")
        return engine

    def resolve_diarization(self, model: DiarizationModel) -> DiarizationEngine:
        if not self.preloaded:
            return StubDiarization()
        engine = self._diar.get(model)
        if engine is None:
            raise TaskFailed("engine_unavailable", f"{model.value} unavailable")
        return engine


_cache = EngineCache()


def get_cache() -> EngineCache:
    return _cache
