"""Кэш движков процесса. Preload при старте, веса общие для слотов."""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable
from enum import Enum
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
from app.schemas import AsrModel, DiarizationModel, EngineStatus, ErrorCode

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

    def _preload_engine[K: Enum, E](
        self,
        wanted: set[str],
        slot: dict[K, E | None],
        key: K,
        factory: Callable[[], E],
    ) -> None:
        name = key.value
        if name not in wanted:
            slot[key] = None
            self.status[name] = EngineStatus.disabled
            return
        t0 = time.perf_counter()
        try:
            slot[key] = factory()
            self.status[name] = EngineStatus.loaded
        except Exception as exc:
            log.warning("%s preload failed: %s: %s", name, type(exc).__name__, exc)
            slot[key] = None
            self.status[name] = EngineStatus.unavailable
        finally:
            _observe_preload(name, t0)

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

        self._preload_engine(
            asr_wanted,
            self._asr,
            AsrModel.whisper,
            lambda: FasterWhisperASR(
                settings.WHISPER_MODEL, models_dir, device=device, compute_type=dtype
            ),
        )
        self._preload_engine(
            asr_wanted,
            self._asr,
            AsrModel.gigaam,
            lambda: GigaAMASR(
                settings.GIGAAM_MODEL,
                models_dir,
                device=device,
                hf_token=settings.HF_TOKEN,
            ),
        )
        self._preload_engine(
            diar_wanted,
            self._diar,
            DiarizationModel.nemo,
            lambda: NemoSortformerDiarizer(
                settings.NEMO_MODEL,
                models_dir,
                device=device,
                hf_token=settings.HF_TOKEN,
            ),
        )
        self._preload_engine(
            diar_wanted,
            self._diar,
            DiarizationModel.pyannote,
            lambda: PyannoteDiarizer(
                settings.PYANNOTE_MODEL,
                models_dir,
                hf_token=settings.HF_TOKEN,
                device=device,
            ),
        )

        self.preloaded = True

    def resolve_asr(self, model: AsrModel) -> ASREngine:
        if not self.preloaded:
            return StubASR()
        engine = self._asr.get(model)
        if engine is None:
            raise TaskFailed(ErrorCode.engine_unavailable, f"{model.value} unavailable")
        return engine

    def resolve_diarization(self, model: DiarizationModel) -> DiarizationEngine:
        if not self.preloaded:
            return StubDiarization()
        engine = self._diar.get(model)
        if engine is None:
            raise TaskFailed(ErrorCode.engine_unavailable, f"{model.value} unavailable")
        return engine


_cache = EngineCache()


def get_cache() -> EngineCache:
    return _cache


def _observe_preload(engine: str, started: float) -> None:
    from app.prometheus_metrics import observe_preload

    observe_preload(engine, time.perf_counter() - started)
