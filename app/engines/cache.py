"""Кэш движков процесса. Preload: полная копия загруженных моделей на слот WORKERS."""

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
from app.engines.hf_offline import huggingface_offline, looks_like_missing_cache
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
        self._asr: dict[AsrModel, list[ASREngine | None]] = {}
        self._diar: dict[DiarizationModel, list[DiarizationEngine | None]] = {}
        self.device = "cpu"
        self.replicas = 1

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
        self.replicas = 1

    def _preload_engine[K: Enum, E](
        self,
        wanted: set[str],
        slot: dict[K, list[E | None]],
        key: K,
        factory: Callable[[], E],
        replicas: int,
    ) -> None:
        name = key.value
        empty: list[E | None] = [None] * replicas
        if name not in wanted:
            slot[key] = empty
            self.status[name] = EngineStatus.disabled
            return
        copies: list[E | None] = []
        t0 = time.perf_counter()
        try:
            for index in range(replicas):
                if index == 0:
                    copies.append(_replica_zero(name, factory))
                else:
                    with huggingface_offline():
                        copies.append(factory())
                    log.info("%s replica %s loaded from local cache", name, index)
        except Exception as exc:
            log.warning("%s preload failed: %s: %s", name, type(exc).__name__, exc)
            slot[key] = empty
            self.status[name] = EngineStatus.unavailable
        else:
            slot[key] = copies
            self.status[name] = EngineStatus.loaded
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
        self.replicas = settings.WORKERS
        asr_wanted = set(settings.asr_families_to_preload())
        diar_wanted = set(settings.diarization_families_to_preload())
        log.info(
            "preload asr=%s diarization=%s device=%s dtype=%s workers=%s",
            settings.PRELOAD_ASR,
            settings.PRELOAD_DIARIZATION,
            device,
            dtype,
            self.replicas,
        )

        self._preload_engine(
            asr_wanted,
            self._asr,
            AsrModel.whisper,
            lambda: FasterWhisperASR(
                settings.WHISPER_MODEL, models_dir, device=device, compute_type=dtype
            ),
            self.replicas,
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
            self.replicas,
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
            self.replicas,
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
            self.replicas,
        )

        self.preloaded = True

    def resolve_asr(self, model: AsrModel, slot: int = 0) -> ASREngine:
        if not self.preloaded:
            return StubASR()
        engine = _replica(self._asr.get(model), slot)
        if engine is None:
            raise TaskFailed(ErrorCode.engine_unavailable, f"{model.value} unavailable")
        return engine

    def resolve_diarization(
        self, model: DiarizationModel, slot: int = 0
    ) -> DiarizationEngine:
        if not self.preloaded:
            return StubDiarization()
        engine = _replica(self._diar.get(model), slot)
        if engine is None:
            raise TaskFailed(ErrorCode.engine_unavailable, f"{model.value} unavailable")
        return engine


def _replica[E](replicas: list[E | None] | None, slot: int) -> E | None:
    if replicas is None or slot < 0 or slot >= len(replicas):
        return None
    return replicas[slot]


def _replica_zero[E](name: str, factory: Callable[[], E]) -> E:
    try:
        with huggingface_offline():
            engine = factory()
    except Exception as exc:
        if not looks_like_missing_cache(exc):
            raise
        log.info("%s replica 0 not in local cache, downloading", name)
        return factory()
    log.info("%s replica 0 loaded from local cache", name)
    return engine


_cache = EngineCache()


def get_cache() -> EngineCache:
    return _cache


def _observe_preload(engine: str, started: float) -> None:
    from app.prometheus_metrics import observe_preload

    observe_preload(engine, time.perf_counter() - started)
