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
from app.engines.asr.parakeet import ParakeetASR
from app.engines.asr.whisper import FasterWhisperASR
from app.engines.base import ASREngine, DiarizationEngine
from app.engines.diarization.nemo import NemoSortformerDiarizer
from app.engines.diarization.pyannote import PyannoteDiarizer
from app.engines.hf_offline import (
    ensure_hub_online,
    configure_models_dir,
    hf_hub_model_cached,
    huggingface_offline,
    looks_like_missing_cache,
)
from app.engines.stubs import StubASR, StubDiarization, StubSerTone, StubTextTone
from app.pipeline import TaskFailed
from app.schemas import AsrModel, DiarizationModel, EngineStatus, ErrorCode
from app.tone.ser_engine import TransformersSerTone
from app.tone.text_engine import TransformersTextTone

log = logging.getLogger(__name__)


class PreloadError(RuntimeError):
    """Запрошенное семейство не загрузилось — процесс не должен принимать задачи."""

    def __init__(self, engines: tuple[str, ...]) -> None:
        self.engines = engines
        names = ", ".join(engines)
        super().__init__(f"Required preload failed (unavailable): {names}")


class EngineCache:
    def __init__(self) -> None:
        self.preloaded = False
        self.status: dict[str, EngineStatus] = {
            "whisper": EngineStatus.unavailable,
            "parakeet": EngineStatus.unavailable,
            "nemo": EngineStatus.unavailable,
            "pyannote": EngineStatus.unavailable,
            "tone_text": EngineStatus.disabled,
            "tone_ser": EngineStatus.disabled,
        }
        self._asr: dict[AsrModel, list[ASREngine | None]] = {}
        self._diar: dict[DiarizationModel, list[DiarizationEngine | None]] = {}
        self._tone_text: list[object | None] = []
        self._tone_ser: list[object | None] = []
        self.device = "cpu"
        self.replicas = 1

    def reset(self) -> None:
        self.preloaded = False
        self.status = {
            "whisper": EngineStatus.loaded,
            "parakeet": EngineStatus.loaded,
            "nemo": EngineStatus.loaded,
            "pyannote": EngineStatus.loaded,
            "tone_text": EngineStatus.loaded,
            "tone_ser": EngineStatus.loaded,
        }
        self._asr.clear()
        self._diar.clear()
        self._tone_text.clear()
        self._tone_ser.clear()
        self.device = "cpu"
        self.replicas = 1

    def _preload_engine[K: Enum, E](
        self,
        wanted: set[str],
        slot: dict[K, list[E | None]],
        key: K,
        factory: Callable[[], E],
        replicas: int,
        *,
        models_dir: str,
        hf_model_id: str | None = None,
    ) -> None:
        name = key.value
        empty: list[E | None] = [None] * replicas
        if name not in wanted:
            slot[key] = empty
            self.status[name] = EngineStatus.disabled
            log.info("preload skip %s (not in PRELOAD_* selection)", name)
            return
        checkpoint = hf_model_id or "—"
        log.info(
            "preload %s begin checkpoint=%s replicas=%d",
            name,
            checkpoint,
            replicas,
        )
        copies: list[E | None] = []
        t0 = time.perf_counter()
        try:
            for index in range(replicas):
                if index == 0:
                    copies.append(
                        _replica_zero(
                            name,
                            factory,
                            models_dir=models_dir,
                            hf_model_id=hf_model_id,
                        )
                    )
                else:
                    with huggingface_offline():
                        copies.append(factory())
                    log.info("preload %s: replica %d ready (local cache)", name, index)
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            log.warning(
                "preload %s failed after %.1fs: %s: %s",
                name,
                elapsed,
                type(exc).__name__,
                exc,
            )
            slot[key] = empty
            self.status[name] = EngineStatus.unavailable
        else:
            slot[key] = copies
            self.status[name] = EngineStatus.loaded
            elapsed = time.perf_counter() - t0
            log.info(
                "preload %s done loaded %d replica(s) in %.1fs",
                name,
                len(copies),
                elapsed,
            )
        finally:
            _observe_preload(name, t0)

    def preload(self, settings: Settings) -> None:
        models_dir = configure_models_dir(settings.MODELS_DIR)
        _prepare_runtime_caches(settings)
        from app.engines.diarization.pyannote import install_soundfile_audio_fallback

        install_soundfile_audio_fallback()
        if settings.HF_TOKEN:
            os.environ["HF_TOKEN"] = settings.HF_TOKEN
            os.environ["HUGGING_FACE_HUB_TOKEN"] = settings.HF_TOKEN
        device, dtype = infer_device(settings.DEVICE)
        self.device = device
        self.replicas = settings.WORKERS
        asr_wanted = set(settings.asr_families_to_preload())
        diar_wanted = set(settings.diarization_families_to_preload())
        tone_wanted = set(settings.tone_families_to_preload())
        plan = _preload_plan_lines(settings, asr_wanted, diar_wanted, tone_wanted)
        preload_t0 = time.perf_counter()
        log.info(
            "engine preload starting models_dir=%s device=%s dtype=%s worker_slots=%d",
            models_dir,
            device,
            dtype,
            self.replicas,
        )
        if plan:
            log.info("engine preload will load: %s", "; ".join(plan))
        else:
            log.info("engine preload will load: (no engine families selected)")
        log.info(
            "engine preload config PRELOAD_ASR=%s PRELOAD_DIARIZATION=%s PRELOAD_TONE=%s",
            settings.PRELOAD_ASR,
            settings.PRELOAD_DIARIZATION,
            settings.PRELOAD_TONE,
        )

        self._preload_engine(
            asr_wanted,
            self._asr,
            AsrModel.whisper,
            lambda: FasterWhisperASR(
                settings.WHISPER_MODEL, models_dir, device=device, compute_type=dtype
            ),
            self.replicas,
            models_dir=models_dir,
            hf_model_id=settings.WHISPER_MODEL,
        )
        self._preload_engine(
            asr_wanted,
            self._asr,
            AsrModel.parakeet,
            lambda: ParakeetASR(
                settings.PARAKEET_MODEL,
                models_dir,
                device=device,
                chunk_sec=settings.PARAKEET_CHUNK_SEC,
                hf_token=settings.HF_TOKEN or None,
            ),
            self.replicas,
            models_dir=models_dir,
            hf_model_id=settings.PARAKEET_MODEL,
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
            models_dir=models_dir,
            hf_model_id=settings.NEMO_MODEL,
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
            models_dir=models_dir,
            hf_model_id=settings.PYANNOTE_MODEL,
        )

        self._preload_tone_text(settings, tone_wanted, device, models_dir)
        self._preload_tone_ser(settings, tone_wanted, device, models_dir)

        required = _required_preload_engines(asr_wanted, diar_wanted, tone_wanted, settings)
        failed = tuple(
            name for name in required if self.status.get(name) is not EngineStatus.loaded
        )
        loaded, unavailable, disabled = _preload_status_summary(self.status)
        elapsed = time.perf_counter() - preload_t0
        if failed:
            log.error(
                "engine preload aborted in %.1fs: required unavailable=[%s] "
                "loaded=[%s] disabled=[%s]",
                elapsed,
                ", ".join(failed),
                loaded,
                disabled,
            )
            raise PreloadError(failed)

        self.preloaded = True
        log.info(
            "engine preload finished in %.1fs loaded=[%s] unavailable=[%s] disabled=[%s]",
            elapsed,
            loaded,
            unavailable,
            disabled,
        )

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

    def _preload_tone_text(
        self,
        settings: Settings,
        wanted: set[str],
        device: str,
        models_dir: str,
    ) -> None:
        replicas = self.replicas
        empty: list[object | None] = [None] * replicas
        if "text" not in wanted or not settings.tone_text_configured():
            self._tone_text = empty
            self.status["tone_text"] = EngineStatus.disabled
            reason = "PRELOAD_TONE" if "text" not in wanted else "TONE_TEXT_MODEL empty"
            log.info("preload skip tone_text (%s)", reason)
            return
        copies: list[object | None] = []
        t0 = time.perf_counter()
        try:
            model_id = settings.TONE_TEXT_MODEL
            log.info(
                "preload tone_text begin checkpoint=%s replicas=%d",
                model_id,
                replicas,
            )
            for index in range(replicas):
                factory = lambda mid=model_id, dev=device, mdir=models_dir: TransformersTextTone(
                    mid, dev, mdir
                )
                if index == 0:
                    copies.append(
                        _replica_zero(
                            "tone_text",
                            factory,
                            models_dir=models_dir,
                            hf_model_id=model_id,
                        )
                    )
                else:
                    with huggingface_offline():
                        copies.append(factory())
                    log.info("preload tone_text: replica %d ready (local cache)", index)
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            log.warning(
                "preload tone_text failed after %.1fs: %s: %s",
                elapsed,
                type(exc).__name__,
                exc,
            )
            self._tone_text = empty
            self.status["tone_text"] = EngineStatus.unavailable
        else:
            self._tone_text = copies
            self.status["tone_text"] = EngineStatus.loaded
            log.info(
                "preload tone_text done loaded %d replica(s) in %.1fs",
                len(copies),
                time.perf_counter() - t0,
            )
        finally:
            _observe_preload("tone_text", t0)

    def _preload_tone_ser(
        self,
        settings: Settings,
        wanted: set[str],
        device: str,
        models_dir: str,
    ) -> None:
        _ = models_dir
        replicas = self.replicas
        empty: list[object | None] = [None] * replicas
        if "ser" not in wanted or not settings.tone_ser_configured():
            self._tone_ser = empty
            self.status["tone_ser"] = EngineStatus.disabled
            reason = "PRELOAD_TONE" if "ser" not in wanted else "TONE_SER_MODEL empty"
            log.info("preload skip tone_ser (%s)", reason)
            return
        copies: list[object | None] = []
        t0 = time.perf_counter()
        try:
            model_id = settings.TONE_SER_MODEL
            log.info(
                "preload tone_ser begin checkpoint=%s replicas=%d",
                model_id,
                replicas,
            )
            for index in range(replicas):
                factory = lambda mid=model_id, dev=device, mdir=models_dir: TransformersSerTone(
                    mid, dev, mdir
                )
                if index == 0:
                    copies.append(
                        _replica_zero(
                            "tone_ser",
                            factory,
                            models_dir=models_dir,
                            hf_model_id=model_id,
                        )
                    )
                else:
                    with huggingface_offline():
                        copies.append(factory())
                    log.info("preload tone_ser: replica %d ready (local cache)", index)
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            log.warning(
                "preload tone_ser failed after %.1fs: %s: %s",
                elapsed,
                type(exc).__name__,
                exc,
            )
            self._tone_ser = empty
            self.status["tone_ser"] = EngineStatus.unavailable
        else:
            self._tone_ser = copies
            self.status["tone_ser"] = EngineStatus.loaded
            log.info(
                "preload tone_ser done loaded %d replica(s) in %.1fs",
                len(copies),
                time.perf_counter() - t0,
            )
        finally:
            _observe_preload("tone_ser", t0)

    def resolve_tone_text(self, slot: int = 0) -> object | None:
        if not self.preloaded:
            return StubTextTone()
        if self.status["tone_text"] is not EngineStatus.loaded:
            return None
        return _replica(self._tone_text, slot)

    def resolve_tone_ser(self, slot: int = 0) -> object | None:
        if not self.preloaded:
            return StubSerTone()
        if self.status["tone_ser"] is not EngineStatus.loaded:
            return None
        return _replica(self._tone_ser, slot)


def _replica[E](replicas: list[E | None] | None, slot: int) -> E | None:
    if replicas is None or slot < 0 or slot >= len(replicas):
        return None
    return replicas[slot]


def _replica_zero[E](
    name: str,
    factory: Callable[[], E],
    *,
    models_dir: str,
    hf_model_id: str | None = None,
) -> E:
    label = hf_model_id or name
    if hf_model_id and not hf_hub_model_cached(models_dir, hf_model_id):
        log.info(
            "preload %s: replica 0 downloading %s into %s",
            name,
            label,
            models_dir,
        )
        ensure_hub_online()
        engine = factory()
        log.info("preload %s: replica 0 ready (downloaded)", name)
        return engine
    try:
        with huggingface_offline():
            engine = factory()
    except Exception as exc:
        if not looks_like_missing_cache(exc):
            raise
        log.info(
            "preload %s: replica 0 downloading %s into %s",
            name,
            label,
            models_dir,
        )
        ensure_hub_online()
        engine = factory()
        log.info("preload %s: replica 0 ready (downloaded)", name)
        return engine
    log.info("preload %s: replica 0 ready (local cache)", name)
    return engine


def _required_preload_engines(
    asr_wanted: set[str],
    diar_wanted: set[str],
    tone_wanted: set[str],
    settings: Settings,
) -> tuple[str, ...]:
    required: list[str] = []
    if "whisper" in asr_wanted:
        required.append("whisper")
    if "parakeet" in asr_wanted:
        required.append("parakeet")
    if "nemo" in diar_wanted:
        required.append("nemo")
    if "pyannote" in diar_wanted:
        required.append("pyannote")
    if "text" in tone_wanted and settings.tone_text_configured():
        required.append("tone_text")
    if "ser" in tone_wanted and settings.tone_ser_configured():
        required.append("tone_ser")
    return tuple(required)


def _preload_plan_lines(
    settings: Settings,
    asr_wanted: set[str],
    diar_wanted: set[str],
    tone_wanted: set[str],
) -> list[str]:
    plan: list[str] = []
    if "whisper" in asr_wanted:
        plan.append(f"whisper={settings.WHISPER_MODEL}")
    if "parakeet" in asr_wanted:
        plan.append(f"parakeet={settings.PARAKEET_MODEL}")
    if "nemo" in diar_wanted:
        plan.append(f"nemo={settings.NEMO_MODEL}")
    if "pyannote" in diar_wanted:
        plan.append(f"pyannote={settings.PYANNOTE_MODEL}")
    if "text" in tone_wanted and settings.tone_text_configured():
        plan.append(f"tone_text={settings.TONE_TEXT_MODEL}")
    if "ser" in tone_wanted and settings.tone_ser_configured():
        plan.append(f"tone_ser={settings.TONE_SER_MODEL}")
    return plan


def _preload_status_summary(status: dict[str, EngineStatus]) -> tuple[str, str, str]:
    loaded = sorted(name for name, state in status.items() if state is EngineStatus.loaded)
    unavailable = sorted(
        name for name, state in status.items() if state is EngineStatus.unavailable
    )
    disabled = sorted(name for name, state in status.items() if state is EngineStatus.disabled)
    return ", ".join(loaded), ", ".join(unavailable), ", ".join(disabled)


_cache = EngineCache()


def get_cache() -> EngineCache:
    return _cache


def _prepare_runtime_caches(settings: Settings) -> None:
    """Numba (librosa/tone) и Matplotlib иначе пишут в $HOME и падают, если каталог недоступен."""
    numba_dir = Path(settings.DATA_DIR).resolve() / "numba_cache"
    mpl_dir = Path(settings.LOG_DIR).resolve() / "mpl"
    numba_dir.mkdir(parents=True, exist_ok=True)
    mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ["NUMBA_CACHE_DIR"] = str(numba_dir)
    os.environ["MPLCONFIGDIR"] = str(mpl_dir)


def _observe_preload(engine: str, started: float) -> None:
    from app.prometheus_metrics import observe_preload

    observe_preload(engine, time.perf_counter() - started)
