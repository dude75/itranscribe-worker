from __future__ import annotations

import logging
import os
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.config import Settings
from app.engines.cache import EngineCache
from app.pipeline import TaskFailed
from app.schemas import AsrModel, DiarizationModel, EngineStatus, ErrorCode


def _spy(name: str, constructed: list[str]):
    class Spy:
        def __init__(self, *args, **kwargs) -> None:
            constructed.append(name)

    return Spy


def test_preload_defaults_all() -> None:
    settings = Settings(_env_file=None)
    assert settings.PRELOAD_ASR == "all"
    assert settings.PRELOAD_DIARIZATION == "all"
    assert settings.DEVICE == "auto"
    assert settings.FFMPEG_TIMEOUT_SEC == 120
    assert settings.TASK_MAX_RESTARTS == 1
    assert settings.MAX_UPLOAD_BYTES == 1024 ** 3
    assert settings.LOG_MAX_BYTES == 5 * 1024 * 1024
    assert settings.LOG_BACKUP_COUNT == 5
    assert settings.WORKERS == 1
    assert settings.asr_families_to_preload() == ("whisper", "gigaam")
    assert settings.diarization_families_to_preload() == ("nemo", "pyannote")


def test_device_normalizes_case() -> None:
    settings = Settings(DEVICE=" CUDA ", _env_file=None)
    assert settings.DEVICE == "cuda"


def test_preload_normalizes_case_and_whitespace() -> None:
    settings = Settings(
        PRELOAD_ASR=" Whisper ",
        PRELOAD_DIARIZATION="PYANNOTE",
        _env_file=None,
    )
    assert settings.PRELOAD_ASR == "whisper"
    assert settings.PRELOAD_DIARIZATION == "pyannote"
    assert settings.asr_families_to_preload() == ("whisper",)
    assert settings.diarization_families_to_preload() == ("pyannote",)


@pytest.mark.parametrize(
    "field, value",
    [
        ("PRELOAD_ASR", "nemo"),
        ("PRELOAD_DIARIZATION", "whisper"),
        ("PRELOAD_ASR", "none"),
        ("DEVICE", "mps"),
    ],
)
def test_preload_rejects_unknown(field: str, value: str) -> None:
    with pytest.raises(ValidationError):
        Settings(**{field: value}, _env_file=None)


def test_task_max_restarts_rejects_negative() -> None:
    with pytest.raises(ValidationError):
        Settings(TASK_MAX_RESTARTS=-1, _env_file=None)


def test_max_upload_bytes_rejects_non_positive() -> None:
    with pytest.raises(ValidationError):
        Settings(MAX_UPLOAD_BYTES=0, _env_file=None)
    with pytest.raises(ValidationError):
        Settings(MAX_UPLOAD_BYTES=-1, _env_file=None)


def test_log_rotation_settings_reject_invalid() -> None:
    with pytest.raises(ValidationError):
        Settings(LOG_MAX_BYTES=0, _env_file=None)
    with pytest.raises(ValidationError):
        Settings(LOG_MAX_BYTES=-1, _env_file=None)
    with pytest.raises(ValidationError):
        Settings(LOG_BACKUP_COUNT=0, _env_file=None)
    with pytest.raises(ValidationError):
        Settings(LOG_BACKUP_COUNT=-1, _env_file=None)


def test_workers_rejects_non_positive() -> None:
    with pytest.raises(ValidationError):
        Settings(WORKERS=0, _env_file=None)
    with pytest.raises(ValidationError):
        Settings(WORKERS=-1, _env_file=None)


def test_preload_skips_constructors_and_marks_disabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    constructed: list[str] = []
    monkeypatch.setattr("app.engines.cache.FasterWhisperASR", _spy("whisper", constructed))
    monkeypatch.setattr("app.engines.cache.GigaAMASR", _spy("gigaam", constructed))
    monkeypatch.setattr("app.engines.cache.NemoSortformerDiarizer", _spy("nemo", constructed))
    monkeypatch.setattr("app.engines.cache.PyannoteDiarizer", _spy("pyannote", constructed))
    monkeypatch.setattr(
        "app.engines.cache.infer_device", lambda *_args, **_kwargs: ("cpu", "float32")
    )

    settings = Settings(
        PRELOAD_ASR="whisper",
        PRELOAD_DIARIZATION="nemo",
        MODELS_DIR=str(tmp_path),
        HF_TOKEN="",
        _env_file=None,
    )
    cache = EngineCache()
    cache.preload(settings)

    assert cache.device == "cpu"
    assert constructed == ["whisper", "nemo"]
    assert cache.status["whisper"] is EngineStatus.loaded
    assert cache.status["nemo"] is EngineStatus.loaded
    assert cache.status["gigaam"] is EngineStatus.disabled
    assert cache.status["pyannote"] is EngineStatus.disabled

    with pytest.raises(TaskFailed) as asr_exc:
        cache.resolve_asr(AsrModel.gigaam)
    assert asr_exc.value.code is ErrorCode.engine_unavailable

    with pytest.raises(TaskFailed) as diar_exc:
        cache.resolve_diarization(DiarizationModel.pyannote)
    assert diar_exc.value.code is ErrorCode.engine_unavailable

    assert cache.resolve_asr(AsrModel.whisper) is not None
    assert cache.resolve_diarization(DiarizationModel.nemo) is not None
    assert cache.replicas == 1


def test_preload_gigaam_unavailable_without_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("app.engines.cache.NemoSortformerDiarizer", _spy("nemo", []))
    monkeypatch.setattr(
        "app.engines.cache.infer_device", lambda *_args, **_kwargs: ("cpu", "float32")
    )

    settings = Settings(
        PRELOAD_ASR="gigaam",
        PRELOAD_DIARIZATION="nemo",
        MODELS_DIR=str(tmp_path),
        HF_TOKEN="",
        _env_file=None,
    )
    cache = EngineCache()
    cache.preload(settings)

    assert cache.status["gigaam"] is EngineStatus.unavailable
    with pytest.raises(TaskFailed) as exc:
        cache.resolve_asr(AsrModel.gigaam)
    assert exc.value.code is ErrorCode.engine_unavailable


def test_preload_all_constructs_every_family(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    constructed: list[str] = []
    monkeypatch.setattr("app.engines.cache.FasterWhisperASR", _spy("whisper", constructed))
    monkeypatch.setattr("app.engines.cache.GigaAMASR", _spy("gigaam", constructed))
    monkeypatch.setattr("app.engines.cache.NemoSortformerDiarizer", _spy("nemo", constructed))
    monkeypatch.setattr("app.engines.cache.PyannoteDiarizer", _spy("pyannote", constructed))
    monkeypatch.setattr(
        "app.engines.cache.infer_device", lambda *_args, **_kwargs: ("cpu", "float32")
    )

    settings = Settings(
        PRELOAD_ASR="all",
        PRELOAD_DIARIZATION="all",
        MODELS_DIR=str(tmp_path),
        _env_file=None,
    )
    cache = EngineCache()
    cache.preload(settings)

    assert constructed == ["whisper", "gigaam", "nemo", "pyannote"]
    assert cache.status == {
        "whisper": EngineStatus.loaded,
        "gigaam": EngineStatus.loaded,
        "nemo": EngineStatus.loaded,
        "pyannote": EngineStatus.loaded,
    }


def test_preload_workers_builds_independent_replicas(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    constructed: list[str] = []
    monkeypatch.setattr("app.engines.cache.FasterWhisperASR", _spy("whisper", constructed))
    monkeypatch.setattr("app.engines.cache.GigaAMASR", _spy("gigaam", constructed))
    monkeypatch.setattr("app.engines.cache.NemoSortformerDiarizer", _spy("nemo", constructed))
    monkeypatch.setattr("app.engines.cache.PyannoteDiarizer", _spy("pyannote", constructed))
    monkeypatch.setattr(
        "app.engines.cache.infer_device", lambda *_args, **_kwargs: ("cpu", "float32")
    )

    settings = Settings(
        PRELOAD_ASR="whisper",
        PRELOAD_DIARIZATION="nemo",
        WORKERS=2,
        MODELS_DIR=str(tmp_path),
        HF_TOKEN="token",
        _env_file=None,
    )
    cache = EngineCache()
    cache.preload(settings)

    assert constructed == ["whisper", "whisper", "nemo", "nemo"]
    assert cache.replicas == 2
    first = cache.resolve_asr(AsrModel.whisper, 0)
    second = cache.resolve_asr(AsrModel.whisper, 1)
    assert first is not second
    assert cache.resolve_diarization(DiarizationModel.nemo, 0) is not cache.resolve_diarization(
        DiarizationModel.nemo, 1
    )
    with pytest.raises(TaskFailed) as exc:
        cache.resolve_asr(AsrModel.whisper, 2)
    assert exc.value.code is ErrorCode.engine_unavailable


def _cache_aware(name: str, downloaded: dict[str, bool], events: list[tuple[str, bool]]):
    class Engine:
        def __init__(self, *args, **kwargs) -> None:
            offline = os.environ.get("HF_HUB_OFFLINE") == "1"
            events.append((name, offline))
            if offline and not downloaded[name]:
                raise RuntimeError(
                    "Cannot find an appropriate cached snapshot folder for the specified revision"
                )
            if not offline:
                downloaded[name] = True

    return Engine


def test_preload_replica_zero_downloads_then_rest_use_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
    events: list[tuple[str, bool]] = []
    downloaded = {"whisper": False, "nemo": False, "gigaam": False, "pyannote": False}
    monkeypatch.setattr(
        "app.engines.cache.FasterWhisperASR", _cache_aware("whisper", downloaded, events)
    )
    monkeypatch.setattr(
        "app.engines.cache.NemoSortformerDiarizer", _cache_aware("nemo", downloaded, events)
    )
    monkeypatch.setattr("app.engines.cache.GigaAMASR", _spy("gigaam", []))
    monkeypatch.setattr("app.engines.cache.PyannoteDiarizer", _spy("pyannote", []))
    monkeypatch.setattr(
        "app.engines.cache.infer_device", lambda *_args, **_kwargs: ("cpu", "float32")
    )

    settings = Settings(
        PRELOAD_ASR="whisper",
        PRELOAD_DIARIZATION="nemo",
        WORKERS=2,
        MODELS_DIR=str(tmp_path),
        HF_TOKEN="token",
        _env_file=None,
    )
    EngineCache().preload(settings)

    assert events == [
        ("whisper", True),
        ("whisper", False),
        ("whisper", True),
        ("nemo", True),
        ("nemo", False),
        ("nemo", True),
    ]
    assert downloaded == {
        "whisper": True,
        "nemo": True,
        "gigaam": False,
        "pyannote": False,
    }
    assert os.environ.get("HF_HUB_OFFLINE") != "1"


def test_preload_uses_cache_for_every_replica_when_already_downloaded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
    events: list[tuple[str, bool]] = []
    downloaded = {"whisper": True, "nemo": True, "gigaam": False, "pyannote": False}
    monkeypatch.setattr(
        "app.engines.cache.FasterWhisperASR", _cache_aware("whisper", downloaded, events)
    )
    monkeypatch.setattr(
        "app.engines.cache.NemoSortformerDiarizer", _cache_aware("nemo", downloaded, events)
    )
    monkeypatch.setattr("app.engines.cache.GigaAMASR", _spy("gigaam", []))
    monkeypatch.setattr("app.engines.cache.PyannoteDiarizer", _spy("pyannote", []))
    monkeypatch.setattr(
        "app.engines.cache.infer_device", lambda *_args, **_kwargs: ("cpu", "float32")
    )

    settings = Settings(
        PRELOAD_ASR="whisper",
        PRELOAD_DIARIZATION="nemo",
        WORKERS=2,
        MODELS_DIR=str(tmp_path),
        HF_TOKEN="token",
        _env_file=None,
    )
    EngineCache().preload(settings)

    assert events == [
        ("whisper", True),
        ("whisper", True),
        ("nemo", True),
        ("nemo", True),
    ]
    assert os.environ.get("HF_HUB_OFFLINE") != "1"


def test_preload_failure_logs_exception_for_every_family(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    class Boom:
        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError("weights missing")

    monkeypatch.setattr("app.engines.cache.FasterWhisperASR", Boom)
    monkeypatch.setattr("app.engines.cache.GigaAMASR", Boom)
    monkeypatch.setattr("app.engines.cache.NemoSortformerDiarizer", Boom)
    monkeypatch.setattr("app.engines.cache.PyannoteDiarizer", Boom)
    monkeypatch.setattr(
        "app.engines.cache.infer_device", lambda *_args, **_kwargs: ("cpu", "float32")
    )

    settings = Settings(
        PRELOAD_ASR="all",
        PRELOAD_DIARIZATION="all",
        WORKERS=2,
        MODELS_DIR=str(tmp_path),
        HF_TOKEN="token",
        _env_file=None,
    )
    with caplog.at_level(logging.WARNING, logger="app.engines.cache"):
        EngineCache().preload(settings)

    messages = [record.getMessage() for record in caplog.records]
    for name in ("whisper", "gigaam", "nemo", "pyannote"):
        expected = f"{name} preload failed: RuntimeError: weights missing"
        assert any(expected == message for message in messages), messages
