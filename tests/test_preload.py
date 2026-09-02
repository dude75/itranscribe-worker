from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from app.config import Settings
from app.engines.cache import EngineCache
from app.pipeline import TaskFailed
from app.schemas import AsrModel, DiarizationModel, EngineStatus


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
    assert asr_exc.value.code == "engine_unavailable"

    with pytest.raises(TaskFailed) as diar_exc:
        cache.resolve_diarization(DiarizationModel.pyannote)
    assert diar_exc.value.code == "engine_unavailable"

    assert cache.resolve_asr(AsrModel.whisper) is not None
    assert cache.resolve_diarization(DiarizationModel.nemo) is not None


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
    assert exc.value.code == "engine_unavailable"


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
