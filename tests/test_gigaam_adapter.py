from __future__ import annotations

import shutil
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from app.audio import infer_device
from app.config import get_settings
from app.engines.asr.gigaam import GigaAMASR


def test_gigaam_requires_token() -> None:
    settings = get_settings()
    with pytest.raises(RuntimeError, match="HF_TOKEN"):
        GigaAMASR(settings.GIGAAM_MODEL, settings.MODELS_DIR, hf_token="")


def _fake_gigaam_modules(monkeypatch: pytest.MonkeyPatch, get_pipeline) -> None:
    class FakeModel:
        _device = "cpu"

    fake = types.ModuleType("gigaam")
    fake.load_model = lambda *args, **kwargs: FakeModel()
    vad = types.ModuleType("gigaam.vad_utils")
    vad.get_pipeline = get_pipeline
    fake.vad_utils = vad
    monkeypatch.setitem(sys.modules, "gigaam", fake)
    monkeypatch.setitem(sys.modules, "gigaam.vad_utils", vad)


def test_gigaam_constructor_warms_vad(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    warmed: list[object] = []

    def get_pipeline(device):
        warmed.append(device)
        return object()

    _fake_gigaam_modules(monkeypatch, get_pipeline)
    GigaAMASR("multilingual_large_ctc", str(tmp_path), hf_token="token", device="cpu")
    assert warmed == ["cpu"]


def test_gigaam_words_preserve_segment_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_gigaam_modules(monkeypatch, lambda _device: object())
    engine = GigaAMASR(
        "multilingual_large_ctc", str(tmp_path), hf_token="token", device="cpu"
    )

    class _Word:
        def __init__(self, start: float, end: float, text: str) -> None:
            self.start = start
            self.end = end
            self.text = text

    class _Segment:
        def __init__(self, words: list[_Word]) -> None:
            self.words = words

    engine._model.transcribe_longform = lambda *_a, **_k: [
        _Segment([_Word(0.0, 0.2, "a"), _Word(0.2, 0.4, "b")]),
        _Segment([_Word(1.0, 1.2, "c")]),
    ]
    words = engine.words("x.wav")
    assert [w.text for w in words] == ["a", "b", "c"]
    assert [w.segment_id for w in words] == [0, 0, 1]


def test_gigaam_constructor_fails_if_vad_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def get_pipeline(_device):
        raise RuntimeError(
            "Model pyannote/segmentation-3.0 was not found locally, "
            "and no HF_TOKEN was provided to download it."
        )

    _fake_gigaam_modules(monkeypatch, get_pipeline)
    with pytest.raises(RuntimeError, match="HF_TOKEN"):
        GigaAMASR("multilingual_large_ctc", str(tmp_path), hf_token="token", device="cpu")


def _speech_wav(path: Path, text: str = "привет мир") -> Path:
    if shutil.which("say") is None:
        pytest.skip("macOS say is not available")
    subprocess.run(
        ["say", "-o", str(path), "--data-format=LEI16@16000", text],
        check=True,
        capture_output=True,
    )
    return path


def _repeat_to_duration(src: Path, dest: Path, seconds: float) -> Path:
    audio, sr = sf.read(src)
    if audio.ndim > 1:
        audio = audio[:, 0]
    need = int(seconds * sr)
    reps = int(np.ceil(need / max(len(audio), 1)))
    long = np.tile(audio, max(reps, 1))[:need]
    sf.write(dest, long, sr)
    return dest


@pytest.fixture(scope="module")
def gigaam_engine() -> GigaAMASR:
    pytest.importorskip("gigaam")
    settings = get_settings()
    device, _dtype = infer_device()
    return GigaAMASR(
        settings.GIGAAM_MODEL,
        settings.MODELS_DIR,
        device=device,
        hf_token=settings.HF_TOKEN,
    )


@pytest.mark.ml
def test_gigaam_short_longform(gigaam_engine: GigaAMASR, tmp_path: Path) -> None:
    wav = _speech_wav(tmp_path / "giga_short.wav")
    words = gigaam_engine.words(str(wav))
    assert words
    assert all(w.end >= w.start and w.text.strip() for w in words)


@pytest.mark.ml
def test_gigaam_long_file(gigaam_engine: GigaAMASR, tmp_path: Path) -> None:
    short = _speech_wav(tmp_path / "giga_piece.wav")
    long = _repeat_to_duration(short, tmp_path / "giga_long.wav", 26.0)
    words = gigaam_engine.words(str(long))
    assert words
    assert all(w.end >= w.start for w in words)
