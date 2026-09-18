from __future__ import annotations

import shutil
import subprocess
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import soundfile as sf

from app.audio import infer_device
from app.config import get_settings
from app.engines.asr.parakeet import ParakeetASR, _configure_parakeet_decoding
from app.engines.base import words_from_parakeet_timestamps


def test_parakeet_timestamps_with_offset_and_stride() -> None:
    word_ts = [
        {"word": "hello", "start_offset": 10, "end_offset": 20},
        {"word": "world", "start": 1.0, "end": 1.5},
    ]
    words = words_from_parakeet_timestamps(
        word_ts,
        time_stride=0.08,
        time_offset=100.0,
        segment_id=2,
    )
    assert [w.text for w in words] == ["hello", "world"]
    assert words[0].start == pytest.approx(100.8)
    assert words[0].end == pytest.approx(101.6)
    assert words[1].start == pytest.approx(101.0)
    assert words[1].end == pytest.approx(101.5)
    assert all(w.segment_id == 2 for w in words)


def test_parakeet_timestamps_skip_blank() -> None:
    word_ts = [{"word": "  "}, {"char": "ok", "start": 0.0, "end": 0.2}]
    words = words_from_parakeet_timestamps(word_ts, time_stride=0.08)
    assert [w.text for w in words] == ["ok"]


def _fake_nemo_modules(monkeypatch: pytest.MonkeyPatch, model) -> None:
    asr = types.ModuleType("nemo.collections.asr")
    models = types.ModuleType("nemo.collections.asr.models")
    models.ASRModel = MagicMock()
    models.ASRModel.from_pretrained = MagicMock(return_value=model)
    asr.models = models
    collections = types.ModuleType("nemo.collections")
    collections.asr = asr
    nemo = types.ModuleType("nemo")
    nemo.collections = collections
    monkeypatch.setitem(sys.modules, "nemo", nemo)
    monkeypatch.setitem(sys.modules, "nemo.collections", collections)
    monkeypatch.setitem(sys.modules, "nemo.collections.asr", asr)
    monkeypatch.setitem(sys.modules, "nemo.collections.asr.models", models)


def test_parakeet_chunks_merge_timestamps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg is not available")

    sr = 16000
    short = np.zeros(int(5 * sr), dtype=np.float32)
    sf.write(tmp_path / "piece.wav", short, sr)
    long = tmp_path / "long.wav"
    reps = int(np.ceil((26 * sr) / len(short)))
    sf.write(long, np.tile(short, reps)[: int(26 * sr)], sr)

    calls: list[str] = []

    class FakeHyp:
        text = "chunk"
        timestamp = {"word": [{"word": "a", "start": 0.0, "end": 0.5}]}

    class FakeModel:
        window_stride = 0.01

        def to(self, device):
            return self

        def eval(self):
            return self

        def transcribe(self, paths, timestamps=True, **_kwargs):
            calls.append(paths[0])
            return [FakeHyp()]

    _fake_nemo_modules(monkeypatch, FakeModel())
    engine = ParakeetASR(
        "nvidia/parakeet-tdt-0.6b-v3",
        str(tmp_path / "models"),
        device="cpu",
        chunk_sec=10.0,
    )
    words = engine.words(str(long))
    assert len(calls) == 3
    assert [w.text for w in words] == ["a", "a", "a"]
    assert words[0].start == pytest.approx(0.0)
    assert words[1].start == pytest.approx(10.0)
    assert words[2].start == pytest.approx(20.0)
    assert [w.segment_id for w in words] == [0, 1, 2]


def test_parakeet_disables_cuda_graph_decoder_before_transcribe() -> None:
    greedy = types.SimpleNamespace(use_cuda_graph_decoder=True)
    beam = types.SimpleNamespace(allow_cuda_graphs=True)
    decoding = types.SimpleNamespace(compute_timestamps=False, greedy=greedy, beam=beam)
    model = types.SimpleNamespace(
        cfg=types.SimpleNamespace(decoding=decoding),
        change_decoding_strategy=MagicMock(),
    )

    _configure_parakeet_decoding(model)

    assert decoding.compute_timestamps is True
    assert greedy.use_cuda_graph_decoder is False
    assert beam.allow_cuda_graphs is False
    model.change_decoding_strategy.assert_called_once_with(decoding, verbose=False)


def test_parakeet_decoding_config_missing_is_noop() -> None:
    _configure_parakeet_decoding(types.SimpleNamespace())


def _speech_wav(path: Path, text: str = "привет мир") -> Path:
    if shutil.which("say") is None:
        pytest.skip("macOS say is not available")
    subprocess.run(
        ["say", "-o", str(path), "--data-format=LEI16@16000", text],
        check=True,
        capture_output=True,
    )
    return path


@pytest.fixture(scope="module")
def parakeet_engine() -> ParakeetASR:
    pytest.importorskip("nemo")
    settings = get_settings()
    device, _dtype = infer_device()
    return ParakeetASR(
        settings.PARAKEET_MODEL,
        settings.MODELS_DIR,
        device=device,
        chunk_sec=settings.PARAKEET_CHUNK_SEC,
        hf_token=settings.HF_TOKEN or None,
    )


@pytest.mark.ml
def test_parakeet_short(parakeet_engine: ParakeetASR, tmp_path: Path) -> None:
    wav = _speech_wav(tmp_path / "parakeet_short.wav")
    words = parakeet_engine.words(str(wav))
    assert words
    assert all(w.end >= w.start and w.text.strip() for w in words)


@pytest.mark.ml
def test_parakeet_long_file_chunked(parakeet_engine: ParakeetASR, tmp_path: Path) -> None:
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg is not available")
    short = _speech_wav(tmp_path / "parakeet_piece.wav")
    audio, sr = sf.read(short)
    if audio.ndim > 1:
        audio = audio[:, 0]
    need = int(26 * sr)
    long = np.tile(audio, int(np.ceil(need / max(len(audio), 1))))[:need]
    long_path = tmp_path / "parakeet_long.wav"
    sf.write(long_path, long, sr)

    engine = ParakeetASR(
        get_settings().PARAKEET_MODEL,
        get_settings().MODELS_DIR,
        device=infer_device()[0],
        chunk_sec=10.0,
        hf_token=get_settings().HF_TOKEN or None,
    )
    words = engine.words(str(long_path))
    assert words
    assert all(w.end >= w.start for w in words)
    assert words[-1].end == pytest.approx(26.0, abs=2.0)
