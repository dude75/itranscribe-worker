from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from app.audio import infer_device
from app.config import get_settings
from app.engines.asr.gigaam import GigaAMASR

pytestmark = pytest.mark.ml
pytest.importorskip("gigaam")


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


def test_gigaam_short_longform(gigaam_engine: GigaAMASR, tmp_path: Path) -> None:
    wav = _speech_wav(tmp_path / "giga_short.wav")
    words = gigaam_engine.words(str(wav))
    assert words
    assert all(w.end >= w.start and w.text.strip() for w in words)


def test_gigaam_long_file(gigaam_engine: GigaAMASR, tmp_path: Path) -> None:
    short = _speech_wav(tmp_path / "giga_piece.wav")
    long = _repeat_to_duration(short, tmp_path / "giga_long.wav", 26.0)
    words = gigaam_engine.words(str(long))
    assert words
    assert all(w.end >= w.start for w in words)
