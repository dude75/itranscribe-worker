from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from app.audio import infer_device
from app.config import get_settings
from app.engines.diarization.pyannote import PyannoteDiarizer


def test_pyannote_requires_token() -> None:
    settings = get_settings()
    with pytest.raises(RuntimeError, match="HF_TOKEN"):
        PyannoteDiarizer(settings.PYANNOTE_MODEL, settings.MODELS_DIR, hf_token="")


def _two_speaker_wav(path: Path) -> Path:
    if shutil.which("say") is None:
        pytest.skip("macOS say is not available")
    a = path.with_name("spk_a.wav")
    b_raw = path.with_name("spk_b_raw.wav")
    b = path.with_name("spk_b.wav")
    phrase_a = " ".join(["hello this is the first speaker talking about the weather"] * 4)
    phrase_b = " ".join(["and now the second speaker replies with a different story"] * 4)
    subprocess.run(
        ["say", "-v", "Samantha", "-o", str(a), "--data-format=LEI16@16000", "--rate=180", phrase_a],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["say", "-v", "Alex", "-o", str(b_raw), "--data-format=LEI16@16000", "--rate=160", phrase_b],
        check=True,
        capture_output=True,
    )
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg не установлен")
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(b_raw),
            "-af", "asetrate=16000*0.72,aresample=16000,atempo=1.15",
            str(b),
        ],
        check=True,
        capture_output=True,
    )
    left, sr = sf.read(a)
    right, _sr2 = sf.read(b)
    if left.ndim > 1:
        left = left[:, 0]
    if right.ndim > 1:
        right = right[:, 0]
    gap = np.zeros(int(sr * 1.0), dtype=left.dtype)
    mixed = np.concatenate([left, gap, right])
    sf.write(path, mixed, sr)
    return path


@pytest.fixture(scope="module")
def pyannote_engine() -> PyannoteDiarizer:
    settings = get_settings()
    device, _dtype = infer_device()
    return PyannoteDiarizer(
        settings.PYANNOTE_MODEL,
        settings.MODELS_DIR,
        hf_token=settings.HF_TOKEN,
        device=device,
    )


def test_pyannote_two_speakers(pyannote_engine: PyannoteDiarizer, tmp_path: Path) -> None:
    wav = _two_speaker_wav(tmp_path / "two_spk.wav")
    segments = pyannote_engine.segments(str(wav), min_speakers=2, max_speakers=2)
    assert segments
    assert all(s.end >= s.start and s.speaker for s in segments)
    speakers = {s.speaker for s in segments}
    assert len(speakers) >= 2
