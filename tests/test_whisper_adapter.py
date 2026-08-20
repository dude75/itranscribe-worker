from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from app.audio import infer_device
from app.config import get_settings
from app.engines.asr.whisper import FasterWhisperASR


def _speech_wav(path: Path, text: str = "one two three four") -> Path:
    if shutil.which("say") is None:
        pytest.skip("macOS say is not available")
    subprocess.run(
        ["say", "-o", str(path), "--data-format=LEI16@16000", text],
        check=True,
        capture_output=True,
    )
    return path


@pytest.fixture(scope="module")
def whisper_engine() -> FasterWhisperASR:
    settings = get_settings()
    device, dtype = infer_device()
    assert device == "cpu" or device == "cuda"
    if device == "cpu":
        assert dtype == "float32"
    return FasterWhisperASR(
        settings.WHISPER_MODEL,
        settings.MODELS_DIR,
        device=device,
        compute_type=dtype,
    )


def test_whisper_words_with_timestamps(whisper_engine: FasterWhisperASR, tmp_path: Path) -> None:
    wav = _speech_wav(tmp_path / "whisper.wav")
    words = whisper_engine.words(str(wav))
    assert words
    for word in words:
        assert word.end >= word.start
        assert word.text.strip()
    settings = get_settings()
    models = Path(settings.MODELS_DIR)
    assert models.exists()
    assert any(models.iterdir())
