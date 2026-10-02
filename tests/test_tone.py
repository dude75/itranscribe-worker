from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from pydantic import ValidationError

from app.config import Settings
from app.schemas import TranscriptLine, coerce_tone_bool
from app.tone.emotions import normalize_emotion_label, valence_from_emotions
from app.tone.prosody import compute_prosody
from app.tone.runner import run_tone_pass
from app.engines.stubs import StubSerTone, StubTextTone


def test_tone_bool_coercion() -> None:
    assert coerce_tone_bool("false") is False
    assert coerce_tone_bool("true") is True
    assert coerce_tone_bool("TRUE") is True
    assert coerce_tone_bool(None) is False
    assert coerce_tone_bool("") is False
    assert coerce_tone_bool(True) is True
    with pytest.raises(ValueError, match="true or false"):
        coerce_tone_bool("full")
    with pytest.raises(ValueError, match="true or false"):
        coerce_tone_bool("1")


def test_tone_layers_independent() -> None:
    settings = Settings(TONE_TEXT_MODEL="", TONE_PROSODY="", TONE_SER_MODEL="", _env_file=None)
    assert settings.tone_any_layer_configured() is False
    settings2 = Settings(TONE_PROSODY="preset:minimal", _env_file=None)
    assert settings2.tone_any_layer_configured() is True
    assert settings2.tone_text_configured() is False


def test_prosody_presets() -> None:
    settings = Settings(TONE_PROSODY="preset:minimal", _env_file=None)
    assert settings.prosody_features() == frozenset({"energy"})
    standard = Settings(TONE_PROSODY="preset:standard", _env_file=None)
    assert standard.prosody_features() == frozenset({"energy", "f0"})
    extended = Settings(TONE_PROSODY="preset:extended", _env_file=None)
    assert extended.prosody_features() == frozenset({"energy", "f0", "tempo", "pauses"})
    empty = Settings(TONE_PROSODY="", _env_file=None)
    assert empty.prosody_features() is None
    with pytest.raises(ValidationError):
        Settings(TONE_PROSODY="preset:unknown", _env_file=None)


def test_prosody_does_not_read_whole_wav(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def forbid_whole_read(*args: object, **kwargs: object) -> None:
        raise AssertionError("prosody must not load entire WAV via sf.read")

    monkeypatch.setattr(sf, "read", forbid_whole_read)
    wav = tmp_path / "seg.wav"
    sr = 16_000
    audio = np.zeros(sr * 10, dtype=np.float32)
    audio[sr * 9 : sr * 10] = 0.2
    sf.write(wav, audio, sr)
    lines = [TranscriptLine(speaker="A", start=9.0, end=10.0, text="late segment")]
    out = compute_prosody(str(wav), lines, features=frozenset({"energy"}))
    assert out[0] is not None
    assert out[0].energy_mean_db is not None


def test_pause_before_sec_same_speaker(tmp_path: Path) -> None:
    wav = tmp_path / "pause.wav"
    sf.write(wav, np.zeros(16_000, dtype=np.float32), 16_000)
    lines = [
        TranscriptLine(speaker="A", start=0.0, end=0.5, text="one two"),
        TranscriptLine(speaker="A", start=1.0, end=1.5, text="three four"),
        TranscriptLine(speaker="B", start=1.5, end=2.0, text="four"),
    ]
    prosody = compute_prosody(str(wav), lines, features=frozenset({"pauses", "tempo"}))
    assert prosody[0] is not None and prosody[0].pause_before_sec is None
    assert prosody[1] is not None and prosody[1].pause_before_sec == pytest.approx(0.5)
    assert prosody[2] is not None and prosody[2].pause_before_sec is None
    assert prosody[1].words_per_sec == pytest.approx(4.0)


def test_run_tone_pass_prosody_only(tmp_path: Path) -> None:
    wav = tmp_path / "tone.wav"
    sr = 16_000
    t = np.linspace(0, 1, sr, dtype=np.float32)
    sf.write(wav, 0.1 * np.sin(2 * np.pi * 220 * t), sr)
    lines = [
        TranscriptLine(speaker="A", start=0.0, end=0.5, text="привет"),
        TranscriptLine(speaker="B", start=0.5, end=1.0, text="мир"),
    ]
    settings = Settings(
        TONE_TEXT_MODEL="",
        TONE_PROSODY="preset:standard",
        TONE_SER_MODEL="",
        _env_file=None,
    )
    result = run_tone_pass(
        settings,
        wav_path=str(wav),
        lines=lines,
        text_engine=None,
        ser_engine=None,
    )
    assert result.layers_applied == ["prosody"]
    assert result.lines[0].tone is not None
    assert result.lines[0].tone.prosody is not None
    assert result.lines[0].tone.valence is None


def test_run_tone_pass_skips_unavailable_text(tmp_path: Path) -> None:
    wav = tmp_path / "tone.wav"
    sr = 16_000
    sf.write(wav, np.zeros(sr, dtype=np.float32), sr)
    lines = [TranscriptLine(speaker="A", start=0.0, end=1.0, text="one")]
    settings = Settings(
        TONE_TEXT_MODEL="example/text",
        TONE_PROSODY="preset:minimal",
        TONE_SER_MODEL="",
        _env_file=None,
    )
    result = run_tone_pass(
        settings,
        wav_path=str(wav),
        lines=lines,
        text_engine=None,
        ser_engine=None,
    )
    assert result.layers_applied == ["prosody"]


def test_run_tone_pass_skips_unavailable_ser(tmp_path: Path) -> None:
    wav = tmp_path / "tone.wav"
    sr = 16_000
    sf.write(wav, np.zeros(sr, dtype=np.float32), sr)
    lines = [TranscriptLine(speaker="A", start=0.0, end=1.0, text="one")]
    settings = Settings(
        TONE_TEXT_MODEL="",
        TONE_PROSODY="preset:minimal",
        TONE_SER_MODEL="example/ser",
        _env_file=None,
    )
    result = run_tone_pass(
        settings,
        wav_path=str(wav),
        lines=lines,
        text_engine=None,
        ser_engine=None,
    )
    assert result.layers_applied == ["prosody"]
    assert "ser" not in result.layers_applied


def test_run_tone_pass_ser_stub(tmp_path: Path) -> None:
    wav = tmp_path / "tone.wav"
    sr = 16_000
    sf.write(wav, np.zeros(sr, dtype=np.float32), sr)
    lines = [TranscriptLine(speaker="A", start=0.0, end=0.5, text="hi")]
    settings = Settings(
        TONE_TEXT_MODEL="",
        TONE_PROSODY="",
        TONE_SER_MODEL="example/ser",
        _env_file=None,
    )
    result = run_tone_pass(
        settings,
        wav_path=str(wav),
        lines=lines,
        text_engine=None,
        ser_engine=StubSerTone(),
    )
    assert result.layers_applied == ["ser"]
    assert result.lines[0].tone is not None
    assert result.lines[0].tone.ser is not None
    assert result.lines[0].tone.ser.emotion == "neutral"


def test_run_tone_pass_text_stub(tmp_path: Path) -> None:
    wav = tmp_path / "tone.wav"
    sr = 16_000
    sf.write(wav, np.zeros(sr, dtype=np.float32), sr)
    lines = [TranscriptLine(speaker=None, start=0.0, end=1.0, text="test")]
    settings = Settings(
        TONE_TEXT_MODEL="x",
        TONE_PROSODY="",
        _env_file=None,
    )
    result = run_tone_pass(
        settings,
        wav_path=str(wav),
        lines=lines,
        text_engine=StubTextTone(),
        ser_engine=None,
    )
    assert result.layers_applied == ["text"]
    assert result.lines[0].tone is not None


def test_valence_from_emotions() -> None:
    v = valence_from_emotions({"joy": 0.9, "anger": 0.1, "sadness": 0.0})
    assert v > 0
    assert normalize_emotion_label("angry") == "anger"
