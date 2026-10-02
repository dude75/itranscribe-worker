"""Tone pass после alignment."""

from __future__ import annotations

import time
from dataclasses import dataclass

from app.config import Settings
from app.schemas import (
    CallSummary,
    ProsodyFeatures,
    SerResult,
    TranscriptLine,
    UtteranceTone,
)
from app.tone.emotions import arousal_from_prosody
from app.tone.prosody import compute_prosody
from app.tone.audio_io import load_segment
from app.tone.summary import build_call_summary
from app.tone.text_engine import TextToneEngine, emotions_to_tone_fields


@dataclass
class ToneRunResult:
    lines: list[TranscriptLine]
    tone_time_sec: float
    tone_ser_time_sec: float | None
    call_summary: CallSummary | None
    layers_applied: list[str]


def run_tone_pass(
    settings: Settings,
    *,
    wav_path: str,
    lines: list[TranscriptLine],
    text_engine: TextToneEngine | None,
    ser_engine: object | None,
) -> ToneRunResult:
    started = time.perf_counter()
    layers_applied: list[str] = []

    emotion_rows: list[dict[str, float]] | None = None
    if settings.tone_text_configured() and text_engine is not None:
        emotion_rows = text_engine.analyze_batch([line.text for line in lines])
        layers_applied.append("text")

    prosody_rows: list[ProsodyFeatures | None] | None = None
    if settings.tone_prosody_configured():
        features = settings.prosody_features()
        if features:
            prosody_rows = compute_prosody(
                wav_path,
                lines,
                features=features,
            )
            layers_applied.append("prosody")

    out_lines: list[TranscriptLine] = []
    for index, line in enumerate(lines):
        valence: float | None = None
        emotions: dict[str, float] = {}
        if emotion_rows is not None:
            valence, emotions = emotions_to_tone_fields(emotion_rows[index])
        prosody = prosody_rows[index] if prosody_rows else None
        arousal = None
        if prosody is not None:
            arousal = arousal_from_prosody(
                prosody.energy_mean_db,
                prosody.f0_hz_median,
                prosody.words_per_sec,
            )
        tone = _tone_from_parts(valence, emotions, arousal, prosody, ser=None)
        out_lines.append(line.model_copy(update={"tone": tone}))

    ser_time: float | None = None
    if settings.tone_ser_configured() and ser_engine is not None:
        ser_started = time.perf_counter()
        enriched: list[TranscriptLine] = []
        for line in out_lines:
            audio, sr = load_segment(wav_path, line.start, line.end)
            emotion, score = ser_engine.analyze_segment(audio, sr)
            ser_result = SerResult(emotion=emotion, score=score)
            tone = line.tone
            if tone is None:
                tone = UtteranceTone(ser=ser_result)
            else:
                tone = tone.model_copy(update={"ser": ser_result})
            enriched.append(line.model_copy(update={"tone": tone}))
        out_lines = enriched
        ser_time = time.perf_counter() - ser_started
        layers_applied.append("ser")

    summary = build_call_summary(out_lines)
    return ToneRunResult(
        lines=out_lines,
        tone_time_sec=time.perf_counter() - started,
        tone_ser_time_sec=ser_time,
        call_summary=summary,
        layers_applied=layers_applied,
    )


def _tone_from_parts(
    valence: float | None,
    emotions: dict[str, float],
    arousal: float | None,
    prosody: ProsodyFeatures | None,
    ser: SerResult | None,
) -> UtteranceTone | None:
    if (
        valence is None
        and not emotions
        and arousal is None
        and prosody is None
        and ser is None
    ):
        return None
    return UtteranceTone(
        valence=valence,
        emotions=emotions,
        arousal=arousal,
        prosody=prosody,
        ser=ser,
    )
