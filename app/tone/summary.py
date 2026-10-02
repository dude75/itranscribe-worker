"""Сводка по звонку."""

from __future__ import annotations

from app.schemas import CallSummary, TranscriptLine, UtteranceTone


def build_call_summary(
    lines: list[TranscriptLine],
    *,
    opening_sec: float = 60.0,
    closing_sec: float = 60.0,
) -> CallSummary | None:
    valences: list[tuple[float, float]] = []
    for line in lines:
        if line.tone is None or line.tone.valence is None:
            continue
        valences.append((line.start, line.tone.valence))
    if not valences:
        return None

    end_time = max(line.end for line in lines) if lines else 0.0
    opening = [v for t, v in valences if t < opening_sec]
    closing = [v for t, v in valences if t >= max(0.0, end_time - closing_sec)]

    opening_valence = float(sum(opening) / len(opening)) if opening else None
    closing_valence = float(sum(closing) / len(closing)) if closing else None
    de_escalation: bool | None = None
    if opening_valence is not None and closing_valence is not None:
        de_escalation = opening_valence < -0.2 and closing_valence > 0.2

    return CallSummary(
        opening_valence=opening_valence,
        closing_valence=closing_valence,
        de_escalation=de_escalation,
    )
