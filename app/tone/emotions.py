"""Нормализация labels и сводные valence/arousal."""

from __future__ import annotations

import re

_EMOTION_ALIASES: dict[str, str] = {
    "no_emotion": "neutral",
    "no emotion": "neutral",
    "neu": "neutral",
    "happiness": "joy",
    "happy": "joy",
    "angry": "anger",
    "sad": "sadness",
}


def normalize_emotion_label(label: str) -> str:
    key = label.strip().lower().replace(" ", "_")
    return _EMOTION_ALIASES.get(key, key)


def normalize_emotion_scores(raw: dict[str, float]) -> dict[str, float]:
    out: dict[str, float] = {}
    for label, score in raw.items():
        name = normalize_emotion_label(str(label))
        out[name] = max(out.get(name, 0.0), float(score))
    return out


def valence_from_emotions(emotions: dict[str, float]) -> float:
    joy = emotions.get("joy", 0.0)
    anger = emotions.get("anger", 0.0)
    sadness = emotions.get("sadness", 0.0)
    fear = emotions.get("fear", 0.0)
    raw = joy - anger - sadness - 0.5 * fear
    return max(-1.0, min(1.0, raw))


def arousal_from_prosody(
    energy_mean_db: float | None,
    f0_hz_median: float | None,
    words_per_sec: float | None,
) -> float | None:
    parts: list[float] = []
    if energy_mean_db is not None:
        # типичный RMS dB для речи ~ -35..-10
        parts.append(max(0.0, min(1.0, (energy_mean_db + 40.0) / 30.0)))
    if f0_hz_median is not None:
        parts.append(max(0.0, min(1.0, (f0_hz_median - 80.0) / 220.0)))
    if words_per_sec is not None:
        parts.append(max(0.0, min(1.0, words_per_sec / 4.0)))
    if not parts:
        return None
    return sum(parts) / len(parts)


def word_count(text: str) -> int:
    return len(re.findall(r"\w+", text, flags=re.UNICODE))
