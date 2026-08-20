"""Склейка слов ASR и сегментов диаризации (ТЗ §8)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from app.engines.base import DiarizationSegment, Word

SPEAKER_UNKNOWN = "speaker_unknown"


@dataclass(frozen=True)
class Utterance:
    speaker: str | None
    start: float
    end: float
    text: str


def _speaker_for_word(word: Word, segments: Sequence[DiarizationSegment]) -> str:
    for seg in segments:
        if word.start >= seg.start and word.end <= seg.end:
            return seg.speaker
    center = (word.start + word.end) / 2.0
    for seg in segments:
        if seg.start <= center <= seg.end:
            return seg.speaker
    return SPEAKER_UNKNOWN


def _canonical_names(labels: Sequence[str]) -> list[str]:
    mapping: dict[str, str] = {}
    result: list[str] = []
    for label in labels:
        if label == SPEAKER_UNKNOWN:
            result.append(SPEAKER_UNKNOWN)
            continue
        if label not in mapping:
            mapping[label] = f"speaker_{len(mapping)}"
        result.append(mapping[label])
    return result


def utterances_from_words(words: Sequence[Word]) -> list[Utterance]:
    """Транскрипт без диаризации: все слова одной репликой, спикер не назначается."""
    if not words:
        return []
    return [
        Utterance(
            speaker=None,
            start=words[0].start,
            end=words[-1].end,
            text=" ".join(word.text for word in words),
        )
    ]


def align(words: Sequence[Word], segments: Sequence[DiarizationSegment]) -> list[Utterance]:
    if not words:
        return []

    raw_speakers = [_speaker_for_word(word, segments) for word in words]
    speakers = _canonical_names(raw_speakers)

    utterances: list[Utterance] = []
    buf: list[Word] = [words[0]]
    current = speakers[0]
    for word, speaker in zip(words[1:], speakers[1:], strict=True):
        if speaker == current:
            buf.append(word)
            continue
        utterances.append(
            Utterance(
                speaker=current,
                start=buf[0].start,
                end=buf[-1].end,
                text=" ".join(w.text for w in buf),
            )
        )
        buf = [word]
        current = speaker
    utterances.append(
        Utterance(
            speaker=current,
            start=buf[0].start,
            end=buf[-1].end,
            text=" ".join(w.text for w in buf),
        )
    )
    return utterances
