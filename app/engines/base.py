"""Протоколы движков ASR и диаризации. Без ML-импортов."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol, Sequence


@dataclass(frozen=True)
class Word:
    start: float
    end: float
    text: str
    segment_id: int | None = None


def words_from_asr_segments(segments: Iterable[object]) -> list[Word]:
    """Плоский список слов Whisper/GigaAM с индексом исходного сегмента."""
    out: list[Word] = []
    for segment_id, segment in enumerate(segments):
        words = getattr(segment, "words", None)
        if words:
            for word in words:
                text = (
                    getattr(word, "text", None) or getattr(word, "word", None) or ""
                ).strip()
                if not text:
                    continue
                out.append(
                    Word(
                        start=float(word.start),
                        end=float(word.end),
                        text=text,
                        segment_id=segment_id,
                    )
                )
            continue
        text = (getattr(segment, "text", None) or "").strip()
        if text:
            out.append(
                Word(
                    start=float(segment.start),
                    end=float(segment.end),
                    text=text,
                    segment_id=segment_id,
                )
            )
    return out


@dataclass(frozen=True)
class DiarizationSegment:
    start: float
    end: float
    speaker: str


class ASREngine(Protocol):
    def words(self, wav_path: str) -> Sequence[Word]:
        """Транскрибация: слова с таймкодами."""
        ...


class DiarizationEngine(Protocol):
    def segments(self, wav_path: str) -> Sequence[DiarizationSegment]:
        """Диаризация: сегменты спикеров."""
        ...
