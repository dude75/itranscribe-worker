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


def words_from_parakeet_timestamps(
    word_ts: Iterable[dict],
    time_stride: float,
    *,
    time_offset: float = 0.0,
    segment_id: int = 0,
) -> list[Word]:
    """Слова из NeMo Parakeet hyp.timestamp['word'] с offset-ами в секундах."""
    out: list[Word] = []
    for stamp in word_ts:
        text = str(stamp.get("word") or stamp.get("char") or "").strip()
        if not text:
            continue
        if "start" in stamp and "end" in stamp:
            start = float(stamp["start"])
            end = float(stamp["end"])
        else:
            start = float(stamp.get("start_offset", 0)) * time_stride
            end = float(stamp.get("end_offset", 0)) * time_stride
        out.append(
            Word(
                start=time_offset + start,
                end=time_offset + end,
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
