"""Протоколы движков ASR и диаризации. Без ML-импортов."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence


@dataclass(frozen=True)
class Word:
    start: float
    end: float
    text: str


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
