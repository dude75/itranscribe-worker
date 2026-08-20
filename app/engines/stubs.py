"""Заглушки ASR и диаризации по Protocol из T-01."""

from __future__ import annotations

from app.audio import audio_duration_sec
from app.engines.base import DiarizationSegment, Word


class StubASR:
    def words(self, wav_path: str) -> list[Word]:
        duration = audio_duration_sec(wav_path)
        mid = duration / 2.0
        return [
            Word(start=0.0, end=mid, text="привет"),
            Word(start=mid, end=duration, text="мир"),
        ]


class StubDiarization:
    def segments(self, wav_path: str) -> list[DiarizationSegment]:
        duration = audio_duration_sec(wav_path)
        mid = duration / 2.0
        return [
            DiarizationSegment(start=0.0, end=mid, speaker="A"),
            DiarizationSegment(start=mid, end=duration, speaker="B"),
        ]
