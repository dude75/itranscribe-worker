"""Просодия по слайсам WAV (CPU)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import librosa
import numpy as np
import soundfile as sf

from app.schemas import ProsodyFeatures, TranscriptLine
from app.tone.audio_io import read_segment
from app.tone.emotions import word_count


@dataclass(frozen=True)
class _SegmentProsody:
    energy_mean_db: float | None = None
    energy_max_db: float | None = None
    f0_hz_median: float | None = None
    words_per_sec: float | None = None
    pause_before_sec: float | None = None


def _rms_db(frame: np.ndarray) -> float:
    if frame.size == 0:
        return float("-inf")
    rms = float(np.sqrt(np.mean(np.square(frame), dtype=np.float64)))
    if rms <= 1e-12:
        return float("-inf")
    return 20.0 * math.log10(rms)


def _resample_chunk(chunk: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr or chunk.size == 0:
        return chunk
    return librosa.resample(chunk, orig_sr=orig_sr, target_sr=target_sr)


def _prosody_from_chunk(
    chunk: np.ndarray,
    sr: int,
    *,
    features: frozenset[str],
    line: TranscriptLine,
    start: float,
    end: float,
    index: int,
    lines: list[TranscriptLine],
) -> _SegmentProsody:
    energy_mean: float | None = None
    energy_max: float | None = None
    if "energy" in features and chunk.size:
        hop = max(1, sr // 100)
        frames = librosa.util.frame(chunk, frame_length=hop * 2, hop_length=hop)
        if frames.size:
            dbs = [_rms_db(frames[:, i]) for i in range(frames.shape[1])]
            dbs = [d for d in dbs if math.isfinite(d)]
            if dbs:
                energy_mean = float(np.mean(dbs))
                energy_max = float(np.max(dbs))

    f0_med: float | None = None
    if "f0" in features and chunk.size > sr // 10:
        f0 = librosa.yin(
            chunk,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"),
            sr=sr,
        )
        voiced = f0[(f0 > 0) & np.isfinite(f0)]
        if voiced.size:
            f0_med = float(np.median(voiced))

    wps: float | None = None
    if "tempo" in features:
        duration = end - start
        words = word_count(line.text)
        if duration > 0 and words:
            wps = words / duration

    pause: float | None = None
    if "pauses" in features and index > 0:
        prev = lines[index - 1]
        if line.speaker == prev.speaker:
            pause = max(0.0, line.start - prev.end)

    return _SegmentProsody(
        energy_mean_db=energy_mean,
        energy_max_db=energy_max,
        f0_hz_median=f0_med,
        words_per_sec=wps,
        pause_before_sec=pause,
    )


def compute_prosody(
    wav_path: str,
    lines: list[TranscriptLine],
    *,
    features: frozenset[str],
    sample_rate: int = 16_000,
) -> list[ProsodyFeatures | None]:
    if not lines or not features:
        return [None] * len(lines)

    needs_audio = "energy" in features or "f0" in features
    raw_segments: list[_SegmentProsody] = []

    if needs_audio:
        with sf.SoundFile(wav_path) as handle:
            native_sr = handle.samplerate
            for index, line in enumerate(lines):
                start = max(0.0, line.start)
                end = max(start, line.end)
                if end > start:
                    chunk = read_segment(handle, start, end)
                    chunk = _resample_chunk(chunk, native_sr, sample_rate)
                else:
                    chunk = np.array([], dtype=np.float32)
                sr = sample_rate
                raw_segments.append(
                    _prosody_from_chunk(
                        chunk,
                        sr,
                        features=features,
                        line=line,
                        start=start,
                        end=end,
                        index=index,
                        lines=lines,
                    )
                )
    else:
        for index, line in enumerate(lines):
            start = max(0.0, line.start)
            end = max(start, line.end)
            raw_segments.append(
                _prosody_from_chunk(
                    np.array([], dtype=np.float32),
                    sample_rate,
                    features=features,
                    line=line,
                    start=start,
                    end=end,
                    index=index,
                    lines=lines,
                )
            )

    return [
        ProsodyFeatures(
            energy_mean_db=seg.energy_mean_db,
            energy_max_db=seg.energy_max_db,
            f0_hz_median=seg.f0_hz_median,
            words_per_sec=seg.words_per_sec,
            pause_before_sec=seg.pause_before_sec,
        )
        for seg in raw_segments
    ]
