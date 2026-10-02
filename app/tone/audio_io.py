"""Чтение сегментов WAV без загрузки всего файла."""

from __future__ import annotations

import numpy as np
import soundfile as sf


def read_segment(handle: sf.SoundFile, start: float, end: float) -> np.ndarray:
    """Прочитать [start, end) секунд из уже открытого файла."""
    sr = handle.samplerate
    i0 = max(0, int(start * sr))
    i1 = max(i0, int(end * sr))
    handle.seek(i0)
    frames = i1 - i0
    data = handle.read(frames, dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    return data


def load_segment(wav_path: str, start: float, end: float) -> tuple[np.ndarray, int]:
    with sf.SoundFile(wav_path) as handle:
        sr = handle.samplerate
        data = read_segment(handle, start, end)
    return data, sr
