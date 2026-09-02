"""Faster-Whisper ASR. Язык не фиксируется."""

from __future__ import annotations

from pathlib import Path

from app.audio import infer_device
from app.engines.base import Word, words_from_asr_segments


class FasterWhisperASR:
    def __init__(
        self,
        model_name: str,
        models_dir: str,
        device: str | None = None,
        compute_type: str | None = None,
    ) -> None:
        from faster_whisper import WhisperModel

        if device is None or compute_type is None:
            device, compute_type = infer_device()
        Path(models_dir).mkdir(parents=True, exist_ok=True)
        self._model = WhisperModel(
            model_name,
            device=device,
            compute_type=compute_type,
            download_root=str(models_dir),
        )

    def words(self, wav_path: str) -> list[Word]:
        segments, _info = self._model.transcribe(
            wav_path,
            word_timestamps=True,
        )
        return words_from_asr_segments(segments)
