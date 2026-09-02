"""GigaAM ASR: всегда transcribe_longform(..., word_timestamps=True). Нужен HF_TOKEN (VAD)."""

from __future__ import annotations

from pathlib import Path

from app.audio import infer_device
from app.engines.base import Word


class GigaAMASR:
    def __init__(
        self,
        model_name: str,
        models_dir: str,
        hf_token: str,
        device: str | None = None,
    ) -> None:
        if not hf_token:
            raise RuntimeError(
                "GigaAM unavailable: HF_TOKEN is empty. "
                "Accept the pyannote/segmentation-3.0 license on Hugging Face and set HF_TOKEN."
            )
        import os

        import gigaam

        os.environ["HF_TOKEN"] = hf_token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token
        if device is None:
            device, _dtype = infer_device()
        Path(models_dir).mkdir(parents=True, exist_ok=True)
        self._model = gigaam.load_model(
            model_name,
            device=device,
            download_root=str(models_dir),
            fp16_encoder=device != "cpu",
        )
        from gigaam.vad_utils import get_pipeline

        get_pipeline(self._model._device)

    def words(self, wav_path: str) -> list[Word]:
        result = self._model.transcribe_longform(wav_path, word_timestamps=True)
        segments = getattr(result, "segments", result)
        out: list[Word] = []
        for segment in segments:
            words = getattr(segment, "words", None)
            if words:
                for word in words:
                    text = (word.text or "").strip()
                    if text:
                        out.append(
                            Word(start=float(word.start), end=float(word.end), text=text)
                        )
                continue
            text = (getattr(segment, "text", None) or "").strip()
            if text:
                out.append(
                    Word(
                        start=float(segment.start),
                        end=float(segment.end),
                        text=text,
                    )
                )
        return out
