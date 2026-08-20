"""PyAnnote speaker-diarization-3.1. Нужен HF_TOKEN."""

from __future__ import annotations

from pathlib import Path

from app.audio import infer_device
from app.engines.base import DiarizationSegment


class PyannoteDiarizer:
    def __init__(
        self,
        model_name: str,
        models_dir: str,
        hf_token: str,
        device: str | None = None,
    ) -> None:
        if not hf_token:
            raise RuntimeError(
                "PyAnnote unavailable: HF_TOKEN is empty. "
                "Accept the model license on Hugging Face and set HF_TOKEN."
            )
        from pyannote.audio import Pipeline

        if device is None:
            device, _dtype = infer_device()
        Path(models_dir).mkdir(parents=True, exist_ok=True)
        pipeline = Pipeline.from_pretrained(
            model_name,
            token=hf_token,
            cache_dir=str(models_dir),
        )
        if pipeline is None:
            raise RuntimeError(
                "PyAnnote unavailable: pipeline did not load (token or license)."
            )
        try:
            import torch

            pipeline.to(torch.device(device))
        except Exception:
            pass
        self._pipeline = pipeline

    def segments(self, wav_path: str, **kwargs) -> list[DiarizationSegment]:
        output = self._pipeline(wav_path, **kwargs)
        annotation = output
        if hasattr(output, "speaker_diarization"):
            annotation = output.speaker_diarization
        out: list[DiarizationSegment] = []
        if hasattr(annotation, "itertracks"):
            for turn, _, speaker in annotation.itertracks(yield_label=True):
                out.append(
                    DiarizationSegment(
                        start=float(turn.start),
                        end=float(turn.end),
                        speaker=str(speaker),
                    )
                )
            return out
        raise RuntimeError("PyAnnote returned an unexpected diarization type")
