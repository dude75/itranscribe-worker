"""PyAnnote speaker-diarization-3.1. Нужен HF_TOKEN."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from app.audio import infer_device
from app.engines.base import DiarizationSegment

log = logging.getLogger(__name__)


def _parameter_device(module: Any) -> Any:
    import torch

    parameters = getattr(module, "parameters", None)
    if callable(parameters):
        try:
            return next(parameters()).device
        except (StopIteration, TypeError, AttributeError):
            pass
    device = getattr(module, "device", None)
    if isinstance(device, torch.device):
        return device
    if isinstance(device, str) and device:
        return torch.device(device)
    return None


def _pipeline_parts(pipeline: Any) -> list[tuple[str, Any]]:
    parts: list[tuple[str, Any]] = [("pipeline", pipeline)]
    segmentation = getattr(pipeline, "_segmentation", None)
    if segmentation is not None:
        parts.append(("_segmentation", getattr(segmentation, "model", segmentation)))
    embedding = getattr(pipeline, "_embedding", None)
    if embedding is not None:
        parts.append(
            (
                "_embedding",
                getattr(embedding, "model_", None)
                or getattr(embedding, "model", None)
                or embedding,
            )
        )
    return parts


def move_pipeline_to_device(pipeline: Any, device: str) -> Any:
    """Переносит пайплайн и проверяет, что сегментация и эмбеддинги на том же устройстве."""
    import torch

    target = torch.device(device)
    try:
        pipeline = pipeline.to(target)
    except Exception as exc:
        raise RuntimeError(
            f"PyAnnote unavailable: failed to move pipeline to {target}: {exc}"
        ) from exc

    mismatched: list[str] = []
    for name, module in _pipeline_parts(pipeline):
        actual = _parameter_device(module)
        if actual is not None and actual.type != target.type:
            mismatched.append(f"{name}={actual}")
    if mismatched:
        raise RuntimeError(
            "PyAnnote unavailable: components not on "
            f"{target}: {', '.join(mismatched)}"
        )
    log.info("pyannote ready device=%s", target)
    return pipeline


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
        self._pipeline = move_pipeline_to_device(pipeline, device)

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
