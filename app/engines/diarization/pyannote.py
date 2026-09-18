"""PyAnnote speaker-diarization-3.1. Нужен HF_TOKEN."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import soundfile as sf

from app.audio import infer_device
from app.engines.base import DiarizationSegment
from app.engines.hf_offline import call_with_local_files_only

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


def _soundfile_waveform(path: str) -> tuple[Any, int]:
    import torch

    data, sample_rate = sf.read(str(path), always_2d=True)
    return torch.from_numpy(data.T.copy()).float(), int(sample_rate)


def _is_torchcodec_error(exc: BaseException) -> bool:
    return "torchcodec" in str(exc).lower()


def install_soundfile_audio_fallback() -> None:
    """PyAnnote Audio I/O через soundfile, если torchcodec не сходится с ffmpeg на хосте."""
    import pyannote.audio.core.io as io_mod
    from pyannote.audio.core.io import Audio

    if getattr(Audio, "_itranscribe_soundfile", False):
        return

    original_call = Audio.__call__
    original_crop = Audio.crop
    original_duration = Audio.get_duration
    original_meta = io_mod.get_audio_metadata

    def get_audio_metadata(file: Any) -> Any:
        try:
            return original_meta(file)
        except RuntimeError as exc:
            if not _is_torchcodec_error(exc):
                raise
            path = file["audio"] if isinstance(file, dict) else file
            info = sf.info(str(path))
            return type(
                "SoundfileMeta",
                (),
                {
                    "duration_seconds_from_header": float(info.duration),
                    "sample_rate": int(info.samplerate),
                },
            )()

    def __call__(self, file: Any) -> Any:
        try:
            return original_call(self, file)
        except RuntimeError as exc:
            if not _is_torchcodec_error(exc):
                raise
            validated = self.validate_file(file)
            if "waveform" in validated:
                raise
            waveform, sample_rate = _soundfile_waveform(str(validated["audio"]))
            return self.downmix_and_resample(
                waveform, sample_rate, channel=validated.get("channel")
            )

    def get_duration(self, file: Any) -> float:
        try:
            return original_duration(self, file)
        except RuntimeError as exc:
            if not _is_torchcodec_error(exc):
                raise
            validated = self.validate_file(file)
            if "waveform" in validated:
                raise
            info = sf.info(str(validated["audio"]))
            return float(info.duration)

    def crop(self, file: Any, segment: Any, mode: str = "raise") -> Any:
        try:
            return original_crop(self, file, segment, mode=mode)
        except RuntimeError as exc:
            if not _is_torchcodec_error(exc):
                raise
            validated = self.validate_file(file)
            if "waveform" in validated:
                raise
            waveform, sample_rate = _soundfile_waveform(str(validated["audio"]))
            return original_crop(
                self,
                {"waveform": waveform, "sample_rate": sample_rate},
                segment,
                mode=mode,
            )

    io_mod.get_audio_metadata = get_audio_metadata
    Audio.__call__ = __call__
    Audio.get_duration = get_duration
    Audio.crop = crop
    Audio._itranscribe_soundfile = True


def _waveform_from_wav(wav_path: str, pipeline: Any) -> dict[str, Any]:
    """soundfile + тензор, без torchcodec (он требует конкретную major-версию ffmpeg)."""
    import torch

    data, sample_rate = sf.read(wav_path, always_2d=True)
    waveform = torch.from_numpy(data.T.copy()).float()
    device = _parameter_device(pipeline)
    if device is not None:
        waveform = waveform.to(device)
    return {"waveform": waveform, "sample_rate": int(sample_rate)}


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
        install_soundfile_audio_fallback()
        pipeline = call_with_local_files_only(
            Pipeline.from_pretrained,
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
        output = self._pipeline(_waveform_from_wav(wav_path, self._pipeline), **kwargs)
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
