"""NVIDIA NeMo Sortformer. Clustering не подменяем."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from app.audio import infer_device
from app.engines.base import DiarizationSegment

# Streaming Sortformer, высокая латентность ≈ офлайн-батч (карточка модели, окно ~30 с).
_STREAMING_OFFLINE = {
    "chunk_len": 340,
    "chunk_right_context": 40,
    "fifo_len": 40,
    "spkcache_update_period": 340,
    "spkcache_len": 188,
}


class NemoUnavailable(RuntimeError):
    """Sortformer не загрузился; сервис остаётся живым."""


def _import_sortformer():
    try:
        from nemo.collections.asr.models import SortformerEncLabelModel

        return SortformerEncLabelModel
    except Exception as exc:
        raise NemoUnavailable(
            "Sortformer unavailable: SortformerEncLabelModel not in this NeMo build. "
            "Clustering is not used as a stand-in."
        ) from exc


def _is_number(value: Any) -> bool:
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def _collect_lines(raw: Any, out: list[Any]) -> None:
    if raw is None:
        return
    if isinstance(raw, str):
        out.append(raw)
        return
    if isinstance(raw, (list, tuple)):
        if (
            len(raw) >= 3
            and not isinstance(raw[0], (list, tuple))
            and _is_number(raw[0])
            and _is_number(raw[1])
        ):
            out.append(raw)
            return
        for item in raw:
            _collect_lines(item, out)


def segments_from_sortformer(raw: Any) -> list[DiarizationSegment]:
    lines: list[Any] = []
    _collect_lines(raw, lines)
    out: list[DiarizationSegment] = []
    for line in lines:
        if isinstance(line, (list, tuple)) and len(line) >= 3:
            start, end, speaker = line[0], line[1], line[2]
        else:
            parts = str(line).split()
            if len(parts) < 3:
                continue
            start, end, speaker = parts[0], parts[1], parts[2]
        try:
            start_f = float(start)
            end_f = float(end)
        except (TypeError, ValueError):
            continue
        out.append(
            DiarizationSegment(start=start_f, end=end_f, speaker=str(speaker))
        )
    out.sort(key=lambda segment: (segment.start, segment.end))
    return out


def _configure_streaming_offline(model: Any) -> None:
    modules = getattr(model, "sortformer_modules", None)
    if modules is None:
        return
    for key, value in _STREAMING_OFFLINE.items():
        if hasattr(modules, key):
            setattr(modules, key, value)
    checker = getattr(modules, "_check_streaming_parameters", None)
    if callable(checker):
        checker()


class NemoSortformerDiarizer:
    def __init__(
        self,
        model_name: str,
        models_dir: str,
        device: str | None = None,
        hf_token: str | None = None,
    ) -> None:
        sortformer_cls = _import_sortformer()
        if device is None:
            device, _dtype = infer_device()
        models_path = Path(models_dir).resolve()
        models_path.mkdir(parents=True, exist_ok=True)
        os.environ["NEMO_CACHE_DIR"] = str(models_path)
        os.environ["HF_HOME"] = str(models_path)
        os.environ["HF_HUB_CACHE"] = str(models_path / "hub")
        if hf_token:
            os.environ["HF_TOKEN"] = hf_token
            os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token

        import torch

        try:
            model = sortformer_cls.from_pretrained(
                model_name,
                map_location=torch.device(device),
                strict=False,
            )
        except Exception as exc:
            raise NemoUnavailable(f"Sortformer unavailable: {exc}") from exc
        model.eval()
        _configure_streaming_offline(model)
        self._model = model

    def segments(self, wav_path: str) -> list[DiarizationSegment]:
        try:
            raw = self._model.diarize(
                audio=wav_path,
                batch_size=1,
                num_workers=0,
                verbose=False,
            )
        except NemoUnavailable:
            raise
        except Exception as exc:
            raise NemoUnavailable(f"Sortformer unavailable: {exc}") from exc
        return segments_from_sortformer(raw)
