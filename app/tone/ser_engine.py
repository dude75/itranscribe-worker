"""Speech emotion recognition (transformers audio-classification)."""

from __future__ import annotations

import logging
from typing import Protocol

import numpy as np
import torch
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

from app.tone.audio_io import load_segment
from app.tone.emotions import normalize_emotion_label

log = logging.getLogger(__name__)


class SerToneEngine(Protocol):
    def analyze_segment(self, audio: np.ndarray, sample_rate: int) -> tuple[str, float]: ...


class TransformersSerTone:
    def __init__(self, model_id: str, device: str) -> None:
        self._model_id = model_id
        self._device = device
        self._extractor = AutoFeatureExtractor.from_pretrained(model_id)
        self._model = AutoModelForAudioClassification.from_pretrained(model_id)
        self._model.to(device)
        self._model.eval()
        config = self._model.config
        id2label = getattr(config, "id2label", {}) or {}
        self._id2label = {int(k): v for k, v in id2label.items()}

    def analyze_segment(self, audio: np.ndarray, sample_rate: int) -> tuple[str, float]:
        if audio.size == 0:
            return "neutral", 0.0
        target_sr = self._extractor.sampling_rate
        if sample_rate != target_sr:
            import librosa

            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=target_sr)
            sample_rate = target_sr
        inputs = self._extractor(
            audio,
            sampling_rate=sample_rate,
            return_tensors="pt",
            padding=True,
        )
        inputs = {key: value.to(self._device) for key, value in inputs.items()}
        with torch.inference_mode():
            logits = self._model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).squeeze(0)
        score, index = torch.max(probs, dim=-1)
        label = self._id2label.get(int(index.item()), str(int(index.item())))
        return normalize_emotion_label(label), float(score.item())


