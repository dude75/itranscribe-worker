"""Text emotion / sentiment (transformers)."""

from __future__ import annotations

import logging
from typing import Protocol

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from app.tone.emotions import normalize_emotion_scores, valence_from_emotions

log = logging.getLogger(__name__)


class TextToneEngine(Protocol):
    def analyze_batch(self, texts: list[str]) -> list[dict[str, float]]: ...


class TransformersTextTone:
    def __init__(self, model_id: str, device: str) -> None:
        self._model_id = model_id
        self._device = device
        self._tokenizer = AutoTokenizer.from_pretrained(model_id)
        self._model = AutoModelForSequenceClassification.from_pretrained(model_id)
        self._model.to(device)
        self._model.eval()
        config = self._model.config
        id2label = getattr(config, "id2label", None) or {}
        num = int(getattr(config, "num_labels", len(id2label)) or 0)
        labels: list[str] = []
        for index in range(num):
            label = id2label.get(index, id2label.get(str(index), str(index)))
            labels.append(str(label))
        self._labels = labels or list(getattr(config, "label2id", {}).keys())

    def analyze_batch(self, texts: list[str]) -> list[dict[str, float]]:
        if not texts:
            return []
        batch_size = 32
        results: list[dict[str, float]] = []
        for offset in range(0, len(texts), batch_size):
            chunk = texts[offset : offset + batch_size]
            inputs = self._tokenizer(
                chunk,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            inputs = {key: value.to(self._device) for key, value in inputs.items()}
            with torch.inference_mode():
                outputs = self._model(**inputs)
                logits = outputs.logits
                if logits.shape[-1] == 1:
                    probs = torch.sigmoid(logits).squeeze(-1)
                    for prob in probs.tolist():
                        results.append({"neutral": 1.0 - prob, "positive": prob})
                    continue
                probs = torch.sigmoid(logits)
            for row in probs:
                scores = {
                    self._labels[i]: float(row[i].item())
                    for i in range(min(len(self._labels), row.numel()))
                }
                results.append(normalize_emotion_scores(scores))
        return results


def emotions_to_tone_fields(emotions: dict[str, float]) -> tuple[float, dict[str, float]]:
    normalized = normalize_emotion_scores(emotions)
    return valence_from_emotions(normalized), normalized
