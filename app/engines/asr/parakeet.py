"""Parakeet TDT ASR через NeMo. Chunking для аудио длиннее PARAKEET_CHUNK_SEC (~24 мин)."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from app.audio import audio_duration_sec, infer_device
from app.engines.base import Word, words_from_parakeet_timestamps
from app.engines.hf_offline import call_with_local_files_only

# NeMo Parakeet ~24 min на один вызов transcribe(); берём запас ниже 1440 с.
DEFAULT_CHUNK_SEC = 1380.0


def _set_decoding_flag(node, key: str, value) -> None:
    if node is None:
        return
    try:
        from omegaconf import open_dict
    except ImportError:
        open_dict = None
    if open_dict is not None:
        try:
            with open_dict(node):
                setattr(node, key, value)
            return
        except Exception:
            pass
    setattr(node, key, value)


def _configure_parakeet_decoding(model) -> None:
    """TDT CUDA-graph decoder + timestamps=True падает illegal memory access на CUDA 12.9.

    timestamps=True в NeMo вызывает change_decoding_strategy() и собирает декодер заново.
    Флаги должны быть в cfg, иначе graphs включатся снова.
    """
    decoding = getattr(getattr(model, "cfg", None), "decoding", None)
    if decoding is None:
        return
    _set_decoding_flag(decoding, "compute_timestamps", True)
    _set_decoding_flag(getattr(decoding, "greedy", None), "use_cuda_graph_decoder", False)
    _set_decoding_flag(getattr(decoding, "beam", None), "allow_cuda_graphs", False)
    change = getattr(model, "change_decoding_strategy", None)
    if callable(change):
        change(decoding, verbose=False)


def _extract_wav_segment(
    src: Path,
    dest: Path,
    start_sec: float,
    duration_sec: float,
) -> None:
    try:
        completed = subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-y",
                "-ss",
                str(start_sec),
                "-t",
                str(duration_sec),
                "-i",
                str(src),
                "-ac",
                "1",
                "-ar",
                "16000",
                str(dest),
            ],
            check=False,
            capture_output=True,
            text=True,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg не найден") from exc
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or "ffmpeg failed")


class ParakeetASR:
    def __init__(
        self,
        model_name: str,
        models_dir: str,
        device: str | None = None,
        chunk_sec: float = DEFAULT_CHUNK_SEC,
        hf_token: str | None = None,
    ) -> None:
        import os

        import torch

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

        import nemo.collections.asr as nemo_asr

        model = call_with_local_files_only(
            nemo_asr.models.ASRModel.from_pretrained,
            model_name,
        )
        self._model = model.to(device).eval()
        _configure_parakeet_decoding(self._model)
        self._time_stride = float(getattr(model, "window_stride", 0.01)) * 8.0
        self._chunk_sec = float(chunk_sec) if chunk_sec > 0 else DEFAULT_CHUNK_SEC

    def words(self, wav_path: str) -> list[Word]:
        duration = audio_duration_sec(wav_path)
        if duration <= self._chunk_sec:
            return self._transcribe_file(wav_path, time_offset=0.0, segment_id=0)

        wav = Path(wav_path)
        chunk_dir = wav.parent / f"{wav.stem}_parakeet_chunks"
        chunk_dir.mkdir(parents=True, exist_ok=True)
        all_words: list[Word] = []
        segment_id = 0
        start = 0.0
        try:
            while start < duration:
                chunk_dur = min(self._chunk_sec, duration - start)
                chunk_path = chunk_dir / f"chunk_{segment_id:04d}.wav"
                _extract_wav_segment(wav, chunk_path, start, chunk_dur)
                all_words.extend(
                    self._transcribe_file(
                        str(chunk_path),
                        time_offset=start,
                        segment_id=segment_id,
                    )
                )
                chunk_path.unlink(missing_ok=True)
                start += chunk_dur
                segment_id += 1
        finally:
            shutil.rmtree(chunk_dir, ignore_errors=True)
        return all_words

    def _transcribe_file(
        self,
        wav_path: str,
        *,
        time_offset: float,
        segment_id: int,
    ) -> list[Word]:
        hyps = self._model.transcribe(
            [wav_path],
            timestamps=True,
            batch_size=1,
            num_workers=0,
        )
        hyp = hyps[0]
        ts = getattr(hyp, "timestamp", None) or {}
        word_ts = ts.get("word") or []
        if word_ts:
            return words_from_parakeet_timestamps(
                word_ts,
                self._time_stride,
                time_offset=time_offset,
                segment_id=segment_id,
            )
        text = (getattr(hyp, "text", None) or str(hyp)).strip()
        if not text:
            return []
        chunk_dur = audio_duration_sec(wav_path)
        return [
            Word(
                start=time_offset,
                end=time_offset + chunk_dur,
                text=text,
                segment_id=segment_id,
            )
        ]
