"""Устройство инференса, tmp аудио, ffmpeg, длительность."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import soundfile as sf

TMP_PREFIX = "tmp_"


def _cuda_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def infer_device(preference: str | None = None) -> tuple[str, str]:
    """cpu/float32 или cuda/float16. preference: auto | cpu | cuda (иначе DEVICE из env)."""
    raw = preference if preference is not None else os.environ.get("DEVICE", "auto")
    choice = (raw or "auto").strip().lower()
    if choice in {"", "auto"}:
        if _cuda_available():
            return "cuda", "float16"
        return "cpu", "float32"
    if choice == "cpu":
        return "cpu", "float32"
    if choice == "cuda":
        if _cuda_available():
            return "cuda", "float16"
        raise RuntimeError(
            "DEVICE=cuda, but CUDA is not available. "
            "Need an NVIDIA driver, nvidia-container-toolkit (in Docker), "
            "and a CUDA-enabled PyTorch wheel."
        )
    raise ValueError(f"DEVICE must be auto, cpu, or cuda (got {raw!r})")


def tmp_dir(task_id: str) -> Path:
    return Path(f"{TMP_PREFIX}{task_id}")


def create_tmp(task_id: str) -> Path:
    path = tmp_dir(task_id)
    path.mkdir(parents=True, exist_ok=True)
    return path


def cleanup_tmp(task_id: str) -> None:
    path = tmp_dir(task_id)
    shutil.rmtree(path, ignore_errors=True)


def cleanup_all_tmp(root: str | Path | None = None) -> None:
    base = Path.cwd() if root is None else Path(root)
    for path in base.glob(f"{TMP_PREFIX}*"):
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)


def audio_duration_sec(wav_path: str | Path) -> float:
    info = sf.info(str(wav_path))
    return float(info.duration)


def _run_ffmpeg(src: Path, dst: Path) -> None:
    try:
        completed = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(src),
                "-ac",
                "1",
                "-ar",
                "16000",
                str(dst),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg не найден") from exc
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or "ffmpeg failed")


CONVERT_SUFFIXES = {".mp3", ".m4a"}


def prepare_wav(src: str | Path, task_id: str) -> Path:
    """Кладёт WAV в tmp_<task_id>/audio.wav. MP3/M4A конвертирует через ffmpeg."""
    src_path = Path(src)
    dest_dir = create_tmp(task_id)
    dest = dest_dir / "audio.wav"
    suffix = src_path.suffix.lower()
    if suffix == ".wav":
        shutil.copy2(src_path, dest)
        return dest
    if suffix in CONVERT_SUFFIXES:
        _run_ffmpeg(src_path, dest)
        return dest
    raise ValueError(f"unsupported audio format: {suffix}")
