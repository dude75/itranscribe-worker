"""Устройство инференса, tmp аудио, ffmpeg, длительность."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import BinaryIO

import soundfile as sf

TMP_PREFIX = "tmp_"
TMP_SUBDIR = "tmp"
UPLOAD_SCRATCH_PREFIX = ".upload_"
UPLOAD_WRITE_CHUNK = 1024 * 1024


class PayloadTooLarge(Exception):
    code = "payload_too_large"


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


def _resolve_data_dir(data_dir: str | Path | None = None) -> Path:
    if data_dir is not None:
        return Path(data_dir)
    from app.config import get_settings

    return Path(get_settings().DATA_DIR)


def tmp_root(data_dir: str | Path | None = None) -> Path:
    return _resolve_data_dir(data_dir) / TMP_SUBDIR


def tmp_dir(task_id: str, data_dir: str | Path | None = None) -> Path:
    return tmp_root(data_dir) / task_id


def create_tmp(task_id: str, data_dir: str | Path | None = None) -> Path:
    path = tmp_dir(task_id, data_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def cleanup_tmp(task_id: str, data_dir: str | Path | None = None) -> None:
    path = tmp_dir(task_id, data_dir)
    shutil.rmtree(path, ignore_errors=True)


def cleanup_all_tmp(data_dir: str | Path | None = None) -> int:
    """Удаляет `{DATA_DIR}/tmp/*`. Возвращает число снятых каталогов задач."""
    root = tmp_root(data_dir)
    if not root.is_dir():
        return 0
    removed = 0
    for path in root.iterdir():
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
            removed += 1
    return removed


def cleanup_tmp_except(keep_ids: set[str], data_dir: str | Path | None = None) -> int:
    """Снимает каталоги `{DATA_DIR}/tmp/<id>/`, кроме keep_ids. Возвращает число удалённых."""
    root = tmp_root(data_dir)
    if not root.is_dir():
        return 0
    removed = 0
    for path in root.iterdir():
        if path.is_dir() and path.name not in keep_ids:
            shutil.rmtree(path, ignore_errors=True)
            removed += 1
    return removed


def cleanup_legacy_cwd_tmp(root: str | Path | None = None) -> int:
    """Удаляет устаревшие `tmp_*` в CWD (старый layout). Возвращает число каталогов."""
    base = Path.cwd() if root is None else Path(root)
    removed = 0
    for path in base.glob(f"{TMP_PREFIX}*"):
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
            removed += 1
    return removed


def write_upload_limited(fileobj: BinaryIO, dest: Path, max_bytes: int) -> int:
    """Пишет fileobj на dest чанками. При превышении max_bytes удаляет dest и бросает PayloadTooLarge."""
    if max_bytes <= 0:
        raise ValueError("max_bytes must be > 0")
    written = 0
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        with dest.open("wb") as out:
            while True:
                chunk = fileobj.read(min(UPLOAD_WRITE_CHUNK, max_bytes - written + 1))
                if not chunk:
                    break
                written += len(chunk)
                if written > max_bytes:
                    raise PayloadTooLarge()
                out.write(chunk)
    except Exception:
        dest.unlink(missing_ok=True)
        raise
    return written


def place_upload(src: Path, dest: Path) -> None:
    """Scratch `.upload_*` переименовывает; иначе копирует. dest.parent должен существовать."""
    if src.name.startswith(UPLOAD_SCRATCH_PREFIX):
        src.replace(dest)
    else:
        shutil.copy2(src, dest)


def cleanup_upload_scratch(data_dir: str | Path | None = None) -> int:
    """Удаляет хвосты `{DATA_DIR}/.upload_*`. Не входит в счётчик purged_tmp."""
    base = _resolve_data_dir(data_dir)
    if not base.is_dir():
        return 0
    removed = 0
    for path in base.glob(f"{UPLOAD_SCRATCH_PREFIX}*"):
        if path.is_file():
            path.unlink(missing_ok=True)
            removed += 1
    return removed


def audio_duration_sec(wav_path: str | Path) -> float:
    info = sf.info(str(wav_path))
    return float(info.duration)


class FfmpegTimeout(RuntimeError):
    """ffmpeg не уложился в FFMPEG_TIMEOUT_SEC."""


def _ffmpeg_timeout_sec(explicit: float | int | None) -> float | None:
    if explicit is None:
        from app.config import get_settings

        value = float(get_settings().FFMPEG_TIMEOUT_SEC)
    else:
        value = float(explicit)
    if value <= 0:
        return None
    return value


def _run_ffmpeg(src: Path, dst: Path, timeout_sec: float | int | None) -> None:
    timeout = _ffmpeg_timeout_sec(timeout_sec)
    try:
        completed = subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
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
            stdin=subprocess.DEVNULL,
            start_new_session=True,
            timeout=timeout,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg не найден") from exc
    except subprocess.TimeoutExpired as exc:
        raise FfmpegTimeout(f"ffmpeg timed out after {timeout}s") from exc
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or "ffmpeg failed")


PREPARE_SUFFIXES = {".wav", ".mp3", ".m4a"}


def prepare_wav(
    src: str | Path,
    task_id: str,
    data_dir: str | Path | None = None,
    timeout_sec: float | int | None = None,
) -> Path:
    """Кладёт моно 16 кГц WAV в {DATA_DIR}/tmp/<task_id>/audio.wav через ffmpeg."""
    src_path = Path(src)
    dest_dir = create_tmp(task_id, data_dir)
    dest = dest_dir / "audio.wav"
    suffix = src_path.suffix.lower()
    if suffix not in PREPARE_SUFFIXES:
        raise ValueError(f"unsupported audio format: {suffix}")
    _run_ffmpeg(src_path, dest, timeout_sec)
    return dest
