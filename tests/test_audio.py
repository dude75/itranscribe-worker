from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from app.config import get_settings
from app.audio import (
    audio_duration_sec,
    cleanup_all_tmp,
    cleanup_tmp,
    create_tmp,
    infer_device,
    prepare_wav,
    tmp_dir,
)

SR = 16000
DURATION = 0.5


@pytest.fixture
def wav_file(tmp_path: Path) -> Path:
    path = tmp_path / "sample.wav"
    samples = np.zeros(int(SR * DURATION), dtype=np.float32)
    sf.write(path, samples, SR)
    return path


def test_infer_device_cpu_without_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DEVICE", raising=False)
    device, dtype = infer_device("auto")
    assert device in {"cpu", "cuda"}
    if device == "cpu":
        assert dtype == "float32"
    else:
        assert dtype == "float16"


def test_infer_device_force_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEVICE", "cpu")
    device, dtype = infer_device()
    assert device == "cpu"
    assert dtype == "float32"
    device, dtype = infer_device("cpu")
    assert device == "cpu"
    assert dtype == "float32"


def test_infer_device_cuda_missing_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEVICE", "cuda")
    monkeypatch.setattr("app.audio._cuda_available", lambda: False)
    with pytest.raises(RuntimeError, match="DEVICE=cuda"):
        infer_device()
    with pytest.raises(RuntimeError, match="DEVICE=cuda"):
        infer_device("cuda")


def test_infer_device_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="DEVICE must be"):
        infer_device("mps")


def test_wav_duration_matches(wav_file: Path) -> None:
    duration = audio_duration_sec(wav_file)
    assert duration == pytest.approx(DURATION, abs=1e-3)


def test_tmp_lives_under_data_dir_not_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    get_settings.cache_clear()
    task_id = "path-check"
    try:
        path = create_tmp(task_id)
        assert path == tmp_path / "tmp" / task_id
        assert path == tmp_dir(task_id)
        assert not (Path.cwd() / f"tmp_{task_id}").exists()
    finally:
        cleanup_tmp(task_id)
        get_settings.cache_clear()


def test_prepare_wav_copies_into_tmp(wav_file: Path, tmp_path: Path) -> None:
    task_id = "audio-wav-1"
    try:
        dest = prepare_wav(wav_file, task_id, tmp_path)
        assert dest.exists()
        assert dest.parent == tmp_path / "tmp" / task_id
        assert dest.parent == tmp_dir(task_id, tmp_path)
        assert not (Path.cwd() / f"tmp_{task_id}").exists()
        assert audio_duration_sec(dest) == pytest.approx(DURATION, abs=1e-3)
    finally:
        cleanup_tmp(task_id, tmp_path)
    assert not tmp_dir(task_id, tmp_path).exists()


def test_cleanup_tmp_after_success_or_error(tmp_path: Path) -> None:
    task_id = "audio-cleanup-1"
    path = create_tmp(task_id, tmp_path)
    (path / "audio.wav").write_bytes(b"x")
    assert path.exists()
    cleanup_tmp(task_id, tmp_path)
    assert not path.exists()
    cleanup_tmp(task_id, tmp_path)


def test_cleanup_all_tmp_removes_leftovers(tmp_path: Path) -> None:
    create_tmp("left-1", tmp_path)
    create_tmp("left-2", tmp_path)
    assert (tmp_path / "tmp" / "left-1").is_dir()
    assert not (tmp_path / "tmp_left-1").exists()
    cleanup_all_tmp(tmp_path)
    assert list((tmp_path / "tmp").glob("*")) == []


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg не установлен")
def test_mp3_converts_to_wav(wav_file: Path, tmp_path: Path) -> None:
    mp3 = tmp_path / "sample.mp3"
    import subprocess

    subprocess.run(
        ["ffmpeg", "-y", "-i", str(wav_file), str(mp3)],
        check=True,
        capture_output=True,
    )
    task_id = "audio-mp3-1"
    try:
        dest = prepare_wav(mp3, task_id, tmp_path)
        assert dest.suffix == ".wav"
        assert dest.parent == tmp_path / "tmp" / task_id
        assert audio_duration_sec(dest) == pytest.approx(DURATION, abs=0.05)
    finally:
        cleanup_tmp(task_id, tmp_path)
    assert not tmp_dir(task_id, tmp_path).exists()


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg не установлен")
def test_m4a_converts_to_wav(wav_file: Path, tmp_path: Path) -> None:
    m4a = tmp_path / "sample.m4a"
    import subprocess

    subprocess.run(
        ["ffmpeg", "-y", "-i", str(wav_file), str(m4a)],
        check=True,
        capture_output=True,
    )
    task_id = "audio-m4a-1"
    try:
        dest = prepare_wav(m4a, task_id, tmp_path)
        assert dest.suffix == ".wav"
        assert dest.parent == tmp_path / "tmp" / task_id
        assert audio_duration_sec(dest) == pytest.approx(DURATION, abs=0.05)
    finally:
        cleanup_tmp(task_id, tmp_path)
    assert not tmp_dir(task_id, tmp_path).exists()
