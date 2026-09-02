from __future__ import annotations

import io
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from app.audio import (
    FfmpegTimeout,
    PayloadTooLarge,
    audio_duration_sec,
    cleanup_all_tmp,
    cleanup_tmp,
    create_tmp,
    infer_device,
    place_upload,
    prepare_wav,
    tmp_dir,
    write_upload_limited,
)
from app.config import get_settings

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


def _assert_mono_16k(path: Path) -> None:
    info = sf.info(str(path))
    assert info.channels == 1
    assert info.samplerate == SR


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg не установлен")
def test_prepare_wav_normalizes_into_tmp(wav_file: Path, tmp_path: Path) -> None:
    task_id = "audio-wav-1"
    try:
        dest = prepare_wav(wav_file, task_id, tmp_path)
        assert dest.exists()
        assert dest.parent == tmp_path / "tmp" / task_id
        assert dest.parent == tmp_dir(task_id, tmp_path)
        assert not (Path.cwd() / f"tmp_{task_id}").exists()
        _assert_mono_16k(dest)
        assert audio_duration_sec(dest) == pytest.approx(DURATION, abs=1e-3)
    finally:
        cleanup_tmp(task_id, tmp_path)
    assert not tmp_dir(task_id, tmp_path).exists()


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg не установлен")
def test_prepare_wav_downmixes_stereo(tmp_path: Path) -> None:
    stereo_sr = 48000
    stereo = tmp_path / "stereo.wav"
    samples = np.stack(
        [
            np.full(int(stereo_sr * DURATION), 0.4, dtype=np.float32),
            np.full(int(stereo_sr * DURATION), -0.4, dtype=np.float32),
        ],
        axis=1,
    )
    sf.write(stereo, samples, stereo_sr)
    task_id = "audio-stereo-1"
    try:
        dest = prepare_wav(stereo, task_id, tmp_path)
        _assert_mono_16k(dest)
        assert audio_duration_sec(dest) == pytest.approx(DURATION, abs=0.05)
    finally:
        cleanup_tmp(task_id, tmp_path)


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
        _assert_mono_16k(dest)
        assert audio_duration_sec(dest) == pytest.approx(DURATION, abs=0.05)
    finally:
        cleanup_tmp(task_id, tmp_path)
    assert not tmp_dir(task_id, tmp_path).exists()


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg не установлен")
def test_m4a_converts_to_wav(wav_file: Path, tmp_path: Path) -> None:
    m4a = tmp_path / "sample.m4a"
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
        _assert_mono_16k(dest)
        assert audio_duration_sec(dest) == pytest.approx(DURATION, abs=0.05)
    finally:
        cleanup_tmp(task_id, tmp_path)
    assert not tmp_dir(task_id, tmp_path).exists()


class _FfmpegOk:
    returncode = 0
    stderr = ""


def test_prepare_wav_ffmpeg_uses_nostdin_and_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wav = tmp_path / "sample.wav"
    wav.write_bytes(b"x")
    captured: dict[str, object] = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        return _FfmpegOk()

    monkeypatch.setattr("app.audio.subprocess.run", fake_run)
    task_id = "audio-ffmpeg-flags"
    try:
        prepare_wav(wav, task_id, tmp_path, timeout_sec=42)
    finally:
        cleanup_tmp(task_id, tmp_path)

    cmd = captured["cmd"]
    assert cmd[:3] == ["ffmpeg", "-nostdin", "-y"]
    assert "-ac" in cmd and cmd[cmd.index("-ac") + 1] == "1"
    assert "-ar" in cmd and cmd[cmd.index("-ar") + 1] == "16000"
    kwargs = captured["kwargs"]
    assert kwargs["timeout"] == 42
    assert kwargs["stdin"] is subprocess.DEVNULL
    assert kwargs["start_new_session"] is True


def test_prepare_wav_ffmpeg_timeout_zero_disables(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mp3 = tmp_path / "sample.mp3"
    mp3.write_bytes(b"x")
    captured: dict[str, object] = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        return _FfmpegOk()

    monkeypatch.setattr("app.audio.subprocess.run", fake_run)
    task_id = "audio-ffmpeg-no-timeout"
    try:
        prepare_wav(mp3, task_id, tmp_path, timeout_sec=0)
    finally:
        cleanup_tmp(task_id, tmp_path)

    assert captured["kwargs"]["timeout"] is None


def test_prepare_wav_ffmpeg_timeout_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mp3 = tmp_path / "sample.mp3"
    mp3.write_bytes(b"x")

    def fake_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=args[0], timeout=kwargs["timeout"])

    monkeypatch.setattr("app.audio.subprocess.run", fake_run)
    task_id = "audio-ffmpeg-timeout"
    try:
        with pytest.raises(FfmpegTimeout, match="timed out after 1"):
            prepare_wav(mp3, task_id, tmp_path, timeout_sec=1)
    finally:
        cleanup_tmp(task_id, tmp_path)


def test_pipeline_maps_ffmpeg_timeout_to_task_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app.config import Settings
    from app.pipeline import run_pipeline
    from app.schemas import AsrModel, TaskStatus
    from app.tasks import TaskStore

    db = tmp_path / "tasks.db"
    store = TaskStore(str(db))
    settings = Settings(
        DATA_DIR=str(tmp_path),
        SQLITE_PATH=str(db),
        LOG_DIR=str(tmp_path / "logs"),
        PERFORMANCE_LOG=str(tmp_path / "logs" / "performance_log.csv"),
        FFMPEG_TIMEOUT_SEC=5,
        _env_file=None,
    )
    upload = tmp_path / "clip.mp3"
    upload.write_bytes(b"x")
    store.create(
        "ffmpeg-timeout-task",
        AsrModel.whisper,
        None,
        settings.WHISPER_MODEL,
        None,
        str(upload),
    )

    def boom(*_args, **_kwargs):
        raise FfmpegTimeout("ffmpeg timed out after 5s")

    monkeypatch.setattr("app.pipeline.prepare_wav", boom)
    try:
        run_pipeline(store, settings, "ffmpeg-timeout-task")
        done = store.get("ffmpeg-timeout-task")
        assert done is not None
        assert done.status is TaskStatus.error
        assert done.error is not None
        assert done.error["code"] == "ffmpeg_timeout"
        assert done.error["message"] == "ffmpeg timed out after 5s"
    finally:
        store.close()


def test_run_pipeline_does_not_call_infer_device(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app.config import Settings
    from app.pipeline import run_pipeline
    from app.schemas import AsrModel, TaskStatus
    from app.tasks import TaskStore

    def boom(*_args, **_kwargs):
        raise RuntimeError("DEVICE=cuda, but CUDA is not available")

    monkeypatch.setattr("app.audio.infer_device", boom)

    db = tmp_path / "tasks.db"
    store = TaskStore(str(db))
    settings = Settings(
        DATA_DIR=str(tmp_path),
        SQLITE_PATH=str(db),
        LOG_DIR=str(tmp_path / "logs"),
        PERFORMANCE_LOG=str(tmp_path / "logs" / "performance_log.csv"),
        _env_file=None,
    )
    store.create(
        "no-device-task",
        AsrModel.whisper,
        None,
        settings.WHISPER_MODEL,
        None,
        str(tmp_path / "missing.wav"),
    )
    try:
        run_pipeline(store, settings, "no-device-task")
        done = store.get("no-device-task")
        assert done is not None
        assert done.status is TaskStatus.error
        assert done.error is not None
        assert done.error["code"] == "missing_upload"
    finally:
        store.close()


def test_write_upload_limited_writes_chunks(tmp_path: Path) -> None:
    dest = tmp_path / "out.bin"
    written = write_upload_limited(io.BytesIO(b"abc"), dest, 10)
    assert written == 3
    assert dest.read_bytes() == b"abc"


def test_write_upload_limited_too_large_removes_dest(tmp_path: Path) -> None:
    dest = tmp_path / "out.bin"
    with pytest.raises(PayloadTooLarge):
        write_upload_limited(io.BytesIO(b"abcdef"), dest, 4)
    assert not dest.exists()


def test_place_upload_renames_scratch(tmp_path: Path) -> None:
    src = tmp_path / ".upload_abc.wav"
    src.write_bytes(b"hi")
    dest = tmp_path / "tmp" / "id" / "upload.wav"
    dest.parent.mkdir(parents=True)
    place_upload(src, dest)
    assert dest.read_bytes() == b"hi"
    assert not src.exists()


def test_place_upload_copies_regular_file(tmp_path: Path) -> None:
    src = tmp_path / "sample.wav"
    src.write_bytes(b"hi")
    dest = tmp_path / "tmp" / "id" / "upload.wav"
    dest.parent.mkdir(parents=True)
    place_upload(src, dest)
    assert dest.read_bytes() == b"hi"
    assert src.exists()
