from __future__ import annotations

import asyncio
import shutil
import sqlite3
import threading
import time
import uuid
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from app.audio import create_tmp, tmp_dir
from app.config import get_settings
from app.engines.stubs import StubASR
from app.main import app
from app.pipeline import checkpoints_for
from app.queueing import TaskRunner
from app.schemas import AsrModel, DiarizationModel, TaskStatus, coerce_optional_diarization
from app.tasks import TaskStore
from app.version import read_version


def _auth_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {get_settings().API_TOKEN}"}


@pytest.fixture
def wav_bytes(tmp_path: Path) -> tuple[str, bytes]:
    path = tmp_path / "sample.wav"
    sf.write(path, np.zeros(8000, dtype=np.float32), 16000)
    return path.name, path.read_bytes()


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ITRANSCRIBE_STUBS", "1")
    monkeypatch.setenv("SQLITE_PATH", str(tmp_path / "tasks.db"))
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("PERFORMANCE_LOG", str(tmp_path / "logs" / "performance_log.csv"))
    monkeypatch.setenv("WORKERS", "1")
    monkeypatch.setenv("WORKER_QUEUE_SIZE", "1")
    get_settings.cache_clear()
    with TestClient(app) as test_client:
        yield test_client
    get_settings.cache_clear()


def _post_transcribe(client: TestClient, wav_bytes: tuple[str, bytes], **data):
    name, payload = wav_bytes
    files = {"file": (name, payload, "audio/wav")}
    return client.post("/transcribe", files=files, data=data, headers=_auth_headers())


def test_health_without_token(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["version"] == read_version()
    assert body["device"] in {"cpu", "cuda"}
    assert set(body["engines"]) == {"whisper", "gigaam", "nemo", "pyannote"}
    assert set(body["engines"].values()) <= {"loaded", "unavailable", "disabled"}


def test_tasks_unauthorized(client: TestClient) -> None:
    response = client.get("/tasks")
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "unauthorized"


def test_invalid_asr_model_422(client: TestClient, wav_bytes: tuple[str, bytes]) -> None:
    response = _post_transcribe(client, wav_bytes, asr_model="nope")
    assert response.status_code == 422


def test_coerce_optional_diarization() -> None:
    assert coerce_optional_diarization(None) is None
    assert coerce_optional_diarization("") is None
    assert coerce_optional_diarization("  ") is None
    assert coerce_optional_diarization("nemo") == "nemo"
    assert coerce_optional_diarization(DiarizationModel.pyannote) is DiarizationModel.pyannote


def test_invalid_diarization_model_422(client: TestClient, wav_bytes: tuple[str, bytes]) -> None:
    response = _post_transcribe(client, wav_bytes, diarization_model="nope")
    assert response.status_code == 422


def test_transcribe_omits_diarization(client: TestClient, wav_bytes: tuple[str, bytes]) -> None:
    created = _post_transcribe(client, wav_bytes, asr_model="whisper")
    assert created.status_code == 202
    assert created.json()["meta"]["diarization_model"] is None
    task_id = created.json()["meta"]["task_id"]

    deadline = time.time() + 5
    result = None
    while time.time() < deadline:
        result = client.get(f"/tasks/{task_id}", headers=_auth_headers())
        assert result.status_code == 200
        if result.json()["status"] in {"success", "error"}:
            break
        time.sleep(0.02)
    assert result is not None
    payload = result.json()
    assert payload["status"] == "success"
    assert payload["meta"]["diarization_model"] is None
    assert payload["meta"]["diarization_checkpoint"] is None
    assert payload["meta"]["diarization_time_sec"] == 0.0
    assert payload["transcript"]
    assert payload["transcript"][0]["speaker"] is None
    assert payload["transcript"][0]["text"] == "привет мир"


def test_transcribe_empty_diarization_skips(
    client: TestClient, wav_bytes: tuple[str, bytes]
) -> None:
    created = _post_transcribe(client, wav_bytes, diarization_model="")
    assert created.status_code == 202
    assert created.json()["meta"]["diarization_model"] is None
    task_id = created.json()["meta"]["task_id"]

    deadline = time.time() + 5
    result = None
    while time.time() < deadline:
        result = client.get(f"/tasks/{task_id}", headers=_auth_headers())
        assert result.status_code == 200
        if result.json()["status"] in {"success", "error"}:
            break
        time.sleep(0.02)
    assert result is not None
    payload = result.json()
    assert payload["status"] == "success"
    assert payload["meta"]["diarization_model"] is None
    assert payload["transcript"][0]["speaker"] is None


def test_transcribe_with_diarization_assigns_speakers(
    client: TestClient, wav_bytes: tuple[str, bytes]
) -> None:
    created = _post_transcribe(client, wav_bytes, diarization_model="pyannote")
    assert created.status_code == 202
    assert created.json()["meta"]["diarization_model"] == "pyannote"
    task_id = created.json()["meta"]["task_id"]

    deadline = time.time() + 5
    result = None
    while time.time() < deadline:
        result = client.get(f"/tasks/{task_id}", headers=_auth_headers())
        assert result.status_code == 200
        if result.json()["status"] in {"success", "error"}:
            break
        time.sleep(0.02)
    assert result is not None
    payload = result.json()
    assert payload["status"] == "success"
    assert payload["meta"]["diarization_model"] == "pyannote"
    speakers = {line["speaker"] for line in payload["transcript"]}
    assert speakers == {"speaker_0", "speaker_1"}


def test_poll_until_success_and_tmp_cleaned(
    client: TestClient, wav_bytes: tuple[str, bytes]
) -> None:
    created = _post_transcribe(client, wav_bytes)
    assert created.status_code == 202
    body = created.json()
    assert body["status"] == "queued"
    task_id = body["meta"]["task_id"]

    deadline = time.time() + 5
    result = None
    while time.time() < deadline:
        result = client.get(f"/tasks/{task_id}", headers=_auth_headers())
        assert result.status_code == 200
        if result.json()["status"] in {"success", "error"}:
            break
        time.sleep(0.02)
    assert result is not None
    payload = result.json()
    assert payload["status"] == "success"
    assert payload["transcript"]
    assert payload["meta"]["rtf"] is not None
    assert not tmp_dir(task_id).exists()
    assert not Path(f"tmp_{task_id}").exists()

    listing = client.get("/tasks", headers=_auth_headers())
    assert listing.status_code == 200
    assert listing.json()[0]["task_id"] == task_id
    assert "transcript" not in listing.json()[0]


def test_queue_running_queued_and_503(
    client: TestClient, wav_bytes: tuple[str, bytes]
) -> None:
    gate = threading.Event()
    original = StubASR.words

    def slow(self, wav_path: str):
        gate.wait(timeout=10)
        return original(self, wav_path)

    StubASR.words = slow  # type: ignore[method-assign]
    try:
        first = _post_transcribe(client, wav_bytes)
        assert first.status_code == 202
        first_id = first.json()["meta"]["task_id"]
        deadline = time.time() + 3
        while time.time() < deadline:
            status = client.get(f"/tasks/{first_id}", headers=_auth_headers()).json()["status"]
            if status == "running":
                break
            time.sleep(0.02)
        else:
            raise AssertionError("first task did not become running")

        second = _post_transcribe(client, wav_bytes)
        assert second.status_code == 202
        assert second.json()["status"] == "queued"
        third = _post_transcribe(client, wav_bytes)
        assert third.status_code == 503
        assert third.json()["error"]["code"] == "queue_full"
    finally:
        gate.set()
        StubASR.words = original  # type: ignore[method-assign]


def test_delete_queued_ok_running_409(
    client: TestClient, wav_bytes: tuple[str, bytes]
) -> None:
    gate = threading.Event()
    original = StubASR.words

    def slow(self, wav_path: str):
        gate.wait(timeout=10)
        return original(self, wav_path)

    StubASR.words = slow  # type: ignore[method-assign]
    try:
        first = _post_transcribe(client, wav_bytes)
        first_id = first.json()["meta"]["task_id"]
        deadline = time.time() + 3
        while time.time() < deadline:
            status = client.get(f"/tasks/{first_id}", headers=_auth_headers()).json()["status"]
            if status == "running":
                break
            time.sleep(0.02)
        else:
            raise AssertionError("first task did not become running")

        second = _post_transcribe(client, wav_bytes)
        second_id = second.json()["meta"]["task_id"]
        deleted = client.delete(f"/tasks/{second_id}", headers=_auth_headers())
        assert deleted.status_code == 200
        missing = client.get(f"/tasks/{second_id}", headers=_auth_headers())
        assert missing.status_code == 404

        running = client.delete(f"/tasks/{first_id}", headers=_auth_headers())
        assert running.status_code == 409
        assert running.json()["error"]["code"] == "task_running"
    finally:
        gate.set()
        StubASR.words = original  # type: ignore[method-assign]


def _isolate_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    workers: str = "1",
    queue_size: str = "4",
    max_restarts: str | None = None,
) -> None:
    monkeypatch.setenv("ITRANSCRIBE_STUBS", "1")
    monkeypatch.setenv("SQLITE_PATH", str(tmp_path / "tasks.db"))
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("PERFORMANCE_LOG", str(tmp_path / "logs" / "performance_log.csv"))
    monkeypatch.setenv("WORKERS", workers)
    monkeypatch.setenv("WORKER_QUEUE_SIZE", queue_size)
    if max_restarts is not None:
        monkeypatch.setenv("TASK_MAX_RESTARTS", max_restarts)
    get_settings.cache_clear()


def _write_wav(path: Path) -> Path:
    sf.write(path, np.zeros(8000, dtype=np.float32), 16000)
    return path


async def _wait_store(store: TaskStore, task_id: str, statuses: set[TaskStatus], timeout: float = 5.0):
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        last = store.get(task_id)
        if last is not None and last.status in statuses:
            return last
        await asyncio.sleep(0.02)
    raise AssertionError(f"timeout waiting {task_id} in {statuses}: {last}")


def _poll_client(client: TestClient, task_id: str, timeout: float = 5.0) -> dict:
    deadline = time.time() + timeout
    result = None
    while time.time() < deadline:
        result = client.get(f"/tasks/{task_id}", headers=_auth_headers())
        assert result.status_code == 200
        if result.json()["status"] in {"success", "error"}:
            return result.json()
        time.sleep(0.02)
    raise AssertionError(f"timeout polling {task_id}: {result.json() if result else None}")


def test_restart_queued_survives_stop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _isolate_env(tmp_path, monkeypatch, workers="1", queue_size="4")
    wav = _write_wav(tmp_path / "sample.wav")
    gate = threading.Event()
    released = threading.Event()
    original = StubASR.words

    def slow(self, wav_path: str):
        gate.wait(timeout=15)
        released.set()
        return original(self, wav_path)

    StubASR.words = slow  # type: ignore[method-assign]

    async def scenario() -> None:
        settings = get_settings()
        runner = TaskRunner(settings)
        await runner.start()
        first = await runner.submit(wav, AsrModel.whisper, None)
        await _wait_store(runner.store, first.task_id, {TaskStatus.running})
        second = await runner.submit(wav, AsrModel.whisper, None)
        assert second.status is TaskStatus.queued
        upload = Path(second.upload_path or "")
        assert upload.is_file()
        assert upload.parent == tmp_path / "tmp" / second.task_id
        await runner.stop()
        assert upload.is_file()
        gate.set()
        released.wait(timeout=5)
        await asyncio.sleep(0.05)
        runner2 = TaskRunner(settings)
        await runner2.start()
        done = await _wait_store(
            runner2.store, second.task_id, {TaskStatus.success, TaskStatus.error}
        )
        assert done.status is TaskStatus.success
        await runner2.stop()

    try:
        asyncio.run(scenario())
    finally:
        gate.set()
        StubASR.words = original  # type: ignore[method-assign]
        get_settings.cache_clear()


def test_restart_running_requeued_not_interrupted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _isolate_env(tmp_path, monkeypatch, workers="1", queue_size="4")
    wav = _write_wav(tmp_path / "sample.wav")
    gate = threading.Event()
    released = threading.Event()
    original = StubASR.words

    def slow(self, wav_path: str):
        gate.wait(timeout=15)
        released.set()
        return original(self, wav_path)

    StubASR.words = slow  # type: ignore[method-assign]

    async def scenario() -> None:
        settings = get_settings()
        runner = TaskRunner(settings)
        await runner.start()
        first = await runner.submit(wav, AsrModel.whisper, None)
        await _wait_store(runner.store, first.task_id, {TaskStatus.running})
        assert Path(first.upload_path or "").is_file()
        await runner.stop()
        assert Path(first.upload_path or "").is_file()
        gate.set()
        released.wait(timeout=5)
        await asyncio.sleep(0.05)
        runner2 = TaskRunner(settings)
        await runner2.start()
        rec = runner2.store.get(first.task_id)
        assert rec is not None
        if rec.status is TaskStatus.error:
            assert rec.error is None or rec.error.get("code") != "interrupted"
        else:
            assert rec.status in {
                TaskStatus.queued,
                TaskStatus.running,
                TaskStatus.success,
            }
        done = await _wait_store(
            runner2.store, first.task_id, {TaskStatus.success, TaskStatus.error}
        )
        assert done.status is TaskStatus.success
        await runner2.stop()

    try:
        asyncio.run(scenario())
    finally:
        gate.set()
        StubASR.words = original  # type: ignore[method-assign]
        get_settings.cache_clear()


def test_restore_missing_upload_errors_not_queued(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _isolate_env(tmp_path, monkeypatch)
    settings = get_settings()
    store = TaskStore(settings.SQLITE_PATH)
    asr_ckpt, diar_ckpt = checkpoints_for(settings, AsrModel.whisper, None)
    missing = store.create(
        "missing-upload-1",
        AsrModel.whisper,
        None,
        asr_ckpt,
        diar_ckpt,
        str(tmp_path / "no-such-upload.wav"),
    )
    store.close()

    async def scenario() -> None:
        runner = TaskRunner(settings)
        await runner.start()
        rec = runner.store.get(missing.task_id)
        assert rec is not None
        assert rec.status is TaskStatus.error
        assert rec.error is not None
        assert rec.error["code"] == "missing_upload"
        assert runner.store.count_queued() == 0
        await runner.stop()

    try:
        asyncio.run(scenario())
    finally:
        get_settings.cache_clear()


def test_restore_fifo_order_and_queue_limit_not_applied(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _isolate_env(tmp_path, monkeypatch, workers="1", queue_size="1")
    wav = _write_wav(tmp_path / "sample.wav")
    settings = get_settings()
    store = TaskStore(settings.SQLITE_PATH)
    asr_ckpt, diar_ckpt = checkpoints_for(settings, AsrModel.whisper, None)
    ids = ["fifo-a", "fifo-b"]
    stamps = ["2026-01-01T00:00:00", "2026-01-01T00:00:01"]
    for task_id, stamp in zip(ids, stamps, strict=True):
        dest_dir = create_tmp(task_id, tmp_path)
        dest = dest_dir / "upload.wav"
        shutil.copy2(wav, dest)
        store.create(task_id, AsrModel.whisper, None, asr_ckpt, diar_ckpt, str(dest))
        with store._lock:
            store._conn.execute(
                "UPDATE tasks SET timestamp = ? WHERE task_id = ?",
                (stamp, task_id),
            )
            store._conn.commit()
    store.close()

    async def scenario() -> None:
        runner = TaskRunner(settings)
        await runner.start()
        first = await _wait_store(
            runner.store, "fifo-a", {TaskStatus.success, TaskStatus.error}
        )
        second = await _wait_store(
            runner.store, "fifo-b", {TaskStatus.success, TaskStatus.error}
        )
        assert first.status is TaskStatus.success
        assert second.status is TaskStatus.success
        assert first.started_at is not None
        assert second.started_at is not None
        assert first.started_at <= second.started_at
        await runner.stop()

    try:
        asyncio.run(scenario())
    finally:
        get_settings.cache_clear()


def _seed_running_with_upload(tmp_path: Path, task_id: str, wav: Path) -> None:
    settings = get_settings()
    store = TaskStore(settings.SQLITE_PATH)
    asr_ckpt, diar_ckpt = checkpoints_for(settings, AsrModel.whisper, None)
    dest_dir = create_tmp(task_id, tmp_path)
    dest = dest_dir / "upload.wav"
    shutil.copy2(wav, dest)
    store.create(task_id, AsrModel.whisper, None, asr_ckpt, diar_ckpt, str(dest))
    assert store.mark_running(task_id)
    store.close()


def test_restore_running_zero_restarts_process_killed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _isolate_env(tmp_path, monkeypatch, max_restarts="0")
    wav = _write_wav(tmp_path / "sample.wav")
    _seed_running_with_upload(tmp_path, "poison-0", wav)

    async def scenario() -> None:
        runner = TaskRunner(get_settings())
        await runner.start()
        rec = runner.store.get("poison-0")
        assert rec is not None
        assert rec.status is TaskStatus.error
        assert rec.error is not None
        assert rec.error["code"] == "process_killed"
        assert rec.attempts == 1
        assert runner.store.count_queued() == 0
        assert not (tmp_path / "tmp" / "poison-0").exists()
        await runner.delete("poison-0")
        assert runner.store.get("poison-0") is None
        await runner.stop()

    try:
        asyncio.run(scenario())
    finally:
        get_settings.cache_clear()


def test_restore_running_under_limit_retries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _isolate_env(tmp_path, monkeypatch)
    wav = _write_wav(tmp_path / "sample.wav")
    _seed_running_with_upload(tmp_path, "retry-1", wav)

    async def scenario() -> None:
        runner = TaskRunner(get_settings())
        await runner.start()
        done = await _wait_store(
            runner.store, "retry-1", {TaskStatus.success, TaskStatus.error}
        )
        assert done.status is TaskStatus.success
        assert done.attempts == 1
        await runner.stop()

    try:
        asyncio.run(scenario())
    finally:
        get_settings.cache_clear()


def test_restore_running_errors_when_attempts_exceed_max(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _isolate_env(tmp_path, monkeypatch)
    wav = _write_wav(tmp_path / "sample.wav")
    _seed_running_with_upload(tmp_path, "poison-2", wav)
    store = TaskStore(get_settings().SQLITE_PATH)
    assert store.bump_attempts("poison-2") == 1
    store.close()

    async def scenario() -> None:
        runner = TaskRunner(get_settings())
        await runner.start()
        rec = runner.store.get("poison-2")
        assert rec is not None
        assert rec.status is TaskStatus.error
        assert rec.error is not None
        assert rec.error["code"] == "process_killed"
        assert rec.attempts == 2
        assert runner.store.count_queued() == 0
        assert not (tmp_path / "tmp" / "poison-2").exists()
        await runner.stop()

    try:
        asyncio.run(scenario())
    finally:
        get_settings.cache_clear()


def test_pipeline_error_does_not_increment_attempts(
    client: TestClient, wav_bytes: tuple[str, bytes]
) -> None:
    original = StubASR.words

    def boom(self, wav_path: str):
        raise RuntimeError("simulated python exception")

    StubASR.words = boom  # type: ignore[method-assign]
    try:
        created = _post_transcribe(client, wav_bytes, asr_model="whisper")
        assert created.status_code == 202
        task_id = created.json()["meta"]["task_id"]
        payload = _poll_client(client, task_id)
        assert payload["status"] == "error"
        assert payload["error"]["code"] == "pipeline_error"
        rec = app.state.runner.store.get(task_id)
        assert rec is not None
        assert rec.attempts == 0
    finally:
        StubASR.words = original  # type: ignore[method-assign]


def test_attempts_column_migrated_on_existing_db(tmp_path: Path) -> None:
    db = tmp_path / "legacy.db"
    conn = sqlite3.connect(str(db))
    conn.execute(
        """
        CREATE TABLE tasks (
            task_id TEXT PRIMARY KEY,
            status TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            asr_model TEXT NOT NULL,
            diarization_model TEXT NOT NULL,
            asr_checkpoint TEXT,
            diarization_checkpoint TEXT,
            started_at TEXT,
            finished_at TEXT,
            audio_duration_sec REAL,
            asr_time_sec REAL,
            diarization_time_sec REAL,
            alignment_time_sec REAL,
            total_time_sec REAL,
            rtf REAL,
            transcript TEXT,
            error TEXT,
            upload_path TEXT
        )
        """
    )
    conn.execute(
        """
        INSERT INTO tasks (
            task_id, status, timestamp, asr_model, diarization_model
        ) VALUES (?, ?, ?, ?, ?)
        """,
        ("legacy-1", "queued", "2026-01-01T00:00:00", "whisper", ""),
    )
    conn.commit()
    conn.close()

    store = TaskStore(str(db))
    rec = store.get("legacy-1")
    assert rec is not None
    assert rec.attempts == 0
    assert store.bump_attempts("legacy-1") == 1
    rec = store.get("legacy-1")
    assert rec is not None
    assert rec.attempts == 1
    store.close()


def test_purge_unauthorized(client: TestClient) -> None:
    response = client.delete("/tasks")
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "unauthorized"


def test_purge_tasks_clears_queue_history_and_orphan_tmp(
    client: TestClient, wav_bytes: tuple[str, bytes], tmp_path: Path
) -> None:
    created = _post_transcribe(client, wav_bytes)
    assert created.status_code == 202
    finished_id = created.json()["meta"]["task_id"]
    payload = _poll_client(client, finished_id)
    assert payload["status"] == "success"

    runner = app.state.runner
    _, file_bytes = wav_bytes
    queued_ids: list[str] = []
    for _ in range(2):
        tid = str(uuid.uuid4())
        dest_dir = create_tmp(tid, tmp_path)
        dest = dest_dir / "upload.wav"
        dest.write_bytes(file_bytes)
        asr_ckpt, diar_ckpt = checkpoints_for(runner.settings, AsrModel.whisper, None)
        runner.store.create(tid, AsrModel.whisper, None, asr_ckpt, diar_ckpt, str(dest))
        queued_ids.append(tid)

    orphan_id = "orphan-no-row"
    create_tmp(orphan_id, tmp_path)
    assert tmp_dir(orphan_id, tmp_path).is_dir()

    response = client.delete("/tasks", headers=_auth_headers())
    assert response.status_code == 200
    body_json = response.json()
    assert body_json["status"] == "ok"
    assert body_json["purged_queued"] == 2
    assert body_json["purged_finished"] == 1
    assert body_json["skipped_running"] == 0
    assert body_json["purged_tmp"] == 3
    listing = client.get("/tasks", headers=_auth_headers())
    assert listing.status_code == 200
    assert listing.json() == []
    for tid in queued_ids + [orphan_id]:
        assert not tmp_dir(tid, tmp_path).exists()


def test_purge_skips_running_returns_200_not_409(
    client: TestClient, wav_bytes: tuple[str, bytes], tmp_path: Path
) -> None:
    gate = threading.Event()
    original = StubASR.words

    def slow(self, wav_path: str):
        gate.wait(timeout=10)
        return original(self, wav_path)

    StubASR.words = slow  # type: ignore[method-assign]
    try:
        first = _post_transcribe(client, wav_bytes)
        first_id = first.json()["meta"]["task_id"]
        deadline = time.time() + 3
        while time.time() < deadline:
            status = client.get(f"/tasks/{first_id}", headers=_auth_headers()).json()["status"]
            if status == "running":
                break
            time.sleep(0.02)
        else:
            raise AssertionError("first task did not become running")

        second = _post_transcribe(client, wav_bytes)
        second_id = second.json()["meta"]["task_id"]
        assert second.json()["status"] == "queued"

        response = client.delete("/tasks", headers=_auth_headers())
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["purged_queued"] == 1
        assert body["skipped_running"] == 1
        assert body["purged_finished"] == 0
        assert body["purged_tmp"] >= 1
        assert not tmp_dir(second_id, tmp_path).exists()
        assert tmp_dir(first_id, tmp_path).is_dir()

        still = client.get(f"/tasks/{first_id}", headers=_auth_headers())
        assert still.status_code == 200
        assert still.json()["status"] == "running"

        listing = client.get("/tasks", headers=_auth_headers())
        ids = {item["task_id"] for item in listing.json()}
        assert first_id in ids
        assert second_id not in ids
    finally:
        gate.set()
        StubASR.words = original  # type: ignore[method-assign]

    done = _poll_client(client, first_id)
    assert done["status"] == "success"


def test_purge_counts_legacy_cwd_tmp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    monkeypatch.chdir(tmp_path)
    _isolate_env(data_dir, monkeypatch)
    get_settings.cache_clear()
    legacy = tmp_path / "tmp_legacy-orphan"
    legacy.mkdir()

    async def scenario() -> None:
        runner = TaskRunner(get_settings())
        await runner.start()
        result = await runner.purge()
        assert result.purged_queued == 0
        assert result.purged_finished == 0
        assert result.skipped_running == 0
        # purged_tmp = {DATA_DIR}/tmp dirs + legacy CWD tmp_* dirs
        assert result.purged_tmp == 1
        await runner.stop()

    try:
        asyncio.run(scenario())
        assert not legacy.exists()
    finally:
        get_settings.cache_clear()
