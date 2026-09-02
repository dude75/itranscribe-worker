from __future__ import annotations

import asyncio
import threading
import time
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient
from prometheus_client import generate_latest
from prometheus_client.parser import text_string_to_metric_families

from app.config import get_settings
from app.engines.stubs import StubASR
from app.main import app
from app.pipeline import checkpoints_for
from app.prometheus_metrics import Metrics, set_active
from app.queueing import TaskRunner
from app.schemas import AsrModel
from app.tasks import TaskStore


def _auth_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {get_settings().API_TOKEN}"}


def _sample(text: str, name: str, labels: dict[str, str] | None = None) -> float | None:
    wanted = labels or {}
    for family in text_string_to_metric_families(text):
        for sample in family.samples:
            if sample.name != name:
                continue
            if all(sample.labels.get(key) == value for key, value in wanted.items()):
                return float(sample.value)
    return None


@pytest.fixture
def wav_bytes(tmp_path: Path) -> tuple[str, bytes]:
    path = tmp_path / "sample.wav"
    sf.write(path, np.zeros(8000, dtype=np.float32), 16000)
    return path.name, path.read_bytes()


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("API_TOKEN", "test")
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


def _wait_task(client: TestClient, task_id: str) -> dict:
    deadline = time.time() + 5
    result = None
    while time.time() < deadline:
        result = client.get(f"/tasks/{task_id}", headers=_auth_headers())
        assert result.status_code == 200
        if result.json()["status"] in {"success", "error"}:
            return result.json()
        time.sleep(0.02)
    raise AssertionError("task did not finish")


def test_metrics_without_token(client: TestClient) -> None:
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]
    body = response.text
    assert _sample(body, "itranscribe_up") == 1.0
    assert _sample(body, "itranscribe_ready") == 1.0
    assert _sample(body, "itranscribe_queue_depth") == 0.0
    assert _sample(body, "itranscribe_worker_slots") == 1.0
    assert _sample(body, "itranscribe_queue_limit") == 1.0


def test_metrics_ignores_bad_token(client: TestClient) -> None:
    response = client.get("/metrics", headers={"Authorization": "Bearer not-the-token"})
    assert response.status_code == 200
    assert _sample(response.text, "itranscribe_up") == 1.0


def test_health_still_public_tasks_still_auth(client: TestClient) -> None:
    assert client.get("/health").status_code == 200
    assert client.get("/tasks").status_code == 401


def test_submitted_and_completed_success(client: TestClient, wav_bytes: tuple[str, bytes]) -> None:
    created = _post_transcribe(client, wav_bytes, asr_model="whisper")
    assert created.status_code == 202
    task_id = created.json()["meta"]["task_id"]
    payload = _wait_task(client, task_id)
    assert payload["status"] == "success"

    body = client.get("/metrics").text
    labels = {"asr_model": "whisper", "diarization_model": "none"}
    assert _sample(body, "itranscribe_tasks_submitted_total", labels) == 1.0
    assert (
        _sample(
            body,
            "itranscribe_tasks_completed_total",
            {**labels, "status": "success"},
        )
        == 1.0
    )
    assert _sample(body, "itranscribe_pipeline_duration_seconds_count", labels) == 1.0


def test_queue_full_increments_rejected(client: TestClient, wav_bytes: tuple[str, bytes]) -> None:
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
        third = _post_transcribe(client, wav_bytes)
        assert third.status_code == 503
        body = client.get("/metrics").text
        assert _sample(body, "itranscribe_queue_rejected_total") == 1.0
        assert "itranscribe_queue_rejected_total" in body
    finally:
        gate.set()
        StubASR.words = original  # type: ignore[method-assign]


def test_http_path_uses_template_not_uuid(
    client: TestClient, wav_bytes: tuple[str, bytes]
) -> None:
    created = _post_transcribe(client, wav_bytes)
    task_id = created.json()["meta"]["task_id"]
    _wait_task(client, task_id)
    body = client.get("/metrics").text
    assert "/tasks/{task_id}" in body
    for line in body.splitlines():
        if 'path="' in line:
            assert task_id not in line


def test_csv_still_written(client: TestClient, wav_bytes: tuple[str, bytes]) -> None:
    created = _post_transcribe(client, wav_bytes)
    _wait_task(client, created.json()["meta"]["task_id"])
    csv_path = Path(get_settings().PERFORMANCE_LOG)
    assert csv_path.is_file()
    lines = csv_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) >= 2


def test_metrics_disabled_does_not_break_transcribe(
    tmp_path: Path, wav_bytes: tuple[str, bytes], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("API_TOKEN", "test")
    monkeypatch.setenv("ITRANSCRIBE_STUBS", "1")
    monkeypatch.setenv("SQLITE_PATH", str(tmp_path / "tasks.db"))
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("PERFORMANCE_LOG", str(tmp_path / "logs" / "performance_log.csv"))
    monkeypatch.setenv("METRICS_ENABLED", "false")
    monkeypatch.setenv("WORKERS", "1")
    monkeypatch.setenv("WORKER_QUEUE_SIZE", "1")
    get_settings.cache_clear()
    with TestClient(app) as test_client:
        created = _post_transcribe(test_client, wav_bytes)
        assert created.status_code == 202
        _wait_task(test_client, created.json()["meta"]["task_id"])
        response = test_client.get("/metrics")
        assert response.status_code == 200
        assert _sample(response.text, "itranscribe_queue_depth") is None
        assert _sample(response.text, "itranscribe_tasks_submitted_total") is None
    get_settings.cache_clear()


def test_restore_missing_upload_metric(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("API_TOKEN", "test")
    monkeypatch.setenv("ITRANSCRIBE_STUBS", "1")
    monkeypatch.setenv("SQLITE_PATH", str(tmp_path / "tasks.db"))
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("PERFORMANCE_LOG", str(tmp_path / "logs" / "performance_log.csv"))
    get_settings.cache_clear()
    settings = get_settings()
    metrics = Metrics(enabled=True)
    set_active(metrics)
    store = TaskStore(settings.SQLITE_PATH)
    asr_ckpt, diar_ckpt = checkpoints_for(settings, AsrModel.whisper, None)
    store.create(
        "missing-upload-metric",
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
        await runner.stop()

    try:
        asyncio.run(scenario())
        body = generate_latest(metrics.registry).decode()
        assert _sample(body, "itranscribe_restore_tasks_total", {"outcome": "missing_upload"}) == 1.0
    finally:
        set_active(None)
        get_settings.cache_clear()


def test_restore_process_killed_metric(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("API_TOKEN", "test")
    monkeypatch.setenv("ITRANSCRIBE_STUBS", "1")
    monkeypatch.setenv("SQLITE_PATH", str(tmp_path / "tasks.db"))
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("PERFORMANCE_LOG", str(tmp_path / "logs" / "performance_log.csv"))
    monkeypatch.setenv("TASK_MAX_RESTARTS", "0")
    get_settings.cache_clear()
    settings = get_settings()
    metrics = Metrics(enabled=True)
    set_active(metrics)
    store = TaskStore(settings.SQLITE_PATH)
    asr_ckpt, diar_ckpt = checkpoints_for(settings, AsrModel.whisper, None)
    dest_dir = tmp_path / "tmp" / "poison-metric"
    dest_dir.mkdir(parents=True)
    dest = dest_dir / "upload.wav"
    sf.write(dest, np.zeros(8000, dtype=np.float32), 16000)
    store.create(
        "poison-metric",
        AsrModel.whisper,
        None,
        asr_ckpt,
        diar_ckpt,
        str(dest),
    )
    assert store.mark_running("poison-metric")
    store.close()

    async def scenario() -> None:
        runner = TaskRunner(settings)
        await runner.start()
        rec = runner.store.get("poison-metric")
        assert rec is not None
        assert rec.status.value == "error"
        await runner.stop()

    try:
        asyncio.run(scenario())
        body = generate_latest(metrics.registry).decode()
        assert _sample(body, "itranscribe_restore_tasks_total", {"outcome": "process_killed"}) == 1.0
    finally:
        set_active(None)
        get_settings.cache_clear()
