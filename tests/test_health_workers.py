from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from app.config import get_settings
from app.engines.stubs import StubASR
from app.main import app
from tests.test_api_tasks import _auth_headers, _isolate_env, _poll_client, _post_transcribe


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
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


@pytest.fixture
def wav_bytes(tmp_path: Path) -> tuple[str, bytes]:
    path = tmp_path / "sample.wav"
    sf.write(path, np.zeros(8000, dtype=np.float32), 16000)
    return path.name, path.read_bytes()


def _assert_workers(body: dict, *, max_workers: int, active: int, available: int) -> None:
    workers = body["workers"]
    assert set(workers) == {"max", "active", "available"}
    assert workers["max"] == max_workers
    assert workers["active"] == active
    assert workers["available"] == available
    assert workers["active"] <= workers["max"]
    assert workers["available"] == workers["max"] - workers["active"]


def test_health_workers_idle(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    _assert_workers(response.json(), max_workers=1, active=0, available=1)


def test_health_workers_max_from_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _isolate_env(tmp_path, monkeypatch, workers="2", queue_size="4")
    from app.main import app

    with TestClient(app) as test_client:
        _assert_workers(test_client.get("/health").json(), max_workers=2, active=0, available=2)
    get_settings.cache_clear()


def test_health_workers_max_alias(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _isolate_env(tmp_path, monkeypatch, queue_size="4")
    monkeypatch.delenv("WORKERS", raising=False)
    monkeypatch.setenv("WORKERS_MAX", "3")
    get_settings.cache_clear()
    from app.main import app

    with TestClient(app) as test_client:
        _assert_workers(test_client.get("/health").json(), max_workers=3, active=0, available=3)
    get_settings.cache_clear()


def test_health_workers_during_and_after_task(
    client: TestClient, wav_bytes: tuple[str, bytes]
) -> None:
    gate = threading.Event()
    original = StubASR.words

    def slow(self, wav_path: str):
        gate.wait(timeout=10)
        return original(self, wav_path)

    StubASR.words = slow  # type: ignore[method-assign]
    try:
        idle = client.get("/health").json()
        _assert_workers(idle, max_workers=1, active=0, available=1)

        created = _post_transcribe(client, wav_bytes)
        assert created.status_code == 202
        task_id = created.json()["meta"]["task_id"]
        deadline = time.time() + 3
        while time.time() < deadline:
            status = client.get(f"/tasks/{task_id}", headers=_auth_headers()).json()["status"]
            if status == "running":
                break
            time.sleep(0.02)
        else:
            raise AssertionError("task did not become running")

        running = client.get("/health").json()
        _assert_workers(running, max_workers=1, active=1, available=0)

        gate.set()
        _poll_client(client, task_id)

        finished = client.get("/health").json()
        _assert_workers(finished, max_workers=1, active=0, available=1)
    finally:
        gate.set()
        StubASR.words = original  # type: ignore[method-assign]
