from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from app.config import get_settings
from app.engines.stubs import StubASR
from app.main import app
from app.schemas import DiarizationModel, coerce_optional_diarization
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
