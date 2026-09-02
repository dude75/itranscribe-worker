from __future__ import annotations

import shutil
import subprocess
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.audio import tmp_dir
from app.config import get_settings
from app.main import app
from app.schemas import AsrModel, DiarizationModel

pytestmark = pytest.mark.ml
pytest.importorskip("torch")


COMBOS = [
    (AsrModel.whisper, DiarizationModel.nemo),
    (AsrModel.whisper, DiarizationModel.pyannote),
    (AsrModel.gigaam, DiarizationModel.nemo),
    (AsrModel.gigaam, DiarizationModel.pyannote),
]


def _auth_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {get_settings().API_TOKEN}"}


def _speech_wav(path: Path) -> Path:
    if shutil.which("say") is None:
        pytest.skip("macOS say is not available")
    subprocess.run(
        [
            "say",
            "-o",
            str(path),
            "--data-format=LEI16@16000",
            "one two three four five",
        ],
        check=True,
        capture_output=True,
    )
    return path


@pytest.fixture(scope="module")
def client():
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("API_TOKEN", "test")
        get_settings.cache_clear()
        with TestClient(app) as test_client:
            yield test_client
        get_settings.cache_clear()


@pytest.fixture(scope="module")
def speech_bytes(tmp_path_factory: pytest.TempPathFactory) -> tuple[str, bytes]:
    wav = _speech_wav(tmp_path_factory.mktemp("smoke") / "smoke.wav")
    return wav.name, wav.read_bytes()


def _poll(client: TestClient, task_id: str, timeout: float = 180.0) -> dict:
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        last = client.get(f"/tasks/{task_id}", headers=_auth_headers())
        assert last.status_code == 200
        body = last.json()
        if body["status"] in {"success", "error"}:
            return body
        time.sleep(0.5)
    raise AssertionError(f"timeout waiting for {task_id}: {last.json() if last else None}")


def test_smoke_four_combinations(client: TestClient, speech_bytes: tuple[str, bytes]) -> None:
    health = client.get("/health")
    assert health.status_code == 200
    engines = health.json()["engines"]
    results: dict[str, str] = {}
    name, payload = speech_bytes

    for asr, diar in COMBOS:
        key = f"{asr.value}+{diar.value}"
        if engines.get(asr.value) != "loaded" or engines.get(diar.value) != "loaded":
            results[key] = "skipped"
            continue
        response = client.post(
            "/transcribe",
            files={"file": (name, payload, "audio/wav")},
            data={"asr_model": asr.value, "diarization_model": diar.value},
            headers=_auth_headers(),
        )
        if response.status_code == 503:
            results[key] = "skipped"
            continue
        assert response.status_code == 202, response.text
        task_id = response.json()["meta"]["task_id"]
        body = _poll(client, task_id)
        if body["status"] == "error" and (body.get("error") or {}).get("code") == "engine_unavailable":
            results[key] = "skipped"
            continue
        assert body["status"] == "success", body
        assert body["transcript"] is not None
        assert not tmp_dir(task_id).exists()
        assert not Path(f"tmp_{task_id}").exists()
        results[key] = "ok"

    print("smoke table:", results)
    assert any(value == "ok" for value in results.values())
    csv_path = Path(get_settings().PERFORMANCE_LOG)
    assert csv_path.exists()
    text = csv_path.read_text(encoding="utf-8")
    assert "task_id" in text.splitlines()[0]
    assert "ok" in results.values() or "skipped" in results.values()
