from __future__ import annotations

from pathlib import Path

from app.schemas import AsrModel, TranscriptLine
from app.tasks import TaskStore


def _seed_queued(store: TaskStore, task_id: str) -> None:
    store.create(
        task_id,
        AsrModel.whisper,
        None,
        "large-v3-turbo",
        None,
        "/tmp/unused.wav",
    )


def test_transcript_encrypted_at_rest(tmp_path: Path) -> None:
    store = TaskStore(str(tmp_path / "tasks.db"))
    try:
        _seed_queued(store, "enc-1")
        store.mark_success(
            "enc-1",
            audio_duration_sec=1.0,
            asr_time_sec=0.1,
            diarization_time_sec=0.0,
            alignment_time_sec=0.0,
            total_time_sec=0.2,
            rtf=0.1,
            transcript=[
                TranscriptLine(speaker=None, start=0.0, end=1.0, text="секретный текст")
            ],
        )
        with store._lock:
            raw = store._conn.execute(
                "SELECT transcript FROM tasks WHERE task_id = ?",
                ("enc-1",),
            ).fetchone()["transcript"]
        assert raw
        assert not raw.lstrip().startswith("[")
        assert "секретный текст" not in raw
        rec = store.get("enc-1")
        assert rec is not None
        assert rec.transcript is not None
        assert rec.transcript[0]["text"] == "секретный текст"
    finally:
        store.close()


def test_transcript_legacy_plaintext_still_reads(tmp_path: Path) -> None:
    store = TaskStore(str(tmp_path / "tasks.db"))
    try:
        _seed_queued(store, "legacy-plain")
        plaintext = '[{"speaker": null, "start": 0.0, "end": 1.0, "text": "старый json"}]'
        with store._lock:
            store._conn.execute(
                "UPDATE tasks SET transcript = ? WHERE task_id = ?",
                (plaintext, "legacy-plain"),
            )
            store._conn.commit()
        rec = store.get("legacy-plain")
        assert rec is not None
        assert rec.transcript is not None
        assert rec.transcript[0]["text"] == "старый json"
    finally:
        store.close()
