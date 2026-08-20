"""SQLite-реестр задач (ТЗ §4.5)."""

from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from app.schemas import AsrModel, DiarizationModel, ErrorDetail, TaskStatus, TranscriptLine


@dataclass
class TaskRecord:
    task_id: str
    status: TaskStatus
    timestamp: str
    asr_model: AsrModel
    diarization_model: DiarizationModel | None
    asr_checkpoint: str | None = None
    diarization_checkpoint: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    audio_duration_sec: float | None = None
    asr_time_sec: float | None = None
    diarization_time_sec: float | None = None
    alignment_time_sec: float | None = None
    total_time_sec: float | None = None
    rtf: float | None = None
    transcript: list[dict[str, Any]] | None = None
    error: dict[str, Any] | None = None
    upload_path: str | None = None


def _now() -> str:
    return datetime.now().isoformat()


class TaskStore:
    def __init__(self, sqlite_path: str) -> None:
        path = Path(sqlite_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._init_schema()

    def _init_schema(self) -> None:
        with self._lock:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS tasks (
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
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def create(
        self,
        task_id: str,
        asr_model: AsrModel,
        diarization_model: DiarizationModel | None,
        asr_checkpoint: str,
        diarization_checkpoint: str | None,
        upload_path: str,
    ) -> TaskRecord:
        timestamp = _now()
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO tasks (
                    task_id, status, timestamp, asr_model, diarization_model,
                    asr_checkpoint, diarization_checkpoint, upload_path
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    task_id,
                    TaskStatus.queued.value,
                    timestamp,
                    asr_model.value,
                    diarization_model.value if diarization_model is not None else "",
                    asr_checkpoint,
                    diarization_checkpoint,
                    upload_path,
                ),
            )
            self._conn.commit()
        record = self.get(task_id)
        assert record is not None
        return record

    def get(self, task_id: str) -> TaskRecord | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM tasks WHERE task_id = ?", (task_id,)
            ).fetchone()
        if row is None:
            return None
        return _row_to_record(row)

    def list_tasks(self, status: TaskStatus | None = None) -> list[TaskRecord]:
        query = "SELECT * FROM tasks"
        params: tuple[object, ...] = ()
        if status is not None:
            query += " WHERE status = ?"
            params = (status.value,)
        query += " ORDER BY timestamp DESC"
        with self._lock:
            rows = self._conn.execute(query, params).fetchall()
        return [_row_to_record(row) for row in rows]

    def count_queued(self) -> int:
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) AS n FROM tasks WHERE status = ?",
                (TaskStatus.queued.value,),
            ).fetchone()
        return int(row["n"])

    def mark_running(self, task_id: str) -> bool:
        with self._lock:
            cur = self._conn.execute(
                """
                UPDATE tasks SET status = ?, started_at = ?
                WHERE task_id = ? AND status = ?
                """,
                (TaskStatus.running.value, _now(), task_id, TaskStatus.queued.value),
            )
            self._conn.commit()
            return cur.rowcount > 0

    def mark_success(
        self,
        task_id: str,
        *,
        audio_duration_sec: float,
        asr_time_sec: float,
        diarization_time_sec: float,
        alignment_time_sec: float,
        total_time_sec: float,
        rtf: float,
        transcript: list[TranscriptLine],
    ) -> None:
        payload = json.dumps([line.model_dump() for line in transcript], ensure_ascii=False)
        with self._lock:
            self._conn.execute(
                """
                UPDATE tasks SET
                    status = ?, finished_at = ?,
                    audio_duration_sec = ?, asr_time_sec = ?,
                    diarization_time_sec = ?, alignment_time_sec = ?,
                    total_time_sec = ?, rtf = ?, transcript = ?, error = NULL
                WHERE task_id = ?
                """,
                (
                    TaskStatus.success.value,
                    _now(),
                    audio_duration_sec,
                    asr_time_sec,
                    diarization_time_sec,
                    alignment_time_sec,
                    total_time_sec,
                    rtf,
                    payload,
                    task_id,
                ),
            )
            self._conn.commit()

    def mark_error(
        self,
        task_id: str,
        error: ErrorDetail,
        *,
        audio_duration_sec: float | None = None,
        asr_time_sec: float | None = None,
        diarization_time_sec: float | None = None,
        alignment_time_sec: float | None = None,
        total_time_sec: float | None = None,
        rtf: float | None = None,
    ) -> None:
        with self._lock:
            self._conn.execute(
                """
                UPDATE tasks SET
                    status = ?, finished_at = ?, error = ?,
                    audio_duration_sec = COALESCE(?, audio_duration_sec),
                    asr_time_sec = COALESCE(?, asr_time_sec),
                    diarization_time_sec = COALESCE(?, diarization_time_sec),
                    alignment_time_sec = COALESCE(?, alignment_time_sec),
                    total_time_sec = COALESCE(?, total_time_sec),
                    rtf = COALESCE(?, rtf)
                WHERE task_id = ?
                """,
                (
                    TaskStatus.error.value,
                    _now(),
                    error.model_dump_json(),
                    audio_duration_sec,
                    asr_time_sec,
                    diarization_time_sec,
                    alignment_time_sec,
                    total_time_sec,
                    rtf,
                    task_id,
                ),
            )
            self._conn.commit()

    def interrupt_running(self) -> list[str]:
        running = self.list_tasks(TaskStatus.running)
        ids = [item.task_id for item in running]
        for task_id in ids:
            self.mark_error(task_id, ErrorDetail(code="interrupted"))
        return ids

    def delete(self, task_id: str) -> bool:
        with self._lock:
            cur = self._conn.execute("DELETE FROM tasks WHERE task_id = ?", (task_id,))
            self._conn.commit()
            return cur.rowcount > 0

    def delete_if_not_running(self, task_id: str) -> TaskStatus | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT status FROM tasks WHERE task_id = ?", (task_id,)
            ).fetchone()
            if row is None:
                return None
            status = TaskStatus(row["status"])
            if status is TaskStatus.running:
                return status
            self._conn.execute("DELETE FROM tasks WHERE task_id = ?", (task_id,))
            self._conn.commit()
            return status

    def purge_expired(self, ttl_sec: int) -> int:
        if ttl_sec <= 0:
            return 0
        cutoff = (datetime.now() - timedelta(seconds=ttl_sec)).isoformat()
        with self._lock:
            cur = self._conn.execute(
                """
                DELETE FROM tasks
                WHERE status IN (?, ?)
                  AND finished_at IS NOT NULL
                  AND finished_at <= ?
                """,
                (TaskStatus.success.value, TaskStatus.error.value, cutoff),
            )
            self._conn.commit()
            return cur.rowcount


def _row_to_record(row: sqlite3.Row) -> TaskRecord:
    transcript = json.loads(row["transcript"]) if row["transcript"] else None
    error = json.loads(row["error"]) if row["error"] else None
    return TaskRecord(
        task_id=row["task_id"],
        status=TaskStatus(row["status"]),
        timestamp=row["timestamp"],
        asr_model=AsrModel(row["asr_model"]),
        diarization_model=(
            DiarizationModel(row["diarization_model"]) if row["diarization_model"] else None
        ),
        asr_checkpoint=row["asr_checkpoint"],
        diarization_checkpoint=row["diarization_checkpoint"],
        started_at=row["started_at"],
        finished_at=row["finished_at"],
        audio_duration_sec=row["audio_duration_sec"],
        asr_time_sec=row["asr_time_sec"],
        diarization_time_sec=row["diarization_time_sec"],
        alignment_time_sec=row["alignment_time_sec"],
        total_time_sec=row["total_time_sec"],
        rtf=row["rtf"],
        transcript=transcript,
        error=error,
        upload_path=row["upload_path"],
    )
