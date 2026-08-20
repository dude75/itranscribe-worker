"""CSV метрик и stdout `metric_event` (ТЗ §10)."""

from __future__ import annotations

import csv
import json
import sys
import threading
from dataclasses import asdict, dataclass
from pathlib import Path

from app.config import get_settings

CSV_FIELDS = [
    "timestamp",
    "task_id",
    "asr_model",
    "diarization_model",
    "asr_checkpoint",
    "diarization_checkpoint",
    "audio_duration_sec",
    "asr_time_sec",
    "diarization_time_sec",
    "alignment_time_sec",
    "total_time_sec",
    "rtf",
]

CSV_HEADER = ",".join(CSV_FIELDS)

_lock = threading.Lock()


@dataclass
class MetricEvent:
    timestamp: str
    task_id: str
    asr_model: str
    diarization_model: str | None = None
    asr_checkpoint: str | None = None
    diarization_checkpoint: str | None = None
    audio_duration_sec: float | None = None
    asr_time_sec: float | None = None
    diarization_time_sec: float | None = None
    alignment_time_sec: float | None = None
    total_time_sec: float | None = None
    rtf: float | None = None
    status: str = "success"


def _csv_cell(value: object) -> str:
    if value is None:
        return ""
    return str(value)


def write_metric(path: str | Path, event: MetricEvent) -> None:
    if not get_settings().PERFORMANCE_LOG_ENABLED:
        return
    csv_path = Path(path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    row = {field: _csv_cell(getattr(event, field)) for field in CSV_FIELDS}
    with _lock:
        new_file = not csv_path.exists() or csv_path.stat().st_size == 0
        with csv_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, lineterminator="\n")
            if new_file:
                writer.writeheader()
            writer.writerow(row)
        payload = {key: value for key, value in asdict(event).items() if value is not None}
        payload["metric_event"] = True
        print(json.dumps(payload, ensure_ascii=False), file=sys.stdout, flush=True)
