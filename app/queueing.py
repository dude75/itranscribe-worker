"""Слоты WORKERS и очередь queued (WORKER_QUEUE_SIZE)."""

from __future__ import annotations

import asyncio
import shutil
import uuid
from pathlib import Path

from app.audio import (
    cleanup_legacy_cwd_tmp,
    cleanup_tmp,
    cleanup_tmp_except,
    cleanup_upload_scratch,
    create_tmp,
)
from app.config import Settings, get_settings
from app.pipeline import checkpoints_for, run_pipeline
from app.prometheus_metrics import observe_restore, observe_submitted
from app.schemas import AsrModel, DiarizationModel, ErrorDetail, PurgeResult, TaskStatus
from app.tasks import TaskRecord, TaskStore


class QueueFullError(Exception):
    code = "queue_full"


class TaskRunningError(Exception):
    code = "task_running"


def _upload_exists(record: TaskRecord) -> bool:
    return bool(record.upload_path) and Path(record.upload_path).is_file()


class TaskRunner:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.store = TaskStore(self.settings.SQLITE_PATH)
        self._slots = asyncio.Semaphore(self.settings.WORKERS)
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._submit_lock = asyncio.Lock()
        self._cancelled: set[str] = set()
        self._tasks: set[asyncio.Task[None]] = set()
        self._dispatcher: asyncio.Task[None] | None = None
        self._ttl_task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        Path(self.settings.LOG_DIR).mkdir(parents=True, exist_ok=True)
        self._restore_unfinished()
        self._dispatcher = asyncio.create_task(self._dispatch_loop())
        self._ttl_task = asyncio.create_task(self._ttl_loop())

    def _restore_unfinished(self) -> None:
        data_dir = self.settings.DATA_DIR
        for record in self.store.list_tasks(TaskStatus.running):
            if not _upload_exists(record):
                self.store.mark_error(record.task_id, ErrorDetail(code="interrupted"))
                observe_restore("interrupted")
                cleanup_tmp(record.task_id, data_dir)
                continue
            attempts = self.store.bump_attempts(record.task_id)
            if attempts > self.settings.TASK_MAX_RESTARTS:
                self.store.mark_error(
                    record.task_id, ErrorDetail(code="process_killed")
                )
                observe_restore("process_killed")
                cleanup_tmp(record.task_id, data_dir)
            else:
                self.store.reset_to_queued(record.task_id)
        for record in self.store.list_tasks(TaskStatus.queued):
            if not _upload_exists(record):
                self.store.mark_error(record.task_id, ErrorDetail(code="missing_upload"))
                observe_restore("missing_upload")
                cleanup_tmp(record.task_id, data_dir)
        # WORKER_QUEUE_SIZE не применяется: после рестарта очередь может быть длиннее лимита.
        for record in self.store.list_queued_fifo():
            self._queue.put_nowait(record.task_id)
            observe_restore("requeued")

    async def stop(self) -> None:
        if self._dispatcher is not None:
            self._dispatcher.cancel()
        if self._ttl_task is not None:
            self._ttl_task.cancel()
        pending = list(self._tasks)
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        # queued/running tmp сохраняем — после рестарта пайплайн стартует с upload-файла.
        for record in self.store.list_tasks():
            if record.status in {TaskStatus.success, TaskStatus.error}:
                cleanup_tmp(record.task_id, self.settings.DATA_DIR)
        self.store.close()

    async def submit(
        self,
        src_path: str | Path,
        asr_model: AsrModel,
        diarization_model: DiarizationModel | None,
    ) -> TaskRecord:
        async with self._submit_lock:
            if self.store.count_queued() >= self.settings.WORKER_QUEUE_SIZE:
                raise QueueFullError()
            task_id = str(uuid.uuid4())
            tmp = create_tmp(task_id, self.settings.DATA_DIR)
            src = Path(src_path)
            dest = tmp / f"upload{src.suffix.lower()}"
            shutil.copy2(src, dest)
            asr_ckpt, diar_ckpt = checkpoints_for(self.settings, asr_model, diarization_model)
            record = self.store.create(
                task_id,
                asr_model,
                diarization_model,
                asr_ckpt,
                diar_ckpt,
                str(dest),
            )
            await self._queue.put(task_id)
            observe_submitted(
                asr_model.value,
                None if diarization_model is None else diarization_model.value,
            )
            return record

    async def delete(self, task_id: str) -> None:
        status = self.store.delete_if_not_running(task_id)
        if status is None:
            raise KeyError(task_id)
        if status is TaskStatus.running:
            raise TaskRunningError()
        if status is TaskStatus.queued:
            self._cancelled.add(task_id)
        cleanup_tmp(task_id, self.settings.DATA_DIR)

    async def purge(self) -> PurgeResult:
        """Снести queued и историю; running не трогать. SQL только в store."""
        async with self._submit_lock:
            running = self.store.list_tasks(TaskStatus.running)
            keep_ids = {item.task_id for item in running}
            queued = self.store.delete_by_statuses((TaskStatus.queued,))
            for record in queued:
                self._cancelled.add(record.task_id)
            finished = self.store.delete_by_statuses(
                (TaskStatus.success, TaskStatus.error)
            )
            purged_tmp = cleanup_tmp_except(keep_ids, self.settings.DATA_DIR)
            purged_tmp += cleanup_legacy_cwd_tmp()
            cleanup_upload_scratch(self.settings.DATA_DIR)
            return PurgeResult(
                status="ok",
                purged_queued=len(queued),
                purged_finished=len(finished),
                purged_tmp=purged_tmp,
                skipped_running=len(running),
            )

    async def _dispatch_loop(self) -> None:
        while True:
            task_id = await self._queue.get()
            task = asyncio.create_task(self._run_one(task_id))
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)

    async def _run_one(self, task_id: str) -> None:
        try:
            if task_id in self._cancelled:
                return
            async with self._slots:
                if task_id in self._cancelled or self.store.get(task_id) is None:
                    return
                await asyncio.to_thread(run_pipeline, self.store, self.settings, task_id)
        finally:
            self._queue.task_done()

    async def _ttl_loop(self) -> None:
        while True:
            self.store.purge_expired(self.settings.TASK_TTL_SEC)
            await asyncio.sleep(30)
