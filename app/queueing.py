"""Слоты WORKERS и очередь queued (WORKER_QUEUE_SIZE)."""

from __future__ import annotations

import asyncio
import shutil
import uuid
from pathlib import Path

from app.audio import cleanup_tmp, create_tmp
from app.config import Settings, get_settings
from app.pipeline import checkpoints_for, run_pipeline
from app.schemas import AsrModel, DiarizationModel, TaskStatus
from app.tasks import TaskRecord, TaskStore


class QueueFullError(Exception):
    code = "queue_full"


class TaskRunningError(Exception):
    code = "task_running"


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
        for task_id in self.store.interrupt_running():
            cleanup_tmp(task_id)
        for record in self.store.list_tasks(TaskStatus.queued):
            await self._queue.put(record.task_id)
        self._dispatcher = asyncio.create_task(self._dispatch_loop())
        self._ttl_task = asyncio.create_task(self._ttl_loop())

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
        for record in self.store.list_tasks():
            cleanup_tmp(record.task_id)
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
            tmp = create_tmp(task_id)
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
            return record

    async def delete(self, task_id: str) -> None:
        status = self.store.delete_if_not_running(task_id)
        if status is None:
            raise KeyError(task_id)
        if status is TaskStatus.running:
            raise TaskRunningError()
        if status is TaskStatus.queued:
            self._cancelled.add(task_id)
        cleanup_tmp(task_id)

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
