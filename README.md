# itranscribe-worker

On-premise **ASR + optional speaker diarization** HTTP service. Submit an audio file, pick ASR (and optionally a diarization family), poll the task until the linear transcript is ready.

**Language:** [English](README.md) · [Русский](README.ru.md)

## What it does

- Input: WAV, MP3, or M4A.
- Output: a **linear** list of utterances (`speaker`, `start`, `end`, `text`) — one phrase at a time, not overlapping JSON. Without diarization, `speaker` is `null`.
- Each task chooses a combination:
  - ASR: `whisper` or `gigaam` (required)
  - Diarization: `nemo` or `pyannote`, or **omit / empty** to skip diarization (transcription only)
- Concrete checkpoints (Whisper size, GigaAM name, PyAnnote pipeline, NeMo models) are set in `.env`, not in the request body.
- One Python process: selected families are loaded once at startup and shared by task slots.

`POST /transcribe` returns **202** with a `task_id`. Fetch the result from `/tasks`.

## Requirements

- Python **3.12**
- Virtualenv at `.venv` (use `./.venv/bin/python` and `./.venv/bin/pip` only)
- **ffmpeg** on `PATH` (MP3/M4A → WAV, GigaAM longform)
- Hugging Face account + **accepted licenses** for PyAnnote 3.1 (`pyannote/speaker-diarization-3.1` and its dependencies). Set `HF_TOKEN` in `.env` (the same token downloads Sortformer from Hugging Face). Without a token/license, PyAnnote is unavailable.
- Disk under `./data` for model weights, SQLite, logs, and the task queue tmp (not committed)



## Install and run

```bash
python3.12 -m venv .venv
./.venv/bin/pip install -U pip
./.venv/bin/pip install -r requirements.txt
./.venv/bin/pip install -r requirements-ml.txt
```

Create a `.env` in the repo root (see table below). Do not commit it. Then:

```bash
./.venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 1
```

Always keep **uvicorn** `--workers 1`. Parallelism of jobs is `WORKERS` in `.env` (slots inside this one process), not extra uvicorn processes.

First start **preloads** the families chosen by `PRELOAD_ASR` and `PRELOAD_DIARIZATION` (default `all` = all four). Weights for skipped families are not downloaded. A failed engine is `unavailable`; a skipped one is `disabled`. The process stays up. The first real task should not download weights again if they already sit in `./data/models`.

Check:

```bash
curl -s http://127.0.0.1:8000/health
```

Docker: [Docker Compose](#docker-compose) (CPU or NVIDIA GPU images).

## `.env`

Copy names into `.env`. **Do not put real tokens in git or in this README.** Changing a value requires a process restart (loaded checkpoints stay in memory until then).


| Variable                  | Meaning                                                                                                                                                                                                                                     |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `API_TOKEN`               | Bearer key for all routes except `/health`. Empty = nobody is authorized. Not the same as `HF_TOKEN`.                                                                                                                                       |
| `HF_TOKEN`                | Hugging Face token: download PyAnnote, VAD used by GigaAM longform, and the NeMo Sortformer checkpoint.                                                                                                                                     |
| `HOST`                    | Bind address (`127.0.0.1` locally; Docker uses `0.0.0.0`).                                                                                                                                                                                  |
| `PORT`                    | HTTP port (default `8000`).                                                                                                                                                                                                                 |
| `DATA_DIR`                | Persistent root (default `./data`): models, SQLite, logs, and queue tmp at `{DATA_DIR}/tmp/<task_id>/`.                                                                                                                                     |
| `MODELS_DIR`              | Model weights / HF cache (default `./data/models`).                                                                                                                                                                                         |
| `SQLITE_PATH`             | Task database (default `./data/tasks.db`). Audio is not stored here; uploads live under `{DATA_DIR}/tmp/`.                                                                                                                                  |
| `LOG_DIR`                 | Application log directory (default `./data/logs`).                                                                                                                                                                                          |
| `PERFORMANCE_LOG`         | Inference metrics CSV (default `./data/logs/performance_log.csv`).                                                                                                                                                                          |
| `LOG_ENABLED`             | Application file log + app logger. Default `true`. `false` / `0` / `no` = off. Does not affect CSV / `metric_event`.                                                                                                                        |
| `PERFORMANCE_LOG_ENABLED` | CSV row + JSON `metric_event` on stdout when a task finishes. Default `true`. `false` / `0` / `no` = off. Does not affect app logs.                                                                                                         |
| `WHISPER_MODEL`           | Faster-Whisper checkpoint name (default `large-v3-turbo`).                                                                                                                                                                                  |
| `GIGAAM_MODEL`            | `gigaam.load_model` name (default `multilingual_large_ctc`).                                                                                                                                                                                |
| `PYANNOTE_MODEL`          | PyAnnote pipeline id (default `pyannote/speaker-diarization-3.1`).                                                                                                                                                                          |
| `NEMO_MODEL`              | Hugging Face id of Sortformer for the `nemo` family (default `nvidia/diar_streaming_sortformer_4spk-v2`, CC-BY-4.0). Maximum 4 speakers.                                                                                                    |
| `PRELOAD_ASR`             | Which ASR families to load and download at startup: `whisper`, `gigaam`, or `all` (default).                                                                                                                                                |
| `PRELOAD_DIARIZATION`     | Which diarization families to load and download at startup: `nemo`, `pyannote`, or `all` (default).                                                                                                                                         |
| `DEVICE`                  | Inference device: `auto` (default), `cpu`, or `cuda`. `auto` uses CUDA when `torch.cuda.is_available()`, otherwise CPU. `cpu` never uses the GPU. `cuda` requires CUDA or the process fails at startup. Docker Compose sets this per image. |
| `WORKERS`                 | How many **tasks** may run at once in this process. Default `1`. Not uvicorn workers; weights are shared.                                                                                                                                   |
| `WORKER_QUEUE_SIZE`       | Max `queued` tasks waiting for a slot. Default `4`. Beyond that: `503` `queue_full`.                                                                                                                                                        |
| `TASK_TTL_SEC`            | Seconds after `success`/`error` before the SQLite row is deleted. `0` = no TTL (delete only via `DELETE`).                                                                                                                                  |


Everything that must survive a restart lives under `./data` (models, `tasks.db`, logs, **and queue tmp** `{DATA_DIR}/tmp/`). Mount that directory in Docker.

After a process restart (or `docker compose restart`) unfinished work is restored from SQLite + those tmp files — **not** resumed mid-pipeline:

- `queued` tasks with an upload file on disk are put back on the in-memory queue (FIFO by `timestamp`). `WORKER_QUEUE_SIZE` is **not** applied on restore, so the queue may be longer than the limit until it drains; new `POST /transcribe` still uses the limit.
- A task that was `running` is set back to `queued` and run from scratch if its upload file still exists. If the file is gone, it finishes as `error` with `interrupted`.
- A `queued` task whose upload file is missing finishes as `error` with `missing_upload` and is not enqueued.
- Graceful shutdown does **not** delete tmp for queued or running tasks. Finished (`success` / `error`) tmp is still cleaned.



## API

All routes except `GET /health` require:

`Authorization: Bearer <API_TOKEN>`

Replace `$TOKEN` and `$HOST` in the examples (`http://127.0.0.1:8000`).

### Health (no token)

```bash
curl -s "$HOST/health"
```

JSON includes `version` (same as `version.txt`), which engines are `loaded`, `unavailable`, or `disabled` (no secrets). `disabled` means the family was left out of `PRELOAD_ASR` / `PRELOAD_DIARIZATION`. `device` is `cpu` or `cuda`.

### Submit a file → 202

```bash
curl -sS -X POST "$HOST/transcribe" \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@./sample.wav" \
  -F "asr_model=whisper" \
  -F "diarization_model=pyannote"
```

`asr_model`: `whisper` (default) or `gigaam`.  
`diarization_model`: `nemo` (Sortformer, max 4 speakers) or `pyannote`. Omit the field or send it empty to skip diarization (ASR only). There is no default family — missing/empty means no speaker map.

### Poll one task

```bash
TASK_ID=4f8b9e12-87c2-4911-bca4-d832e12cf900
curl -sS "$HOST/tasks/$TASK_ID" -H "Authorization: Bearer $TOKEN"
```

`status` is `queued` | `running` | `success` | `error`. On success, `transcript` is filled. On a **task** error (engine, file, inference) HTTP is still **200** with `"status": "error"` and an `error` object — keep polling the same URL. Unknown id → **404**.

Example poll loop:

```bash
TASK_ID=$(curl -sS -X POST "$HOST/transcribe" \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@./sample.wav" \
  -F "asr_model=whisper" \
  -F "diarization_model=pyannote" | python3 -c "import sys,json; print(json.load(sys.stdin)['meta']['task_id'])")

while true; do
  body=$(curl -sS "$HOST/tasks/$TASK_ID" -H "Authorization: Bearer $TOKEN")
  status=$(printf '%s' "$body" | python3 -c "import sys,json; print(json.load(sys.stdin)['status'])")
  echo "$status"
  case "$status" in success|error) printf '%s\n' "$body"; break ;; esac
  sleep 2
done
```



### List tasks

```bash
curl -sS "$HOST/tasks" -H "Authorization: Bearer $TOKEN"
curl -sS "$HOST/tasks?status=success" -H "Authorization: Bearer $TOKEN"
```

Newest first. No transcript in the list.

### Delete one task

```bash
curl -sS -X DELETE "$HOST/tasks/$TASK_ID" -H "Authorization: Bearer $TOKEN"
```

- `queued` / `success` / `error` → **200**, row removed (queued also drops tmp audio).
- `running` → **409** `task_running` (in-flight inference is not cancelled).



### Purge queue and history

```bash
curl -sS -X DELETE "$HOST/tasks" -H "Authorization: Bearer $TOKEN"
```

Clears the whole queue and finished history. **Does not** cancel a task that is currently `running` (those rows and their tmp stay; HTTP **200**, not **409**). Also removes orphan dirs under `{DATA_DIR}/tmp/` plus leftover CWD `tmp_`* and `{DATA_DIR}/.upload_*`. Does not touch `models/`, `tasks.db`, or logs.

JSON **200**:

```json
{
  "status": "ok",
  "purged_queued": 0,
  "purged_finished": 0,
  "purged_tmp": 0,
  "skipped_running": 0
}
```

`purged_tmp` is the number of task directories removed under `{DATA_DIR}/tmp/` plus any legacy CWD `tmp_*` directories removed.

## Docker Compose

Two images from the same `Dockerfile`: **CPU** (`itranscribe-worcker:cpu`) and **NVIDIA GPU** (`itranscribe-worcker:gpu`). Compose sets `DEVICE` per image (`cpu` / `cuda`). Do not run both stacks on port `8000` at the same time.

### Prepare

1. Copy `.env.example` → `.env` and fill `API_TOKEN` / `HF_TOKEN` (see `[.env](#env)`).
2. Create `./data` if it does not exist (weights, SQLite, logs, queue tmp). Compose mounts `./data:/data`.
3. **GPU only:** NVIDIA driver on the host and [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html). Check: `nvidia-smi` and `docker run --rm --gpus all nvidia/cuda:12.6.3-base-ubuntu24.04 nvidia-smi`.



### Run

CPU:

```bash
docker compose up --build
```

GPU:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```

Add `-d` to run in the background (`docker compose logs -f` for logs). Published port: `8000:8000`. First start preloads engines (same as local). Weights stay in `./data/models` on the host.

Then the same API `curl` examples against `http://127.0.0.1:8000`.

```bash
docker compose down
docker compose -f docker-compose.yml -f docker-compose.gpu.yml down
```

`./data` on the host is not deleted.

## Typical errors


| What you see                                                    | Meaning                                                                                                                                   |
| --------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| HTTP **401**, `error.code = unauthorized`                       | Missing/wrong `Authorization: Bearer …`, or empty `API_TOKEN`.                                                                            |
| HTTP **503**, `error.code = queue_full`                         | Too many `queued` tasks (`WORKER_QUEUE_SIZE`). Wait or raise the limit and restart.                                                       |
| HTTP **200**, `status=error`, `error.code = engine_unavailable` | Requested family is `unavailable` or `disabled` in `/health`. Switch `asr_model` / `diarization_model`, or change preload and restart.    |
| HTTP **200**, `status=error`, `error.code = missing_upload`     | Upload file for a queued/restored task is gone from `{DATA_DIR}/tmp/`.                                                                    |
| HTTP **200**, `status=error`, `error.code = interrupted`        | Process died while the task was `running` and the upload file was missing after restart.                                                  |
| HTTP **422**                                                    | Invalid `asr_model` / `diarization_model` (`whisper`/`gigaam`; `nemo`/`pyannote`). Empty `diarization_model` is valid (skip diarization). |


