# itranscribe-worcker

**Версия:** `0.0.1`

Локальный (on-premise) HTTP-сервис **ASR + опциональная диаризация спикеров**. Отправляете аудио, выбираете ASR (и при необходимости семейство диаризации), опрашиваете задачу, пока не будет готов линейный транскрипт.

**Язык:** [English](README.md) · [Русский](README.ru.md)

## Что это

- На вход: WAV, MP3 или M4A.
- На выход: **линейный** список реплик (`speaker`, `start`, `end`, `text`) — в один момент одна фраза, без параллельных реплик в JSON. Без диаризации `speaker` равен `null`.
- На каждой задаче выбирается комбинация:
  - ASR: `whisper` или `gigaam` (обязательно)
  - Диаризация: `nemo` или `pyannote`, либо **не указывать / пусто** — только транскрибация, без карты спикеров
- Конкретные чекпоинты (размер Whisper, имя GigaAM, пайплайн PyAnnote, модели NeMo) задаются в `.env`, не в теле запроса.
- Один процесс Python: выбранные семейства грузятся один раз при старте и общие для слотов задач.

`POST /transcribe` отвечает **202** и `task_id`. Результат забирается через `/tasks`.

## Требования

- Python **3.12**
- Виртуальное окружение **`.venv`** (только `./.venv/bin/python` и `./.venv/bin/pip`)
- **ffmpeg** в `PATH` (MP3/M4A → WAV, GigaAM longform)
- Аккаунт Hugging Face и **принятые лицензии** PyAnnote 3.1 (`pyannote/speaker-diarization-3.1` и зависимости). В `.env` нужен `HF_TOKEN` (им же качается Sortformer с Hugging Face). Без токена/лицензии PyAnnote недоступен.
- Диск под `./data` для весов, SQLite и логов (в git не коммитится)

## Установка и запуск

```bash
python3.12 -m venv .venv
./.venv/bin/pip install -U pip
./.venv/bin/pip install -r requirements.txt
./.venv/bin/pip install -r requirements-ml.txt
```

Создайте `.env` в корне репозитория (таблица ниже). Файл не коммитить. Затем:

```bash
./.venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 1
```

Всегда **uvicorn `--workers 1`**. Параллелизм задач — это `WORKERS` в `.env` (слоты внутри этого процесса), а не дополнительные процессы uvicorn.

При первом старте идёт **preload** семейств из `PRELOAD_ASR` и `PRELOAD_DIARIZATION` (по умолчанию `all` = все четыре). Веса пропущенных семейств не скачиваются. Сбой загрузки — `unavailable`; семейство выключено в preload — `disabled`. Процесс живой. Повторная задача не должна качать веса, если они уже лежат в `./data/models`.

Проверка:

```bash
curl -s http://127.0.0.1:8000/health
```

Docker: [Docker Compose](#docker-compose) (образы CPU или NVIDIA GPU).

## `.env`

Имена переменных — в `.env`. **Реальные токены не класть в git и не копировать в README.** Смена значения требует перезапуска процесса (уже загруженные чекпоинты остаются в памяти до рестарта).

| Переменная | Смысл |
|------------|--------|
| `API_TOKEN` | Bearer-ключ для всех маршрутов, кроме `/health`. Пустой = никто не пройдёт. Не путать с `HF_TOKEN`. |
| `HF_TOKEN` | Токен Hugging Face: скачать PyAnnote, VAD для GigaAM longform и чекпоинт NeMo Sortformer. |
| `HOST` | Интерфейс (`127.0.0.1` локально; в Docker — `0.0.0.0`). |
| `PORT` | HTTP-порт (по умолчанию `8000`). |
| `DATA_DIR` | Корень персистентных данных (по умолчанию `./data`). |
| `MODELS_DIR` | Веса / кэш HF (по умолчанию `./data/models`). |
| `SQLITE_PATH` | БД задач (по умолчанию `./data/tasks.db`). Аудио сюда не пишется. |
| `LOG_DIR` | Каталог прикладных логов (по умолчанию `./data/logs`). |
| `PERFORMANCE_LOG` | CSV метрик инференса (по умолчанию `./data/logs/performance_log.csv`). |
| `LOG_ENABLED` | Прикладной лог-файл + app-logger. По умолчанию `true`. `false` / `0` / `no` — выкл. Не трогает CSV / `metric_event`. |
| `PERFORMANCE_LOG_ENABLED` | Строка CSV + JSON `metric_event` в stdout при завершении задачи. По умолчанию `true`. `false` / `0` / `no` — выкл. Не трогает прикладные логи. |
| `WHISPER_MODEL` | Имя Faster-Whisper (по умолчанию `large-v3-turbo`). |
| `GIGAAM_MODEL` | Имя для `gigaam.load_model` (по умолчанию `multilingual_large_ctc`). |
| `PYANNOTE_MODEL` | Id пайплайна PyAnnote (по умолчанию `pyannote/speaker-diarization-3.1`). |
| `NEMO_MODEL` | Hugging Face id Sortformer для семейства `nemo` (по умолчанию `nvidia/diar_streaming_sortformer_4spk-v2`, лицензия CC-BY-4.0). Максимум 4 спикера. |
| `PRELOAD_ASR` | Какие ASR поднимать и скачивать при старте: `whisper`, `gigaam` или `all` (по умолчанию). |
| `PRELOAD_DIARIZATION` | Какие диаризации поднимать и скачивать при старте: `nemo`, `pyannote` или `all` (по умолчанию). |
| `DEVICE` | Устройство инференса: `auto` (по умолчанию), `cpu` или `cuda`. `auto` берёт CUDA, если `torch.cuda.is_available()`, иначе CPU. `cpu` — никогда GPU. `cuda` — только GPU; нет CUDA — процесс не стартует. В Docker Compose значение задаётся образом. |
| `WORKERS` | Сколько **задач** можно считать сразу в этом процессе. По умолчанию `1`. Это не воркеры uvicorn; веса общие. |
| `WORKER_QUEUE_SIZE` | Сколько задач может висеть в `queued`. По умолчанию `4`. Сверх лимита: `503` `queue_full`. |
| `TASK_TTL_SEC` | Через сколько секунд после `success`/`error` удалить строку из SQLite. `0` — не удалять по TTL (только `DELETE`). |

Всё, что должно пережить рестарт, лежит в `./data` (модели, `tasks.db`, логи). В Docker монтируйте этот каталог.

## API

Все маршруты, кроме `GET /health`, требуют:

`Authorization: Bearer <API_TOKEN>`

В примерах подставьте `$TOKEN` и `$HOST` (`http://127.0.0.1:8000`).

### Health (без токена)

```bash
curl -s "$HOST/health"
```

В JSON — `version` (как в `version.txt`) и какие движки `loaded`, `unavailable` или `disabled` (без секретов). `disabled` — семейство не входило в `PRELOAD_ASR` / `PRELOAD_DIARIZATION`. Поле `device` — `cpu` или `cuda`.

### Постановка файла → 202

```bash
curl -sS -X POST "$HOST/transcribe" \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@./sample.wav" \
  -F "asr_model=whisper" \
  -F "diarization_model=pyannote"
```

`asr_model`: `whisper` (по умолчанию) или `gigaam`.  
`diarization_model`: `nemo` (Sortformer, максимум 4 спикера) или `pyannote`. Не указывайте поле или передайте пустую строку, чтобы не делать диаризацию (только ASR). Дефолтного семейства нет: нет поля / пусто = без карты спикеров.

### Опрос одной задачи

```bash
TASK_ID=4f8b9e12-87c2-4911-bca4-d832e12cf900
curl -sS "$HOST/tasks/$TASK_ID" -H "Authorization: Bearer $TOKEN"
```

`status`: `queued` | `running` | `success` | `error`. При успехе заполнен `transcript`. Ошибка **задачи** (движок, файл, инференс) — HTTP всё равно **200**, `"status": "error"` и объект `error`; опрашивайте тот же URL. Нет такого id → **404**.

Пример цикла опроса:

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

### Список задач

```bash
curl -sS "$HOST/tasks" -H "Authorization: Bearer $TOKEN"
curl -sS "$HOST/tasks?status=success" -H "Authorization: Bearer $TOKEN"
```

Новые сверху. Transcript в списке нет.

### Удаление

```bash
curl -sS -X DELETE "$HOST/tasks/$TASK_ID" -H "Authorization: Bearer $TOKEN"
```

- `queued` / `success` / `error` → **200**, строка удалена (для queued ещё снимается tmp-аудио).
- `running` → **409** `task_running` (идущий инференс не прерывается).

## Docker Compose

Два образа из одного `Dockerfile`: **CPU** (`itranscribe-worcker:cpu`) и **NVIDIA GPU** (`itranscribe-worcker:gpu`). Compose сам ставит `DEVICE` (`cpu` / `cuda`). Не поднимайте оба стека на порту `8000` одновременно.

### Подготовка

1. Скопируйте `.env.example` → `.env` и заполните `API_TOKEN` / `HF_TOKEN` (см. [`.env`](#env)).
2. Каталог `./data` (веса, SQLite, логи). Compose монтирует `./data:/data`.
3. **Только GPU:** драйвер NVIDIA на хосте и [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html). Проверка: `nvidia-smi` и `docker run --rm --gpus all nvidia/cuda:12.6.3-base-ubuntu24.04 nvidia-smi`.

### Запуск

CPU:

```bash
docker compose up --build
```

GPU:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```

Добавьте `-d`, чтобы запустить в фоне (`docker compose logs -f` для логов). Порт: `8000:8000`. Первый старт — preload движков (как локально). Веса остаются в `./data/models` на хосте.

Дальше те же `curl` к API на `http://127.0.0.1:8000`.

```bash
docker compose down
docker compose -f docker-compose.yml -f docker-compose.gpu.yml down
```

Каталог `./data` на хосте не удаляется.

## Типичные ошибки

| Что видно | Смысл |
|-----------|--------|
| HTTP **401**, `error.code = unauthorized` | Нет / неверный `Authorization: Bearer …`, или пустой `API_TOKEN`. |
| HTTP **503**, `error.code = queue_full` | Слишком много задач в `queued` (`WORKER_QUEUE_SIZE`). Подождите или увеличьте лимит и перезапустите. |
| HTTP **200**, `status=error`, `error.code = engine_unavailable` | Запрошенное семейство `unavailable` или `disabled` в `/health`. Смените `asr_model` / `diarization_model` или поменяйте preload и перезапустите. |
| HTTP **422** | Неверный `asr_model` / `diarization_model` (`whisper`/`gigaam`; `nemo`/`pyannote`). Пустой `diarization_model` допустим (без диаризации). |
