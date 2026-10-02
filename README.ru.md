# itranscribe-worker

Локальный (on-premise) HTTP-сервис **ASR + опциональная диаризация спикеров**. Отправляете аудио, выбираете ASR (и при необходимости семейство диаризации), опрашиваете задачу, пока не будет готов линейный транскрипт.

**Язык:** [English](README.md) · [Русский](README.ru.md)

## Что это

- На вход: WAV, MP3, M4A, FLAC, OGG, Opus или WebM.
- На выход: **линейный** список реплик (`speaker`, `start`, `end`, `text`) — в один момент одна фраза, без параллельных реплик в JSON. Без диаризации `speaker` равен `null`.
- На каждой задаче выбирается комбинация:
  - ASR: `whisper` или `parakeet` (обязательно)
  - Диаризация: `nemo` или `pyannote`, либо **не указывать / пусто** — только транскрибация, без карты спикеров
- Конкретные чекпоинты (размер Whisper, id Parakeet NeMo, пайплайн PyAnnote, модели NeMo диаризации) задаются в `.env`, не в теле запроса.
- Один процесс Python: каждый слот `WORKERS` — полная копия в памяти всех моделей, которые подняли через `PRELOAD_*`. На диске файлы одни.

`POST /transcribe` отвечает **202** и `task_id`. Результат забирается через `/tasks`.

## Требования

- Python **3.12** (не 3.13/3.14 и не системный `python3`, если это другая версия)
- Виртуальное окружение `.venv`: команды только через `./.venv/bin/python` и `./.venv/bin/pip`
- **pip 25.3** — ставится в venv перед `requirements*.txt` (как в Docker)
- **ffmpeg** в `PATH` (все загрузки → моно 16 kHz WAV)
- Аккаунт Hugging Face и **принятые лицензии** PyAnnote 3.1 (`pyannote/speaker-diarization-3.1` и зависимости). В `.env` нужен `HF_TOKEN` (им же качается Sortformer с Hugging Face). Без токена/лицензии PyAnnote недоступен.
- Диск под `./data` для весов, SQLite, логов и tmp очереди задач (в git не коммитится)



## Установка и запуск

```bash
python3.12 -m venv .venv
./.venv/bin/pip install "pip==25.3"
./.venv/bin/pip install -r requirements.txt
./.venv/bin/pip install -r requirements-ml.txt
```

Проверка окружения: `./.venv/bin/python --version` → **3.12.x**, `./.venv/bin/pip --version` → **pip 25.3**.

Создайте `.env` в корне репозитория (таблица ниже). Файл не коммитить. Затем:

```bash
./.venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 1
```

Всегда **uvicorn** `--workers 1`. Параллелизм задач — это `WORKERS` в `.env` (слоты внутри этого процесса), а не дополнительные процессы uvicorn.

## Тесты

Тот же Python, что в Docker (**только 3.12**). Из корня репозитория:

```bash
PYTHONPATH=. ./.venv/bin/python -m pytest
```

Опционально: `pytest -m "not ml"` — без тяжёлых ML-интеграций; для полного прогона нужен `requirements-ml.txt`.

При первом старте идёт **preload** семейств из `PRELOAD_ASR` и `PRELOAD_DIARIZATION` (по умолчанию `all` = все ASR и диаризация). У каждого семейства первая копия берёт веса из `MODELS_DIR`, если они уже есть, иначе качает с Hugging Face; остальные копии `WORKERS` того же семейства читают только этот кэш (без etag в Hub). Веса пропущенных семейств не скачиваются. Сбой загрузки **запрошенного** семейства — процесс **не стартует** (`PreloadError` в логах). Семейство не в `PRELOAD_*` — `disabled`, на старт не влияет. Повторная задача не должна качать веса, если они уже лежат в `./data/models`.

Проверка:

```bash
curl -s http://127.0.0.1:8000/health
```

Docker: [Docker Compose](#docker-compose) (образы CPU или NVIDIA GPU).

## `.env`

Имена переменных — в `.env`. **Реальные токены не класть в git и не копировать в README.** Смена значения требует перезапуска процесса (уже загруженные чекпоинты остаются в памяти до рестарта).


| Переменная                | Смысл                                                                                                                                                                                                                                                |
| ------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `API_TOKEN`               | Bearer-ключ для всех маршрутов, кроме `/health`. Пустой = никто не пройдёт. Не путать с `HF_TOKEN`. Им же запечатывается `transcript` в SQLite (см. ниже).                                                                                    |
| `HF_TOKEN`                | Токен Hugging Face: PyAnnote, чекпоинты Parakeet/NeMo и tone-модели при необходимости.                                                                                                                                                              |
| `HOST`                    | Интерфейс (`127.0.0.1` локально; в Docker — `0.0.0.0`).                                                                                                                                                                                              |
| `PORT`                    | HTTP-порт (по умолчанию `8000`).                                                                                                                                                                                                                     |
| `DATA_DIR`                | Корень персистентных данных (по умолчанию `./data`): модели, SQLite, логи и tmp очереди `{DATA_DIR}/tmp/<task_id>/`.                                                                                                                                 |
| `MODELS_DIR`              | Веса / кэш HF (по умолчанию `./data/models`).                                                                                                                                                                                                        |
| `SQLITE_PATH`             | БД задач (по умолчанию `./data/tasks.db`). Аудио сюда не пишется; загрузки лежат в `{DATA_DIR}/tmp/`. Колонка `transcript` на диске зашифрована (см. ниже).                                                                                          |
| `LOG_DIR`                 | Каталог прикладных логов (по умолчанию `./data/logs`).                                                                                                                                                                                               |
| `PERFORMANCE_LOG`         | CSV метрик инференса (по умолчанию `./data/logs/performance_log.csv`).                                                                                                                                                                               |
| `LOG_ENABLED`             | Прикладной лог-файл + app-logger. По умолчанию `true`. `false` / `0` / `no` — выкл. Не трогает CSV / `metric_event`.                                                                                                                                 |
| `LOG_MAX_BYTES`           | Ротация `app.log` при превышении размера в байтах. По умолчанию `5242880` (5 MiB).                                                                                                                                                                   |
| `LOG_BACKUP_COUNT`        | Сколько архивов хранить (`app.log.1` … `app.log.N`). По умолчанию `5`.                                                                                                                                                                               |
| `PERFORMANCE_LOG_ENABLED` | Строка CSV + JSON `metric_event` в stdout при завершении задачи. По умолчанию `true`. `false` / `0` / `no` — выкл. Не трогает прикладные логи.                                                                                                       |
| `METRICS_ENABLED`         | Прикладные метрики Prometheus на `GET /metrics`. По умолчанию `true`. `false` / `0` / `no` — только process collectors; endpoint остаётся.                                                                                                           |
| `WHISPER_MODEL`           | Имя Faster-Whisper (по умолчанию `large-v3-turbo`).                                                                                                                                                                                                  |
| `PARAKEET_MODEL`          | Id чекпоинта NeMo ASR (по умолчанию `nvidia/parakeet-tdt-0.6b-v3`). Мультиязычный, с пунктуацией.                                                                                                                                                   |
| `PARAKEET_CHUNK_SEC`      | Макс. секунд на один вызов Parakeet `transcribe()` (по умолчанию `1380`, ~23 мин). Более длинное аудио режется ffmpeg, таймкоды склеиваются.                                                                                                         |
| `PYANNOTE_MODEL`          | Id пайплайна PyAnnote (по умолчанию `pyannote/speaker-diarization-3.1`).                                                                                                                                                                             |
| `NEMO_MODEL`              | Hugging Face id Sortformer для семейства `nemo` (по умолчанию `nvidia/diar_streaming_sortformer_4spk-v2`, лицензия CC-BY-4.0). Максимум 4 спикера.                                                                                                   |
| `PRELOAD_ASR`             | Какие ASR поднимать и скачивать при старте: `whisper`, `parakeet`, `all` (по умолчанию) или подмножество через запятую (`whisper,parakeet`).                                                                                                                                                       |
| `PRELOAD_DIARIZATION`     | Какие диаризации поднимать и скачивать при старте: `nemo`, `pyannote`, `all` (по умолчанию) или подмножество через запятую (`nemo,pyannote`).                                                                                                                                                      |
| `TONE_TEXT_MODEL`         | HF id эмоций по тексту. **Пусто** — слой выключен.                                                                                                                                                                                                                                                 |
| `TONE_PROSODY`            | **Пусто** — слой выключен. Иначе `preset:minimal\|standard\|extended` или список `energy,f0,tempo,pauses`. Поля попадают в `transcript[].tone.prosody` (отдельных строк в таймлайне не добавляется): **energy** — громкость; **f0** — высота голоса; **tempo** — слова/с; **pauses** — `pause_before_sec` (пауза перед репликой того же `speaker`). |
| `TONE_SER_MODEL`          | HF id SER по аудио. **Пусто** — слой выключен.                                                                                                                                                                                                                                                      |
| `PRELOAD_TONE`            | `none` (default), `text`, `ser`, `all` / `text,ser` — preload нейросетей на слот `WORKERS` (только если задан соответствующий id). Просodия preload не требует.                                                                                                                                     |
| `DEVICE`                  | Устройство инференса: `auto` (по умолчанию), `cpu` или `cuda`. `auto` берёт CUDA, если `torch.cuda.is_available()`, иначе CPU. `cpu` — никогда GPU. `cuda` — только GPU; нет CUDA — процесс не стартует. В Docker Compose значение задаётся образом. |
| `WORKERS`                 | Сколько **задач** можно считать сразу в этом процессе. По умолчанию `1`. Это не воркеры uvicorn. Слот — полная копия в памяти всех загруженных моделей (RAM/VRAM × `WORKERS`); на диске файлы одни.                                          |
| `WORKERS_MAX`             | Синоним `WORKERS` (то же значение). В `GET /health` попадает в `workers.max` для Capacity в [idigest-hub](https://github.com/dude75/idigest-hub).                                                                                                    |
| `WORKER_QUEUE_SIZE`       | Сколько задач может висеть в `queued`. По умолчанию `4`. Сверх лимита: `503` `queue_full`.                                                                                                                                                           |
| `MAX_UPLOAD_BYTES`        | Максимум тела `POST /transcribe` в байтах (`Content-Length` и стрим файла). По умолчанию `1073741824` (1 GiB). Сверх лимита: HTTP **413** `payload_too_large`.                                                                                        |
| `ALLOWED_UPLOAD_SUFFIXES` | Расширения загрузки: `all` (по умолчанию — wav, mp3, m4a, flac, ogg, opus, webm) или подмножество через запятую (`wav,mp3,m4a`). Неизвестные значения — ошибка при старте. Запрещённое расширение — HTTP **400** `invalid_file`.                        |
| `TASK_TTL_SEC`            | Через сколько секунд после `success`/`error` удалить строку из SQLite. `0` — не удалять по TTL (только `DELETE`).                                                                                                                                    |
| `FFMPEG_TIMEOUT_SEC`      | Сколько секунд дать ffmpeg на нормализацию любой загрузки (WAV/MP3/M4A/FLAC/OGG/Opus/WebM) в моно 16 кГц WAV. По умолчанию `120`. По таймауту задача уходит в `error` с кодом `ffmpeg_timeout`, процесс ffmpeg убивается. `0` — без лимита.                               |
| `TASK_TIMEOUT_SEC`        | Сколько секунд дать всей задаче (ffmpeg + ASR + диаризация + alignment). По умолчанию `14400` (4 часа). По таймауту задача уходит в `error` с кодом `task_timeout`; текущий этап доигрывается, следующие не стартуют. Нативный инференс посреди вызова не прерывается. `0` — без лимита. |
| `TASK_MAX_RESTARTS`       | Сколько раз задачу, найденную в `running` после смерти процесса, вернуть в `queued`. По умолчанию `1` (одна повторная попытка). Дальше — `error` с кодом `process_killed`. `0` — сразу ошибка при первом restore. CUDA OOM — обычное Python-исключение (`pipeline_error`), в этот счётчик не входит. |

### Рекомендуемые Hugging Face id для tone

Не SLA: перед prod прогоните smoke на своих звонках. A/B — смена id в `.env` и рестарт.

**`TONE_TEXT_MODEL`** (эмоции / sentiment по тексту реплики):

| | **ru** | **en** | **multilingual** |
| --- | --- | --- | --- |
| **lite** | `cointegrated/rubert-tiny2-cedr-emotion-detection` | `j-hartmann/emotion-english-distilroberta-base` | `cardiffnlp/twitter-xlm-roberta-base-sentiment` |
| **heavy** | `cointegrated/rubert-base-cased-sentiment` | `j-hartmann/emotion-english-roberta-large` | `cardiffnlp/twitter-xlm-roberta-large-sentiment-multilingual` |

**`TONE_SER_MODEL`** (эмоция по аудио-слайсу реплики):

| | **ru** | **en** | **multilingual** |
| --- | --- | --- | --- |
| **lite** | `superb/wav2vec2-base-superb-er` * | `superb/wav2vec2-base-superb-er` | `iic/emotion2vec_plus_base` * |
| **heavy** | `ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition` * | `ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition` | `iic/emotion2vec_plus_large` * |

\* SER чаще обучен на англ./acted corpus; на русской телефонии качество может просесть — проверяйте. Для ru-only инстанса разумный старт: text **lite** + просodия (`TONE_PROSODY`), SER опционально.

В `.env` подставляется **один** id на переменную; колонка таблицы — ориентир по языку аудитории.

Всё, что должно пережить рестарт, лежит в `./data` (модели, `tasks.db`, логи **и tmp очереди** `{DATA_DIR}/tmp/`). В Docker монтируйте этот каталог. Контейнер Compose пишет в него от uid/gid **1001** (см. [Docker Compose](#docker-compose)).

Колонка `transcript` в `tasks.db` хранится в Fernet (AES-128-CBC + HMAC). Ключ — `SHA-256(API_TOKEN)`, не сырой токен. `GET /tasks/{id}` по-прежнему отдаёт открытый JSON; в списке задач транскрипта нет. Метаданные, `error` и tmp-аудио не шифруются. Это защита только от утечки `tasks.db` без `.env`. Смена `API_TOKEN` делает уже зашифрованные строки нечитаемыми до TTL или `DELETE`; строки, записанные до этой версии, остаются обычным JSON и читаются как раньше.

После рестарта процесса (или `docker compose restart`) незавершённые задачи восстанавливаются из SQLite и этих tmp-файлов — **не** с середины пайплайна:

- Задачи в `queued` с файлом на диске снова ставятся во внутреннюю очередь (FIFO по `timestamp`). `WORKER_QUEUE_SIZE` при восстановлении **не** применяется: очередь может быть длиннее лимита, пока не разгребётся; новые `POST /transcribe` по-прежнему смотрят на лимит.
- Задача, которая была `running`, возвращается в `queued` и считается заново, если upload-файл на месте, не больше `TASK_MAX_RESTARTS` раз (по умолчанию `1`). Ещё одна смерть процесса после этого — `error` с кодом `process_killed` (kernel OOM-kill, нативный segfault; не CUDA OOM). Если файла нет — `error` с кодом `interrupted`.
- `queued` без файла на диске завершается как `error` с кодом `missing_upload` и в RAM-очередь не попадает.
- Корректное завершение процесса **не** удаляет tmp у queued и running. Tmp завершённых (`success` / `error`) по-прежнему чистится.



## API

Все маршруты, кроме `GET /health`, требуют:

`Authorization: Bearer <API_TOKEN>`

В примерах подставьте `$TOKEN` и `$HOST` (`http://127.0.0.1:8000`).

### Health (без токена)

```bash
curl -s "$HOST/health"
curl -s "$HOST/health" | jq '{version, device, engines, workers}'
```

В JSON — `version` (как в `version.txt`) и какие движки `loaded`, `unavailable` или `disabled` (без секретов). `disabled` — семейство не входило в `PRELOAD_ASR` / `PRELOAD_DIARIZATION`. Поле `device` — `cpu` или `cuda`. Блок `workers` — параллельная ёмкость процесса:

```json
{
  "status": "ok",
  "version": "0.1.3",
  "engines": { "whisper": "loaded", "parakeet": "loaded", "nemo": "loaded", "pyannote": "loaded", "tone_text": "disabled", "tone_ser": "disabled" },
  "device": "cuda",
  "workers": {
    "max": 2,
    "active": 0,
    "available": 2
  }
}
```

| Поле | Смысл |
| ---- | ----- |
| `workers.max` | Параллельных слотов транскрипции на этом процессе (`WORKERS` / `WORKERS_MAX`). |
| `workers.active` | Задач, занимающих слот (`running`). |
| `workers.available` | Свободных слотов: `max - active`. |

[idigest-hub](https://github.com/dude75/idigest-hub) читает `workers.*` для Capacity transcribe-нод (вместо legacy fallback 1/1, когда размер пула неизвестен).

### Метрики

```bash
curl -s -H "Authorization: Bearer $TOKEN" "$HOST/metrics"
```

Текст Prometheus. Process collectors и прикладные gauges/counters/histograms (очередь, движки, тайминги задач). Тот же Bearer, что и у остального API.

Grafana: импорт [`grafana/dashboards/itranscribe-worker.json`](grafana/dashboards/itranscribe-worker.json) (Dashboards → New → Import), datasource — Prometheus, который скрейпит этот endpoint. Пример scrape:

```yaml
scrape_configs:
  - job_name: itranscribe-worker
    metrics_path: /metrics
    scrape_interval: 15s
    authorization:
      credentials: "<API_TOKEN>"
    static_configs:
      - targets: ["127.0.0.1:8000"]
```

Дашборд: очередь, движки, пайплайн/RTF, HTTP, процесс/диск. Задачи только с ASR идут с `diarization_model="none"`. HTTP-панели без scrape `/metrics`. CSV `PERFORMANCE_LOG` сюда не входит.

### Постановка файла → 202

```bash
curl -sS -X POST "$HOST/transcribe" \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@./sample.wav" \
  -F "asr_model=whisper" \
  -F "diarization_model=pyannote"
```

`asr_model`: `whisper` (по умолчанию) или `parakeet`.  
`diarization_model`: `nemo` (Sortformer, максимум 4 спикера) или `pyannote`. Не указывайте поле или передайте пустую строку, чтобы не делать диаризацию (только ASR). Дефолтного семейства нет: нет поля / пусто = без карты спикеров. Для длинных файлов, где важна скорость, в запросе берите `nemo`. `pyannote` — когда важнее его карта спикеров, а не минимальное время.

`tone`: `true` или `false` (по умолчанию `false`). При `true` считаются **непустые** слои в `.env` (`TONE_TEXT_MODEL`, `TONE_PROSODY`, `TONE_SER_MODEL`) **независимо**: недоступный preload (text/ser) **пропускается**, остальные слои и транскрипт не падают; в логах warning. Нет ни одного слоя в `.env` — как `false`, `meta.tone_skipped=true`. Если все настроенные слои недоступны — `success`, `meta.tone_skipped=true`, `meta.tone_layers` пустой или отсутствует. На репликах optional `tone`; при успехе optional `call_summary`, `meta.tone_layers`. Интерпретация полей для людей/LLM: [docs/tone/llm-interpretation.ru.md](docs/tone/llm-interpretation.ru.md).

Пресеты просодии: `minimal` = energy; `standard` = energy + f0; `extended` = + tempo + pauses. Паузы не создают новые интервалы в `transcript` — только метрика на существующей реплике.

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

### Удаление одной задачи

```bash
curl -sS -X DELETE "$HOST/tasks/$TASK_ID" -H "Authorization: Bearer $TOKEN"
```

- `queued` / `success` / `error` → **200**, строка удалена (для queued ещё снимается tmp-аудио).
- `running` → **409** `task_running` (идущий инференс не прерывается).



### Полная очистка очереди и истории

```bash
curl -sS -X DELETE "$HOST/tasks" -H "Authorization: Bearer $TOKEN"
```

Сносит всю очередь и завершённую историю. **Не** отменяет задачу в `running` (строка и её tmp остаются; HTTP **200**, не **409**). Также удаляет сиротские каталоги в `{DATA_DIR}/tmp/`, хвосты CWD `tmp_`* и `{DATA_DIR}/.upload_*`. Не трогает `models/`, `tasks.db` и логи.

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

`purged_tmp` — число каталогов задач, удалённых из `{DATA_DIR}/tmp/`, плюс число снятых устаревших CWD `tmp_*`.

## Docker Compose

Два образа из одного `Dockerfile`: **CPU** (`itranscribe-worker:cpu`) и **NVIDIA GPU** (`itranscribe-worker:gpu`). Compose сам ставит `DEVICE` (`cpu` / `cuda`). Не поднимайте оба стека на порту `8000` одновременно.

### Подготовка

1. Скопируйте `.env.example` → `.env` и заполните `API_TOKEN` / `HF_TOKEN` (см. `[.env](#env)`).
2. Каталог `./data` (веса, SQLite, логи, tmp очереди). Compose монтирует `./data:/data`.
   Процесс в контейнере идёт как **uid/gid 1001** (не root). Этому пользователю нужна запись в `./data`.
   Если каталог уже заполнял старый контейнер от root, один раз поправьте владельца:

   ```bash
   sudo chown -R 1001:1001 ./data
   ```

   Не ставьте `chmod 777`. `docker compose down` каталог `./data` не удаляет.
3. **Только GPU:** драйвер NVIDIA на хосте и [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html). Проверка: `nvidia-smi` и `docker run --rm --gpus all nvidia/cuda:12.9.2-base-ubuntu24.04 nvidia-smi`.



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

Каталог `./data` на хосте не удаляется. После перехода с образа от root выполните `sudo chown -R 1001:1001 ./data` перед следующим `up`, если в логах `Permission denied` на `/data`.

## Типичные ошибки


| Что видно                                                       | Смысл                                                                                                                                            |
| --------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| HTTP **401**, `error.code = unauthorized`                       | Нет / неверный `Authorization: Bearer …`, или пустой `API_TOKEN`.                                                                                |
| `GET /tasks/{id}` 500 после смены `API_TOKEN`                   | Зашифрованный `transcript` запечатан предыдущим токеном. Подождите `TASK_TTL_SEC` или `DELETE` эти задачи.                                       |
| HTTP **503**, `error.code = queue_full`                         | Слишком много задач в `queued` (`WORKER_QUEUE_SIZE`). Подождите или увеличьте лимит и перезапустите.                                             |
| HTTP **413**, `error.code = payload_too_large`                  | Тело `POST /transcribe` больше `MAX_UPLOAD_BYTES` (по умолчанию 1 GiB).                                                                          |
| HTTP **200**, `status=error`, `error.code = engine_unavailable` | Запрошенное семейство `unavailable` или `disabled` в `/health`. Смените `asr_model` / `diarization_model` или поменяйте preload и перезапустите. |
| HTTP **200**, `status=error`, `error.code = missing_upload`     | Upload-файл queued/восстановленной задачи пропал из `{DATA_DIR}/tmp/`.                                                                           |
| HTTP **200**, `status=error`, `error.code = interrupted`        | Процесс умер, пока задача была `running`, и после рестарта файла не оказалось.                                                                   |
| HTTP **200**, `status=error`, `error.code = process_killed`     | Процесс умер на `running` больше `TASK_MAX_RESTARTS` раз (kernel OOM-kill / нативный segfault). Воркер остаётся живым. CUDA OOM — это `pipeline_error`. |
| HTTP **200**, `status=error`, `error.code = ffmpeg_timeout`     | ffmpeg не успел нормализовать загрузку в моно 16 кГц WAV за `FFMPEG_TIMEOUT_SEC`. Процесс конвертера убивается, слот воркера освобождается.        |
| HTTP **200**, `status=error`, `error.code = task_timeout`       | Задача не уложилась в `TASK_TIMEOUT_SEC` (по умолчанию 4 часа). Следующие этапы не стартуют; слот воркера освобождается, когда текущий этап вернётся. |
| HTTP **422**                                                    | Неверный `asr_model` / `diarization_model` (`whisper`/`parakeet`; `nemo`/`pyannote`). Пустой `diarization_model` допустим (без диаризации). |
| `pip install` падает на ML-зависимостях                         | Обычно venv не на **Python 3.12** (например 3.14). Пересоздайте: `python3.12 -m venv .venv`, `pip==25.3`, затем `requirements*.txt`. |
| Whisper / `pipeline_error`: `open() got an unexpected keyword argument 'metadata_errors'` | Случайно стоит **PyAV 19** (`av==19`). Нужно **`av>=14.2,<19`** (см. `requirements-ml.txt`): `./.venv/bin/pip install 'av>=14.2.0,<19'`. |
| UserWarning: `torchcodec is not installed correctly` (macOS) | **torchcodec 0.10** совместим с FFmpeg **4–8**; Homebrew часто ставит **9** (`libavutil.61`). Загрузки у нас уже в **моно 16 kHz WAV** (ffmpeg CLI); PyAnnote читает WAV через **soundfile**, не через torchcodec. Предупреждение на import можно игнорировать. Нужен «чистый» torchcodec — поставьте FFmpeg 8 (`brew install ffmpeg@8`) и добавьте его `lib` в `DYLD_LIBRARY_PATH`. |
| `Permission denied` на `/data/...` (`tasks.db`, `models`, `logs`, `tmp`) | Хостовый `./data` недоступен uid 1001. Выполните `sudo chown -R 1001:1001 ./data` и перезапустите. Не ставьте `chmod 777`. |


