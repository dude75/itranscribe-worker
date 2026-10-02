---
document: itranscribe-worker tone interpretation
audience: LLM / аналитик, читающий JSON задачи GET /tasks/{id}
language: ru
when_to_use: >-
  Пользователь дал transcript с полем tone, call_summary или спрашивает про эмоции/просодию/звонок.
  Не выдумывай метрики — опирайся только на поля JSON и правила ниже.
---

# Интерпретация tone (itranscribe-worker)

## Контекст API

- Tone считается **после** ASR, diarization (если была) и alignment — на **готовых репликах** `transcript[]`.
- Включение: `tone=true` в `POST /transcribe` + непустые слои в `.env` (`TONE_TEXT_MODEL`, `TONE_PROSODY`, `TONE_SER_MODEL`).
- **`meta.tone_text_model`**, **`meta.tone_ser_model`** — HF id, зафиксированные при создании задачи (`null`, если слой выключен в `.env`).
- **`meta.tone_prosody_param`** — список фич просодии через запятую (например `energy,f0`), без `preset:*` (`null`, если `TONE_PROSODY` пустой).
- **`meta.tone_layers`** — какие слои **реально** отработали (например `["text", "prosody", "ser"]`). Если слой в `.env` есть, но preload был `unavailable`, он **пропускается** (warning в логах сервера), пайплайн не падает.
- **`meta.tone_skipped`**: `true` — tone запросили, но нечего считать (нет слоёв в `.env`) или **ни один** настроенный слой не применился.

Поля `tone` на реплике **опциональны**: отсутствие `tone` или `null` = tone для этой задачи не считался или слои не дали данных.

---

## Структура JSON

```text
task
├── meta.tone_requested, meta.tone_text_model, meta.tone_ser_model, meta.tone_prosody_param, meta.tone_layers, meta.tone_skipped, meta.tone_time_sec, ...
├── call_summary?          # сводка по звонку (если есть valence на репликах)
└── transcript[]
    └── tone?              # UtteranceTone на одну реплику
        ├── valence?
        ├── arousal?
        ├── emotions?      # text
        ├── prosody?       # CPU / аудио слайс [start,end]
        └── ser?           # audio emotion
```

---

## Слои и поля

| Слой (.env) | Поля в `tone` | Источник данных |
|-------------|---------------|-----------------|
| **text** (`TONE_TEXT_MODEL`) | `emotions`, `valence` | Текст реплики, HF classification |
| **prosody** (`TONE_PROSODY`) | `prosody`, `arousal` | WAV-слайс реплики, librosa (CPU) |
| **ser** (`TONE_SER_MODEL`) | `ser` | WAV-слайс реплики, HF audio classification |

`valence` **не** берётся из SER и **не** из prosody — только из **text** (`emotions`).

`arousal` **не** из text/SER — только из **prosody** (агрегат energy, F0, tempo).

---

## Поля реплики (`transcript[].tone`)

### `emotions` (object: label → float 0…1)

- **Что:** scores классов эмоций **по тексту** (модель из `TONE_TEXT_MODEL`).
- **Labels:** после нормализации часто `neutral`, `joy`, `sadness`, `anger`, `fear`, `surprise` (зависит от модели; алиасы вроде `happy` → `joy`).
- **Как говорить пользователю:** «модель текста склоняется к …»; не утверждай психологическое состояние человека.
- **Не путать:** это не вероятности с суммой 1, но обычно один класс доминирует.

### `valence` (float ≈ −1…1)

- **Что:** одномерная «позитивность текста», производная от `emotions`.
- **Формула (код):** `joy - anger - sadness - 0.5 * fear`, clip to [−1, 1].
- **Интерпретация:** >0 — больше радости/меньше явного негатива в тексте; <0 — доминирует anger/sadness/fear; около 0 — нейтрально или смешанно.
- **Если `emotions` только neutral:** valence часто около 0.

### `prosody` (object)

Считается на интервале **`[line.start, line.end]`** (моно 16 kHz WAV после ffmpeg).

| Поле | Смысл | Заметки для LLM |
|------|--------|-----------------|
| `energy_mean_db` | Средняя громкость (RMS, dB) | Типичная речь часто ~−35…−10 dB; сравнивай **внутри звонка**. |
| `energy_max_db` | Пик громкости в реплике | Резкие акценты, повышение голоса. |
| `f0_hz_median` | Медиана F0, Гц | «Высота» голоса; зависит от пола/микрофона; только относительные сравнения. |
| `words_per_sec` | слова / длительность | Темп; из текста ASR и таймкодов. |
| `pause_before_sec` | пауза перед репликой | Только если **тот же `speaker`**, что у предыдущей строки; иначе `null`. **Не** отдельная строка transcript. |

Пресеты `TONE_PROSODY`: `minimal` = energy; `standard` = energy+f0; `extended` = + tempo + pauses.

### `arousal` (float 0…1)

- **Что:** условная «активность» по голосу/темпу.
- **Формула (код):** среднее нормализованных: energy (линейно из dB), F0, words_per_sec — каждый в [0,1].
- **Интерпретация:** выше — громче/быстрее/выше F0 в совокупности; ниже — спокойнее. Не «злость» и не «радость».

### `ser` (`emotion`, `score`)

- **Что:** класс эмоции **по аудио** слайса (модель `TONE_SER_MODEL`).
- **`score`:** уверенность модели для выбранного класса (0…1).
- **Ограничение:** многие SER-модели обучены не на вашей телефонии/языке — **может расходиться с text**. Описывай оба как «сигналы моделей», не как ground truth.

---

## Сводка звонка (`call_summary`)

Строится только если есть **`tone.valence`** хотя бы на части реплик (нужен **text**-слой).

| Поле | Правило (код) | Смысл для LLM |
|------|----------------|---------------|
| `opening_valence` | Среднее valence реплик с `start < 60` с | «Тон текста в первой минуте» (грубо). |
| `closing_valence` | Среднее valence с `start >= duration - 60` с | «Тон текста в последней минуте». |
| `de_escalation` | `true` iff opening < **−0.2** AND closing > **+0.2** | Эвристика «было заметно негativно → стало заметно позитивнее». **Не** общий «успех звонка». |

Окно **60 с** — константа в коде, не настраивается через `.env`.

SER и prosody в `call_summary` **не участвуют**.

---

## Правила для LLM (DO / DON'T)

**DO**

- Смотри **`meta.tone_layers`** и явно говори, какие сигналы доступны (text / prosody / ser).
- Сравнивай реплики и спикеров **внутри одного звонка**, не с абсолютными порогами из других доменов.
- При расхождении text vs ser формулируй: «текст звучит нейтрально, акустическая модель тоже neutral» или «текст negative, SER neutral — возможное несоответствие моделей».
- Упоминай, что метрики зависят от ASR (ошибки в тексте → ошибки text-tone).

**DON'T**

- Не диагностируй психику, не ставь медицинские/HR-вердикты («депрессия», «агрессивный клиент») только по tone.
- Не утверждай причинно-следственные связи («из-за pause 2 с клиент недоволен») без других данных.
- Не интерпретируй отсутствующие поля как «нейтрально» — они **не измерялись**.
- Не используй `de_escalation: false` как «эскалация была» — это только «не выполнились пороги de-escalation».

---

## Шаблон краткого разбора для пользователя

1. **Доступные слои:** … (`meta.tone_layers`).
2. **Звонок:** opening/closing valence, de_escalation (если есть `call_summary`).
3. **Эпизоды:** 1–3 реплики с крайними valence / arousal / pause / расхождением text vs ser.
4. **Оговорки:** модели, язык, качество ASR.

---

## Пример (фрагмент)

```json
"tone": {
  "valence": -0.01,
  "arousal": 0.48,
  "emotions": { "neutral": 0.93, "joy": 0.008, ... },
  "prosody": {
    "energy_mean_db": -46.3,
    "f0_hz_median": 649,
    "words_per_sec": 1.72,
    "pause_before_sec": null
  },
  "ser": { "emotion": "neutral", "score": 0.99 }
}
```

**Краткая интерпретация:** текст реплики для rubert-модели преимущественно **neutral**, valence ≈ 0; по голосу умеренная активность (arousal ~0.5), не громкая реплика; SER согласуется с neutral. Без других реплик звонок целиком не обобщать.

---

## Ссылки в репозитории

- Конфиг: `README.ru.md` — `TONE_*`, `PRELOAD_TONE`, таблицы HF-моделей.
- Схемы: `app/schemas.py` — `UtteranceTone`, `ProsodyFeatures`, `CallSummary`.
- Формулы: `app/tone/emotions.py`, `app/tone/summary.py`.
- English version: [llm-interpretation.md](./llm-interpretation.md).
