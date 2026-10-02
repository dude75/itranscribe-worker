---
document: itranscribe-worker tone interpretation
audience: LLM / analyst reading GET /tasks/{id} JSON
language: en
when_to_use: >-
  User provided a transcript with tone, call_summary, or asks about emotions/prosody.
  Do not invent metrics — use only JSON fields and the rules below.
---

# Interpreting tone (itranscribe-worker)

See the full Russian reference (same structure): [llm-interpretation.ru.md](./llm-interpretation.ru.md).

## API context

- Tone runs **after** ASR, optional diarization, and alignment on `transcript[]` lines.
- Request: `tone=true` on `POST /transcribe` plus non-empty layers in `.env` (`TONE_TEXT_MODEL`, `TONE_PROSODY`, `TONE_SER_MODEL`).
- **`meta.tone_text_model`**, **`meta.tone_ser_model`**: HF ids configured on the server when the task was created (`null` if the layer is off).
- **`meta.tone_prosody_param`**: comma-separated prosody features (e.g. `energy,f0`), not a `preset:*` label (`null` if `TONE_PROSODY` is empty).
- **`meta.tone_layers`**: layers that **actually** ran. Unavailable preload (text/ser) is **skipped** with a server warning; the task still succeeds.
- **`meta.tone_skipped`**: `true` if tone was requested but no layers in `.env`, or no layer could be applied.

## Layers → fields

| Layer | `tone` fields | Source |
|-------|---------------|--------|
| **text** | `emotions`, `valence` | Utterance text (HF model) |
| **prosody** | `prosody`, `arousal` | Audio slice `[start, end]` (librosa) |
| **ser** | `ser` | Audio slice (HF audio classification) |

- **`valence`**: from **text** only (`joy - anger - sadness - 0.5*fear`, clipped to [−1, 1]).
- **`arousal`**: from **prosody** only (normalized energy, F0, words/sec).

## `call_summary`

Requires **text** (`valence` on some lines).

- **`opening_valence`**: mean valence for lines with `start < 60s`.
- **`closing_valence`**: mean valence for lines in the last 60s of audio.
- **`de_escalation`**: `true` only if opening < −0.2 **and** closing > +0.2 (heuristic de-escalation, not “call failed”).

## LLM rules

**Do:** respect `meta.tone_layers`; compare within one call; treat text vs SER as separate model signals; note ASR errors affect text tone.

**Don't:** clinical/HR verdicts from tone alone; causal claims from pauses alone; treat missing fields as “neutral”; read `de_escalation: false` as “escalation happened”.

## Code references

- `app/schemas.py`, `app/tone/emotions.py`, `app/tone/summary.py`
- Config: `README.md` (TONE_*)
