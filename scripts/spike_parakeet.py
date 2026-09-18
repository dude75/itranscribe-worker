#!/usr/bin/env python3
"""Spike: Parakeet TDT 0.6B v3 через NeMo — go/no-go перед интеграцией в itranscribe-worker."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MODELS_DIR = Path(os.environ.get("MODELS_DIR", ROOT / "data" / "models"))
MODEL_NAME = os.environ.get("PARAKEET_MODEL", "nvidia/parakeet-tdt-0.6b-v3")
DEVICE = os.environ.get("DEVICE", "auto").strip().lower()
SPIKE_MAX_SEC = float(os.environ.get("SPIKE_MAX_SEC", "1440"))  # Parakeet ~24 min


def _load_dotenv() -> None:
    env_path = ROOT / ".env"
    if not env_path.is_file():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def _resolve_device() -> str:
    import torch

    if DEVICE == "cpu":
        return "cpu"
    if DEVICE == "cuda":
        if not torch.cuda.is_available():
            sys.exit("DEVICE=cuda, но CUDA недоступна")
        return "cuda"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _speech_wav(path: Path, text: str, voice: str | None = None) -> Path:
    if shutil.which("say") is None:
        sys.exit(
            f"Нет {path}. Установите SPIKE_WAV_RU/SPIKE_WAV_EN или запустите на macOS с `say`."
        )
    cmd = ["say", "-o", str(path), "--data-format=LEI16@16000", text]
    if voice:
        cmd[1:1] = ["-v", voice]
    subprocess.run(cmd, check=True, capture_output=True)
    return path


def _wav_duration(path: Path) -> float:
    from app.audio import audio_duration_sec

    return audio_duration_sec(path)


def _prepare_mono_wav(src: Path, tmp: Path) -> Path:
    """Моно 16 kHz WAV, как в пайплайне itranscribe-worker."""
    from app.audio import prepare_wav

    suffix = src.suffix.lower()
    if suffix == ".wav":
        import soundfile as sf

        info = sf.info(str(src))
        if info.channels == 1 and info.samplerate == 16000:
            return src
    wav = prepare_wav(src, "spike_parakeet", ROOT / "data", timeout_sec=0)
    return wav


def _maybe_trim(wav: Path, tmp: Path) -> Path:
    duration = _wav_duration(wav)
    if SPIKE_MAX_SEC <= 0 or duration <= SPIKE_MAX_SEC:
        return wav
    trimmed = tmp / f"{wav.stem}_trim.wav"
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-y",
            "-i",
            str(wav),
            "-t",
            str(SPIKE_MAX_SEC),
            str(trimmed),
        ],
        check=True,
        capture_output=True,
    )
    print(
        f"WARN: {duration:.0f}s > {SPIKE_MAX_SEC:.0f}s — spike берёт первые "
        f"{SPIKE_MAX_SEC / 60:.0f} мин (лимит Parakeet). "
        f"SPIKE_MAX_SEC=0 чтобы не резать."
    )
    return trimmed


def _technical_audio() -> Path | None:
    technical = ROOT / "technical"
    if not technical.is_dir():
        return None
    exts = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}
    files = sorted(
        p for p in technical.iterdir() if p.is_file() and p.suffix.lower() in exts
    )
    return files[0] if files else None


def _pick_wavs(tmp: Path) -> list[tuple[str, Path]]:
    out: list[tuple[str, Path]] = []
    ru = os.environ.get("SPIKE_WAV_RU")
    en = os.environ.get("SPIKE_WAV_EN")
    technical = _technical_audio()
    if ru:
        out.append(("ru-custom", Path(ru)))
    elif technical is not None:
        out.append(("technical", technical))
    else:
        out.append(("ru-say", _speech_wav(tmp / "spike_ru.wav", "привет мир это тест")))
    if en:
        out.append(("en-custom", Path(en)))
    elif ru is None and technical is None:
        out.append(("en-say", _speech_wav(tmp / "spike_en.wav", "hello world this is a test")))
    return out


def _stamp_seconds(stamp: dict, time_stride: float) -> tuple[float, float]:
    if "start" in stamp and "end" in stamp:
        return float(stamp["start"]), float(stamp["end"])
    start = float(stamp.get("start_offset", 0)) * time_stride
    end = float(stamp.get("end_offset", 0)) * time_stride
    return start, end


def _word_text(stamp: dict) -> str:
    return str(stamp.get("word") or stamp.get("char") or "").strip()


def _print_words(word_ts: list[dict], time_stride: float, limit: int = 12) -> None:
    shown = 0
    for index, stamp in enumerate(word_ts):
        text = _word_text(stamp)
        if not text:
            continue
        start, end = _stamp_seconds(stamp, time_stride)
        print(f"    {start:6.2f}-{end:6.2f}s  {text!r}")
        shown += 1
        if shown >= limit:
            rest = sum(1 for s in word_ts[index + 1 :] if _word_text(s))
            if rest:
                print(f"    ... ещё {rest} слов")
            break


def main() -> None:
    _load_dotenv()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(MODELS_DIR)
    os.environ["HF_HUB_CACHE"] = str(MODELS_DIR / "hub")
    os.environ["NEMO_CACHE_DIR"] = str(MODELS_DIR)
    if token := os.environ.get("HF_TOKEN"):
        os.environ["HUGGING_FACE_HUB_TOKEN"] = token

    print("=== Parakeet v3 spike ===")
    print(f"repo:       {ROOT}")
    print(f"model:      {MODEL_NAME}")
    print(f"models_dir: {MODELS_DIR}")

    try:
        import nemo
        import nemo.collections.asr as nemo_asr
        import torch
    except ImportError as exc:
        sys.exit(f"NeMo/torch не установлены: {exc}\n  pip install -r requirements-ml.txt")

    print(f"nemo:       {getattr(nemo, '__version__', '?')}")
    print(f"torch:      {torch.__version__}")
    device = _resolve_device()
    print(f"device:     {device}")

    tmp = ROOT / "data" / "tmp" / "spike_parakeet"
    tmp.mkdir(parents=True, exist_ok=True)
    cases = _pick_wavs(tmp)

    print("\n--- load model ---")
    t0 = time.perf_counter()
    try:
        model = nemo_asr.models.ASRModel.from_pretrained(MODEL_NAME)
    except Exception as exc:
        sys.exit(f"FAIL load: {type(exc).__name__}: {exc}")
    model = model.to(device).eval()
    load_sec = time.perf_counter() - t0
    print(f"OK load in {load_sec:.1f}s")

    time_stride = float(getattr(model, "window_stride", 0.01)) * 8.0
    print(f"time_stride: {time_stride} (offset -> секунды, если нужно)")

    for label, src in cases:
        if not src.is_file():
            sys.exit(f"Файл не найден: {src}")
        print(f"\n--- prepare [{label}] {src.name} ---")
        wav = _maybe_trim(_prepare_mono_wav(src, tmp), tmp)
        duration = _wav_duration(wav)
        print(f"wav:        {wav} ({duration:.1f}s)")

        print(f"\n--- transcribe [{label}] {src.name} ---")

        t0 = time.perf_counter()
        try:
            hyps = model.transcribe([str(wav)], timestamps=True)
        except Exception as exc:
            print(f"FAIL transcribe: {type(exc).__name__}: {exc}")
            continue
        infer_sec = time.perf_counter() - t0

        hyp = hyps[0]
        text = getattr(hyp, "text", str(hyp))
        print(f"text:       {text!r}")
        print(f"infer:      {infer_sec:.2f}s  (~{duration / infer_sec:.1f}x realtime)")

        ts = getattr(hyp, "timestamp", None) or {}
        word_ts = ts.get("word") or []
        seg_ts = ts.get("segment") or []
        print(f"words:      {len(word_ts)}")
        print(f"segments:   {len(seg_ts)}")

        if word_ts:
            print("  sample word timestamps:")
            _print_words(word_ts, time_stride)
        else:
            print("  WARN: word timestamps пустые — для itranscribe-worker это blocker")

        if seg_ts:
            first = seg_ts[0]
            if "start" in first:
                s, e = float(first["start"]), float(first["end"])
            else:
                s, e = _stamp_seconds(first, time_stride)
            seg_text = first.get("segment") or first.get("word") or "?"
            print(f"  first segment: {s:.2f}-{e:.2f}s {seg_text!r}")

    print("\n=== checklist ===")
    print("[ ] модель загрузилась на вашем nemo_toolkit")
    print("[ ] русский текст кириллицей, не транслит")
    print("[ ] word timestamps не пустые")
    print("[ ] скорость приемлема vs GigaAM/Whisper")
    print("[ ] длинный файл (>24 min) — отдельный spike, нужен chunking")
    print("\nGo/no-go: если load OK + RU кириллица + timestamps — имеет смысл интеграция.")


if __name__ == "__main__":
    main()
