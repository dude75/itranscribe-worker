#!/usr/bin/env python3
"""Сравнение ASR: whisper vs parakeet на одном файле."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@dataclass
class RunResult:
    engine: str
    checkpoint: str
    load_sec: float
    infer_sec: float
    duration_sec: float
    word_count: int
    text_preview: str
    error: str | None = None

    @property
    def realtime(self) -> float | None:
        if self.error or self.infer_sec <= 0:
            return None
        return self.duration_sec / self.infer_sec


def _load_dotenv() -> None:
    env_path = ROOT / ".env"
    if not env_path.is_file():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _resolve_input() -> Path:
    if path := os.environ.get("SPIKE_WAV") or os.environ.get("SPIKE_WAV_RU"):
        return Path(path)
    technical = ROOT / "technical" / "test_ru.mp3"
    if technical.is_file():
        return technical
    sys.exit("Укажите SPIKE_WAV=path/to/audio или положите technical/test_ru.mp3")


def _prepare_wav(src: Path, tmp: Path, max_sec: float) -> Path:
    from app.audio import audio_duration_sec, prepare_wav

    wav = prepare_wav(src, "spike_asr_compare", ROOT / "data", timeout_sec=0)
    duration = audio_duration_sec(wav)
    if max_sec <= 0 or duration <= max_sec:
        return Path(wav)
    trimmed = tmp / "compare_trim.wav"
    subprocess.run(
        ["ffmpeg", "-nostdin", "-y", "-i", str(wav), "-t", str(max_sec), str(trimmed)],
        check=True,
        capture_output=True,
    )
    return trimmed


def _preview(text: str, limit: int = 200) -> str:
    one = " ".join(text.split())
    if len(one) <= limit:
        return one
    return one[: limit - 3] + "..."


def _run_whisper(wav: Path, models_dir: str, device: str, dtype: str, model_name: str) -> RunResult:
    from app.audio import audio_duration_sec
    from app.engines.asr.whisper import FasterWhisperASR

    duration = audio_duration_sec(wav)
    t0 = time.perf_counter()
    try:
        engine = FasterWhisperASR(model_name, models_dir, device=device, compute_type=dtype)
    except Exception as exc:
        return RunResult("whisper", model_name, 0, 0, duration, 0, "", str(exc))
    load_sec = time.perf_counter() - t0

    t0 = time.perf_counter()
    try:
        words = engine.words(str(wav))
    except Exception as exc:
        return RunResult("whisper", model_name, load_sec, 0, duration, 0, "", str(exc))
    infer_sec = time.perf_counter() - t0
    text = " ".join(w.text for w in words)
    return RunResult("whisper", model_name, load_sec, infer_sec, duration, len(words), _preview(text))


def _run_parakeet(
    wav: Path, models_dir: str, device: str, model_name: str, chunk_sec: float, hf_token: str
) -> RunResult:
    from app.audio import audio_duration_sec
    from app.engines.asr.parakeet import ParakeetASR

    duration = audio_duration_sec(wav)
    t0 = time.perf_counter()
    try:
        engine = ParakeetASR(
            model_name,
            models_dir,
            device=device,
            chunk_sec=chunk_sec,
            hf_token=hf_token or None,
        )
    except Exception as exc:
        return RunResult("parakeet", model_name, 0, 0, duration, 0, "", str(exc))
    load_sec = time.perf_counter() - t0

    t0 = time.perf_counter()
    try:
        words = engine.words(str(wav))
    except Exception as exc:
        return RunResult("parakeet", model_name, load_sec, 0, duration, 0, "", str(exc))
    infer_sec = time.perf_counter() - t0
    text = " ".join(w.text for w in words)
    return RunResult(
        "parakeet", model_name, load_sec, infer_sec, duration, len(words), _preview(text)
    )


def _print_result(r: RunResult) -> None:
    print(f"\n--- {r.engine} ({r.checkpoint}) ---")
    if r.error:
        print(f"FAIL: {r.error}")
        return
    rt = r.realtime
    rt_s = f"{rt:.1f}x" if rt else "?"
    print(f"load:   {r.load_sec:.1f}s")
    print(f"infer:  {r.infer_sec:.2f}s  (~{rt_s} realtime)")
    print(f"words:  {r.word_count}")
    print(f"text:   {r.text_preview}")


def main() -> None:
    _load_dotenv()
    from app.audio import infer_device
    from app.config import get_settings

    settings = get_settings()
    models_dir = str(Path(settings.MODELS_DIR).resolve())
    os.environ["HF_HOME"] = models_dir
    os.environ["HF_HUB_CACHE"] = models_dir
    os.environ["NEMO_CACHE_DIR"] = models_dir
    if settings.HF_TOKEN:
        os.environ["HF_TOKEN"] = settings.HF_TOKEN
        os.environ["HUGGING_FACE_HUB_TOKEN"] = settings.HF_TOKEN

    max_sec = float(os.environ.get("SPIKE_MAX_SEC", "120"))
    engines = os.environ.get("SPIKE_ENGINES", "whisper,parakeet").split(",")
    engines = [e.strip().lower() for e in engines if e.strip()]

    device, dtype = infer_device(settings.DEVICE)
    src = _resolve_input()
    tmp = ROOT / "data" / "tmp" / "spike_asr_compare"
    tmp.mkdir(parents=True, exist_ok=True)
    wav = _prepare_wav(src, tmp, max_sec)

    from app.audio import audio_duration_sec

    duration = audio_duration_sec(wav)

    print("=== ASR compare spike ===")
    print(f"source:     {src}")
    print(f"wav:        {wav} ({duration:.1f}s)")
    print(f"device:     {device} ({dtype})")
    print(f"engines:    {', '.join(engines)}")

    results: list[RunResult] = []
    if "whisper" in engines:
        results.append(
            _run_whisper(wav, models_dir, device, dtype, settings.WHISPER_MODEL)
        )
    if "parakeet" in engines:
        results.append(
            _run_parakeet(
                wav,
                models_dir,
                device,
                settings.PARAKEET_MODEL,
                settings.PARAKEET_CHUNK_SEC,
                settings.HF_TOKEN,
            )
        )

    for r in results:
        _print_result(r)

    ok = [r for r in results if not r.error]
    if ok:
        print("\n=== summary (infer only) ===")
        print(f"{'engine':<10} {'load,s':>8} {'infer,s':>8} {'realtime':>10} {'words':>6}")
        for r in sorted(ok, key=lambda x: x.infer_sec):
            rt = r.realtime or 0
            print(
                f"{r.engine:<10} {r.load_sec:8.1f} {r.infer_sec:8.2f} {rt:9.1f}x {r.word_count:6d}"
            )
        fastest = min(ok, key=lambda x: x.infer_sec)
        print(f"\nfastest infer: {fastest.engine} ({fastest.infer_sec:.2f}s)")


if __name__ == "__main__":
    main()
