"""Временный offline Hugging Face на время конструктора реплики.

HF_HUB_OFFLINE читается библиотеками при импорте: кроме env патчим
уже загруженные `*.HF_HUB_OFFLINE`. Модули, импортированные внутри
`with`, синхронизируем при выходе. Вложенные with безопасны.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

_OFFLINE_ENV_KEY = "HF_HUB_OFFLINE"
_TRUE = {"1", "true", "yes"}
_CACHE_MISS_MARKERS = (
    "localentrynotfound",
    "offlinemodeisenabled",
    "not found locally",
    "cached snapshot",
    "local_files_only",
    "hf_hub_offline",
    "offline mode",
    "outgoing traffic has been disabled",
    "can't load feature extractor",
    "can't load tokenizer",
    "couldn't find file",
)
_NOT_CACHE_MISS_MARKERS = (
    "out of memory",
    "hf_token is empty",
)
_CONST_TARGETS = (
    ("huggingface_hub.constants", "HF_HUB_OFFLINE"),
    ("transformers.utils.hub", "HF_HUB_OFFLINE"),
)
_MODELS_DIR_ENV_KEYS = (
    "HF_HOME",
    "HF_HUB_CACHE",
    "HUGGINGFACE_HUB_CACHE",
    "NEMO_CACHE_DIR",
)

_depth = 0
_stack: list[dict[str, str | None]] = []


def huggingface_offline_enabled() -> bool:
    return os.environ.get("HF_HUB_OFFLINE", "").strip().lower() in _TRUE


def looks_like_missing_cache(exc: BaseException) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    if any(marker in text for marker in _NOT_CACHE_MISS_MARKERS):
        return False
    return any(marker in text for marker in _CACHE_MISS_MARKERS)


def configure_models_dir(models_dir: str) -> str:
    """Единый каталог весов до первого hub-скачивания (env + уже импортированный huggingface_hub)."""
    resolved = str(Path(models_dir).resolve())
    Path(resolved).mkdir(parents=True, exist_ok=True)
    for key in _MODELS_DIR_ENV_KEYS:
        os.environ[key] = resolved
    _sync_hub_cache_constants(resolved)
    return resolved


def hf_hub_model_cached(models_dir: str, model_id: str) -> bool:
    """Есть snapshot Hugging Face Hub под MODELS_DIR (как у pyannote Pipeline)."""
    normalized = model_id.strip()
    if not normalized:
        return False
    folder_name = "models--" + normalized.replace("/", "--")
    snapshots = Path(models_dir).resolve() / folder_name / "snapshots"
    if not snapshots.is_dir():
        return False
    return any(entry.is_dir() for entry in snapshots.iterdir())


def call_with_local_files_only(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Передаёт local_files_only=True, если API это принимает."""
    if not huggingface_offline_enabled():
        return fn(*args, **kwargs)
    try:
        return fn(*args, **kwargs, local_files_only=True)
    except TypeError as exc:
        if "local_files_only" not in str(exc):
            raise
        return fn(*args, **kwargs)


@contextmanager
def huggingface_offline() -> Iterator[None]:
    global _depth
    if _depth == 0:
        _stack.append(_env_snapshot())
        os.environ[_OFFLINE_ENV_KEY] = "1"
        _sync_loaded_constants_from_env()
    _depth += 1
    try:
        yield
    finally:
        _depth -= 1
        if _depth == 0:
            _restore_env(_stack.pop())
            _sync_loaded_constants_from_env()


def _env_snapshot() -> dict[str, str | None]:
    return {_OFFLINE_ENV_KEY: os.environ.get(_OFFLINE_ENV_KEY)}


def _restore_env(saved: dict[str, str | None]) -> None:
    value = saved[_OFFLINE_ENV_KEY]
    if value is None:
        os.environ.pop(_OFFLINE_ENV_KEY, None)
    else:
        os.environ[_OFFLINE_ENV_KEY] = value


def _sync_loaded_constants_from_env() -> None:
    offline = huggingface_offline_enabled()
    for module_name, attr in _CONST_TARGETS:
        module = sys.modules.get(module_name)
        if module is None or not hasattr(module, attr):
            continue
        setattr(module, attr, offline)


def _sync_hub_cache_constants(models_dir: str) -> None:
    module = sys.modules.get("huggingface_hub.constants")
    if module is None:
        return
    module.HF_HOME = models_dir
    module.HF_HUB_CACHE = models_dir
    module.HUGGINGFACE_HUB_CACHE = models_dir
    module.hf_cache_home = models_dir


def ensure_hub_online() -> None:
    os.environ.pop(_OFFLINE_ENV_KEY, None)
    _sync_loaded_constants_from_env()
