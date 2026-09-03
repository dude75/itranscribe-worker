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
from typing import Any

_ENV_KEYS = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
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
)
_NOT_CACHE_MISS_MARKERS = (
    "out of memory",
    "hf_token is empty",
)
_CONST_TARGETS = (
    ("huggingface_hub.constants", "HF_HUB_OFFLINE"),
    ("transformers.utils.hub", "HF_HUB_OFFLINE"),
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
        for key in _ENV_KEYS:
            os.environ[key] = "1"
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
    return {key: os.environ.get(key) for key in _ENV_KEYS}


def _restore_env(saved: dict[str, str | None]) -> None:
    for key, value in saved.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def _sync_loaded_constants_from_env() -> None:
    offline = huggingface_offline_enabled()
    for module_name, attr in _CONST_TARGETS:
        module = sys.modules.get(module_name)
        if module is None or not hasattr(module, attr):
            continue
        setattr(module, attr, offline)
