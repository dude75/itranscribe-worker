"""Версия сервиса из `version.txt` в корне репозитория."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

_VERSION_FILE = Path(__file__).resolve().parent.parent / "version.txt"


@lru_cache
def read_version() -> str:
    return _VERSION_FILE.read_text(encoding="utf-8").strip()
