"""Уборка: {DATA_DIR}/tmp и устаревшие tmp_* в корне репозитория."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from app.audio import cleanup_all_tmp, cleanup_legacy_cwd_tmp

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = REPO_ROOT / "data"


def _sweep_task_tmp() -> None:
    cleanup_all_tmp(DEFAULT_DATA_DIR)
    cleanup_legacy_cwd_tmp(REPO_ROOT)
    from app.config import get_settings

    data_dir = Path(get_settings().DATA_DIR)
    if data_dir.resolve() != DEFAULT_DATA_DIR.resolve():
        cleanup_all_tmp(data_dir)


@pytest.fixture(scope="session", autouse=True)
def _cleanup_task_tmp_dirs_session() -> Iterator[None]:
    _sweep_task_tmp()
    yield
    _sweep_task_tmp()


@pytest.fixture(autouse=True)
def _cleanup_task_tmp_dirs() -> Iterator[None]:
    yield
    _sweep_task_tmp()
