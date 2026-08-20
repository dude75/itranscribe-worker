"""Общая уборка: tmp_<task_id> в корне репозитория не должны переживать прогон."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from app.audio import cleanup_all_tmp

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session", autouse=True)
def _cleanup_task_tmp_dirs_session() -> Iterator[None]:
    cleanup_all_tmp(REPO_ROOT)
    yield
    cleanup_all_tmp(REPO_ROOT)


@pytest.fixture(autouse=True)
def _cleanup_task_tmp_dirs() -> Iterator[None]:
    yield
    cleanup_all_tmp(REPO_ROOT)
