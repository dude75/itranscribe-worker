"""Общие фикстуры. Не трогать живой {DATA_DIR}/tmp сервиса."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from app.config import get_settings


@pytest.fixture(scope="session", autouse=True)
def _test_api_token() -> Iterator[None]:
    """Герметичный Bearer, без зависимости от локального .env."""
    mp = pytest.MonkeyPatch()
    mp.setenv("API_TOKEN", "test")
    get_settings.cache_clear()
    yield
    mp.undo()
    get_settings.cache_clear()
