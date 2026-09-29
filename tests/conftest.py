"""Общие фикстуры. Не трогать живой {DATA_DIR}/tmp сервиса."""

from __future__ import annotations

import sys
from collections.abc import Iterator

import pytest

from app.config import get_settings


def pytest_configure(config: pytest.Config) -> None:
    if sys.version_info[:2] != (3, 12):
        pytest.exit(
            f"Tests require Python 3.12 (same as Docker); got {sys.version.split()[0]}. "
            "Use: python3.12 -m venv .venv && ./.venv/bin/pip install -r requirements.txt "
            "-r requirements-ml.txt",
            returncode=1,
        )


@pytest.fixture(scope="session", autouse=True)
def _test_runtime_env(tmp_path_factory: pytest.TempPathFactory) -> Iterator[None]:
    """Герметичный Bearer и кэши; PRELOAD не зависит от локального .env."""
    root = tmp_path_factory.mktemp("runtime")
    numba_dir = root / "numba_cache"
    mpl_dir = root / "mpl"
    numba_dir.mkdir()
    mpl_dir.mkdir()
    mp = pytest.MonkeyPatch()
    mp.setenv("API_TOKEN", "test")
    mp.setenv("PRELOAD_ASR", "all")
    mp.setenv("PRELOAD_DIARIZATION", "all")
    mp.setenv("NUMBA_CACHE_DIR", str(numba_dir))
    mp.setenv("MPLCONFIGDIR", str(mpl_dir))
    get_settings.cache_clear()
    yield
    mp.undo()
    get_settings.cache_clear()
