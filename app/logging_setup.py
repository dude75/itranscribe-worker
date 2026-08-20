"""Прикладные логи процесса (ТЗ §10). Независимы от CSV/`metric_event`."""

from __future__ import annotations

import logging
from pathlib import Path

from app.config import Settings

APP_LOGGER_NAME = "app"
LOG_FILENAME = "app.log"


def setup_logging(settings: Settings) -> None:
    logger = logging.getLogger(APP_LOGGER_NAME)
    logger.disabled = False
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()
    if not settings.LOG_ENABLED:
        logger.setLevel(logging.CRITICAL + 1)
        logger.addHandler(logging.NullHandler())
        logger.propagate = False
        return
    log_dir = Path(settings.LOG_DIR)
    log_dir.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    file_handler = logging.FileHandler(log_dir / LOG_FILENAME, encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.propagate = False
