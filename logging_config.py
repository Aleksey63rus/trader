"""
Настройка логирования INVESTOR (файл + консоль).
"""
from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

import constants as C


def setup_logging(level: str = "INFO") -> None:
    """Создаёт каталог logs/, настраивает root-логгер."""
    log_dir = Path(__file__).resolve().parent / C.LOG_DIR
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / "investor.log"

    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Убрать дублирующиеся хендлеры при повторном вызове
    root.handlers.clear()

    fmt = logging.Formatter(C.LOG_FORMAT, datefmt=C.LOG_DATE_FORMAT)

    fh = RotatingFileHandler(
        log_file,
        maxBytes=C.LOG_MAX_BYTES,
        backupCount=C.LOG_BACKUP_COUNT,
        encoding="utf-8",
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(getattr(logging, level.upper(), logging.INFO))
    sh.setFormatter(fmt)

    root.addHandler(fh)
    root.addHandler(sh)

    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
