"""Загрузка и подготовка данных (CSV → DataFrame, индикаторы)."""
from __future__ import annotations

import logging
from pathlib import Path

from analysis.indicators import add_indicators
from backtesting.engine import load_csv

log = logging.getLogger("investor.data")


def load_ohlc_csv(path: str | Path):
    """Загрузка CSV через движок бэктеста."""
    return load_csv(Path(path))


def prepare_frame(df: object) -> object:
    """Добавляет atr и rsi."""
    return add_indicators(df)  # type: ignore[arg-type]


class DataManager:
    def __init__(self, database_path: str | Path | None = None) -> None:
        from data.db import get_database_path

        self.database_path = Path(database_path) if database_path else get_database_path()

    def load_csv_enriched(self, path: str | Path) -> object:
        df = load_csv(Path(path))
        out = add_indicators(df)
        log.info("Загружено %s строк из %s", len(out), path)
        return out
