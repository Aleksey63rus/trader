"""
SQLite схема: 4 таблицы — candles, signals, trades, logs.
Использует aiosqlite для async-доступа.
"""

import sqlite3
from pathlib import Path

import aiosqlite

from config import DB_PATH

DDL = """
CREATE TABLE IF NOT EXISTS candles (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol    TEXT NOT NULL,
    interval  TEXT NOT NULL,
    datetime  TEXT NOT NULL,
    open      REAL NOT NULL,
    high      REAL NOT NULL,
    low       REAL NOT NULL,
    close     REAL NOT NULL,
    volume    REAL NOT NULL,
    atr       REAL,
    UNIQUE(symbol, interval, datetime)
);

CREATE TABLE IF NOT EXISTS signals (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol              TEXT NOT NULL,
    created_at          TEXT NOT NULL,
    direction           TEXT NOT NULL,
    entry_price         REAL NOT NULL,
    stop_loss           REAL NOT NULL,
    take_profit         REAL NOT NULL,
    wave_structure_json TEXT,
    reason              TEXT,
    acted_on            INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS trades (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    signal_id    INTEGER REFERENCES signals(id),
    symbol       TEXT NOT NULL,
    side         TEXT NOT NULL,
    qty          REAL NOT NULL,
    entry_price  REAL NOT NULL,
    entry_time   TEXT NOT NULL,
    exit_price   REAL,
    exit_time    TEXT,
    commission   REAL DEFAULT 0,
    pnl          REAL,
    status       TEXT NOT NULL DEFAULT 'OPEN'
);

CREATE TABLE IF NOT EXISTS logs (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    level      TEXT NOT NULL,
    module     TEXT NOT NULL,
    message    TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_candles_symbol_dt ON candles(symbol, interval, datetime);
CREATE INDEX IF NOT EXISTS idx_trades_status     ON trades(status);
"""


def init_db_sync(db_path: Path = DB_PATH) -> None:
    """Синхронная инициализация БД (используется при старте приложения)."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.executescript(DDL)
        conn.commit()


async def init_db(db_path: Path = DB_PATH) -> None:
    """Асинхронная инициализация БД."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    async with aiosqlite.connect(db_path) as db:
        await db.executescript(DDL)
        await db.commit()


async def get_db(db_path: Path = DB_PATH) -> aiosqlite.Connection:
    """Открывает и возвращает соединение с БД (вызывающий код должен закрыть его)."""
    db = await aiosqlite.connect(db_path)
    db.row_factory = aiosqlite.Row
    return db


def init_database(db_path: Path = DB_PATH) -> None:
    """Алиас для main.py / совместимости с ТЗ."""
    init_db_sync(db_path)


def get_database_path() -> Path:
    return DB_PATH


async def log_event(level: str, module: str, message: str, db_path: Path = DB_PATH) -> None:
    """Записывает событие в таблицу logs."""
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            "INSERT INTO logs (level, module, message) VALUES (?, ?, ?)",
            (level, module, message),
        )
        await db.commit()
