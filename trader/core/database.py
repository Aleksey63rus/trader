"""
SQLite база данных — инициализация и CRUD-операции.
Таблицы: portfolios, trades, settings, alerts_log
"""
from __future__ import annotations
import sqlite3
import json
from pathlib import Path
from typing import Any, Optional

DB_PATH = Path(__file__).parent.parent / "data" / "trader.db"


def get_conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db() -> None:
    with get_conn() as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS portfolios (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            name            TEXT    NOT NULL,
            strategy_id     TEXT    NOT NULL,
            initial_capital REAL    NOT NULL DEFAULT 100000,
            current_capital REAL    NOT NULL DEFAULT 100000,
            reinvest        INTEGER NOT NULL DEFAULT 1,
            active          INTEGER NOT NULL DEFAULT 1,
            created_at      TEXT    NOT NULL DEFAULT (datetime('now')),
            updated_at      TEXT    NOT NULL DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS trades (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            portfolio_id INTEGER NOT NULL REFERENCES portfolios(id),
            strategy_id  TEXT    NOT NULL,
            ticker       TEXT    NOT NULL,
            direction    TEXT    NOT NULL DEFAULT 'LONG',
            entry_date   TEXT    NOT NULL,
            exit_date    TEXT,
            entry_px     REAL    NOT NULL,
            exit_px      REAL,
            shares       REAL    NOT NULL,
            pnl_rub      REAL,
            pnl_pct      REAL,
            reason       TEXT,
            hold_days    REAL,
            status       TEXT    NOT NULL DEFAULT 'OPEN',
            created_at   TEXT    NOT NULL DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS settings (
            key        TEXT PRIMARY KEY,
            value      TEXT NOT NULL,
            updated_at TEXT NOT NULL DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS alerts_log (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            event_type TEXT NOT NULL,
            ticker     TEXT,
            message    TEXT NOT NULL,
            sent       INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS backtest_results (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            strategy_id  TEXT  NOT NULL,
            portfolio_id INTEGER,
            run_date     TEXT  NOT NULL DEFAULT (datetime('now')),
            config       TEXT  NOT NULL,
            result_json  TEXT  NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_trades_portfolio ON trades(portfolio_id);
        CREATE INDEX IF NOT EXISTS idx_trades_ticker    ON trades(ticker);
        CREATE INDEX IF NOT EXISTS idx_trades_status    ON trades(status);
        CREATE INDEX IF NOT EXISTS idx_alerts_sent      ON alerts_log(sent);
        """)


# ── Portfolios ────────────────────────────────────────────────────────────────

def create_portfolio(name: str, strategy_id: str, capital: float, reinvest: bool = True) -> int:
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO portfolios (name, strategy_id, initial_capital, current_capital, reinvest) VALUES (?,?,?,?,?)",
            (name, strategy_id, capital, capital, int(reinvest))
        )
        return cur.lastrowid


def get_portfolios() -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute("SELECT * FROM portfolios ORDER BY id").fetchall()
    return [dict(r) for r in rows]


def get_portfolio(pid: int) -> Optional[dict]:
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM portfolios WHERE id=?", (pid,)).fetchone()
    return dict(row) if row else None


def update_portfolio_capital(pid: int, capital: float) -> None:
    with get_conn() as conn:
        conn.execute(
            "UPDATE portfolios SET current_capital=?, updated_at=datetime('now') WHERE id=?",
            (capital, pid)
        )


def delete_portfolio(pid: int) -> None:
    with get_conn() as conn:
        conn.execute("DELETE FROM trades WHERE portfolio_id=?", (pid,))
        conn.execute("DELETE FROM portfolios WHERE id=?", (pid,))


# ── Trades ────────────────────────────────────────────────────────────────────

def add_trade(portfolio_id: int, trade: dict) -> int:
    with get_conn() as conn:
        cur = conn.execute("""
            INSERT INTO trades
                (portfolio_id, strategy_id, ticker, entry_date, entry_px, shares, status, direction)
            VALUES (?,?,?,?,?,?,?,?)
        """, (
            portfolio_id,
            trade.get("strategy_id", ""),
            trade["ticker"],
            trade["entry_date"],
            trade["entry_px"],
            trade["shares"],
            "OPEN",
            trade.get("direction", "LONG"),
        ))
        return cur.lastrowid


def close_trade(trade_id: int, exit_date: str, exit_px: float,
                pnl_rub: float, pnl_pct: float, reason: str, hold_days: float) -> None:
    with get_conn() as conn:
        conn.execute("""
            UPDATE trades SET exit_date=?, exit_px=?, pnl_rub=?, pnl_pct=?,
                reason=?, hold_days=?, status='CLOSED'
            WHERE id=?
        """, (exit_date, exit_px, pnl_rub, pnl_pct, reason, hold_days, trade_id))


def get_trades(portfolio_id: Optional[int] = None, status: Optional[str] = None,
               ticker: Optional[str] = None, limit: int = 500) -> list[dict]:
    clauses, params = [], []
    if portfolio_id: clauses.append("portfolio_id=?"); params.append(portfolio_id)
    if status:       clauses.append("status=?");       params.append(status)
    if ticker:       clauses.append("ticker=?");       params.append(ticker)
    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    with get_conn() as conn:
        rows = conn.execute(
            f"SELECT * FROM trades {where} ORDER BY id DESC LIMIT ?",
            params + [limit]
        ).fetchall()
    return [dict(r) for r in rows]


def get_pnl_by_period(portfolio_id: int, start: str, end: str) -> dict:
    with get_conn() as conn:
        rows = conn.execute("""
            SELECT ticker, SUM(pnl_rub) AS pnl_rub, SUM(pnl_pct) AS pnl_pct_sum,
                   AVG(pnl_pct) AS pnl_pct_avg, COUNT(*) AS n_trades,
                   SUM(CASE WHEN pnl_rub > 0 THEN 1 ELSE 0 END) AS n_wins
            FROM trades WHERE portfolio_id=? AND status='CLOSED'
              AND exit_date >= ? AND exit_date <= ?
            GROUP BY ticker ORDER BY pnl_rub DESC
        """, (portfolio_id, start, end)).fetchall()
    total_row = dict
    total = {"pnl_rub": 0.0, "n_trades": 0, "n_wins": 0}
    result = []
    for r in rows:
        d = dict(r)
        total["pnl_rub"] += d["pnl_rub"] or 0
        total["n_trades"] += d["n_trades"]
        total["n_wins"]   += d["n_wins"]
        result.append(d)
    total["win_rate"] = round(total["n_wins"] / total["n_trades"] * 100, 1) if total["n_trades"] else 0
    return {"by_ticker": result, "total": total}


# ── Settings ──────────────────────────────────────────────────────────────────

def get_setting(key: str, default: Any = None) -> Any:
    with get_conn() as conn:
        row = conn.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
    if row is None:
        return default
    try:
        return json.loads(row[0])
    except Exception:
        return row[0]


def set_setting(key: str, value: Any) -> None:
    v = json.dumps(value) if not isinstance(value, str) else value
    with get_conn() as conn:
        conn.execute("""
            INSERT INTO settings (key, value, updated_at) VALUES (?,?,datetime('now'))
            ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at
        """, (key, v))


def get_all_settings() -> dict:
    with get_conn() as conn:
        rows = conn.execute("SELECT key, value FROM settings").fetchall()
    result = {}
    for r in rows:
        try:
            result[r[0]] = json.loads(r[1])
        except Exception:
            result[r[0]] = r[1]
    return result


# ── Alerts ────────────────────────────────────────────────────────────────────

def log_alert(event_type: str, message: str, ticker: Optional[str] = None) -> int:
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO alerts_log (event_type, ticker, message) VALUES (?,?,?)",
            (event_type, ticker, message)
        )
        return cur.lastrowid


def mark_alert_sent(alert_id: int) -> None:
    with get_conn() as conn:
        conn.execute("UPDATE alerts_log SET sent=1 WHERE id=?", (alert_id,))


def get_pending_alerts() -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM alerts_log WHERE sent=0 ORDER BY id LIMIT 100"
        ).fetchall()
    return [dict(r) for r in rows]


# ── Backtest Results ──────────────────────────────────────────────────────────

def save_backtest_result(strategy_id: str, config: dict, result: dict,
                         portfolio_id: Optional[int] = None) -> int:
    with get_conn() as conn:
        cur = conn.execute("""
            INSERT INTO backtest_results (strategy_id, portfolio_id, config, result_json)
            VALUES (?,?,?,?)
        """, (strategy_id, portfolio_id, json.dumps(config), json.dumps(result)))
        return cur.lastrowid


def get_backtest_results(strategy_id: Optional[str] = None, limit: int = 20) -> list[dict]:
    where = "WHERE strategy_id=?" if strategy_id else ""
    params = [strategy_id] if strategy_id else []
    with get_conn() as conn:
        rows = conn.execute(
            f"SELECT id, strategy_id, portfolio_id, run_date, config FROM backtest_results {where} ORDER BY id DESC LIMIT ?",
            params + [limit]
        ).fetchall()
    return [dict(r) for r in rows]


def get_backtest_result(result_id: int) -> Optional[dict]:
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM backtest_results WHERE id=?", (result_id,)).fetchone()
    if not row:
        return None
    d = dict(row)
    d["result"] = json.loads(d.pop("result_json"))
    d["config"]  = json.loads(d["config"])
    return d
