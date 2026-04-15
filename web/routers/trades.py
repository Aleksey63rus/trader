"""Журнал сделок из SQLite (если есть записи)."""
from __future__ import annotations

import sqlite3

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

from data.db import get_database_path

router = APIRouter(tags=["trades"])


@router.get("/trades", response_class=HTMLResponse)
async def trades_page() -> str:
    path = get_database_path()
    rows: list[tuple] = []
    if path.is_file():
        conn = sqlite3.connect(path)
        try:
            cur = conn.execute(
                "SELECT id, symbol, side, quantity, entry_price, exit_price, pnl, status, exit_reason "
                "FROM trades ORDER BY id DESC LIMIT 200"
            )
            rows = cur.fetchall()
        except sqlite3.Error:
            rows = []
        finally:
            conn.close()

    body_rows = "".join(
        f"<tr><td>{r[0]}</td><td>{r[1]}</td><td>{r[2]}</td><td>{r[3]}</td>"
        f"<td>{r[4]}</td><td>{r[5]}</td><td>{r[6]}</td><td>{r[7]}</td><td>{r[8]}</td></tr>"
        for r in rows
    )
    return f"""<!DOCTYPE html>
<html lang="ru"><head><meta charset="utf-8"/><title>Сделки</title>
<style>body{{font-family:system-ui;margin:2rem}} table{{border-collapse:collapse}} td,th{{border:1px solid #ccc;padding:6px}}</style>
</head><body>
<h1>Журнал сделок</h1>
<p><a href="/">На главную</a></p>
<table><thead><tr>
<th>id</th><th>symbol</th><th>side</th><th>qty lots</th><th>entry</th><th>exit</th><th>pnl</th><th>status</th><th>reason</th>
</tr></thead><tbody>{body_rows or '<tr><td colspan="9">Нет данных</td></tr>'}</tbody></table>
</body></html>"""


@router.get("/api/trades")
async def trades_api() -> dict:
    path = get_database_path()
    if not path.is_file():
        return {"trades": []}
    conn = sqlite3.connect(path)
    try:
        conn.row_factory = sqlite3.Row
        cur = conn.execute(
            "SELECT * FROM trades ORDER BY id DESC LIMIT 500"
        )
        return {"trades": [dict(r) for r in cur.fetchall()]}
    except sqlite3.Error:
        return {"trades": []}
    finally:
        conn.close()
