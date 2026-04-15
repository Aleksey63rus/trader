"""Главная страница и статус приложения."""
from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

import config

router = APIRouter()


@router.get("/", response_class=HTMLResponse, tags=["ui"])
async def index() -> str:
    return f"""<!DOCTYPE html>
<html lang="ru">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>INVESTOR</title>
  <style>
    body {{ font-family: system-ui, sans-serif; max-width: 720px; margin: 2rem auto; padding: 0 1rem; }}
    code {{ background: #f0f0f0; padding: 0.2em 0.4em; border-radius: 4px; }}
    a {{ color: #0a58ca; }}
  </style>
</head>
<body>
  <h1>INVESTOR</h1>
  <p>Сервер запущен. Документация API: <a href="/docs">/docs</a></p>
  <p>Хост: <code>{config.APP_HOST}</code>, порт: <code>{config.APP_PORT}</code></p>
  <ul>
    <li><a href="/chart/SBER/view">График (демо, Plotly)</a></li>
    <li><a href="/trades">Журнал сделок</a></li>
    <li><a href="/api/settings">Настройки (JSON)</a></li>
    <li>Бэктест демо: POST в Swagger <code>/api/backtest/demo</code></li>
  </ul>
  <p>Полное ТЗ: файл <code>full_tz.md</code> в корне проекта.</p>
</body>
</html>
"""


@router.get("/api/health", tags=["api"])
async def health() -> dict:
    return {"status": "ok", "project": "INVESTOR", "phase": "analysis_backtest_web"}
