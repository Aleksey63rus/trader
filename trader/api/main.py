"""
FastAPI backend — главный файл приложения.
Запуск: uvicorn api.main:app --reload --port 8000
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import pandas as pd
from datetime import datetime
from typing import Optional

from core.database import (
    init_db, get_portfolios, get_portfolio, create_portfolio, delete_portfolio,
    get_trades, get_pnl_by_period, get_setting, set_setting, get_all_settings,
    save_backtest_result, get_backtest_results, get_backtest_result,
)
from core.data_loader import load_csv, resample_tf

# ── Инициализация ─────────────────────────────────────────────────────────────
app = FastAPI(title="Investor Terminal", version="1.0.0",
              description="Профессиональный торговый терминал с бэктестингом")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

init_db()

# ── Статика фронтенда ─────────────────────────────────────────────────────────
FRONTEND_DIST = Path(__file__).parent.parent / "frontend" / "dist"

if FRONTEND_DIST.exists():
    app.mount("/assets", StaticFiles(directory=str(FRONTEND_DIST / "assets")), name="assets")

    @app.get("/", include_in_schema=False)
    def serve_index():
        return FileResponse(str(FRONTEND_DIST / "index.html"))

# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/api/health", tags=["System"])
def health():
    return {"status": "ok", "timestamp": datetime.now().isoformat(), "version": "1.0.0"}

# ── Portfolios ────────────────────────────────────────────────────────────────

@app.get("/api/portfolios", tags=["Portfolios"])
def list_portfolios():
    return get_portfolios()


@app.post("/api/portfolios", tags=["Portfolios"])
def create_portfolio_endpoint(
    name: str, strategy_id: str,
    capital: float = 100_000, reinvest: bool = True
):
    pid = create_portfolio(name, strategy_id, capital, reinvest)
    return get_portfolio(pid)


@app.get("/api/portfolios/{pid}", tags=["Portfolios"])
def get_portfolio_endpoint(pid: int):
    p = get_portfolio(pid)
    if not p:
        raise HTTPException(404, "Портфель не найден")
    return p


@app.delete("/api/portfolios/{pid}", tags=["Portfolios"])
def delete_portfolio_endpoint(pid: int):
    delete_portfolio(pid)
    return {"ok": True}

# ── Trades ────────────────────────────────────────────────────────────────────

@app.get("/api/trades", tags=["Trades"])
def list_trades(
    portfolio_id: Optional[int] = None,
    status: Optional[str] = None,
    ticker: Optional[str] = None,
    limit: int = 200,
):
    return get_trades(portfolio_id, status, ticker, limit)


@app.get("/api/trades/pnl", tags=["Trades"])
def pnl_by_period(
    portfolio_id: int,
    start: str = "2022-01-01",
    end: str = "2099-12-31",
):
    return get_pnl_by_period(portfolio_id, start, end)

# ── Backtest ──────────────────────────────────────────────────────────────────

# Основная папка с данными — общая для всего проекта
_CANDIDATE_DIRS = [
    Path("c:/investor/data"),
    Path(__file__).parent.parent.parent / "data",   # c:\investor\data от trader\api\
    Path(__file__).parent.parent / "data",           # c:\investor\trader\data
]
DATA_DIR = next((d for d in _CANDIDATE_DIRS if d.exists() and any(d.glob("*.csv"))),
                Path("c:/investor/data"))


def _load_ticker_data(ticker: str) -> Optional[pd.DataFrame]:
    """Пробует найти CSV по тикеру в папке data/."""
    for p in DATA_DIR.glob(f"{ticker}_*.csv"):
        df = load_csv(str(p))
        if df is not None and not df.empty:
            return df
    return None


@app.post("/api/backtest/{strategy_id}", tags=["Backtest"])
async def run_backtest_endpoint(strategy_id: str, body: dict):
    """
    Запускает бэктест стратегии.
    Body: { tickers: [...], config: {...}, initial_capital: float }
    """
    tickers_raw = body.get("tickers", [])
    config  = body.get("config", {})
    capital = float(body.get("initial_capital", 100_000))
    reinvest = bool(body.get("reinvest", True))

    # Принимаем и строку "SBER,GAZP" и список ["SBER","GAZP"]
    if isinstance(tickers_raw, str):
        tickers = [t.strip().upper() for t in tickers_raw.split(",") if t.strip()]
    else:
        tickers = [t.strip().upper() for t in tickers_raw if t.strip()]

    if not tickers:
        raise HTTPException(400, "Нужно указать хотя бы один тикер")

    try:
        if strategy_id == "atr_bo_daily":
            from strategies.atr_bo_daily import run_backtest, ATRBOConfig
            cfg = ATRBOConfig.from_dict(config)
            data = {t: _load_ticker_data(t) for t in tickers}
            data = {t: df for t, df in data.items() if df is not None}
            if not data:
                raise HTTPException(400, "Данные не найдены в папке data/")
            result = run_backtest(data, cfg, capital)

        elif strategy_id in ("mtf_v1", "mtf_v2"):
            TF_LIST = ["1H", "4H", "8H", "12H", "D"]
            tf_data_all = {}
            for ticker in tickers:
                tf_data_all[ticker] = {}
                for tf in TF_LIST:
                    # Ищем файлы по имени типа SBER_1H_...csv или SBER_...1H.csv
                    for pat in [f"{ticker}_{tf}_*.csv", f"{ticker}_*_{tf}.csv", f"{ticker}_{tf}.csv"]:
                        for p in DATA_DIR.glob(pat):
                            df = load_csv(str(p))
                            if df is not None and not df.empty:
                                tf_data_all[ticker][tf] = df
                                break
                    # Если дневных нет — пробуем ресемплить из 1H
                    if tf == "D" and "D" not in tf_data_all[ticker] and "1H" in tf_data_all[ticker]:
                        tf_data_all[ticker]["D"] = resample_tf(tf_data_all[ticker]["1H"], "D")

            if strategy_id == "mtf_v1":
                from strategies.mtf_v1 import run_backtest, MTFv1Config
                cfg = MTFv1Config.from_dict(config)
            else:
                from strategies.mtf_v2 import run_backtest, MTFv2Config
                cfg = MTFv2Config.from_dict(config)

            result = run_backtest(tf_data_all, cfg, capital)

        else:
            raise HTTPException(400, f"Стратегия '{strategy_id}' не найдена")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Ошибка бэктеста: {e!r}")

    # Сохраняем результат
    res_id = save_backtest_result(strategy_id, config, result)
    result["backtest_result_id"] = res_id
    return result


@app.get("/api/backtest/results", tags=["Backtest"])
def list_backtest_results(strategy_id: Optional[str] = None, limit: int = 20):
    return get_backtest_results(strategy_id, limit)


@app.get("/api/backtest/results/{result_id}", tags=["Backtest"])
def get_backtest_result_endpoint(result_id: int):
    r = get_backtest_result(result_id)
    if not r:
        raise HTTPException(404, "Результат не найден")
    return r

# ── Upload Data ───────────────────────────────────────────────────────────────

@app.post("/api/upload", tags=["Data"])
async def upload_csv(file: UploadFile = File(...)):
    """Загружает CSV-файл с данными тикера в папку data/."""
    if not file.filename.endswith(".csv"):
        raise HTTPException(400, "Только CSV файлы")
    content = await file.read()
    dest = DATA_DIR / file.filename
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(content)
    # Валидация
    try:
        df = load_csv(str(dest))
        if df is None or df.empty:
            dest.unlink(missing_ok=True)
            raise HTTPException(400, "Файл не распознан как OHLCV данные")
        rows = len(df)
        span = f"{df.index[0]}..{df.index[-1]}"
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(400, f"Ошибка разбора файла: {e}")
    return {"filename": file.filename, "rows": rows, "period": span, "saved_to": str(dest)}


def _quick_csv_info(path: Path) -> dict:
    """Быстрое чтение мета-информации CSV без полного парсинга."""
    stat = path.stat()
    size_kb = round(stat.st_size / 1024, 1)
    try:
        # Читаем только первую и последнюю строки для определения периода
        with open(path, "r", encoding="utf-8-sig", errors="replace") as fh:
            lines = fh.readlines()
        # Пропускаем заголовок
        data_lines = [l for l in lines if l.strip() and not l.startswith("<") and not l.lower().startswith("date")]
        rows = len(data_lines)
        # Дата из строки Finam: TICKER;PER;DATE;TIME;... → поле [2] = DD/MM/YY
        def extract_date(line: str) -> str:
            parts = line.replace(";", ",").split(",")
            # Ищем поле с датой (содержит "/" или "-" и цифры)
            for p in parts[:5]:
                p = p.strip().strip('"')
                if "/" in p and len(p) >= 6:
                    # DD/MM/YY или DD/MM/YYYY
                    d = p.split("/")
                    if len(d) == 3:
                        yr = d[2] if len(d[2]) == 4 else "20" + d[2]
                        return f"{yr}-{d[1].zfill(2)}-{d[0].zfill(2)}"
                if "-" in p and len(p) >= 8:
                    return p[:10]
                if len(p) == 8 and p.isdigit():
                    return f"{p[:4]}-{p[4:6]}-{p[6:]}"
            return ""
        date_from = extract_date(data_lines[0])  if data_lines else ""
        date_to   = extract_date(data_lines[-1]) if data_lines else ""
        return {"filename": path.name, "size_kb": size_kb, "rows": rows,
                "from": date_from, "to": date_to}
    except Exception:
        return {"filename": path.name, "size_kb": size_kb, "rows": 0, "from": "", "to": ""}


@app.get("/api/data/files", tags=["Data"])
def list_data_files():
    """Список CSV файлов в папке data/ (быстрая версия без полного парсинга)."""
    csv_files = sorted(DATA_DIR.glob("*.csv"))
    return [_quick_csv_info(f) for f in csv_files]

# ── Settings ──────────────────────────────────────────────────────────────────

# Ключи настроек, содержащие секреты — маскируются при чтении
_SECRET_KEYS = {"broker_secret", "max_token"}


@app.get("/api/settings", tags=["Settings"])
def get_settings():
    s = get_all_settings()
    for k in _SECRET_KEYS:
        if k in s and s[k]:
            v = str(s[k])
            s[k] = v[:4] + "****" if len(v) > 4 else "****"
    return s


@app.post("/api/settings", tags=["Settings"])
def update_settings(body: dict):
    for k, v in body.items():
        set_setting(k, v)
    return {"ok": True, "updated": list(body.keys())}

# ── MAX Bot Test ───────────────────────────────────────────────────────────────

@app.post("/api/max/test", tags=["Settings"])
async def test_max(body: dict):
    """Тест отправки сообщения в мессенджер MAX (platform-api.max.ru)."""
    token   = body.get("token") or get_setting("max_token", "")
    user_id = body.get("user_id") or get_setting("max_user_id", "")
    if not token:
        raise HTTPException(400, "Не указан MAX Bot Token. Получите его на business.max.ru → Чат-боты → Интеграция.")
    if not user_id:
        raise HTTPException(400, "Не указан User ID получателя.")

    # Сохраняем актуальные данные для теста
    import core.database as db
    db.set_setting("max_token", token)
    db.set_setting("max_user_id", user_id)

    # Напрямую вызываем _send_max с подробной диагностикой
    from bot.max_bot import _parse_user_id
    import aiohttp

    try:
        uid = _parse_user_id(user_id)
    except ValueError as e:
        raise HTTPException(400, f"Неверный формат User ID: {e}. Ожидается число, напр. 631107420238.")

    url = f"https://platform-api.max.ru/messages?user_id={uid}"
    headers = {"Authorization": token, "Content-Type": "application/json"}
    payload = {
        "text": (
            "INVESTOR Terminal\n"
            "────────────────────────\n"
            "✓ Подключение к MAX успешно.\n"
            "Уведомления о сделках настроены и работают."
        )
    }
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url, json=payload, headers=headers,
                timeout=aiohttp.ClientTimeout(total=10),
            ) as r:
                resp_body = {}
                try:
                    resp_body = await r.json()
                except Exception:
                    resp_body = {"raw": await r.text()}
                if r.status in (200, 201):
                    return {"ok": True, "user_id": uid}
                # Детальная ошибка от MAX API
                detail = resp_body.get("message") or resp_body.get("error") or str(resp_body)
                raise HTTPException(400,
                    f"MAX API вернул статус {r.status}: {detail}. "
                    f"Проверьте токен и убедитесь, что пользователь написал боту первым."
                )
    except HTTPException:
        raise
    except aiohttp.ClientConnectorError:
        raise HTTPException(503, "Нет подключения к MAX API (platform-api.max.ru). Проверьте интернет.")
    except Exception as e:
        raise HTTPException(500, f"Ошибка при отправке: {type(e).__name__}: {e}")

# ── Broker Test ────────────────────────────────────────────────────────────────

@app.post("/api/broker/test", tags=["Settings"])
async def test_broker():
    """Проверка подключения к брокеру."""
    broker     = get_setting("broker", "")
    client_id  = get_setting("broker_client_id", "")
    secret     = get_setting("broker_secret", "")
    mode       = get_setting("broker_mode", "sandbox")
    api_url    = get_setting("broker_api_url", "")

    if not broker or broker == "manual":
        return {"ok": True, "message": "Ручной режим — брокер не требуется."}
    if not client_id or not secret:
        raise HTTPException(400, "Не заданы Client ID или Secret. Заполните форму в Администрировании.")

    # Пробуем простой GET-запрос на /ping или аналог
    import aiohttp
    try:
        ping_url = api_url.rstrip("/") + "/ping" if api_url else ""
        if not ping_url:
            return {"ok": False, "message": "URL брокера не задан."}
        async with aiohttp.ClientSession() as session:
            async with session.get(ping_url, timeout=aiohttp.ClientTimeout(total=5)) as r:
                return {
                    "ok": r.status < 500,
                    "message": f"HTTP {r.status} от {broker.upper()} [{mode}]",
                }
    except Exception as e:
        return {"ok": False, "message": f"Нет ответа от {api_url}: {e}"}

# ── Strategies Info ───────────────────────────────────────────────────────────

STRATEGIES_META = {
    "atr_bo_daily": {
        "id": "atr_bo_daily",
        "name": "ATR Breakout Daily",
        "timeframe": "Дневной (D)",
        "annual_return_hist": 13.9,
        "max_drawdown_hist": -12.9,
        "win_rate_hist": 69.4,
        "sharpe_hist": 1.19,
        "description": (
            "Торгует сильные пробои дневных баров. Вход: свеча-пробой ≥1.5×ATR "
            "с подтверждением EMA200, RSI(52-82), ADX≥22, Volume×1.5. "
            "Trailing Stop 2×ATR, максимальное удержание 45 дней. "
            "Лучший Sharpe Ratio среди всех стратегий."
        ),
        "data_required": "Дневные OHLCV данные (CSV)",
    },
    "mtf_v1": {
        "id": "mtf_v1",
        "name": "MTF Trend Confirmation v1",
        "timeframe": "1H, 4H, 8H, 12H, D",
        "annual_return_hist": 11.8,
        "max_drawdown_hist": -13.8,
        "win_rate_hist": 47.4,
        "sharpe_hist": 0.94,
        "description": (
            "Многотаймфреймная стратегия. Вход когда 3+ TF из 5 подтверждают рост. "
            "SL по фракталам Билла Вильямса (D). ZigZag фильтр: только восходящая фаза. "
            "Trailing Stop 2×ATR. Максимальное удержание 30 дней."
        ),
        "data_required": "Часовые OHLCV данные (CSV, все TF автоматически)",
    },
    "mtf_v2": {
        "id": "mtf_v2",
        "name": "MTF Trend Confirmation v2",
        "timeframe": "1H, 4H, 8H, 12H, D",
        "annual_return_hist": 9.5,
        "max_drawdown_hist": -28.1,
        "win_rate_hist": 48.3,
        "sharpe_hist": 0.71,
        "description": (
            "Улучшенная MTF с 10-компонентным score (EMA50, Bollinger Bands, "
            "Momentum, Stochastic, Fractal Breakout). Дневной TF весит вдвое. "
            "Вход только при пробое фрактального максимума D. "
            "Более строгие фильтры — меньше сделок, лучший WR."
        ),
        "data_required": "Часовые OHLCV данные (CSV, все TF автоматически)",
    },
}


@app.get("/api/strategies", tags=["Strategies"])
def list_strategies():
    return list(STRATEGIES_META.values())


@app.get("/api/strategies/{strategy_id}", tags=["Strategies"])
def get_strategy(strategy_id: str):
    s = STRATEGIES_META.get(strategy_id)
    if not s:
        raise HTTPException(404, "Стратегия не найдена")
    return s


@app.get("/api/strategies/{strategy_id}/default-config", tags=["Strategies"])
def get_default_config(strategy_id: str):
    if strategy_id == "atr_bo_daily":
        from strategies.atr_bo_daily import ATRBOConfig
        return ATRBOConfig().to_dict()
    elif strategy_id == "mtf_v1":
        from strategies.mtf_v1 import MTFv1Config
        return MTFv1Config().to_dict()
    elif strategy_id == "mtf_v2":
        from strategies.mtf_v2 import MTFv2Config
        return MTFv2Config().to_dict()
    raise HTTPException(404, "Стратегия не найдена")
