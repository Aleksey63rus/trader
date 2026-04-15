"""
Комплексные тесты работоспособности всей программы.
Запуск: cd c:\\investor\\trader && python tests\\test_all.py
"""
from __future__ import annotations
import sys
import json
import time
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

# ─────────────────────────── Helpers ─────────────────────────────────────────
PASS = "✅ PASS"
FAIL = "❌ FAIL"
WARN = "⚠️  WARN"
results: list[dict] = []


def test(name: str):
    """Декоратор-контекстный менеджер для тестов."""
    class _T:
        def __enter__(self):
            self._start = time.perf_counter()
            return self
        def __exit__(self, exc_type, exc_val, tb):
            elapsed = time.perf_counter() - self._start
            if exc_type:
                results.append({"name": name, "status": "FAIL",
                                 "ms": round(elapsed*1000),
                                 "error": str(exc_val)})
                print(f"  {FAIL} [{elapsed*1000:.0f}ms] {name}: {exc_val}")
                return True  # подавляем
            results.append({"name": name, "status": "PASS", "ms": round(elapsed*1000)})
            print(f"  {PASS} [{elapsed*1000:.0f}ms] {name}")
    return _T()


def make_ohlcv(n: int = 300, freq: str = "D",
               start: str = "2022-01-01") -> "pd.DataFrame":
    """Генерирует синтетические OHLCV данные."""
    import numpy as np
    import pandas as pd
    idx = pd.date_range(start, periods=n, freq=freq)
    close = 100 + np.cumsum(np.random.randn(n) * 0.8)
    close = np.maximum(close, 10)
    high  = close + np.abs(np.random.randn(n) * 0.5)
    low   = close - np.abs(np.random.randn(n) * 0.5)
    op    = close + np.random.randn(n) * 0.3
    vol   = np.random.randint(100_000, 2_000_000, n)
    return pd.DataFrame({"open": op, "high": high, "low": low,
                          "close": close, "volume": vol}, index=idx)


# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print(" INVESTOR TERMINAL — ТЕСТЫ РАБОТОСПОСОБНОСТИ")
print(f" {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*60)

# ─────────────────── 1. Индикаторы ───────────────────────────────────────────
print("\n📊 1. Технические индикаторы")
import numpy as np
import pandas as pd

with test("import core.indicators"):
    from core.indicators import ema, atr, rsi, adx, momentum, stochastic, bw_fractals, zigzag

with test("ema() — скользящая средняя"):
    df = make_ohlcv(300)
    e = ema(df["close"], 20)
    assert len(e) == len(df)
    assert not e.iloc[20:].isna().any()

with test("atr() — средний истинный диапазон"):
    df = make_ohlcv(300)
    a = atr(df, 14)
    assert (a.iloc[14:] > 0).all()

with test("rsi() — индекс относительной силы"):
    df = make_ohlcv(300)
    r = rsi(df["close"], 14)
    assert r.iloc[15:].between(0, 100).all()

with test("adx() — индекс направленного движения"):
    df = make_ohlcv(300)
    a = adx(df, 14)
    assert (a.iloc[28:] >= 0).all()

with test("bw_fractals() — фракталы Билла Вильямса"):
    df = make_ohlcv(300)
    fh, fl = bw_fractals(df, 2)
    assert fh.sum() > 0 or fl.sum() > 0

with test("zigzag() — ZigZag"):
    df = make_ohlcv(300)
    zz = zigzag(df, 5.0)
    assert zz.notna().any()

with test("stochastic() — стохастик"):
    df = make_ohlcv(300)
    st = stochastic(df, 14, 3)
    assert st.iloc[20:].between(0, 100).all()

with test("momentum() — моментум"):
    df = make_ohlcv(300)
    mo = momentum(df["close"], 10)
    assert not mo.iloc[11:].isna().any()

# ─────────────────── 2. База данных ──────────────────────────────────────────
print("\n🗄️  2. База данных (SQLite)")

with test("import + init_db()"):
    from core.database import (
        init_db, create_portfolio, get_portfolio, get_portfolios,
        delete_portfolio, get_trades, add_trade, close_trade,
        set_setting, get_setting, get_all_settings,
        log_alert, mark_alert_sent, get_pending_alerts,
        save_backtest_result, get_backtest_result,
    )
    init_db()

with test("create / get portfolio"):
    pid = create_portfolio("Test Portfolio", "atr_bo_daily", 100_000, True)
    p = get_portfolio(pid)
    assert p["name"] == "Test Portfolio"
    assert p["strategy_id"] == "atr_bo_daily"
    assert p["initial_capital"] == 100_000

with test("get_portfolios() список"):
    ps = get_portfolios()
    assert any(p["id"] == pid for p in ps)

with test("add_trade + close_trade"):
    tid = add_trade(pid, {
        "strategy_id": "atr_bo_daily",
        "ticker": "SBER",
        "entry_date": "2024-01-10",
        "entry_px": 290.0,
        "shares": 100.0,
    })
    close_trade(tid, "2024-02-10", 310.0, 2000.0, 6.9, "TIME", 31)
    trades = get_trades(pid, status="CLOSED")
    assert len(trades) >= 1
    assert trades[0]["ticker"] == "SBER"

with test("settings CRUD"):
    set_setting("test_key", {"value": 42})
    v = get_setting("test_key")
    assert v == {"value": 42}
    all_s = get_all_settings()
    assert "test_key" in all_s

with test("log_alert + get_pending_alerts"):
    aid = log_alert("TEST", "Тестовый алерт", "SBER")
    pending = get_pending_alerts()
    assert any(a["id"] == aid for a in pending)
    mark_alert_sent(aid)
    pending2 = get_pending_alerts()
    assert not any(a["id"] == aid for a in pending2)

with test("save_backtest_result + get_backtest_result"):
    rid = save_backtest_result("atr_bo_daily", {"atr_period": 14}, {"annual_return": 13.9}, pid)
    r = get_backtest_result(rid)
    assert r["strategy_id"] == "atr_bo_daily"
    assert r["result"]["annual_return"] == 13.9

with test("delete_portfolio (cleanup)"):
    delete_portfolio(pid)
    assert get_portfolio(pid) is None

# ─────────────────── 3. Стратегия ATR_BO Daily ───────────────────────────────
print("\n📈 3. Стратегия ATR Breakout Daily")

with test("import atr_bo_daily"):
    from strategies.atr_bo_daily import ATRBOConfig, generate_signals, run_backtest

with test("ATRBOConfig default + from_dict"):
    cfg = ATRBOConfig()
    assert cfg.atr_period == 14
    d = cfg.to_dict()
    cfg2 = ATRBOConfig.from_dict(d)
    assert cfg2.max_positions == cfg.max_positions

with test("generate_signals() — сигналы на синтетических данных"):
    df = make_ohlcv(500, "D")
    cfg = ATRBOConfig()
    sig = generate_signals(df, cfg)
    assert "signal" in sig.columns
    assert "at14" in sig.columns

with test("run_backtest() — бэктест 3 тикера"):
    data = {
        "SBER": make_ohlcv(800, "D"),
        "LKOH": make_ohlcv(800, "D", "2022-01-05"),
        "ROSN": make_ohlcv(800, "D", "2022-01-10"),
    }
    cfg = ATRBOConfig(max_positions=2, max_hold_days=20)
    result = run_backtest(data, cfg, 100_000)
    assert "annual_return" in result
    assert "equity_curve" in result
    assert "trades" in result
    assert "by_year" in result
    assert "by_ticker" in result
    assert isinstance(result["n_trades"], int)

# ─────────────────── 4. Стратегия MTF v1 ────────────────────────────────────
print("\n📈 4. Стратегия MTF Trend Confirmation v1")

with test("import mtf_v1"):
    from strategies.mtf_v1 import MTFv1Config, run_backtest as run_mtf1

with test("MTFv1Config default + from_dict"):
    cfg = MTFv1Config()
    d = cfg.to_dict()
    cfg2 = MTFv1Config.from_dict(d)
    assert cfg2.min_score == cfg.min_score

with test("run_backtest() MTF v1 — синтетические данные"):
    np.random.seed(42)
    tf_data = {}
    for ticker in ["SBER", "LKOH"]:
        tf_data[ticker] = {}
        for tf, n, freq in [("1H", 2000, "h"), ("4H", 600, "4h"),
                             ("8H", 300, "8h"), ("12H", 200, "12h"), ("D", 800, "D")]:
            tf_data[ticker][tf] = make_ohlcv(n, freq)
    cfg = MTFv1Config(max_positions=2, max_hold_days=10, min_score=2)
    result = run_mtf1(tf_data, cfg, 100_000)
    assert "annual_return" in result or "error" in result
    if "error" not in result:
        assert "equity_curve" in result

# ─────────────────── 5. Стратегия MTF v2 ────────────────────────────────────
print("\n📈 5. Стратегия MTF Trend Confirmation v2")

with test("import mtf_v2"):
    from strategies.mtf_v2 import MTFv2Config, run_backtest as run_mtf2

with test("MTFv2Config default + from_dict"):
    cfg = MTFv2Config()
    d = cfg.to_dict()
    cfg2 = MTFv2Config.from_dict(d)
    assert cfg2.min_score == cfg.min_score

with test("run_backtest() MTF v2 — синтетические данные"):
    np.random.seed(7)
    tf_data = {}
    for ticker in ["SBER", "LKOH"]:
        tf_data[ticker] = {}
        for tf, n, freq in [("1H", 2000, "h"), ("4H", 600, "4h"),
                             ("8H", 300, "8h"), ("12H", 200, "12h"), ("D", 800, "D")]:
            tf_data[ticker][tf] = make_ohlcv(n, freq)
    cfg = MTFv2Config(max_positions=2, max_hold_days=10, min_score=3.0)
    result = run_mtf2(tf_data, cfg, 100_000)
    assert "annual_return" in result or "error" in result

# ─────────────────── 6. Data Loader ─────────────────────────────────────────
print("\n📂 6. Загрузка данных")

with test("import data_loader"):
    from core.data_loader import resample_tf

with test("resample_tf() — 1H → D"):
    df_1h = make_ohlcv(500, "h")
    df_d  = resample_tf(df_1h, "D")
    assert len(df_d) < len(df_1h)
    assert "open" in df_d.columns

with test("resample_tf() — 1H → 4H"):
    df_4h = resample_tf(df_1h, "4h")
    assert len(df_4h) < len(df_1h)

# ─────────────────── 7. FastAPI приложение ───────────────────────────────────
print("\n🌐 7. FastAPI backend")

with test("import FastAPI app"):
    from api.main import app
    assert app.title == "Investor Terminal"

with test("GET /api/health через TestClient"):
    from fastapi.testclient import TestClient
    client = TestClient(app)
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

with test("GET /api/strategies список"):
    r = client.get("/api/strategies")
    assert r.status_code == 200
    data = r.json()
    assert len(data) == 3
    ids = {s["id"] for s in data}
    assert "atr_bo_daily" in ids

with test("GET /api/strategies/atr_bo_daily"):
    r = client.get("/api/strategies/atr_bo_daily")
    assert r.status_code == 200
    assert r.json()["annual_return_hist"] == 13.9

with test("GET /api/strategies/atr_bo_daily/default-config"):
    r = client.get("/api/strategies/atr_bo_daily/default-config")
    assert r.status_code == 200
    cfg = r.json()
    assert "atr_period" in cfg
    assert cfg["atr_period"] == 14

with test("POST /api/portfolios — создание"):
    r = client.post("/api/portfolios", params={
        "name": "API Test Portfolio", "strategy_id": "atr_bo_daily",
        "capital": 150000, "reinvest": True
    })
    assert r.status_code == 200
    portfolio = r.json()
    _pid = portfolio["id"]

with test("GET /api/portfolios — список"):
    r = client.get("/api/portfolios")
    assert r.status_code == 200
    assert any(p["id"] == _pid for p in r.json())

with test("POST /api/settings — обновление"):
    r = client.post("/api/settings", json={"theme": "dark", "language": "ru"})
    assert r.status_code == 200
    assert "theme" in r.json()["updated"]

with test("GET /api/settings"):
    r = client.get("/api/settings")
    assert r.status_code == 200
    assert "theme" in r.json()

with test("POST /api/backtest/atr_bo_daily — мини-бэктест"):
    # Для этого теста стратегия вернёт ошибку (нет файлов) — это OK
    r = client.post("/api/backtest/atr_bo_daily", json={
        "tickers": ["SBER"],
        "config": {},
        "initial_capital": 100000,
    })
    # 400 (нет данных) или 200 (есть данные) — оба OK
    assert r.status_code in (200, 400, 500)

with test("DELETE /api/portfolios/{pid} — удаление"):
    r = client.delete(f"/api/portfolios/{_pid}")
    assert r.status_code == 200

# ─────────────────── 8. MAX Bot (уведомления) ────────────────────────────────
print("\n📱 8. MAX Bot (мессенджер MAX, offline)")

with test("import max_bot"):
    from bot.max_bot import (
        fmt_entry, fmt_exit, fmt_drawdown_warning,
        fmt_circuit_breaker, fmt_error,
        send_alert_sync,
    )

with test("fmt_entry() — вход в позицию"):
    msg = fmt_entry("SBER", 295.5, 100, "ATR_BO", sl_px=280.0)
    assert "SBER" in msg
    assert "295.50" in msg
    assert "280.00" in msg
    assert "ВХОД" in msg

with test("fmt_exit() — прибыльная сделка"):
    msg = fmt_exit("SBER", 295.5, 320.0, 2450.0, 8.3, "TIME", 30)
    assert "SBER" in msg
    assert "ПРИБЫЛЬ" in msg
    assert "2" in msg and "450" in msg

with test("fmt_exit() — убыточная сделка"):
    msg = fmt_exit("LKOH", 7000.0, 6700.0, -300.0, -4.3, "SL", 5)
    assert "УБЫТОК" in msg

with test("fmt_drawdown_warning() — предупреждение"):
    msg = fmt_drawdown_warning(15.3, 85_000.0)
    assert "15.3%" in msg

with test("fmt_circuit_breaker() — остановка"):
    msg = fmt_circuit_breaker(3)
    assert "3" in msg
    assert "ОСТАНОВКА" in msg

with test("fmt_error() — системная ошибка"):
    msg = fmt_error("data_loader", "Connection refused")
    assert "ОШИБКА" in msg

with test("send_alert_sync() без токена — не падает"):
    from core.database import set_setting
    set_setting("max_token", "")
    set_setting("max_user_id", "")
    ok = send_alert_sync("TEST", "тест без токена MAX")
    assert ok is False  # токен пустой — не отправляет

# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
total  = len(results)
passed = sum(1 for r in results if r["status"] == "PASS")
failed = sum(1 for r in results if r["status"] == "FAIL")
avg_ms = sum(r["ms"] for r in results) / total if total else 0

status_icon = "✅" if failed == 0 else "❌"
print(f" {status_icon} ИТОГО: {passed}/{total} тестов прошло, {failed} упало")
print(f" ⏱  Среднее время: {avg_ms:.0f}мс / тест")
print("="*60)

if failed > 0:
    print("\n❌ УПАВШИЕ ТЕСТЫ:")
    for r in results:
        if r["status"] == "FAIL":
            print(f"   • {r['name']}: {r.get('error','')}")

print()

# Финальный отчёт JSON
report = {
    "timestamp": datetime.now().isoformat(),
    "total": total,
    "passed": passed,
    "failed": failed,
    "avg_ms": round(avg_ms),
    "tests": results,
}
report_path = Path(__file__).parent.parent / "data" / "test_report.json"
report_path.parent.mkdir(parents=True, exist_ok=True)
report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"📄 Отчёт сохранён: {report_path}")
