"""
Momentum Filter Trader v2.0 — точка входа
==========================================

Режимы запуска:
    python main.py web            — запустить FastAPI-дашборд
    python main.py bt <csv>       — быстрый CLI бэктест (все схемы)
    python main.py test           — запустить тесты
"""
from __future__ import annotations

import sys
from pathlib import Path


def _web():
    import uvicorn
    from config import APP_HOST, APP_PORT, LOG_LEVEL
    print(f"Запускаем http://{APP_HOST}:{APP_PORT}  (Ctrl+C для остановки)")
    uvicorn.run("web.app:app", host=APP_HOST, port=APP_PORT,
                reload=True, log_level=LOG_LEVEL.lower())


def _backtest(csv_path: str):
    from core.data_loader import load_csv
    from core.strategy import BacktestEngine

    df     = load_csv(csv_path).between_time("07:00", "23:00")
    ticker = Path(csv_path).stem.split("_")[0].upper()
    engine = BacktestEngine()
    print(f"\n{'Ticker':<6} {'Scheme':<50} {'WR':>7} {'PF':>7} {'Total%':>8} {'Sharpe':>7}")
    print("-" * 90)
    results = engine.run_scheme_comparison(df, ticker)
    for key in "ABCDEFG":
        r = results[key]
        print(f"{r.ticker:<6} {r.scheme_label:<50} "
              f"{r.wr*100:>6.1f}% {r.profit_factor:>7.2f} "
              f"{r.total_pct:>7.1f}% {r.sharpe:>7.2f}")
    best = max(results.values(), key=lambda r: r.wr)
    print(f"\n✓ Лучшая схема по WR: {best.scheme_label} — WR={best.wr*100:.1f}%")


def _test():
    import subprocess
    sys.exit(subprocess.call([sys.executable, "-m", "pytest",
                              "tests/", "-v", "--tb=short"]))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "web"
    if cmd == "web":
        _web()
    elif cmd == "bt":
        if len(sys.argv) < 3:
            print("Использование: python main.py bt <path/to/file.csv>")
            sys.exit(1)
        _backtest(sys.argv[2])
    elif cmd == "test":
        _test()
    else:
        print(__doc__)
