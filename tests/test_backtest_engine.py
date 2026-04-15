"""Тесты бэктест-движка (текущий API)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from backtesting.engine import BacktestEngine


def test_backtest_engine_run_completes() -> None:
    idx = pd.date_range("2024-01-01", periods=220, freq="h")
    rng = np.random.default_rng(42)
    base = 250 + np.cumsum(rng.normal(0, 0.5, size=len(idx)))
    df = pd.DataFrame(
        {
            "open": base,
            "high": base + rng.uniform(0.3, 1.0, size=len(idx)),
            "low": base - rng.uniform(0.3, 1.0, size=len(idx)),
            "close": base,
            "volume": 1000.0,
        },
        index=idx,
    )
    eng = BacktestEngine(df, min_window=60)
    res = eng.run()
    assert res.total_trades >= 0
    assert len(res.equity_curve) > 0
