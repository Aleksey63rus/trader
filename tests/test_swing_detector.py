"""Тесты детектора свингов."""
from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.indicators import add_indicators
from analysis.swing_detector import find_swings


def test_find_swings_returns_alternating() -> None:
    n = 120
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    x = np.linspace(0, 8 * np.pi, n)
    price = 200 + 15 * np.sin(x) + np.linspace(0, 20, n)
    df = pd.DataFrame(
        {
            "open": price,
            "high": price + 0.6,
            "low": price - 0.6,
            "close": price,
            "volume": 1000.0,
        },
        index=idx,
    )
    df = add_indicators(df)
    swings = find_swings(df, lookback=3, atr_multiplier=0.25)
    assert len(swings) >= 4
    kinds = [s.kind for s in swings[:8]]
    assert "HIGH" in kinds and "LOW" in kinds
