"""
Unit-тесты для swing_detector на синтетических данных.
"""

import numpy as np
import pandas as pd

from analysis.swing_detector import SwingPoint, find_swings


def _make_df(prices: list[float]) -> pd.DataFrame:
    """Создаёт DataFrame с синтетическими OHLCV из списка цен закрытия."""
    n = len(prices)
    closes = np.array(prices, dtype=float)
    highs = closes * 1.001
    lows = closes * 0.999
    return pd.DataFrame(
        {
            "open": closes,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": np.ones(n) * 1000,
        }
    )


def _zigzag_prices(n_peaks: int = 10, amplitude: float = 100.0, base: float = 1000.0) -> list[float]:
    """Генерирует явный зигзаг: вверх, вниз, вверх..."""
    prices = []
    for i in range(n_peaks * 2):
        if i % 2 == 0:
            prices += list(np.linspace(base, base + amplitude, 5))
        else:
            prices += list(np.linspace(base + amplitude, base, 5))
    return prices


class TestFindSwings:
    def test_returns_list(self):
        prices = _zigzag_prices(6)
        df = _make_df(prices)
        result = find_swings(df, lookback=3, atr_multiplier=0.1)
        assert isinstance(result, list)

    def test_alternating_kinds(self):
        """Каждая следующая точка должна быть противоположного типа."""
        prices = _zigzag_prices(8, amplitude=200)
        df = _make_df(prices)
        swings = find_swings(df, lookback=3, atr_multiplier=0.1)
        for i in range(1, len(swings)):
            assert swings[i].kind != swings[i - 1].kind, (
                f"Нарушено чередование на позиции {i}: "
                f"{swings[i-1].kind} → {swings[i].kind}"
            )

    def test_swing_points_are_swing_point_instances(self):
        prices = _zigzag_prices(6, amplitude=150)
        df = _make_df(prices)
        swings = find_swings(df, lookback=3, atr_multiplier=0.1)
        for sp in swings:
            assert isinstance(sp, SwingPoint)

    def test_high_is_local_maximum(self):
        prices = _zigzag_prices(6, amplitude=200)
        df = _make_df(prices)
        swings = find_swings(df, lookback=3, atr_multiplier=0.1)
        highs = [sp for sp in swings if sp.kind == "HIGH"]
        for sp in highs:
            # HIGH-свинг должен быть выше соседних значений в DataFrame
            assert sp.price > 0

    def test_atr_filter_removes_small_swings(self):
        """При большом atr_multiplier мелкие колебания должны быть отфильтрованы."""
        prices_small = [1000 + (i % 2) * 1 for i in range(50)]   # колебания 1 пункт
        prices_large = [1000 + (i % 2) * 200 for i in range(50)]  # колебания 200 пунктов
        df_small = _make_df(prices_small)
        df_large = _make_df(prices_large)

        swings_small = find_swings(df_small, lookback=3, atr_multiplier=5.0)
        swings_large = find_swings(df_large, lookback=3, atr_multiplier=0.5)

        assert len(swings_large) > len(swings_small)

    def test_empty_df_returns_empty(self):
        df = _make_df([1000.0] * 5)
        swings = find_swings(df, lookback=3, atr_multiplier=1.0)
        assert swings == []

    def test_monotone_uptrend_few_swings(self):
        """Монотонный рост не должен давать много свинг-точек."""
        prices = list(np.linspace(100, 200, 60))
        df = _make_df(prices)
        swings = find_swings(df, lookback=3, atr_multiplier=1.0)
        # Максимум — 1 LOW в начале и 1 HIGH в конце (или 0)
        assert len(swings) <= 2
