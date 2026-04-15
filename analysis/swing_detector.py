"""
Детектор свинг-точек (локальных экстремумов) для волнового анализа Эллиотта.

Алгоритм:
1. Находим локальные максимумы/минимумы в окне lookback свечей.
2. Применяем ATR-фильтр: размах между соседними свингами >= atr_multiplier × ATR.
3. Обеспечиваем чередование: HIGH → LOW → HIGH → ...
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from analysis.indicators import calc_atr
from config import ATR_MULTIPLIER, ATR_PERIOD, LOOKBACK


def _find_local_extrema(series: pd.Series, lookback: int, is_high: bool) -> list[int]:
    """
    iloc-индексы локальных максимумов (is_high=True) или минимумов.
    Точка — экстремум, если она лучшая в окне [i-lookback, i+lookback].
    """
    values = series.values
    n = len(values)
    result = []
    for i in range(lookback, n - lookback):
        window = values[i - lookback : i + lookback + 1]
        if is_high:
            if values[i] == np.max(window):
                result.append(i)
        else:
            if values[i] == np.min(window):
                result.append(i)
    return result


@dataclass
class SwingPoint:
    idx: int            # позиция в DataFrame (iloc-индекс)
    timestamp: object   # datetime
    price: float
    kind: Literal["HIGH", "LOW"]

    def __repr__(self) -> str:
        return f"SwingPoint({self.kind} @ {self.price:.4f}, idx={self.idx})"


def find_swings(
    df: pd.DataFrame,
    lookback: int = LOOKBACK,
    atr_period: int = ATR_PERIOD,
    atr_multiplier: float = ATR_MULTIPLIER,
) -> list[SwingPoint]:
    """
    Возвращает список чередующихся свинг-точек из DataFrame.

    Параметры
    ---------
    df : DataFrame с колонками high, low, close (индекс — datetime или RangeIndex).
    lookback : окно для поиска локального экстремума (количество свечей с каждой стороны).
    atr_period : период ATR14 для фильтрации шума.
    atr_multiplier : минимальный размах = atr_multiplier × ATR.

    Возвращает
    ----------
    list[SwingPoint] — чередующиеся HIGH/LOW точки в хронологическом порядке.
    """
    df = df.reset_index(drop=True)
    atr = calc_atr(df, atr_period).bfill()

    highs = _find_local_extrema(df["high"], lookback, is_high=True)
    lows = _find_local_extrema(df["low"], lookback, is_high=False)

    # Объединяем и сортируем по индексу
    candidates: list[SwingPoint] = []
    for i in highs:
        candidates.append(
            SwingPoint(
                idx=i,
                timestamp=df.index[i] if hasattr(df.index, "__iter__") else i,
                price=float(df["high"].iloc[i]),
                kind="HIGH",
            )
        )
    for i in lows:
        candidates.append(
            SwingPoint(
                idx=i,
                timestamp=df.index[i] if hasattr(df.index, "__iter__") else i,
                price=float(df["low"].iloc[i]),
                kind="LOW",
            )
        )

    candidates.sort(key=lambda p: p.idx)

    # Чередование + ATR-фильтр
    swings: list[SwingPoint] = []
    for point in candidates:
        if not swings:
            swings.append(point)
            continue

        last = swings[-1]

        if point.kind == last.kind:
            # Оставляем более экстремальный
            if point.kind == "HIGH" and point.price > last.price:
                swings[-1] = point
            elif point.kind == "LOW" and point.price < last.price:
                swings[-1] = point
            continue

        # ATR-фильтр: размах между соседними свингами
        avg_atr = float(atr.iloc[last.idx : point.idx + 1].mean())
        swing_range = abs(point.price - last.price)
        if avg_atr > 0 and swing_range < atr_multiplier * avg_atr:
            # Слишком мелкий размах — игнорируем новую точку
            continue

        swings.append(point)

    return swings
