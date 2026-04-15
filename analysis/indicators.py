"""
Технические индикаторы: ATR и RSI.
Работают с pandas DataFrame с колонками OHLCV.
"""

import numpy as np
import pandas as pd


def calc_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Average True Range (Wilder's smoothing).

    Ожидает колонки: high, low, close.
    Возвращает Series с именем 'atr'.
    """
    high = df["high"]
    low = df["low"]
    prev_close = df["close"].shift(1)

    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    # Первое значение — простое среднее, затем Wilder's smoothing
    atr = tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    atr.name = "atr"
    return atr


def calc_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Relative Strength Index (Wilder's smoothing).

    Ожидает колонку: close.
    Возвращает Series с именем 'rsi'.
    """
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)

    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi.name = "rsi"
    return rsi


def add_indicators(df: pd.DataFrame, atr_period: int = 14, rsi_period: int = 14) -> pd.DataFrame:
    """Добавляет колонки 'atr' и 'rsi' к DataFrame. Возвращает копию."""
    df = df.copy()
    df["atr"] = calc_atr(df, atr_period)
    df["rsi"] = calc_rsi(df, rsi_period)
    return df
