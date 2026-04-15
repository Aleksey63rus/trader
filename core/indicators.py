"""
core.indicators — все технические индикаторы стратегии.
Чистые функции: принимают pd.Series / pd.DataFrame, возвращают pd.Series.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range (EWM-сглаживание)."""
    h, l, pc = df["high"], df["low"], df["close"].shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def ema(s: pd.Series, period: int) -> pd.Series:
    """Exponential Moving Average."""
    return s.ewm(span=period, adjust=False).mean()


def rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Relative Strength Index."""
    d = df["close"].diff()
    g = d.clip(lower=0).ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    l = (-d).clip(lower=0).ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    return 100 - 100 / (1 + g / l.replace(0, np.nan))


def efficiency_ratio(close: pd.Series, period: int = 20) -> pd.Series:
    """
    Kaufman Efficiency Ratio: 0 = pure noise/range, 1 = pure trend.
    ER = |net move over N bars| / sum(|each bar move|)
    """
    direction  = (close - close.shift(period)).abs()
    volatility = close.diff().abs().rolling(period).sum()
    return (direction / volatility.replace(0, np.nan)).fillna(0).clip(0, 1)


def macd_histogram(df: pd.DataFrame,
                   fast: int = 12, slow: int = 26, sig: int = 9) -> pd.Series:
    """MACD histogram (MACD line − Signal line)."""
    m = ema(df["close"], fast) - ema(df["close"], slow)
    return m - ema(m, sig)


def directional_index(df: pd.DataFrame, period: int = 14):
    """Returns (DI+, DI−) as a tuple of pd.Series."""
    h, l, pc = df["high"], df["low"], df["close"].shift(1)
    tr  = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    dmp = (h - h.shift(1)).clip(lower=0)
    dmn = (l.shift(1) - l).clip(lower=0)
    dmp[(h - h.shift(1)) < (l.shift(1) - l)] = 0
    dmn[(l.shift(1) - l) < (h - h.shift(1))] = 0
    a14  = tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    di_p = 100 * dmp.ewm(alpha=1 / period, min_periods=period, adjust=False).mean() / a14
    di_n = 100 * dmn.ewm(alpha=1 / period, min_periods=period, adjust=False).mean() / a14
    return di_p, di_n


def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average Directional Index."""
    di_p, di_n = directional_index(df, period)
    dx = 100 * (di_p - di_n).abs() / (di_p + di_n).replace(0, np.nan)
    return dx.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def volume_ratio(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Current volume relative to N-bar average."""
    return df["volume"] / df["volume"].rolling(period).mean()


def atr_ratio(df: pd.DataFrame, fast: int = 5, slow: int = 14) -> pd.Series:
    """ATR(fast) / ATR(slow) — shows whether volatility is expanding."""
    return atr(df, fast) / atr(df, slow)
