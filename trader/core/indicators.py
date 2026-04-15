"""Общие технические индикаторы, используемые всеми стратегиями."""
from __future__ import annotations
import numpy as np
import pandas as pd


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(span=n, adjust=False).mean()


def rsi(s: pd.Series, n: int = 14) -> pd.Series:
    d = s.diff()
    g = d.clip(lower=0).ewm(span=n, adjust=False).mean()
    ls = (-d).clip(lower=0).ewm(span=n, adjust=False).mean()
    return 100 - 100 / (1 + g / (ls + 1e-10))


def adx(df: pd.DataFrame, n: int = 14) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    dm_p = (h - h.shift()).clip(lower=0).where((h - h.shift()) > (l.shift() - l), 0)
    dm_m = (l.shift() - l).clip(lower=0).where((l.shift() - l) > (h - h.shift()), 0)
    at = tr.ewm(span=n, adjust=False).mean()
    di_p = 100 * dm_p.ewm(span=n, adjust=False).mean() / (at + 1e-10)
    di_m = 100 * dm_m.ewm(span=n, adjust=False).mean() / (at + 1e-10)
    dx = 100 * (di_p - di_m).abs() / (di_p + di_m + 1e-10)
    return dx.ewm(span=n, adjust=False).mean()


def volume_ratio(df: pd.DataFrame, n: int = 20) -> pd.Series:
    v = df["volume"]
    return v / (v.rolling(n).mean() + 1e-10)


def bb_mid(s: pd.Series, n: int = 20) -> pd.Series:
    return s.rolling(n).mean()


def momentum(s: pd.Series, n: int = 10) -> pd.Series:
    return (s - s.shift(n)) / (s.shift(n) + 1e-10) * 100


def stochastic(df: pd.DataFrame, k: int = 14, d: int = 3) -> pd.Series:
    h14 = df["high"].rolling(k).max()
    l14 = df["low"].rolling(k).min()
    k_ = 100 * (df["close"] - l14) / (h14 - l14 + 1e-10)
    return k_.rolling(d).mean()


def bw_fractals(df: pd.DataFrame, n: int = 2) -> tuple[pd.Series, pd.Series]:
    """Фракталы Билла Вильямса: (fractal_high, fractal_low)."""
    h, l = df["high"].values, df["low"].values
    fh = np.zeros(len(df), dtype=bool)
    fl = np.zeros(len(df), dtype=bool)
    for i in range(n, len(df) - n):
        if h[i] > max(h[i - n:i]) and h[i] > max(h[i + 1:i + n + 1]):
            fh[i] = True
        if l[i] < min(l[i - n:i]) and l[i] < min(l[i + 1:i + n + 1]):
            fl[i] = True
    return pd.Series(fh, index=df.index), pd.Series(fl, index=df.index)


def zigzag(df: pd.DataFrame, deviation_pct: float = 5.0) -> pd.Series:
    """Классический ZigZag по минимальному отклонению."""
    h, l = df["high"].values, df["low"].values
    n = len(df)
    zz = np.full(n, np.nan)
    direction, lpi, lpp = 0, 0, (h[0] + l[0]) / 2
    for i in range(1, n):
        if direction == 0:
            if h[i] > lpp * (1 + deviation_pct / 100):
                direction, lpi, lpp, zz[i] = 1, i, h[i], h[i]
            elif l[i] < lpp * (1 - deviation_pct / 100):
                direction, lpi, lpp, zz[i] = -1, i, l[i], l[i]
        elif direction == 1:
            if h[i] > lpp:
                zz[lpi] = np.nan; lpi, lpp, zz[i] = i, h[i], h[i]
            elif l[i] < lpp * (1 - deviation_pct / 100):
                direction, lpi, lpp, zz[i] = -1, i, l[i], l[i]
        else:
            if l[i] < lpp:
                zz[lpi] = np.nan; lpi, lpp, zz[i] = i, l[i], l[i]
            elif h[i] > lpp * (1 + deviation_pct / 100):
                direction, lpi, lpp, zz[i] = 1, i, h[i], h[i]
    return pd.Series(zz, index=df.index)
