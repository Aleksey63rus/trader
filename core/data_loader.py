"""
core.data_loader — загрузка OHLCV данных.

Поддерживает:
  - CSV в формате Финам (разделитель ';', дата DD/MM/YY, время HHMMSS)
  - CSV в общем формате (разделитель ',', колонки datetime/date+time/open/high/low/close/volume)
  - Загрузка с MOEX ISS API (часовой интервал)
"""
from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests


# ── CSV loader ────────────────────────────────────────────────────────────────
def load_csv(path: str | Path) -> pd.DataFrame:
    """
    Загружает OHLCV из CSV-файла.
    Автоматически определяет формат (Финам vs. generic).
    Возвращает DataFrame с DatetimeIndex (UTC-наивный, локальное биржевое время).
    """
    path = Path(path)
    raw  = path.read_text(encoding="utf-8", errors="replace")
    sep  = ";" if raw.count(";") > raw.count(",") else ","
    df   = pd.read_csv(path, sep=sep)
    df.columns = [c.strip().lower().lstrip("<").rstrip(">") for c in df.columns]

    # ── Parse datetime ────────────────────────────────────────────────────────
    if "date" in df.columns and "time" in df.columns:
        date_s = df["date"].astype(str).str.strip()
        time_s = df["time"].astype(str).str.zfill(6)
        sample = date_s.iloc[0]
        fmt    = "%d/%m/%y %H%M%S" if "/" in sample else "%Y%m%d %H%M%S"
        df["datetime"] = pd.to_datetime(date_s + " " + time_s,
                                        format=fmt, errors="coerce")
    elif "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    else:
        raise ValueError(f"Cannot detect datetime columns in {path.name}")

    # ── Normalise volume column ───────────────────────────────────────────────
    if "vol" in df.columns and "volume" not in df.columns:
        df = df.rename(columns={"vol": "volume"})

    df = (df.set_index("datetime")
            .sort_index()
            [["open", "high", "low", "close", "volume"]]
            .astype(float)
            .dropna())
    return df


# ── MOEX ISS downloader ───────────────────────────────────────────────────────
ISS_BASE   = "https://iss.moex.com/iss/engines/stock/markets/shares/boards/TQBR/securities"
_HEADERS   = {"User-Agent": "MomentumFilterTrader/2.0"}
_PAGE_SIZE = 500
# MOEX interval codes: 10=10min, 60=1H, 24=D, 7=W, 31=M, 4=Q
_INTERVAL_MAP = {"1H": 60, "D": 24, "W": 7}


def fetch_moex(ticker: str,
               from_date: datetime,
               to_date:   datetime,
               interval:  str = "1H") -> pd.DataFrame:
    """
    Скачивает OHLCV с MOEX ISS.

    Args:
        ticker:    тикер (SBER, ROSN, …)
        from_date: начало периода
        to_date:   конец периода
        interval:  '1H' | 'D'

    Returns:
        DataFrame с DatetimeIndex, колонки open/high/low/close/volume
    """
    iv_code = _INTERVAL_MAP.get(interval, 60)
    url     = f"{ISS_BASE}/{ticker}/candles.json"
    frames  = []
    start   = 0

    while True:
        params = {
            "interval": iv_code,
            "from":     from_date.strftime("%Y-%m-%d"),
            "till":     to_date.strftime("%Y-%m-%d"),
            "start":    start,
        }
        resp = requests.get(url, params=params, headers=_HEADERS, timeout=20)
        resp.raise_for_status()
        payload = resp.json()["candles"]
        rows    = payload["data"]
        cols    = payload["columns"]
        if not rows:
            break
        frames.append(pd.DataFrame(rows, columns=cols))
        start += len(rows)
        if len(rows) < _PAGE_SIZE:
            break
        time.sleep(0.3)

    if not frames:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    df = pd.concat(frames, ignore_index=True)
    df["datetime"] = pd.to_datetime(df["begin"])
    df = (df.set_index("datetime")
            .sort_index()
            [["open", "high", "low", "close", "volume"]]
            .astype(float))
    return df


def save_csv(df: pd.DataFrame, path: str | Path) -> Path:
    """Сохраняет DataFrame в Finam-совместимый CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["<TICKER>;<PER>;<DATE>;<TIME>;<OPEN>;<HIGH>;<LOW>;<CLOSE>;<VOL>"]
    ticker = path.stem.split("_")[0]
    for dt, row in df.iterrows():
        lines.append(
            f"{ticker};60;{dt.strftime('%d/%m/%y')};{dt.strftime('%H%M%S')};"
            f"{row['open']};{row['high']};{row['low']};{row['close']};{int(row['volume'])}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
