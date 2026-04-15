"""Загрузка и нормализация рыночных данных из разных источников."""
from __future__ import annotations
from pathlib import Path
from typing import Optional
import pandas as pd
import aiohttp


def _parse_finam_csv(path: Path) -> pd.DataFrame:
    """Формат Финам: <TICKER>;<PER>;<DATE>;<TIME>;<OPEN>;<HIGH>;<LOW>;<CLOSE>;<VOL>"""
    df = pd.read_csv(path, sep=";")
    df.columns = [c.strip("<>").lower() for c in df.columns]

    def parse_dt(row):
        d = str(row["date"])
        t = str(int(row["time"])).zfill(6)
        return pd.to_datetime(d + " " + t, format="%d/%m/%y %H%M%S", errors="coerce")

    df.index = df.apply(parse_dt, axis=1)
    df = df[["open", "high", "low", "close", "vol"]].rename(columns={"vol": "volume"})
    return df.dropna().sort_index()


def _parse_generic_csv(path: Path) -> pd.DataFrame:
    """Попытка разобрать CSV с автоопределением формата."""
    for sep in [";", ",", "\t"]:
        try:
            df = pd.read_csv(path, sep=sep, nrows=5)
            if len(df.columns) >= 4:
                break
        except Exception:
            continue

    df = pd.read_csv(path, sep=sep)
    df.columns = [c.strip("<>").lower().strip() for c in df.columns]

    # Ищем колонки с датой и ценами
    date_cols = [c for c in df.columns if "date" in c or "time" in c or "дата" in c]
    price_cols = {"open": None, "high": None, "low": None, "close": None}
    for c in df.columns:
        for key in price_cols:
            if key in c and price_cols[key] is None:
                price_cols[key] = c

    if all(v is not None for v in price_cols.values()):
        if date_cols:
            try:
                df.index = pd.to_datetime(df[date_cols[0]], errors="coerce")
            except Exception:
                df.index = pd.RangeIndex(len(df))
        vol_col = next((c for c in df.columns if "vol" in c or "объём" in c), None)
        cols = {k: price_cols[k] for k in price_cols}
        result = df[[price_cols["open"], price_cols["high"],
                     price_cols["low"], price_cols["close"]]].copy()
        result.columns = ["open", "high", "low", "close"]
        if vol_col:
            result["volume"] = df[vol_col]
        else:
            result["volume"] = 0
        return result.dropna().sort_index()
    raise ValueError(f"Не удалось определить формат файла: {path.name}")


def load_csv(path: str | Path, ticker: str = "", tf: str = "") -> pd.DataFrame:
    """
    Загружает CSV-файл с автоопределением формата.
    Поддерживает Финам, MOEX ISS, MetaTrader, TradingView.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Файл не найден: {path}")

    # Пробуем Финам формат
    try:
        df = _parse_finam_csv(path)
        if len(df) > 0 and isinstance(df.index[0], pd.Timestamp):
            return df
    except Exception:
        pass

    # Общий формат
    return _parse_generic_csv(path)


async def fetch_moex_history(
    ticker: str,
    tf: str = "D",
    start: str = "2022-01-01",
    end: str = "",
    session: Optional[aiohttp.ClientSession] = None,
) -> pd.DataFrame:
    """
    Загружает исторические данные с MOEX ISS API.
    tf: 'D' (дневной), '60' (часовой), '240' (4H)
    """
    interval_map = {"D": 24, "1H": 60, "4H": 240, "8H": 480, "12H": 720}
    interval = interval_map.get(tf, 24)

    if not end:
        from datetime import date
        end = date.today().isoformat()

    url = (
        f"https://iss.moex.com/iss/engines/stock/markets/shares/securities/"
        f"{ticker}/candles.json"
        f"?from={start}&till={end}&interval={interval}&start=0"
    )

    close_session = False
    if session is None:
        session = aiohttp.ClientSession()
        close_session = True

    all_rows = []
    try:
        start_idx = 0
        while True:
            req_url = url + f"&start={start_idx}"
            async with session.get(req_url, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                data = await resp.json()
            candles = data.get("candles", {})
            cols = candles.get("columns", [])
            rows = candles.get("data", [])
            if not rows:
                break
            all_rows.extend(rows)
            if len(rows) < 500:
                break
            start_idx += len(rows)
    finally:
        if close_session:
            await session.close()

    if not all_rows:
        return pd.DataFrame()

    cols_lower = [c.lower() for c in cols]
    df = pd.DataFrame(all_rows, columns=cols_lower)

    # Колонки MOEX: open, close, high, low, value, volume, begin, end
    rename = {"begin": "datetime", "value": "turnover"}
    df = df.rename(columns=rename)
    df.index = pd.to_datetime(df["datetime"], errors="coerce")
    df = df[["open", "high", "low", "close", "volume"]].dropna().sort_index()
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna()


def resample_tf(df: pd.DataFrame, target_tf: str) -> pd.DataFrame:
    """Ресэмплирует данные к нужному таймфрейму."""
    tf_map = {
        "1H": "1h", "4H": "4h", "8H": "8h", "12H": "12h",
        "D": "1D", "W": "1W",
    }
    rule = tf_map.get(target_tf, target_tf)
    return df.resample(rule).agg(
        open=("open", "first"), high=("high", "max"),
        low=("low", "min"), close=("close", "last"),
        volume=("volume", "sum"),
    ).dropna()
