"""
Download OHLCV data from MOEX ISS API (official, free, no auth required).
https://iss.moex.com/iss/reference/

Supports: any interval (60=hourly, 24=daily, 10=10min, etc.)
Saves CSV compatible with the existing load_csv() format.
"""

import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

ISS_BASE = "https://iss.moex.com/iss/engines/stock/markets/shares/boards/TQBR/securities"
HEADERS  = {"User-Agent": "Mozilla/5.0 (investor-bot/1.0)"}
PAGE_SIZE = 500   # ISS max rows per request


def fetch_candles(
    ticker: str,
    interval: int,        # 60 = 1H, 24 = 1D, 10 = 10min
    from_date: datetime,
    to_date:   datetime,
) -> pd.DataFrame:
    """Fetch all candles with automatic pagination."""
    url    = f"{ISS_BASE}/{ticker}/candles.json"
    frames = []
    start  = 0

    while True:
        params = {
            "interval": interval,
            "from":     from_date.strftime("%Y-%m-%d"),
            "till":     to_date.strftime("%Y-%m-%d"),
            "start":    start,
        }
        resp = requests.get(url, params=params, headers=HEADERS, timeout=20)
        resp.raise_for_status()

        data  = resp.json()["candles"]
        cols  = data["columns"]
        rows  = data["data"]
        if not rows:
            break

        df = pd.DataFrame(rows, columns=cols)
        frames.append(df)
        start += len(rows)

        if len(rows) < PAGE_SIZE:
            break
        time.sleep(0.3)

    if not frames:
        return pd.DataFrame()

    result = pd.concat(frames, ignore_index=True)
    result["datetime"] = pd.to_datetime(result["begin"])
    result = result.set_index("datetime").sort_index()
    result = result[["open", "high", "low", "close", "volume"]].astype(float)
    return result


def save_finam_format(df: pd.DataFrame, ticker: str, interval_code: str, out_dir: Path) -> Path:
    """Save in Finam-compatible CSV format (semicolon separated)."""
    f_from = df.index[0].strftime("%y%m%d")
    f_to   = df.index[-1].strftime("%y%m%d")
    fname  = f"{ticker}_{f_from}_{f_to}_1H.csv"
    path   = out_dir / fname

    lines = ["<TICKER>;<PER>;<DATE>;<TIME>;<OPEN>;<HIGH>;<LOW>;<CLOSE>;<VOL>"]
    for dt, row in df.iterrows():
        date_s = dt.strftime("%d/%m/%y")
        time_s = dt.strftime("%H%M%S")
        lines.append(
            f"{ticker};{interval_code};{date_s};{time_s};"
            f"{row['open']};{row['high']};{row['low']};{row['close']};{int(row['volume'])}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def download_all(
    tickers: list[str],
    from_date: datetime,
    to_date:   datetime,
    interval:  int = 60,
    out_dir:   Path = Path(r"c:\investor\data"),
) -> dict[str, Path]:
    print("\nMOEX ISS API downloader")
    print(f"Interval : {interval}min  ({'1H' if interval == 60 else str(interval)+'min'})")
    print(f"Period   : {from_date.date()} .. {to_date.date()}")
    print(f"Tickers  : {tickers}\n")

    saved = {}
    for ticker in tickers:
        print(f"  {ticker:<6} ... ", end="", flush=True)
        try:
            df = fetch_candles(ticker, interval, from_date, to_date)
            if df.empty:
                print("NO DATA")
                continue

            path = save_finam_format(df, ticker, "60", out_dir)
            print(f"{len(df):>6} rows  {df.index[0].date()} .. {df.index[-1].date()}  -> {path.name}")
            saved[ticker] = path
        except Exception as e:
            print(f"ERROR: {e}")
        time.sleep(0.5)

    print(f"\nDownloaded {len(saved)}/{len(tickers)} tickers.")
    return saved


if __name__ == "__main__":
    TICKERS   = ["SBER", "ROSN", "LKOH", "MGNT", "YNDX"]
    FROM_DATE = datetime(2022, 1, 1)
    TO_DATE   = datetime(2026, 3, 20)

    paths = download_all(TICKERS, FROM_DATE, TO_DATE, interval=60)

    # Quick validation
    if paths:
        print("\nValidation:")
        from backtesting.engine import load_csv
        for ticker, path in paths.items():
            try:
                df = load_csv(path)
                # Trading hours filter: 07:00-23:00 MSK
                df = df.between_time("07:00", "23:00")
                rows_per_day = df.groupby(df.index.date).size()
                print(f"  {ticker}: {len(df)} rows, avg {rows_per_day.mean():.0f} candles/day, "
                      f"price range {df['close'].min():.1f}..{df['close'].max():.1f}")
            except Exception as e:
                print(f"  {ticker}: validation error: {e}")
