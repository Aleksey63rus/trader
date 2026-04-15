"""
Загрузка 1H и D данных с MOEX ISS для 12 тикеров.
Запуск: python download_all.py
"""
import time
import sys
from datetime import datetime
from pathlib import Path

import requests
import pandas as pd

FROM_DT = datetime(2022, 1, 10)
TO_DT   = datetime(2026, 3, 21)

TICKERS = ["GAZP","LKOH","MGNT","MTLR","NLMK","NVTK","ROSN","SBER","T","TGKA","YDEX","OZPH"]
# Альтернативные тикеры на случай смены названия
ALT = {"OZPH": "OZON", "T": "TCSG"}

ISS_BASE  = "https://iss.moex.com/iss/engines/stock/markets/shares/boards/TQBR/securities"
HEADERS   = {"User-Agent": "MomentumFilterTrader/2.0"}
PAGE_SIZE = 500

INTERVAL_CODE = {"1H": 60, "D": 24}


def fetch(ticker: str, interval: str) -> pd.DataFrame:
    iv  = INTERVAL_CODE[interval]
    url = f"{ISS_BASE}/{ticker}/candles.json"
    frames, start = [], 0
    while True:
        params = {"interval": iv, "from": FROM_DT.strftime("%Y-%m-%d"),
                  "till": TO_DT.strftime("%Y-%m-%d"), "start": start}
        r = requests.get(url, params=params, headers=HEADERS, timeout=30)
        r.raise_for_status()
        payload = r.json()["candles"]
        rows, cols = payload["data"], payload["columns"]
        if not rows:
            break
        frames.append(pd.DataFrame(rows, columns=cols))
        start += len(rows)
        if len(rows) < PAGE_SIZE:
            break
        time.sleep(0.25)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df["datetime"] = pd.to_datetime(df["begin"])
    return (df.set_index("datetime").sort_index()
              [["open","high","low","close","volume"]].astype(float))


def save(df: pd.DataFrame, path: Path, ticker: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["<TICKER>;<PER>;<DATE>;<TIME>;<OPEN>;<HIGH>;<LOW>;<CLOSE>;<VOL>"]
    per   = "60" if "1H" in str(path) else "D"
    for dt, row in df.iterrows():
        t = dt.strftime("%H%M%S") if per == "60" else "000000"
        lines.append(
            f"{ticker};{per};{dt.strftime('%d/%m/%y')};{t};"
            f"{row['open']};{row['high']};{row['low']};{row['close']};{int(row['volume'])}"
        )
    path.write_text("\n".join(lines)+"\n", encoding="utf-8")


def download_one(ticker: str, interval: str) -> tuple:
    out = Path(f"{ticker}_2022_2026_{interval}.csv")
    if out.exists():
        print(f"  {interval} {ticker}: уже есть ({out.stat().st_size//1024} KB) — пропуск")
        df = pd.read_csv(out)
        return ticker, len(df), "CACHED"

    # Try primary ticker, then alternative
    for tk in [ticker, ALT.get(ticker, ticker)]:
        try:
            df = fetch(tk, interval)
            if len(df) < 10:
                continue
            save(df, out, ticker)
            print(f"  {interval} {ticker}: {len(df)} bars → {out}")
            return ticker, len(df), "OK"
        except Exception as e:
            print(f"  {interval} {ticker} ({tk}): ошибка {e}")
    return ticker, 0, "ERR"


if __name__ == "__main__":
    print(f"Загрузка данных {FROM_DT.date()} → {TO_DT.date()}")
    print("=" * 55)
    summary = {}
    for iv in ["1H", "D"]:
        print(f"\n[{iv}]")
        for tk in TICKERS:
            tk_r, n, status = download_one(tk, iv)
            summary.setdefault(tk_r, {})[iv] = (n, status)
            sys.stdout.flush()
            time.sleep(0.3)

    print("\n\n=== ИТОГ ===")
    print(f"{'Тикер':<8} {'1H бар':>8} {'D бар':>7}")
    print("-"*25)
    for tk in TICKERS:
        d = summary.get(tk, {})
        h1 = d.get("1H", (0,"?"))
        dd = d.get("D",  (0,"?"))
        print(f"{tk:<8} {h1[0]:>8,} {dd[0]:>7,}")
