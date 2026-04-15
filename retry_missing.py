"""Повторное скачивание SNGS и MOEX."""
import requests
import pandas as pd
from pathlib import Path
import time

DATA_DIR  = Path("c:/investor/data")
FROM_DATE = "2022-01-01"
TO_DATE   = "2026-03-22"


def fetch_candles(ticker: str, interval: int) -> pd.DataFrame | None:
    url = (f"https://iss.moex.com/iss/engines/stock/markets/shares/"
           f"securities/{ticker}/candles.json")
    all_rows, start = [], 0
    while True:
        try:
            r = requests.get(url, params={"from": FROM_DATE, "till": TO_DATE,
                                          "interval": interval, "start": start},
                             timeout=40)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            print(f"  ОШИБКА: {e}")
            return None
        candles = data.get("candles", {})
        rows = candles.get("data", [])
        cols = candles.get("columns", [])
        if not rows:
            break
        all_rows.append(pd.DataFrame(rows, columns=cols))
        start += len(rows)
        if len(rows) < 500:
            break
        time.sleep(0.2)
    if not all_rows:
        return None
    df = pd.concat(all_rows, ignore_index=True)
    df["datetime"] = pd.to_datetime(df["begin"])
    df = df.set_index("datetime")[["open","high","low","close","volume"]]
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna()


def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    return df.resample(rule).agg(
        {"open":"first","high":"max","low":"min","close":"last","volume":"sum"}
    ).dropna(subset=["open","close"])


def save_finam(df: pd.DataFrame, ticker: str, per: str, path: Path):
    rows = []
    for dt, row in df.iterrows():
        rows.append({
            "<TICKER>": ticker, "<PER>": per,
            "<DATE>":   dt.strftime("%d/%m/%y"),
            "<TIME>":   dt.strftime("%H%M%S"),
            "<OPEN>":   f"{row['open']:.2f}",
            "<HIGH>":   f"{row['high']:.2f}",
            "<LOW>":    f"{row['low']:.2f}",
            "<CLOSE>":  f"{row['close']:.2f}",
            "<VOL>":    str(int(row["volume"])),
        })
    pd.DataFrame(rows).to_csv(path, sep=";", index=False)
    print(f"  Saved {path.name} ({len(rows)} bars)")


for ticker in ["SNGS", "MOEX"]:
    print(f"\n{'='*40}")
    print(f"  {ticker}")
    print(f"{'='*40}")

    print("  1H: скачиваю...", end=" ", flush=True)
    df1h = fetch_candles(ticker, 60)
    if df1h is None or len(df1h) < 50:
        print("ОШИБКА — пропускаю")
        continue
    print(f"OK ({len(df1h)} баров)")
    save_finam(df1h, ticker, "1H", DATA_DIR / f"{ticker}_2022_2026_1H.csv")

    for tf, rule in [("4H","4h"), ("8H","8h"), ("12H","12h")]:
        p = DATA_DIR / f"{ticker}_2022_2026_{tf}.csv"
        if not p.exists():
            dfx = resample_ohlcv(df1h, rule)
            save_finam(dfx, ticker, tf, p)
        else:
            print(f"  {tf}: уже есть — пропуск")

    p_d = DATA_DIR / f"{ticker}_2022_2026_D.csv"
    if not p_d.exists():
        print("  D:  скачиваю...", end=" ", flush=True)
        dfd = fetch_candles(ticker, 24)
        if dfd is None or len(dfd) < 10:
            print("нет данных ISS — ресемплю из 1H")
            dfd = resample_ohlcv(df1h, "D")
        else:
            print(f"OK ({len(dfd)} баров)")
        save_finam(dfd, ticker, "D", p_d)
    else:
        print("  D:  уже есть — пропуск")

    time.sleep(2)

print("\n\nФинальная проверка:")
for ticker in ["SBER","LKOH","GAZP","YDEX","T","GMKN","PLZL","NVTK","TATN","VTBR",
               "ROSN","SNGS","X5","MOEX","OZON","CHMF","NLMK","ALRS","MGNT","RTKM","MTLR","OZPH"]:
    files = sorted([f.stem.split("_")[-1] for f in DATA_DIR.glob(f"{ticker}_2022_2026_*.csv")])
    ok = "✓" if len(files) == 5 else "⚠"
    print(f"  {ok} {ticker:8s}: {' '.join(files) if files else 'НЕТ'}")
