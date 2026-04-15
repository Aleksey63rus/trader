"""
Скачивает данные 4H, 8H (синтетический), 12H (синтетический) по 30 тикерам.
MOEX ISS поддерживает интервалы: 1,10,60 минут; 24ч (день); 7 (неделя); 31 (месяц).
4H и 12H строятся из 60-минутных баров ресемплингом.
"""
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime
import time

DATA_DIR  = Path("c:/investor/data")
FROM_DATE = "2022-01-01"
TO_DATE   = datetime.today().strftime("%Y-%m-%d")

TICKERS = [
    "GAZP","LKOH","NVTK","ROSN","SNGS","SNGSP",
    "SBER","SBERP","T","VTBR",
    "GMKN","NLMK","MTLR","CHMF","MAGN","RUAL","ALRS","PLZL",
    "YDEX","OZON","MGNT","X5",
    "TATN","TATNP",
    "AFLT","TGKA","IRAO","MTSS","PHOR","OZPH",
]

def fetch_hourly(ticker: str) -> pd.DataFrame | None:
    """Скачивает часовые OHLCV с MOEX ISS (interval=60)."""
    url = (f"https://iss.moex.com/iss/engines/stock/markets/shares/"
           f"securities/{ticker}/candles.json")
    all_rows = []
    start = 0
    while True:
        params = {"from": FROM_DATE, "till": TO_DATE,
                  "interval": 60, "start": start}
        try:
            r = requests.get(url, params=params, timeout=20)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            print(f"    ошибка: {e}")
            return None
        candles = data.get("candles", {})
        rows    = candles.get("data", [])
        cols    = candles.get("columns", [])
        if not rows:
            break
        all_rows.append(pd.DataFrame(rows, columns=cols))
        start += len(rows)
        if len(rows) < 500:
            break
        time.sleep(0.15)
    if not all_rows:
        return None
    df = pd.concat(all_rows, ignore_index=True)
    df["datetime"] = pd.to_datetime(df["begin"])
    df = (df.set_index("datetime")
           .rename(columns={"volume": "volume"})
          [["open","high","low","close","volume"]])
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna()


def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Ресемплинг часовых данных в нужный ТФ."""
    agg = {"open": "first", "high": "max",
           "low": "min",   "close": "last", "volume": "sum"}
    return df.resample(rule).agg(agg).dropna(subset=["open","close"])


def save_finam(df: pd.DataFrame, ticker: str, per: str, path: Path):
    rows = []
    for dt, row in df.iterrows():
        rows.append({
            "<TICKER>": ticker, "<PER>": per,
            "<DATE>":   dt.strftime("%d/%m/%y"),
            "<TIME>":   dt.strftime("%H%M%S"),
            "<OPEN>":   f"{row['open']:.2f}",  "<HIGH>": f"{row['high']:.2f}",
            "<LOW>":    f"{row['low']:.2f}",   "<CLOSE>":f"{row['close']:.2f}",
            "<VOL>":    str(int(row["volume"])),
        })
    pd.DataFrame(rows).to_csv(path, sep=";", index=False)


if __name__ == "__main__":
    print(f"Скачивание 1H → ресемплинг 4H/8H/12H по {len(TICKERS)} тикерам")
    print(f"Период: {FROM_DATE} → {TO_DATE}")
    print("=" * 60)

    ok_list, skip_list, fail_list = [], [], []

    for ticker in TICKERS:
        p4h  = DATA_DIR / f"{ticker}_2022_2026_4H.csv"
        p8h  = DATA_DIR / f"{ticker}_2022_2026_8H.csv"
        p12h = DATA_DIR / f"{ticker}_2022_2026_12H.csv"

        if p4h.exists() and p8h.exists() and p12h.exists():
            print(f"  {ticker:8s}: уже есть — пропуск")
            skip_list.append(ticker)
            continue

        # Пробуем использовать уже скачанный 1H файл
        h1_path = DATA_DIR / f"{ticker}_2022_2026_1H.csv"
        df1h = None

        if h1_path.exists() and h1_path.stat().st_size > 50000:
            try:
                raw = pd.read_csv(h1_path, sep=";")
                raw.columns = [c.strip("<>").lower() for c in raw.columns]
                raw["datetime"] = pd.to_datetime(
                    raw["date"].astype(str) + " " + raw["time"].astype(str).str.zfill(6),
                    format="%d/%m/%y %H%M%S", errors="coerce")
                raw = (raw.dropna(subset=["datetime"])
                          .set_index("datetime")
                          .rename(columns={"vol":"volume"})
                         [["open","high","low","close","volume"]])
                for c in raw.columns:
                    raw[c] = pd.to_numeric(raw[c], errors="coerce")
                df1h = raw.dropna()
                print(f"  {ticker:8s}: 1H из кэша ({len(df1h)} баров)", end=" ")
            except Exception:
                df1h = None

        if df1h is None:
            print(f"  {ticker:8s}: скачиваю 1H...", end=" ", flush=True)
            df1h = fetch_hourly(ticker)
            if df1h is None or len(df1h) < 100:
                print("ОШИБКА")
                fail_list.append(ticker)
                continue
            print(f"OK ({len(df1h)} баров)", end=" ")
            time.sleep(0.3)

        # Ресемплинг
        df4h  = resample_ohlcv(df1h, "4h")
        df8h  = resample_ohlcv(df1h, "8h")
        df12h = resample_ohlcv(df1h, "12h")

        save_finam(df4h,  ticker, "4H",  p4h)
        save_finam(df8h,  ticker, "8H",  p8h)
        save_finam(df12h, ticker, "12H", p12h)

        print(f"→ 4H:{len(df4h)} 8H:{len(df8h)} 12H:{len(df12h)}")
        ok_list.append(ticker)
        time.sleep(0.2)

    print()
    print("=" * 60)
    print(f"Готово:   {len(ok_list)} тикеров")
    print(f"Пропуск:  {len(skip_list)} тикеров")
    print(f"Ошибки:   {len(fail_list)} тикеров: {fail_list}")
