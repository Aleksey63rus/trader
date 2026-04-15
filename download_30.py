"""
Скачивает дневные данные по 30 ликвидным тикерам MOEX (индекс IMOEX + доп.)
через MOEX ISS API и сохраняет в формате Finam CSV.
"""
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime
import time

DATA_DIR = Path("c:/investor/data")

# 30 тикеров: топ IMOEX + дополнительные ликвидные
TICKERS_30 = [
    # Уже есть (пропускаем если файл существует)
    "GAZP", "LKOH", "MGNT", "MTLR", "NLMK", "NVTK",
    "OZPH", "ROSN", "SBER", "T", "TGKA", "YDEX",
    # Новые из IMOEX
    "GMKN",   # Норильский никель
    "TATN",   # Татнефть
    "TATNP",  # Татнефть преф
    "PLZL",   # Полюс (золото)
    "VTBR",   # ВТБ
    "OZON",   # Озон
    "X5",     # X5 Group
    "SBERP",  # Сбер преф
    "SNGS",   # Сургутнефтегаз
    "SNGSP",  # Сургутнефтегаз преф
    "AFLT",   # Аэрофлот
    "ALRS",   # АЛРОСА
    "CHMF",   # Северсталь
    "IRAO",   # Интер РАО
    "MAGN",   # ММК
    "MTSS",   # МТС
    "PHOR",   # ФосАгро
    "RUAL",   # РУСАЛ
]

FROM_DATE = "2022-01-01"
TO_DATE   = datetime.today().strftime("%Y-%m-%d")
INTERVAL  = 24   # дневной = 24

def fetch_moex_daily(ticker: str) -> pd.DataFrame | None:
    """Скачивает дневные OHLCV данные с MOEX ISS."""
    url = (
        f"https://iss.moex.com/iss/engines/stock/markets/shares/securities/"
        f"{ticker}/candles.json"
    )
    params = {
        "from":     FROM_DATE,
        "till":     TO_DATE,
        "interval": INTERVAL,
        "start":    0,
    }

    all_rows = []
    while True:
        try:
            r = requests.get(url, params=params, timeout=15)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            print(f"    Ошибка запроса {ticker}: {e}")
            return None

        candles = data.get("candles", {})
        cols    = candles.get("columns", [])
        rows    = candles.get("data", [])

        if not rows:
            break

        df_chunk = pd.DataFrame(rows, columns=cols)
        all_rows.append(df_chunk)

        # Пагинация
        params["start"] += len(rows)
        if len(rows) < 500:
            break
        time.sleep(0.2)

    if not all_rows:
        return None

    df = pd.concat(all_rows, ignore_index=True)
    df = df.rename(columns={
        "open": "open", "high": "high", "low": "low",
        "close": "close", "volume": "volume", "begin": "datetime"
    })
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.set_index("datetime")[["open","high","low","close","volume"]]
    df = df.dropna()
    return df


def save_finam_csv(df: pd.DataFrame, ticker: str, path: Path):
    """Сохраняет в формат Finam CSV совместимый с нашим загрузчиком."""
    rows = []
    for dt, row in df.iterrows():
        rows.append({
            "<TICKER>": ticker,
            "<PER>":    "D",
            "<DATE>":   dt.strftime("%d/%m/%y"),
            "<TIME>":   "000000",
            "<OPEN>":   f"{row['open']:.2f}",
            "<HIGH>":   f"{row['high']:.2f}",
            "<LOW>":    f"{row['low']:.2f}",
            "<CLOSE>":  f"{row['close']:.2f}",
            "<VOL>":    str(int(row["volume"])),
        })
    out = pd.DataFrame(rows)
    out.to_csv(path, sep=";", index=False)


if __name__ == "__main__":
    print(f"Скачивание дневных данных по {len(TICKERS_30)} тикерам")
    print(f"Период: {FROM_DATE} → {TO_DATE}")
    print("=" * 55)

    ok, skip, fail = [], [], []

    for ticker in TICKERS_30:
        out_path = DATA_DIR / f"{ticker}_2022_2026_D.csv"

        # Пропускаем если уже есть
        if out_path.exists():
            size = out_path.stat().st_size
            if size > 5000:
                print(f"  {ticker:8s}: уже есть ({size//1024}KB) — пропуск")
                skip.append(ticker)
                continue

        print(f"  {ticker:8s}: скачиваю...", end=" ", flush=True)
        df = fetch_moex_daily(ticker)

        if df is None or len(df) < 50:
            print("ОШИБКА (нет данных)")
            fail.append(ticker)
            continue

        save_finam_csv(df, ticker, out_path)
        print(f"OK — {len(df)} баров → {out_path.name}")
        ok.append(ticker)
        time.sleep(0.3)

    print()
    print("=" * 55)
    print(f"Скачано:  {len(ok)} тикеров — {ok}")
    print(f"Пропуск:  {len(skip)} тикеров — {skip}")
    print(f"Ошибки:   {len(fail)} тикеров — {fail}")
