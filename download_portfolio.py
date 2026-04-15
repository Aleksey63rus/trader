"""
Скачивает и обновляет данные для портфеля из 22 эмитентов.
Тикеры: SBER, LKOH, GAZP, YDEX, T, GMKN, PLZL, NVTK, TATN, VTBR, ROSN,
        SNGS, X5, MOEX, OZON, CHMF, NLMK, ALRS, MGNT, RTKM, MTLR, OZPH

Таймфреймы: 1H, 4H, 8H, 12H, D
- 1H и D  скачиваются с MOEX ISS напрямую
- 4H, 8H, 12H — ресемплинг из 1H

Уже существующие файлы пропускаются (--force для перезаписи).
"""
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime
import time
import sys

DATA_DIR  = Path("c:/investor/data")
FROM_DATE = "2022-01-01"
TO_DATE   = datetime.today().strftime("%Y-%m-%d")
FORCE     = "--force" in sys.argv  # перезаписать все

PORTFOLIO = [
    "SBER", "LKOH", "GAZP", "YDEX", "T",
    "GMKN", "PLZL", "NVTK", "TATN", "VTBR",
    "ROSN", "SNGS", "X5",   "MOEX", "OZON",
    "CHMF", "NLMK", "ALRS", "MGNT", "RTKM",
    "MTLR", "OZPH",
]

# MOEX ISS: interval=60 → 1H, interval=24 → D
MOEX_INTERVAL = {"1H": 60, "D": 24}


# ─── MOEX ISS fetch ───────────────────────────────────────────────────────────

def fetch_candles(ticker: str, interval: int) -> pd.DataFrame | None:
    """Скачивает свечи с MOEX ISS с пагинацией."""
    url = (f"https://iss.moex.com/iss/engines/stock/markets/shares/"
           f"securities/{ticker}/candles.json")
    all_rows = []
    start = 0
    while True:
        params = {"from": FROM_DATE, "till": TO_DATE,
                  "interval": interval, "start": start}
        try:
            r = requests.get(url, params=params, timeout=30)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            print(f"    MOEX ошибка: {e}")
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
           [["open", "high", "low", "close", "volume"]])
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna()


def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    agg = {"open": "first", "high": "max",
           "low":  "min",   "close": "last", "volume": "sum"}
    return df.resample(rule).agg(agg).dropna(subset=["open", "close"])


def save_finam(df: pd.DataFrame, ticker: str, per: str, path: Path):
    """Сохраняет в формате Finam CSV (разделитель ;)."""
    rows = []
    for dt, row in df.iterrows():
        rows.append({
            "<TICKER>": ticker,
            "<PER>":    per,
            "<DATE>":   dt.strftime("%d/%m/%y"),
            "<TIME>":   dt.strftime("%H%M%S"),
            "<OPEN>":   f"{row['open']:.2f}",
            "<HIGH>":   f"{row['high']:.2f}",
            "<LOW>":    f"{row['low']:.2f}",
            "<CLOSE>":  f"{row['close']:.2f}",
            "<VOL>":    str(int(row["volume"])),
        })
    pd.DataFrame(rows).to_csv(path, sep=";", index=False)
    print(f"      ✓ {path.name} ({len(rows)} баров)")


def load_existing_1h(ticker: str) -> pd.DataFrame | None:
    """Пытается загрузить уже скачанный 1H файл."""
    p = DATA_DIR / f"{ticker}_2022_2026_1H.csv"
    if not p.exists() or p.stat().st_size < 10000:
        return None
    try:
        raw = pd.read_csv(p, sep=";")
        raw.columns = [c.strip("<>").lower() for c in raw.columns]
        # Поддерживаем оба формата: DATE и TIME отдельно или datetime
        if "date" in raw.columns and "time" in raw.columns:
            raw["datetime"] = pd.to_datetime(
                raw["date"].astype(str) + " " + raw["time"].astype(str).str.zfill(6),
                format="%d/%m/%y %H%M%S", errors="coerce")
        elif "begin" in raw.columns:
            raw["datetime"] = pd.to_datetime(raw["begin"])
        else:
            return None
        raw = (raw.dropna(subset=["datetime"])
                  .set_index("datetime")
                  .rename(columns={"vol": "volume"})
                 [["open", "high", "low", "close", "volume"]])
        for c in raw.columns:
            raw[c] = pd.to_numeric(raw[c], errors="coerce")
        return raw.dropna()
    except Exception as e:
        print(f"      предупреждение: не удалось прочитать кэш 1H: {e}")
        return None


# ─── Main ─────────────────────────────────────────────────────────────────────

def process_ticker(ticker: str) -> str:
    """Скачивает и сохраняет все ТФ для тикера. Возвращает статус."""
    year_tag = "2022_2026"
    paths = {
        "1H":  DATA_DIR / f"{ticker}_{year_tag}_1H.csv",
        "4H":  DATA_DIR / f"{ticker}_{year_tag}_4H.csv",
        "8H":  DATA_DIR / f"{ticker}_{year_tag}_8H.csv",
        "12H": DATA_DIR / f"{ticker}_{year_tag}_12H.csv",
        "D":   DATA_DIR / f"{ticker}_{year_tag}_D.csv",
    }

    # Определяем что нужно скачать/создать
    missing_tfs = [tf for tf, p in paths.items() if not p.exists() or FORCE]
    if not missing_tfs:
        print(f"  {ticker:8s}: все файлы есть — пропуск")
        return "skip"

    print(f"  {ticker:8s}: нужны [{', '.join(missing_tfs)}]")

    # ── 1H ──────────────────────────────────────────────────────────────────
    df1h = None
    need_1h = "1H" in missing_tfs or any(tf in missing_tfs for tf in ["4H","8H","12H"])

    if need_1h:
        df1h = load_existing_1h(ticker)
        if df1h is not None:
            print(f"      1H: из кэша ({len(df1h)} баров)")
        else:
            print("      1H: скачиваю с MOEX ISS...", end=" ", flush=True)
            df1h = fetch_candles(ticker, 60)
            if df1h is None or len(df1h) < 50:
                print("ОШИБКА — нет данных")
                return "fail"
            print(f"OK ({len(df1h)} баров)")
            time.sleep(0.3)

        # Сохраняем 1H если отсутствует
        if "1H" in missing_tfs:
            save_finam(df1h, ticker, "1H", paths["1H"])

        # Ресемплинг → 4H, 8H, 12H
        for tf, rule in [("4H","4h"), ("8H","8h"), ("12H","12h")]:
            if tf in missing_tfs:
                dfx = resample_ohlcv(df1h, rule)
                save_finam(dfx, ticker, tf, paths[tf])

    # ── D (дневные) ──────────────────────────────────────────────────────────
    if "D" in missing_tfs:
        print("      D:  скачиваю с MOEX ISS...", end=" ", flush=True)
        dfd = fetch_candles(ticker, 24)
        if dfd is None or len(dfd) < 10:
            # Резервный вариант: ресемплинг из 1H
            if df1h is not None:
                print("нет данных с ISS, ресемплю из 1H...")
                dfd = resample_ohlcv(df1h, "D")
            else:
                print("ОШИБКА — нет данных")
                return "fail"
        else:
            print(f"OK ({len(dfd)} баров)")
        save_finam(dfd, ticker, "D", paths["D"])
        time.sleep(0.3)

    return "ok"


if __name__ == "__main__":
    DATA_DIR.mkdir(exist_ok=True)
    print(f"{'='*60}")
    print("  Портфельный загрузчик данных MOEX")
    print(f"  Эмитентов: {len(PORTFOLIO)}")
    print(f"  Период: {FROM_DATE} → {TO_DATE}")
    print(f"  Папка: {DATA_DIR}")
    if FORCE:
        print("  Режим: FORCE (перезапись всех файлов)")
    print(f"{'='*60}")

    results = {"ok": [], "skip": [], "fail": []}

    for i, ticker in enumerate(PORTFOLIO, 1):
        print(f"\n[{i:2d}/{len(PORTFOLIO)}] {ticker}")
        status = process_ticker(ticker)
        results[status].append(ticker)
        time.sleep(0.2)

    print(f"\n{'='*60}")
    print(f"  Готово:    {len(results['ok'])} тикеров: {results['ok']}")
    print(f"  Пропущено: {len(results['skip'])} тикеров: {results['skip']}")
    print(f"  Ошибки:    {len(results['fail'])} тикеров: {results['fail']}")
    print(f"{'='*60}")

    # Итоговая сводка файлов
    print(f"\n  Файлы в {DATA_DIR}:")
    total = 0
    for t in PORTFOLIO:
        files = list(DATA_DIR.glob(f"{t}_2022_2026_*.csv"))
        tfs = sorted([f.stem.split("_")[-1] for f in files])
        status_icon = "✓" if len(tfs) == 5 else "⚠"
        print(f"    {status_icon} {t:8s}: {' '.join(tfs) if tfs else 'НЕТ ДАННЫХ'}")
        total += len(files)
    print(f"\n  Итого файлов портфеля: {total} из {len(PORTFOLIO)*5} возможных")
