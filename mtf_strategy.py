"""
=============================================================================
MTF TREND CONFIRMATION STRATEGY
Многотаймфреймная стратегия с подтверждением тренда
=============================================================================

КОНЦЕПЦИЯ (по вашей идее):
  1H  → первичный сигнал входа (ATR Breakout)
  4H  → подтверждение: тренд продолжается
  8H  → дополнительное подтверждение
  12H → усиление тренда
  D   → мастер-фильтр + установка SL по фракталам Б.Вильямса
  W   → (не качаем, берём из D resampling) долгосрочное направление

ИНСТРУМЕНТЫ:
  • Фракталы Билла Вильямса (BW Fractals) — hi/lo точки разворота
    Fractal High: бар с максимумом выше 2 баров слева и 2 справа
    Fractal Low:  бар с минимумом ниже 2 баров слева и 2 справа
    При пробитии фрактала → конец тренда/коррекция

  • ZigZag — выявляет структуру волн (пиков и впадин)
    Параметр: минимальное движение 5% для смены направления

  • Волны Эллиотта (упрощённая версия через ZigZag):
    Идентифицируем 5-волновой импульс или 3-волновую коррекцию
    Оцениваем на какой волне находимся сейчас

  • MTF Score (0-5):
    +1 за каждый TF где есть бычий сигнал (1H, 4H, 8H, 12H, D)
    Вход только при score ≥ 3, удерживаем при score ≥ 1

ВЫХОД:
  • SL = последний фрактальный минимум на дневном ТФ
  • TP = trailing stop 2×ATR(D)
  • Принудительный выход: пробой фрактального минимума ИЛИ ZigZag разворот

=============================================================================
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import numpy as np
import pandas as pd

DATA_DIR   = Path("c:/investor/data")
COMMISSION = 0.0005   # 0.05%
SLIPPAGE   = 0.0003   # 0.03%

# Тикеры с 1H данными
TICKERS_1H = [
    "SBER", "GAZP", "LKOH", "NVTK", "NLMK",
    "MGNT", "ROSN", "MTLR", "OZPH", "YDEX", "T",
]

# ══════════════════════════════════════════════════════════════════════════════
# ЗАГРУЗКА ДАННЫХ
# ══════════════════════════════════════════════════════════════════════════════
def load_tf(ticker: str, tf: str) -> Optional[pd.DataFrame]:
    """
    tf: '1H', '4H', '8H', '12H', 'D'
    Возвращает DataFrame с колонками: open, high, low, close, volume
    """
    # Для 1H — ищем файл
    if tf == "1H":
        candidates = list(DATA_DIR.glob(f"{ticker}_*_1H.csv"))
        # Предпочитаем файл 2022_2026
        for c in candidates:
            if "2022_2026" in c.name:
                path = c; break
        else:
            path = candidates[0] if candidates else None
        if not path:
            return None

        df = pd.read_csv(path, sep=";")
        # Колонки: <TICKER>, <PER>, <DATE>, <TIME>, <OPEN>, <HIGH>, <LOW>, <CLOSE>, <VOL>
        df.columns = [c.strip("<>").lower() for c in df.columns]
        # Парсим дату+время: DATE формат DD/MM/YY, TIME = HHMMSS (60000 = 06:00:00)
        def parse_dt(row):
            d = str(row["date"])  # DD/MM/YY
            t = str(int(row["time"])).zfill(6)  # HHMMSS
            return pd.to_datetime(d + " " + t, format="%d/%m/%y %H%M%S",
                                  errors="coerce")
        df.index = df.apply(parse_dt, axis=1)
        df = df[["open","high","low","close","vol"]].rename(columns={"vol":"volume"})
        df = df.dropna().sort_index()
        return df

    # Для 4H, 8H, 12H, D — загружаем соответствующий файл
    tf_map = {"4H": "4H", "8H": "8H", "12H": "12H", "D": "D"}
    candidates = list(DATA_DIR.glob(f"{ticker}_2022_2026_{tf_map[tf]}.csv"))
    if not candidates:
        candidates = list(DATA_DIR.glob(f"{ticker}_*_{tf_map[tf]}.csv"))
    if not candidates:
        return None

    path = candidates[0]
    df = pd.read_csv(path, sep=";")
    df.columns = [c.strip("<>").lower() for c in df.columns]

    def parse_dt2(row):
        d = str(row["date"])
        t = str(int(row["time"])).zfill(6)
        return pd.to_datetime(d + " " + t, format="%d/%m/%y %H%M%S",
                              errors="coerce")
    df.index = df.apply(parse_dt2, axis=1)
    df = df[["open","high","low","close","vol"]].rename(columns={"vol":"volume"})
    df = df.dropna().sort_index()
    return df


# ══════════════════════════════════════════════════════════════════════════════
# ИНДИКАТОРЫ
# ══════════════════════════════════════════════════════════════════════════════
def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()

def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(span=n, adjust=False).mean()

def rsi(s: pd.Series, n: int = 14) -> pd.Series:
    d = s.diff()
    g = d.clip(lower=0).ewm(span=n, adjust=False).mean()
    ls= (-d).clip(lower=0).ewm(span=n, adjust=False).mean()
    return 100 - 100/(1 + g/(ls+1e-10))

def adx(df: pd.DataFrame, n: int = 14) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    tr  = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    dm_p = (h - h.shift()).clip(lower=0).where((h-h.shift()) > (l.shift()-l), 0)
    dm_m = (l.shift() - l).clip(lower=0).where((l.shift()-l) > (h-h.shift()), 0)
    atr_ = tr.ewm(span=n, adjust=False).mean()
    di_p = 100 * dm_p.ewm(span=n, adjust=False).mean() / (atr_+1e-10)
    di_m = 100 * dm_m.ewm(span=n, adjust=False).mean() / (atr_+1e-10)
    dx   = 100 * (di_p-di_m).abs() / (di_p+di_m+1e-10)
    return dx.ewm(span=n, adjust=False).mean()

def volume_ratio(df: pd.DataFrame, n: int = 20) -> pd.Series:
    v = df["volume"]
    return v / v.rolling(n).mean()


# ── Фракталы Билла Вильямса ────────────────────────────────────────────────────
def bw_fractals(df: pd.DataFrame, n: int = 2) -> tuple[pd.Series, pd.Series]:
    """
    Возвращает (fractal_high, fractal_low) — boolean Series.
    Fractal High: High[i] > High[i-n..i-1] AND High[i] > High[i+1..i+n]
    n=2 — классические фракталы Билла Вильямса (2 бара с каждой стороны)
    """
    h = df["high"]
    l = df["low"]

    frac_h = pd.Series(False, index=df.index)
    frac_l = pd.Series(False, index=df.index)

    for i in range(n, len(df) - n):
        center_h = h.iloc[i]
        center_l = l.iloc[i]

        left_h  = h.iloc[i-n:i]
        right_h = h.iloc[i+1:i+n+1]
        left_l  = l.iloc[i-n:i]
        right_l = l.iloc[i+1:i+n+1]

        if (center_h > left_h.max()) and (center_h > right_h.max()):
            frac_h.iloc[i] = True
        if (center_l < left_l.min()) and (center_l < right_l.min()):
            frac_l.iloc[i] = True

    return frac_h, frac_l


def fractal_sl(df: pd.DataFrame, current_idx: int, lookback: int = 20) -> float:
    """
    SL = последний фрактальный минимум на n баров назад.
    Это уровень где Билл Вильямс рекомендует ставить стоп.
    """
    _, frac_l = bw_fractals(df, n=2)
    # Берём только фракталы до текущего бара
    past_fractals = frac_l.iloc[max(0, current_idx-lookback):current_idx]
    lows_at_fractals = df["low"].iloc[max(0, current_idx-lookback):current_idx]
    fractal_lows = lows_at_fractals[past_fractals]
    if len(fractal_lows) == 0:
        # Фракталов нет — используем минимум за lookback
        return float(df["low"].iloc[max(0,current_idx-lookback):current_idx].min())
    # Возвращаем последний (ближайший) фрактальный минимум
    return float(fractal_lows.iloc[-1])


# ── ZigZag ─────────────────────────────────────────────────────────────────────
def zigzag(df: pd.DataFrame, deviation_pct: float = 5.0) -> pd.Series:
    """
    Классический ZigZag: выявляет значимые пики и впадины.
    deviation_pct: минимальное движение в % для смены направления.
    Возвращает Series: NaN где нет точки ZigZag,
    иначе цена разворота (high или low).
    """
    h = df["high"].values
    l = df["low"].values
    n = len(df)

    zz = np.full(n, np.nan)
    direction = 0  # 1=вверх, -1=вниз, 0=неопределено
    last_pivot_idx   = 0
    last_pivot_price = (h[0] + l[0]) / 2

    for i in range(1, n):
        if direction == 0:
            if h[i] > last_pivot_price * (1 + deviation_pct/100):
                direction = 1
                last_pivot_idx   = i
                last_pivot_price = h[i]
                zz[i] = h[i]
            elif l[i] < last_pivot_price * (1 - deviation_pct/100):
                direction = -1
                last_pivot_idx   = i
                last_pivot_price = l[i]
                zz[i] = l[i]
        elif direction == 1:  # движемся вверх
            if h[i] > last_pivot_price:
                # Обновляем максимум
                zz[last_pivot_idx] = np.nan
                last_pivot_idx   = i
                last_pivot_price = h[i]
                zz[i] = h[i]
            elif l[i] < last_pivot_price * (1 - deviation_pct/100):
                # Разворот вниз
                direction = -1
                last_pivot_idx   = i
                last_pivot_price = l[i]
                zz[i] = l[i]
        elif direction == -1:  # движемся вниз
            if l[i] < last_pivot_price:
                # Обновляем минимум
                zz[last_pivot_idx] = np.nan
                last_pivot_idx   = i
                last_pivot_price = l[i]
                zz[i] = l[i]
            elif h[i] > last_pivot_price * (1 + deviation_pct/100):
                # Разворот вверх
                direction = 1
                last_pivot_idx   = i
                last_pivot_price = h[i]
                zz[i] = h[i]

    return pd.Series(zz, index=df.index)


def zigzag_wave_count(zz: pd.Series, n_recent: int = 10) -> dict:
    """
    Упрощённый счётчик волн Эллиотта через ZigZag.
    Возвращает: wave_count, current_direction, last_pivots
    """
    pivots = zz.dropna()
    if len(pivots) < 2:
        return {"wave_count": 0, "direction": 0, "stage": "unknown"}

    recent = pivots.iloc[-min(n_recent, len(pivots)):]
    vals   = recent.values
    idxs   = recent.index

    # Считаем направления между соседними пивотами
    directions = []
    for i in range(1, len(vals)):
        directions.append(1 if vals[i] > vals[i-1] else -1)

    # Текущее направление — последний зигзаг
    current_dir = directions[-1] if directions else 0

    # Подсчёт волн (чередующихся движений)
    wave_count = len(directions)

    # Определяем стадию: если последние 5 волн восходящие (нечётные = up)
    # это импульс; если 3 волны — коррекция
    if wave_count >= 5:
        # Проверяем паттерн 5 волн
        last5 = directions[-5:]
        impulse = (last5[0] > 0 and last5[1] < 0 and last5[2] > 0 and
                   last5[3] < 0 and last5[4] > 0)
        if impulse:
            stage = "wave5_impulse_complete"  # ждём коррекцию
        elif all(d > 0 for d in last5[::2]) and all(d < 0 for d in last5[1::2]):
            stage = "impulse_in_progress"
        else:
            stage = "correction"
    elif wave_count >= 3:
        last3 = directions[-3:]
        abc = (last3[0] < 0 and last3[1] > 0 and last3[2] < 0)
        if abc:
            stage = "abc_correction_complete"  # хорошая точка для покупки
        else:
            stage = "wave_forming"
    else:
        stage = "early"

    # Последний pivot price и индекс
    last_low  = min(vals[-4:]) if len(vals) >= 4 else vals.min()
    last_high = max(vals[-4:]) if len(vals) >= 4 else vals.max()

    return {
        "wave_count":   wave_count,
        "direction":    current_dir,
        "stage":        stage,
        "last_low":     float(last_low),
        "last_high":    float(last_high),
        "n_pivots":     len(pivots),
    }


# ══════════════════════════════════════════════════════════════════════════════
# БЫЧИЙ СИГНАЛ НА КАЖДОМ ТАЙМФРЕЙМЕ
# ══════════════════════════════════════════════════════════════════════════════
def bullish_signal_tf(df: pd.DataFrame, tf_name: str) -> pd.Series:
    """
    Возвращает Series[bool]: True на барах с бычьим сигналом.
    Использует ATR Breakout + EMA200 + RSI + ADX с параметрами под TF.
    """
    if df is None or len(df) < 210:
        return pd.Series(dtype=bool)

    c     = df["close"]
    at14  = atr(df, 14)
    at5   = atr(df, 5)
    e200  = ema(c, 200)
    rsi14 = rsi(c, 14)
    adx14 = adx(df, 14)
    vol_r = volume_ratio(df, 20)
    bm    = (c - c.shift(1)).clip(lower=0)  # bullish move

    # Параметры немного мягче на старших TF (чтобы не требовать пробоя)
    if tf_name == "1H":
        sig = ((c > e200) & (bm >= 1.2*at14) & (rsi14 >= 50) &
               (rsi14 <= 80) & (adx14 >= 20) & (vol_r >= 1.3))
    elif tf_name == "4H":
        sig = ((c > e200) & (bm >= 1.0*at14) & (rsi14 >= 50) &
               (rsi14 <= 82) & (adx14 >= 18))
    elif tf_name == "8H":
        sig = ((c > e200) & (rsi14 >= 52) & (adx14 >= 18))
    elif tf_name == "12H":
        sig = ((c > e200) & (rsi14 >= 50) & (adx14 >= 15))
    elif tf_name == "D":
        sig = ((c > e200) & (rsi14 >= 48) & (adx14 >= 15))
    else:
        sig = (c > e200)

    return sig


def get_mtf_score_at(
    ticker: str,
    dt: pd.Timestamp,
    tf_data: dict[str, pd.DataFrame],
    tf_sigs: dict[str, pd.Series],
) -> dict:
    """
    Для заданного момента времени вычисляет MTF Score и состояние рынка.
    Возвращает score, fractal_sl, wave_info, active_tfs.
    """
    score     = 0
    active    = []
    TF_ORDER  = ["1H", "4H", "8H", "12H", "D"]

    for tf in TF_ORDER:
        if tf not in tf_data or tf_data[tf] is None:
            continue
        df_tf  = tf_data[tf]
        sig_tf = tf_sigs[tf]

        # Находим последний доступный бар на этом TF до момента dt
        past = sig_tf[sig_tf.index <= dt]
        if len(past) == 0:
            continue

        # Сигнал на последнем баре
        if bool(past.iloc[-1]):
            score += 1
            active.append(tf)

    # Фрактальный SL (с дневного TF)
    frac_sl = None
    if "D" in tf_data and tf_data["D"] is not None:
        df_d   = tf_data["D"]
        past_d = df_d[df_d.index <= dt]
        if len(past_d) >= 10:
            frac_sl = fractal_sl(past_d, len(past_d)-1, lookback=20)

    # ZigZag анализ (на дневном TF)
    wave_info = {}
    if "D" in tf_data and tf_data["D"] is not None:
        df_d   = tf_data["D"]
        past_d = df_d[df_d.index <= dt]
        if len(past_d) >= 50:
            zz      = zigzag(past_d, deviation_pct=5.0)
            wave_info = zigzag_wave_count(zz)

    return {
        "score":    score,
        "active":   active,
        "frac_sl":  frac_sl,
        "wave":     wave_info,
    }


# ══════════════════════════════════════════════════════════════════════════════
# ЗАГРУЗКА ВСЕХ ДАННЫХ
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("  MTF TREND CONFIRMATION STRATEGY")
print("  Многотаймфреймная стратегия с фракталами Билла Вильямса + ZigZag")
print("=" * 70)
print("\nЗагрузка данных по таймфреймам...")

ALL_TF_DATA: dict[str, dict[str, pd.DataFrame]] = {}
ALL_TF_SIGS: dict[str, dict[str, pd.Series]]    = {}

TF_LIST = ["1H", "4H", "8H", "12H", "D"]

for ticker in TICKERS_1H:
    tf_data = {}
    tf_sigs = {}
    ok_tfs  = []

    for tf in TF_LIST:
        df = load_tf(ticker, tf)
        if df is not None and len(df) >= 200:
            tf_data[tf] = df
            tf_sigs[tf] = bullish_signal_tf(df, tf)
            ok_tfs.append(tf)

    if "1H" in tf_data and "D" in tf_data:
        ALL_TF_DATA[ticker] = tf_data
        ALL_TF_SIGS[ticker] = tf_sigs
        print(f"  {ticker:6s}: TFs загружены = {ok_tfs} | "
              f"1H баров: {len(tf_data.get('1H',[])):5d} | "
              f"D баров: {len(tf_data.get('D',[])):4d}")
    else:
        print(f"  {ticker:6s}: пропущен (нет 1H или D)")

print(f"\n  Итого тикеров: {len(ALL_TF_DATA)}")


# ══════════════════════════════════════════════════════════════════════════════
# ГЕНЕРАЦИЯ СИГНАЛОВ MTF
# ══════════════════════════════════════════════════════════════════════════════
print("\nГенерация MTF сигналов...")

MIN_SCORE = 3   # минимальный счёт для входа
EXIT_SCORE = 1  # если score < этого — выходим

@dataclass
class MTFSignal:
    dt:        pd.Timestamp
    ticker:    str
    score:     int
    active_tfs: list
    frac_sl:   Optional[float]
    wave_stage: str
    entry_px:  float


def generate_mtf_signals(ticker: str) -> list[MTFSignal]:
    """
    Сканируем 1H данные. На каждом часовом баре проверяем MTF Score.
    Сигнал: score ≥ MIN_SCORE И нет открытой позиции.
    """
    tf_data = ALL_TF_DATA[ticker]
    tf_sigs = ALL_TF_SIGS[ticker]
    df_1h   = tf_data["1H"]

    signals = []
    in_position = False
    entry_score = 0

    # Предрасчёт сигналов на других TF для быстрого доступа
    # Для каждого TF строим "последний активный сигнал до момента dt"
    # Используем resample и ffill для выравнивания на 1H timeline

    # Выравниваем все TF на 1H ось
    aligned: dict[str, pd.Series] = {}
    for tf in TF_LIST:
        if tf == "1H" or tf not in tf_sigs:
            continue
        sig_tf = tf_sigs[tf]
        # Reindex на 1H, заполняем предыдущим значением (последний известный сигнал)
        sig_aligned = sig_tf.reindex(
            df_1h.index, method="ffill"
        ).fillna(False)
        aligned[tf] = sig_aligned

    # Дневные данные для фрактального SL и ZigZag
    df_d  = tf_data.get("D")
    at14_d = atr(df_d, 14) if df_d is not None else None

    # Предрасчёт ZigZag на дневных данных
    zz_d = zigzag(df_d, deviation_pct=5.0) if df_d is not None else None

    # Предрасчёт фракталов на дневных данных
    frac_h_d, frac_l_d = bw_fractals(df_d, n=2) if df_d is not None else (None, None)
    fractal_lows_d = df_d["low"][frac_l_d] if (df_d is not None and frac_l_d is not None) else None

    for i, (dt, row) in enumerate(df_1h.iterrows()):
        if i < 5:
            continue

        # Считаем score
        score = 0
        active_tfs = []

        # 1H сигнал
        if bool(tf_sigs["1H"].iloc[i-1]):  # предыдущий бар сформирован
            score += 1
            active_tfs.append("1H")

        # Старшие TF
        for tf in ["4H", "8H", "12H", "D"]:
            if tf in aligned and bool(aligned[tf].iloc[i]):
                score += 1
                active_tfs.append(tf)

        if not in_position and score >= MIN_SCORE:
            # Определяем фрактальный SL
            if df_d is not None and fractal_lows_d is not None:
                past_fl = fractal_lows_d[fractal_lows_d.index <= dt]
                frac_sl_val = float(past_fl.iloc[-1]) if len(past_fl) > 0 else None
            else:
                frac_sl_val = None

            # ZigZag волновая стадия
            wave_stage = "unknown"
            if zz_d is not None:
                past_zz = zz_d[zz_d.index <= dt]
                wi = zigzag_wave_count(past_zz)
                wave_stage = wi.get("stage", "unknown")

            # Не входим если волны 5 уже завершены (ждём коррекцию)
            if wave_stage == "wave5_impulse_complete":
                continue

            signals.append(MTFSignal(
                dt=dt, ticker=ticker, score=score,
                active_tfs=active_tfs, frac_sl=frac_sl_val,
                wave_stage=wave_stage,
                entry_px=float(df_1h["open"].iloc[i]) * (1+SLIPPAGE),
            ))
            in_position = True
            entry_score = score

        elif in_position and score < EXIT_SCORE:
            in_position = False

    return signals


# Генерируем сигналы для всех тикеров
all_signals: dict[str, list[MTFSignal]] = {}
for ticker in ALL_TF_DATA:
    sigs = generate_mtf_signals(ticker)
    all_signals[ticker] = sigs
    if sigs:
        scores = [s.score for s in sigs]
        print(f"  {ticker:6s}: {len(sigs):3d} сигналов | "
              f"avg score: {np.mean(scores):.1f} | "
              f"score 5/4/3: "
              f"{sum(1 for s in sigs if s.score==5)}/"
              f"{sum(1 for s in sigs if s.score==4)}/"
              f"{sum(1 for s in sigs if s.score==3)}")
    else:
        print(f"  {ticker:6s}: нет сигналов")


# ══════════════════════════════════════════════════════════════════════════════
# БЭКТЕСТ MTF СТРАТЕГИИ
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  БЭКТЕСТ MTF СТРАТЕГИИ")
print("=" * 70)

INITIAL_CAP  = 100_000.0
MAX_POSITIONS = 4
RISK_PCT      = 0.20   # 20% капитала на позицию
TRAIL_MULT    = 2.0    # trailing stop = 2×ATR(D)
MAX_HOLD_DAYS = 45


@dataclass
class Position:
    ticker:    str
    entry_dt:  pd.Timestamp
    entry_px:  float
    shares:    float
    sl_px:     float    # фрактальный SL
    trail_sl:  float    # trailing SL (max из фрактального и trailing)
    score:     int
    wave_stage: str


@dataclass
class Trade:
    ticker:     str
    entry_dt:   pd.Timestamp
    exit_dt:    pd.Timestamp
    entry_px:   float
    exit_px:    float
    pnl_rub:    float
    pnl_pct:    float
    reason:     str
    score:      int
    wave_stage: str
    hold_days:  float


def run_mtf_backtest(min_score: int = 3, trail_mult: float = 2.0,
                     max_hold_days: int = 45) -> dict:
    free_cash  = INITIAL_CAP
    positions: dict[str, Position] = {}
    trades:    list[Trade] = []
    equity     = [INITIAL_CAP]
    peak_eq    = INITIAL_CAP
    max_dd     = 0.0

    # Собираем все торговые часовые бары для итерации
    # Используем 1H данные первого тикера как опорный timeline
    master_ticks = set()
    for ticker, tf_data in ALL_TF_DATA.items():
        master_ticks.update(tf_data["1H"].index)
    timeline = sorted(master_ticks)

    # Индексы 1H для быстрого доступа
    IDX_1H = {t: {d: i for i, d in enumerate(df.index)}
               for t, df in {t: ALL_TF_DATA[t]["1H"] for t in ALL_TF_DATA}.items()}

    # Предрасчёт ATR(D) выровненных на 1H
    ATR_D_ALIGNED: dict[str, pd.Series] = {}
    for ticker, tf_data in ALL_TF_DATA.items():
        df_d = tf_data.get("D")
        if df_d is not None:
            at_d = atr(df_d, 14)
            ATR_D_ALIGNED[ticker] = at_d.reindex(
                tf_data["1H"].index, method="ffill"
            ).ffill()

    # Предрасчёт фрактальных минимумов на D
    FRAC_SL_SERIES: dict[str, pd.Series] = {}
    for ticker, tf_data in ALL_TF_DATA.items():
        df_d = tf_data.get("D")
        if df_d is not None:
            _, frac_l_d = bw_fractals(df_d, n=2)
            fl_d = df_d["low"].where(frac_l_d).ffill()
            FRAC_SL_SERIES[ticker] = fl_d.reindex(
                tf_data["1H"].index, method="ffill"
            ).ffill()

    # Сигналы по дате для быстрого доступа
    SIGS_BY_DT: dict[str, dict] = {t: {} for t in ALL_TF_DATA}
    for ticker, sigs in all_signals.items():
        for s in sigs:
            SIGS_BY_DT[ticker][s.dt] = s

    # Итерируемся по часовым барам
    for dt in timeline:
        # ── Обновляем открытые позиции ────────────────────────────────────────
        to_close = []
        for ticker, pos in positions.items():
            df_1h = ALL_TF_DATA[ticker]["1H"]
            idx = IDX_1H[ticker].get(dt)
            if idx is None or idx <= 0:
                continue
            if dt <= pos.entry_dt:
                continue

            hi  = float(df_1h["high"].iloc[idx])
            lo  = float(df_1h["low"].iloc[idx])
            op  = float(df_1h["open"].iloc[idx])
            cls = float(df_1h["close"].iloc[idx])

            # Trailing SL на основе ATR(D)
            if ticker in ATR_D_ALIGNED:
                at_d_val = float(ATR_D_ALIGNED[ticker].iloc[idx])
                trail = cls - trail_mult * at_d_val
                if trail > pos.trail_sl:
                    pos.trail_sl = trail

            # Текущий SL = max(фрактальный SL, trailing SL)
            curr_sl = max(pos.sl_px, pos.trail_sl)

            # Обновляем фрактальный SL (он может двигаться вверх по мере появления новых фракталов)
            if ticker in FRAC_SL_SERIES:
                new_frac_sl = float(FRAC_SL_SERIES[ticker].iloc[idx])
                if new_frac_sl > pos.sl_px:
                    pos.sl_px = new_frac_sl
                    curr_sl = max(pos.sl_px, pos.trail_sl)

            # Проверяем причины закрытия
            hold_days = (dt - pos.entry_dt).total_seconds() / 86400
            reason = exit_px = None

            if lo <= curr_sl:
                reason  = "SL_FRACTAL"
                exit_px = max(curr_sl * (1-SLIPPAGE), lo)
            elif hold_days >= max_hold_days:
                reason  = "TIME"
                exit_px = op * (1-SLIPPAGE)

            if reason:
                cash = exit_px * pos.shares * (1-COMMISSION)
                cost = pos.entry_px * pos.shares * (1+COMMISSION)
                pnl  = cash - cost
                pnl_pct = pnl / cost * 100

                free_cash += cash
                trades.append(Trade(
                    ticker=ticker, entry_dt=pos.entry_dt, exit_dt=dt,
                    entry_px=pos.entry_px, exit_px=exit_px,
                    pnl_rub=pnl, pnl_pct=pnl_pct, reason=reason,
                    score=pos.score, wave_stage=pos.wave_stage,
                    hold_days=hold_days,
                ))
                to_close.append(ticker)

        for t in to_close:
            positions.pop(t, None)

        # ── Новые входы ────────────────────────────────────────────────────────
        for ticker in ALL_TF_DATA:
            if len(positions) >= MAX_POSITIONS:
                break
            if ticker in positions:
                continue

            sig = SIGS_BY_DT[ticker].get(dt)
            if sig is None or sig.score < min_score:
                continue

            df_1h = ALL_TF_DATA[ticker]["1H"]
            idx   = IDX_1H[ticker].get(dt)
            if idx is None:
                continue

            entry_px = float(df_1h["open"].iloc[idx]) * (1+SLIPPAGE)

            # Фрактальный SL
            frac_sl_val = sig.frac_sl
            if frac_sl_val is None or frac_sl_val <= 0:
                if ticker in FRAC_SL_SERIES:
                    frac_sl_val = float(FRAC_SL_SERIES[ticker].iloc[idx])
                else:
                    frac_sl_val = entry_px * 0.90

            # Проверяем что SL не слишком далеко (макс 25% от entry)
            sl_dist_pct = (entry_px - frac_sl_val) / entry_px
            if sl_dist_pct > 0.25:
                frac_sl_val = entry_px * 0.85  # ограничиваем 15%

            # Размер позиции
            pos_val = sum(
                float(ALL_TF_DATA[t]["1H"]["close"].iloc[IDX_1H[t].get(dt,-1)])
                * p.shares
                for t, p in positions.items()
                if IDX_1H[t].get(dt) is not None
            )
            total_cap = free_cash + pos_val
            alloc     = min(total_cap * RISK_PCT, free_cash * 0.95)
            if alloc <= 0:
                continue

            shares = alloc / entry_px
            cost   = shares * entry_px * (1+COMMISSION)
            if cost > free_cash:
                continue

            free_cash -= cost
            positions[ticker] = Position(
                ticker=ticker, entry_dt=dt, entry_px=entry_px,
                shares=shares, sl_px=frac_sl_val,
                trail_sl=frac_sl_val,  # начальный trailing = фрактальный SL
                score=sig.score, wave_stage=sig.wave_stage,
            )

        # Equity update
        pos_val = 0.0
        for ticker, pos in positions.items():
            idx = IDX_1H[ticker].get(dt)
            if idx is not None:
                px = float(ALL_TF_DATA[ticker]["1H"]["close"].iloc[idx])
            else:
                px = pos.entry_px
            pos_val += px * pos.shares
        eq = free_cash + pos_val
        equity.append(eq)
        if eq > peak_eq:
            peak_eq = eq
        dd = (peak_eq - eq) / peak_eq * 100
        if dd > max_dd:
            max_dd = dd

    # Принудительное закрытие
    last_dt = timeline[-1]
    for ticker, pos in list(positions.items()):
        cls = float(ALL_TF_DATA[ticker]["1H"]["close"].iloc[-1])
        ep  = cls * (1-SLIPPAGE)
        cash = ep * pos.shares * (1-COMMISSION)
        cost = pos.entry_px * pos.shares * (1+COMMISSION)
        pnl  = cash - cost
        pnl_pct = pnl / cost * 100
        free_cash += cash
        trades.append(Trade(
            ticker=ticker, entry_dt=pos.entry_dt, exit_dt=last_dt,
            entry_px=pos.entry_px, exit_px=ep,
            pnl_rub=pnl, pnl_pct=pnl_pct, reason="FORCED",
            score=pos.score, wave_stage=pos.wave_stage,
            hold_days=(last_dt-pos.entry_dt).total_seconds()/86400,
        ))

    # Статистика
    final     = free_cash
    total_pnl = (final-INITIAL_CAP)/INITIAL_CAP*100
    n_days    = (timeline[-1]-timeline[0]).days
    ann       = ((final/INITIAL_CAP)**(365/max(n_days,1))-1)*100

    pnls   = np.array([t.pnl_pct for t in trades])
    n_tr   = len(trades)
    n_win  = (pnls>0).sum()
    wr     = n_win/n_tr*100 if n_tr else 0
    wins   = pnls[pnls>0]; losses = pnls[pnls<=0]
    pf     = wins.sum()/(-losses.sum()+1e-9) if len(losses) else 99.0

    eq_arr = np.array(equity)
    dr     = np.diff(eq_arr)/(eq_arr[:-1]+1e-9)
    sharpe = (dr.mean()/(dr.std()+1e-9))*np.sqrt(252*6.5)  # 6.5 часов в дне

    by_reason: dict[str,dict] = {}
    for t in trades:
        s = by_reason.setdefault(t.reason, {"n":0,"wins":0,"pnl":0.0})
        s["n"]+=1; s["pnl"]+=t.pnl_pct
        if t.pnl_pct>0: s["wins"]+=1

    by_score: dict[int,dict] = {}
    for t in trades:
        s = by_score.setdefault(t.score, {"n":0,"wins":0,"pnl":0.0})
        s["n"]+=1; s["pnl"]+=t.pnl_pct
        if t.pnl_pct>0: s["wins"]+=1

    # Годовая разбивка
    by_year: dict[str,float] = {}
    running = INITIAL_CAP
    for t in sorted(trades, key=lambda x: x.exit_dt):
        yr = str(t.exit_dt)[:4]
        running += t.pnl_rub
        by_year[yr] = running

    return dict(
        final=final, total_pnl=total_pnl, ann=ann, max_dd=-max_dd,
        sharpe=sharpe, trades=n_tr, wr=wr, pf=min(pf,99),
        avg_win=float(wins.mean()) if len(wins) else 0,
        avg_loss=float(losses.mean()) if len(losses) else 0,
        by_reason=by_reason, by_score=by_score,
        by_year=by_year, trades_list=trades,
    )


# ── Запуск с разными параметрами ───────────────────────────────────────────────
print("\nТест параметров MTF стратегии...")
print(f"  {'min_score':10s} {'trail':6s} {'hold':5s} {'Сделок':7s} "
      f"{'WR%':6s} {'PF':5s} {'ANN%':6s} {'MaxDD':7s} {'Sharpe':7s} {'Итог ₽':10s}")
print("  " + "─" * 75)

configs = [
    (3, 2.0, 45),
    (3, 2.5, 45),
    (3, 2.0, 30),
    (4, 2.0, 45),
    (4, 2.5, 60),
    (5, 2.0, 60),
]

best_score_val = -999; best_r = None
for min_sc, trail, hold in configs:
    r = run_mtf_backtest(min_sc, trail, hold)
    sc = r["ann"]*0.6 + r["sharpe"]*5 - abs(r["max_dd"])*0.4
    mk = " ◄" if sc > best_score_val else ""
    if sc > best_score_val:
        best_score_val = sc; best_r = r
    print(f"  score≥{min_sc}   trail={trail}  h={hold:2d}   {r['trades']:7d} "
          f"{r['wr']:5.1f}% {r['pf']:5.2f} {r['ann']:>+5.1f}% "
          f"{r['max_dd']:>+6.1f}% {r['sharpe']:>6.2f}  {r['final']:>10,.0f}{mk}")


# ── Детальный разбор победителя ────────────────────────────────────────────────
if best_r:
    print(f"\n{'═'*70}")
    print("  ЛУЧШАЯ MTF КОНФИГУРАЦИЯ — ДЕТАЛЬНЫЙ ОТЧЁТ")
    print(f"{'═'*70}")
    print(f"  Начальный капитал:  {INITIAL_CAP:>10,.0f} ₽")
    print(f"  Итоговый капитал:   {best_r['final']:>10,.0f} ₽")
    print(f"  Суммарная прибыль:  {best_r['total_pnl']:>+9.1f}%")
    print(f"  CAGR:               {best_r['ann']:>+9.1f}%")
    print(f"  MaxDD:              {best_r['max_dd']:>+9.1f}%")
    print(f"  Sharpe:             {best_r['sharpe']:>9.2f}")
    print(f"  Trades:             {best_r['trades']:>9d}")
    print(f"  WR:                 {best_r['wr']:>9.1f}%")
    print(f"  PF:                 {best_r['pf']:>9.2f}")
    print(f"  Avg Win/Loss:       {best_r['avg_win']:>+6.2f}% / {best_r['avg_loss']:>+6.2f}%")

    print("\n  ПО ГОДАМ:")
    print(f"  {'Год':6s} {'Капитал':>12s} {'Прибыль':>12s} {'%':>8s}")
    print("  " + "─" * 42)
    prev = INITIAL_CAP
    for yr in sorted(best_r["by_year"]):
        cap = best_r["by_year"][yr]
        pr  = cap - prev
        pct = pr/prev*100 if prev > 0 else 0
        print(f"  {yr}   {cap:>12,.0f} ₽  {pr:>+10,.0f} ₽  {pct:>+7.1f}%")
        prev = cap
    total_pr = best_r["final"] - INITIAL_CAP
    print("  " + "─" * 42)
    print(f"  {'ИТОГО':6s} {best_r['final']:>12,.0f} ₽  {total_pr:>+10,.0f} ₽  "
          f"{best_r['total_pnl']:>+7.1f}%")

    print("\n  ПО ПРИЧИНАМ ВЫХОДА:")
    print(f"  {'Причина':12s} {'N':5s} {'WR%':7s} {'Avg%':8s}")
    print("  " + "─" * 36)
    for reason, s in sorted(best_r["by_reason"].items()):
        wr_r = s["wins"]/s["n"]*100 if s["n"] else 0
        avg  = s["pnl"]/s["n"] if s["n"] else 0
        print(f"  {reason:12s} {s['n']:5d} {wr_r:>6.1f}% {avg:>+7.2f}%")

    print("\n  ПО SCORE (качество сигнала):")
    print(f"  {'Score':6s} {'N':5s} {'WR%':7s} {'Avg%':8s}")
    print("  " + "─" * 30)
    for sc, s in sorted(best_r["by_score"].items()):
        wr_r = s["wins"]/s["n"]*100 if s["n"] else 0
        avg  = s["pnl"]/s["n"] if s["n"] else 0
        print(f"  {sc:6d} {s['n']:5d} {wr_r:>6.1f}% {avg:>+7.2f}%")

    # Топ-10 лучших сделок
    tl_s = sorted(best_r["trades_list"], key=lambda t: t.pnl_pct, reverse=True)
    print("\n  ТОП-10 ЛУЧШИХ СДЕЛОК:")
    print(f"  {'Тикер':6s} {'Вход':18s} {'Выход':18s} {'Дни':5s} {'P&L%':8s} {'Причина':12s} {'Score':6s}")
    for t in tl_s[:10]:
        print(f"  {t.ticker:6s} {str(t.entry_dt)[:16]:18s} {str(t.exit_dt)[:16]:18s} "
              f"{t.hold_days:4.0f}д {t.pnl_pct:>+7.1f}% {t.reason:12s} {t.score}")
    print("\n  ХУДШИЕ 5 СДЕЛОК:")
    for t in tl_s[-5:]:
        print(f"  {t.ticker:6s} {str(t.entry_dt)[:16]:18s} {str(t.exit_dt)[:16]:18s} "
              f"{t.hold_days:4.0f}д {t.pnl_pct:>+7.1f}% {t.reason:12s} {t.score}")

    # По тикерам
    by_t: dict[str,list] = {}
    for t in best_r["trades_list"]: by_t.setdefault(t.ticker,[]).append(t.pnl_pct)
    print("\n  ПО ТИКЕРАМ:")
    print(f"  {'Тикер':6s} {'N':4s} {'WR%':6s} {'Total%':8s} {'Avg%':7s}")
    print("  " + "─" * 36)
    for tk, pnls in sorted(by_t.items(), key=lambda x: sum(x[1]), reverse=True):
        pa = np.array(pnls)
        print(f"  {tk:6s} {len(pnls):4d} {(pa>0).mean()*100:5.1f}% "
              f"{pa.sum():>+7.1f}% {pa.mean():>+6.2f}%")

print(f"\n{'═'*70}")
print("  СРАВНЕНИЕ с предыдущей дневной стратегией:")
print("  Дневная ATR_BO (B): ANN=+13.9%, MaxDD=-12.9%, Sharpe=1.19")
print(f"  MTF (лучший):       ANN={best_r['ann']:>+.1f}%, "
      f"MaxDD={best_r['max_dd']:>+.1f}%, Sharpe={best_r['sharpe']:.2f}")
print(f"{'═'*70}")
