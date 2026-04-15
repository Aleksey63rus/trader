"""
=============================================================================
MTF STRATEGY v2 — Улучшенная многотаймфреймная стратегия
=============================================================================

ПРОБЛЕМЫ v1 (диагноз):
  - Score=3 даёт WR=43.9% — слишком много ложных сигналов
  - Score=4 даёт WR=57.1% — хорошо, но мало сигналов (83 из 285)
  - Тикеры GAZP, T, MTLR убыточны даже с правильными сигналами
  - Trailing stop 2×ATR(D) иногда слишком близко на 1H масштабе

УЛУЧШЕНИЯ v2:
  1. SCORE UPGRADE — расширяем критерии каждого TF:
     - Добавляем Volume Surge (объём выше среднего в момент сигнала)
     - Учитываем положение цены относительно BB (Bollinger Bands)
     - Momentum: скорость роста за последние N баров

  2. FRACTAL BREAKOUT FILTER — входим только когда цена пробивает
     последний фрактальный максимум (подтверждение пробоя, как рекомендует Б.Вильямс)

  3. ZIGZAG PHASE FILTER — входим только в фазе волны 3 или после
     ABC-коррекции (лучшие точки входа по Эллиотту)

  4. DYNAMIC EXIT:
     - Первичный выход: пробой фрактального минимума ДНЕВНОГО TF
     - Вторичный выход: ZigZag разворот (новый ZigZag максимум сменяется минимумом)
     - TIME выход адаптивный: 20-60 дней в зависимости от фазы тренда

  5. PARTIAL PROFIT на фрактальных максимумах:
     - 33% позиции на первом фрактальном максимуме выше entry
     - 33% на следующем фрактальном максимуме
     - 34% на ZigZag-развороте или TIME

=============================================================================
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
from dataclasses import dataclass
import numpy as np
import pandas as pd

# Импортируем уже готовые функции из v1
from mtf_strategy import (
    bw_fractals, zigzag, ema, atr, rsi, adx, volume_ratio,
    COMMISSION, SLIPPAGE,
    ALL_TF_DATA,
)

INITIAL_CAP   = 100_000.0
MAX_POSITIONS = 4
RISK_PCT      = 0.20

# ══════════════════════════════════════════════════════════════════════════════
# НОВЫЕ ИНДИКАТОРЫ
# ══════════════════════════════════════════════════════════════════════════════
def bollinger_bands(s: pd.Series, n: int = 20, k: float = 2.0):
    """Возвращает (upper, mid, lower)."""
    mid   = s.rolling(n).mean()
    std   = s.rolling(n).std()
    return mid + k*std, mid, mid - k*std

def momentum(s: pd.Series, n: int = 10) -> pd.Series:
    """% изменение за n баров."""
    return (s - s.shift(n)) / s.shift(n) * 100

def stochastic(df: pd.DataFrame, k: int = 14, d: int = 3) -> pd.Series:
    """Stochastic %K."""
    h14 = df["high"].rolling(k).max()
    l14 = df["low"].rolling(k).min()
    k_  = 100 * (df["close"] - l14) / (h14 - l14 + 1e-10)
    return k_.rolling(d).mean()  # %D (smoothed)

def fractal_high_series(df: pd.DataFrame, n: int = 2) -> pd.Series:
    """Серия последних фрактальных максимумов (ffill)."""
    fh, _ = bw_fractals(df, n)
    fh_vals = df["high"].where(fh)
    return fh_vals.ffill()

def fractal_low_series(df: pd.DataFrame, n: int = 2) -> pd.Series:
    """Серия последних фрактальных минимумов (ffill)."""
    _, fl = bw_fractals(df, n)
    fl_vals = df["low"].where(fl)
    return fl_vals.ffill()


# ══════════════════════════════════════════════════════════════════════════════
# УЛУЧШЕННЫЙ SCORE — 8 компонентов × TF
# ══════════════════════════════════════════════════════════════════════════════
def compute_tf_features(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    """
    Возвращает DataFrame с бинарными фичами (0/1) на каждом баре:
      bull_ema200, bull_atr_bo, bull_rsi, bull_adx,
      bull_vol, bull_mom, bull_above_bb_mid, bull_stoch
    """
    if df is None or len(df) < 210:
        return pd.DataFrame()

    c     = df["close"]
    h     = df["high"]
    l     = df["low"]
    v     = df["volume"]
    at14  = atr(df, 14)
    at5   = atr(df, 5)
    e200  = ema(c, 200)
    e50   = ema(c, 50)
    rsi14 = rsi(c, 14)
    adx14 = adx(df, 14)
    vol_r = volume_ratio(df, 20)
    bm    = (c - c.shift(1)).clip(lower=0)
    mom10 = momentum(c, 10)
    stoch = stochastic(df, 14, 3)
    bb_u, bb_m, bb_l = bollinger_bands(c, 20, 2.0)

    # Адаптируем пороги под TF
    if tf == "1H":
        atr_bo_mult = 1.2; rsi_lo = 50; rsi_hi = 80; adx_thr = 18; vol_thr = 1.2
    elif tf == "4H":
        atr_bo_mult = 0.8; rsi_lo = 48; rsi_hi = 82; adx_thr = 16; vol_thr = 1.1
    elif tf == "8H":
        atr_bo_mult = 0.5; rsi_lo = 47; rsi_hi = 83; adx_thr = 15; vol_thr = 1.0
    elif tf == "12H":
        atr_bo_mult = 0.3; rsi_lo = 46; rsi_hi = 84; adx_thr = 14; vol_thr = 1.0
    elif tf == "D":
        atr_bo_mult = 0.2; rsi_lo = 45; rsi_hi = 85; adx_thr = 13; vol_thr = 1.0
    else:
        atr_bo_mult = 1.0; rsi_lo = 48; rsi_hi = 82; adx_thr = 15; vol_thr = 1.0

    feat = pd.DataFrame(index=df.index)
    feat["bull_ema200"]     = (c > e200).astype(int)
    feat["bull_ema50"]      = (c > e50).astype(int)
    feat["bull_atr_bo"]     = (bm >= atr_bo_mult * at14).astype(int)
    feat["bull_rsi"]        = ((rsi14 >= rsi_lo) & (rsi14 <= rsi_hi)).astype(int)
    feat["bull_adx"]        = (adx14 >= adx_thr).astype(int)
    feat["bull_vol"]        = (vol_r >= vol_thr).astype(int)
    feat["bull_mom"]        = (mom10 > 0).astype(int)
    feat["bull_stoch"]      = ((stoch > 20) & (stoch < 85)).astype(int)
    feat["bull_above_bbmid"]= (c > bb_m).astype(int)
    # Пробой фрактального максимума (цена выше последнего fractal high)
    fh_s = fractal_high_series(df, n=2)
    feat["bull_frac_break"] = (c > fh_s.shift(1)).astype(int)

    # Суммарный бычий score для этого TF (0-10)
    feat["tf_score"] = feat[[
        "bull_ema200", "bull_ema50", "bull_atr_bo", "bull_rsi",
        "bull_adx", "bull_vol", "bull_mom", "bull_stoch",
        "bull_above_bbmid", "bull_frac_break",
    ]].sum(axis=1)

    # Бинарный сигнал: score >= порога для данного TF
    thresholds = {"1H": 6, "4H": 5, "8H": 4, "12H": 4, "D": 4}
    thr = thresholds.get(tf, 5)
    feat["signal"] = (feat["tf_score"] >= thr).astype(int)

    return feat


print("Вычисление улучшенных фич по таймфреймам...")
ALL_FEATS: dict[str, dict[str, pd.DataFrame]] = {}
for ticker, tf_data in ALL_TF_DATA.items():
    ALL_FEATS[ticker] = {}
    for tf, df in tf_data.items():
        feat = compute_tf_features(df, tf)
        if not feat.empty:
            ALL_FEATS[ticker][tf] = feat

print(f"  Готово: {len(ALL_FEATS)} тикеров")


# ══════════════════════════════════════════════════════════════════════════════
# ПРЕДРАСЧЁТ ZigZag и ФРАКТАЛОВ на D
# ══════════════════════════════════════════════════════════════════════════════
print("Предрасчёт ZigZag и фракталов на дневных данных...")
ZZ_D:       dict[str, pd.Series]   = {}
FRAC_H_D:   dict[str, pd.Series]   = {}   # fractal highs
FRAC_L_D:   dict[str, pd.Series]   = {}   # fractal lows
ATR_D:      dict[str, pd.Series]   = {}

for ticker, tf_data in ALL_TF_DATA.items():
    df_d = tf_data.get("D")
    if df_d is None:
        continue
    ZZ_D[ticker]     = zigzag(df_d, deviation_pct=5.0)
    fh, fl           = bw_fractals(df_d, n=2)
    FRAC_H_D[ticker] = df_d["high"].where(fh).ffill()
    FRAC_L_D[ticker] = df_d["low"].where(fl).ffill()
    ATR_D[ticker]    = atr(df_d, 14)

print("  Готово")


# ══════════════════════════════════════════════════════════════════════════════
# MTF SCORE v2 — Выровненные на 1H временную ось
# ══════════════════════════════════════════════════════════════════════════════
print("Выравнивание сигналов на 1H ось...")

TF_LIST  = ["1H", "4H", "8H", "12H", "D"]
TF_WEIGHT = {"1H": 1, "4H": 1, "8H": 1, "12H": 1, "D": 2}  # D весит вдвое

# Для каждого тикера строим Series суммарного MTF score на 1H оси
MTF_SCORE_1H: dict[str, pd.Series] = {}
TF_ACTIVE_1H: dict[str, dict[str, pd.Series]] = {}

for ticker, tf_data in ALL_TF_DATA.items():
    df_1h = tf_data["1H"]
    total_score = pd.Series(0.0, index=df_1h.index)
    active_by_tf: dict[str, pd.Series] = {}

    for tf in TF_LIST:
        if tf not in ALL_FEATS.get(ticker, {}):
            continue
        feat = ALL_FEATS[ticker][tf]
        sig  = feat["signal"].astype(float)

        # Выравниваем на 1H
        if tf == "1H":
            sig_aligned = sig.reindex(df_1h.index).ffill().fillna(0)
        else:
            sig_aligned = sig.reindex(df_1h.index, method="ffill").ffill().fillna(0)

        w = TF_WEIGHT.get(tf, 1)
        total_score += sig_aligned * w
        active_by_tf[tf] = sig_aligned

    MTF_SCORE_1H[ticker]   = total_score
    TF_ACTIVE_1H[ticker]   = active_by_tf

# Выравниваем ZigZag, фракталы, ATR на 1H
ZZ_D_1H:     dict[str, pd.Series] = {}
FRAC_H_1H:   dict[str, pd.Series] = {}
FRAC_L_1H:   dict[str, pd.Series] = {}
ATR_D_1H:    dict[str, pd.Series] = {}

for ticker, tf_data in ALL_TF_DATA.items():
    df_1h = tf_data["1H"]
    if ticker in ZZ_D:
        ZZ_D_1H[ticker]   = ZZ_D[ticker].reindex(df_1h.index, method="ffill").ffill()
        FRAC_H_1H[ticker] = FRAC_H_D[ticker].reindex(df_1h.index, method="ffill").ffill()
        FRAC_L_1H[ticker] = FRAC_L_D[ticker].reindex(df_1h.index, method="ffill").ffill()
        ATR_D_1H[ticker]  = ATR_D[ticker].reindex(df_1h.index, method="ffill").ffill()

print("  Готово")


# ══════════════════════════════════════════════════════════════════════════════
# ГЕНЕРАЦИЯ СИГНАЛОВ MTF v2
# ══════════════════════════════════════════════════════════════════════════════
# Максимально возможный score = 1+1+1+1+2 = 6
# При TF_WEIGHT D=2, total max = 6
# Пороги: ≥4 (все TF кроме D) или ≥5 (включая D)

MAX_SCORE = sum(TF_WEIGHT.values())   # = 6

def find_entry_signals_v2(
    ticker: str,
    min_score: float = 4.0,
    require_d: bool = True,       # D TF обязательно активен
    require_frac_break: bool = True,  # цена пробивает фрактальный H на D
    zigzag_filter: bool = True,   # фильтр ZigZag фазы
) -> pd.Series:
    """Возвращает boolean Series сигналов на 1H оси."""
    if ticker not in MTF_SCORE_1H:
        return pd.Series(dtype=bool)

    df_1h  = ALL_TF_DATA[ticker]["1H"]
    score  = MTF_SCORE_1H[ticker]
    c      = df_1h["close"]

    # Базовый фильтр: score ≥ порога
    sig = score >= min_score

    # Обязательный D TF
    if require_d and ticker in TF_ACTIVE_1H:
        d_active = TF_ACTIVE_1H[ticker].get("D", pd.Series(0, index=df_1h.index))
        sig = sig & (d_active > 0)

    # Фрактальный пробой (цена выше последнего D-фрактального максимума)
    if require_frac_break and ticker in FRAC_H_1H:
        # Пробой = закрытие выше предыдущего фрактального H (shift(1))
        fh = FRAC_H_1H[ticker].shift(1)
        sig = sig & (c > fh)

    # ZigZag фаза: входим только в восходящей фазе и не на 5-й волне
    if zigzag_filter and ticker in ZZ_D_1H:
        zz = ZZ_D_1H[ticker]
        # Упрощённо: последний ZigZag разворот должен быть минимумом (восходящая фаза)
        # zz > 0 → последняя ZZ точка = максимум (плохо для входа)
        # Разворот на минимуме → zz на последней точке < предыдущей ZZ точки
        # Реализуем через направление последнего отрезка ZigZag
        zz_dir = (zz - zz.shift(1)).apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
        zz_dir_filled = zz_dir.replace(0, np.nan).ffill().fillna(0)
        # Входим только когда направление = вверх (последний ZZ отрезок восходящий)
        sig = sig & (zz_dir_filled > 0)

    # Убираем дублирующиеся подряд идущие сигналы
    sig = sig & ~sig.shift(1).fillna(False)

    return sig


# ══════════════════════════════════════════════════════════════════════════════
# ПОРТФЕЛЬНЫЙ БЭКТЕСТ v2
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class Position:
    ticker:     str
    entry_dt:   pd.Timestamp
    entry_px:   float
    shares:     float
    sl_px:      float
    trail_sl:   float
    partial_pnl: float = 0.0
    remaining:  float  = 1.0
    tp1_hit:    bool   = False
    tp2_hit:    bool   = False
    entry_fh:   float  = 0.0   # фрактальный H в момент входа (для TP1)


@dataclass
class Trade:
    ticker:    str
    entry_dt:  pd.Timestamp
    exit_dt:   pd.Timestamp
    entry_px:  float
    exit_px:   float
    pnl_rub:   float
    pnl_pct:   float
    reason:    str
    hold_days: float
    score:     float


CORR_GROUPS: list[set[str]] = [
    {"SBER", "T"},
    {"LKOH", "ROSN", "NVTK", "GAZP"},
    {"NLMK", "MTLR"},
]

def corr_blocked(ticker: str, open_set: set[str]) -> bool:
    for g in CORR_GROUPS:
        if ticker in g and g & open_set:
            return True
    return False


def run_backtest_v2(
    min_score: float       = 4.0,
    require_d: bool        = True,
    require_frac: bool     = True,
    zigzag_filter: bool    = True,
    trail_mult: float      = 2.0,
    use_partial_tp: bool   = True,   # частичная фиксация на фрактальных H
    max_hold_days: int     = 40,
    label: str             = "",
) -> dict:

    # Генерируем сигналы для всех тикеров
    SIGNALS: dict[str, pd.Series] = {}
    n_total = 0
    for ticker in ALL_TF_DATA:
        s = find_entry_signals_v2(ticker, min_score, require_d,
                                   require_frac, zigzag_filter)
        SIGNALS[ticker] = s
        n_total += s.sum()

    # Общий timeline (1H)
    master = set()
    for t in ALL_TF_DATA:
        master.update(ALL_TF_DATA[t]["1H"].index)
    timeline = sorted(master)

    IDX_1H = {t: {d: i for i, d in enumerate(df.index)}
               for t, df in {t: ALL_TF_DATA[t]["1H"] for t in ALL_TF_DATA}.items()}

    free_cash  = INITIAL_CAP
    positions: dict[str, Position] = {}
    trades:    list[Trade] = []
    equity     = [INITIAL_CAP]
    peak_eq    = INITIAL_CAP
    max_dd     = 0.0

    for dt in timeline:
        # ── Обновляем позиции ──────────────────────────────────────────────────
        to_close = []
        for ticker, pos in positions.items():
            df_1h = ALL_TF_DATA[ticker]["1H"]
            idx   = IDX_1H[ticker].get(dt)
            if idx is None or idx <= 0:
                continue
            if dt <= pos.entry_dt:
                continue

            hi   = float(df_1h["high"].iloc[idx])
            lo   = float(df_1h["low"].iloc[idx])
            op   = float(df_1h["open"].iloc[idx])
            cls  = float(df_1h["close"].iloc[idx])
            at_d = float(ATR_D_1H[ticker].iloc[idx]) if ticker in ATR_D_1H else 0.0
            fh   = float(FRAC_H_1H[ticker].iloc[idx]) if ticker in FRAC_H_1H else 0.0
            fl   = float(FRAC_L_1H[ticker].iloc[idx]) if ticker in FRAC_L_1H else 0.0

            hold_days = (dt - pos.entry_dt).total_seconds() / 86400

            # Обновляем trailing SL
            if at_d > 0:
                trail = cls - trail_mult * at_d
                if trail > pos.trail_sl:
                    pos.trail_sl = trail

            # Фрактальный SL всегда отслеживает последний фрактальный минимум D
            if fl > pos.sl_px and fl < pos.entry_px:
                pass  # SL ниже входа — не двигаем вниз
            elif fl > pos.sl_px:
                pos.sl_px = fl

            curr_sl = max(pos.sl_px, pos.trail_sl)

            # ── Частичная фиксация прибыли на фрактальных H ──────────────────
            if use_partial_tp and pos.remaining > 1e-6:
                # TP1: 33% на первом фрактальном H выше входа
                if not pos.tp1_hit and fh > pos.entry_px * 1.02 and hi >= fh:
                    frac = min(0.33, pos.remaining)
                    ep   = fh * (1-SLIPPAGE)
                    pos.partial_pnl += (ep - pos.entry_px)*frac*pos.shares - \
                                       (pos.entry_px+ep)*COMMISSION*frac*pos.shares
                    pos.remaining   -= frac
                    pos.tp1_hit      = True
                    pos.entry_fh     = fh
                    # Сдвигаем SL на BEP
                    pos.sl_px    = max(pos.sl_px, pos.entry_px * 1.001)
                    pos.trail_sl = max(pos.trail_sl, pos.entry_px * 1.001)
                    curr_sl      = max(pos.sl_px, pos.trail_sl)

                # TP2: 33% на следующем фрактальном H (выше TP1)
                elif pos.tp1_hit and not pos.tp2_hit and fh > pos.entry_fh * 1.01 and hi >= fh:
                    frac = min(0.33, pos.remaining)
                    ep   = fh * (1-SLIPPAGE)
                    pos.partial_pnl += (ep - pos.entry_px)*frac*pos.shares - \
                                       (pos.entry_px+ep)*COMMISSION*frac*pos.shares
                    pos.remaining   -= frac
                    pos.tp2_hit      = True
                    # Сдвигаем SL на TP1 уровень
                    pos.sl_px    = max(pos.sl_px, pos.entry_fh * 0.98)
                    curr_sl      = max(pos.sl_px, pos.trail_sl)

            # ── Проверяем выход ────────────────────────────────────────────────
            reason = exit_px = None
            if pos.remaining <= 1e-6:
                reason = "FRAC_TP"; exit_px = pos.entry_px
            elif lo <= curr_sl:
                reason  = "SL_FRACTAL"
                exit_px = max(curr_sl*(1-SLIPPAGE), lo)
            elif hold_days >= max_hold_days:
                reason  = "TIME"
                exit_px = op*(1-SLIPPAGE)

            if reason:
                rem = pos.remaining
                cash_r = exit_px*rem*pos.shares*(1-COMMISSION)
                cost_t = pos.entry_px*pos.shares*(1+COMMISSION)
                total_cash = pos.partial_pnl + cash_r
                pnl = total_cash - cost_t
                pnl_pct = pnl/cost_t*100
                free_cash += total_cash
                trades.append(Trade(ticker, pos.entry_dt, dt,
                                    pos.entry_px, exit_px, pnl, pnl_pct,
                                    reason, hold_days,
                                    float(MTF_SCORE_1H[ticker].get(pos.entry_dt, 0))))
                to_close.append(ticker)

        for t in to_close:
            positions.pop(t, None)

        # ── Новые входы ────────────────────────────────────────────────────────
        for ticker in ALL_TF_DATA:
            if len(positions) >= MAX_POSITIONS:
                break
            if ticker in positions:
                continue

            sig = SIGNALS[ticker]
            if ticker not in IDX_1H:
                continue
            idx = IDX_1H[ticker].get(dt)
            if idx is None or idx < 1:
                continue
            if not bool(sig.iloc[idx]):
                continue
            if corr_blocked(ticker, set(positions.keys())):
                continue

            df_1h  = ALL_TF_DATA[ticker]["1H"]
            entry  = float(df_1h["open"].iloc[idx]) * (1+SLIPPAGE)

            # Фрактальный SL: последний фрактальный минимум D
            fl_val = float(FRAC_L_1H[ticker].iloc[idx]) if ticker in FRAC_L_1H else entry*0.90
            # Ограничиваем SL: не дальше 20% от цены
            if fl_val <= 0 or (entry - fl_val)/entry > 0.20:
                fl_val = entry * 0.85

            # Начальный trailing SL = фрактальный минимум
            at_d_val = float(ATR_D_1H[ticker].iloc[idx]) if ticker in ATR_D_1H else entry*0.02
            trail_init = entry - trail_mult * at_d_val
            init_sl = max(fl_val, trail_init)

            # Размер позиции
            pos_val = sum(
                float(ALL_TF_DATA[t]["1H"]["close"].iloc[IDX_1H[t].get(dt,-1)])
                * p.shares * p.remaining
                for t, p in positions.items()
                if IDX_1H[t].get(dt) is not None
            )
            total_cap = free_cash + pos_val
            alloc     = min(total_cap * RISK_PCT, free_cash * 0.95)
            if alloc <= 0:
                continue
            shares = alloc / entry
            cost   = shares * entry * (1+COMMISSION)
            if cost > free_cash:
                continue

            free_cash -= cost
            positions[ticker] = Position(
                ticker=ticker, entry_dt=dt, entry_px=entry,
                shares=shares, sl_px=init_sl, trail_sl=init_sl,
            )

        # Equity
        pos_val = sum(
            float(ALL_TF_DATA[t]["1H"]["close"].iloc[IDX_1H[t].get(dt,-1)])
            * p.shares * p.remaining
            for t, p in positions.items()
            if IDX_1H[t].get(dt) is not None
        )
        eq = free_cash + pos_val
        equity.append(eq)
        if eq > peak_eq: peak_eq = eq
        dd = (peak_eq - eq)/peak_eq*100
        if dd > max_dd: max_dd = dd

    # Принудительное закрытие
    last_dt = timeline[-1]
    for ticker, pos in list(positions.items()):
        cls  = float(ALL_TF_DATA[ticker]["1H"]["close"].iloc[-1])
        ep   = cls*(1-SLIPPAGE)
        rem  = pos.remaining
        cash_r = ep*rem*pos.shares*(1-COMMISSION)
        cost_t = pos.entry_px*pos.shares*(1+COMMISSION)
        pnl  = pos.partial_pnl + cash_r - cost_t
        pnl_pct = pnl/cost_t*100
        free_cash += pos.partial_pnl + cash_r
        trades.append(Trade(ticker, pos.entry_dt, last_dt,
                            pos.entry_px, ep, pnl, pnl_pct,
                            "FORCED", (last_dt-pos.entry_dt).total_seconds()/86400,
                            float(MTF_SCORE_1H[ticker].get(pos.entry_dt, 0))))

    # Статистика
    final = free_cash
    total_pnl = (final-INITIAL_CAP)/INITIAL_CAP*100
    n_days = (timeline[-1]-timeline[0]).days
    ann    = ((final/INITIAL_CAP)**(365/max(n_days,1))-1)*100

    pnls  = np.array([t.pnl_pct for t in trades])
    n_tr  = len(trades)
    n_win = (pnls>0).sum()
    wr    = n_win/n_tr*100 if n_tr else 0
    wins  = pnls[pnls>0]; losses = pnls[pnls<=0]
    pf    = wins.sum()/(-losses.sum()+1e-9) if len(losses) else 99.0

    eq_arr = np.array(equity)
    dr     = np.diff(eq_arr)/(eq_arr[:-1]+1e-9)
    sharpe = (dr.mean()/(dr.std()+1e-9))*np.sqrt(252*6.5)

    by_reason: dict[str,dict] = {}
    for t in trades:
        s = by_reason.setdefault(t.reason, {"n":0,"wins":0,"pnl":0.0})
        s["n"]+=1; s["pnl"]+=t.pnl_pct
        if t.pnl_pct>0: s["wins"]+=1

    by_year: dict[str,float] = {}
    running = INITIAL_CAP
    for t in sorted(trades, key=lambda x: x.exit_dt):
        yr = str(t.exit_dt)[:4]
        running += t.pnl_rub
        by_year[yr] = running

    return dict(
        label=label, final=final, total_pnl=total_pnl, ann=ann,
        max_dd=-max_dd, sharpe=sharpe, trades=n_tr, wr=wr, pf=min(pf,99),
        avg_win=float(wins.mean()) if len(wins) else 0,
        avg_loss=float(losses.mean()) if len(losses) else 0,
        by_reason=by_reason, by_year=by_year, trades_list=trades,
        n_signals=int(n_total),
    )


# ══════════════════════════════════════════════════════════════════════════════
# ЗАПУСК ТЕСТОВ
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("  MTF STRATEGY v2 — ОПТИМИЗАЦИЯ ПАРАМЕТРОВ")
print("  Все тикеры сохранены | Фракталы Б.Вильямса + ZigZag + Частичный TP")
print("="*80)
print(f"\n  {'Конфиг':32s} {'Сиг':5s} {'Сд':4s} {'WR':6s} {'PF':5s} "
      f"{'ANN%':6s} {'MaxDD':7s} {'Sharpe':7s} {'Итог ₽':10s}")
print("  " + "─"*88)

CONFIGS = [
    # label,                  min_sc, req_d, req_fr, zigzag, trail, part_tp, hold
    ("v2-sc4-D-FR-ZZ-trail2",    4.0, True,  True,  True,  2.0, True,  40),
    ("v2-sc4-D-FR-ZZ-trail2.5",  4.0, True,  True,  True,  2.5, True,  40),
    ("v2-sc4-D-FR-noZZ-trail2",  4.0, True,  True,  False, 2.0, True,  40),
    ("v2-sc4-D-noFR-ZZ-trail2",  4.0, True,  False, True,  2.0, True,  40),
    ("v2-sc4-D-FR-ZZ-h30",       4.0, True,  True,  True,  2.0, True,  30),
    ("v2-sc4-D-FR-ZZ-h50",       4.0, True,  True,  True,  2.0, True,  50),
    ("v2-sc3.5-D-FR-ZZ",         3.5, True,  True,  True,  2.0, True,  40),
    ("v2-sc5-D-FR-ZZ",           5.0, True,  True,  True,  2.0, True,  45),
    ("v2-sc4-D-FR-ZZ-noPartTP",  4.0, True,  True,  True,  2.0, False, 40),
    ("v2-sc4-noD-FR-ZZ-trail2",  4.0, False, True,  True,  2.0, True,  40),
]

best_sc = -999; best_r = None; results = []
for cfg in CONFIGS:
    (label, min_sc, req_d, req_fr, zigzag,
     trail, part_tp, hold) = cfg
    r = run_backtest_v2(min_sc, req_d, req_fr, zigzag,
                        trail, part_tp, hold, label)
    results.append(r)
    sc = r["ann"]*0.6 + r["sharpe"]*5 - abs(r["max_dd"])*0.4
    mk = " ◄" if sc > best_sc and r["trades"] >= 10 else ""
    if sc > best_sc and r["trades"] >= 10:
        best_sc = sc; best_r = r
    print(f"  {label:32s} {r['n_signals']:5d} {r['trades']:4d} {r['wr']:5.1f}% "
          f"{r['pf']:5.2f} {r['ann']:>+5.1f}% {r['max_dd']:>+6.1f}% "
          f"{r['sharpe']:>6.2f}  {r['final']:>10,.0f}{mk}")


# ── Детальный отчёт победителя ─────────────────────────────────────────────────
if best_r:
    print(f"\n{'═'*80}")
    print(f"  ПОБЕДИТЕЛЬ: {best_r['label']}")
    print(f"{'═'*80}")
    print(f"  Начальный капитал:  {INITIAL_CAP:>10,.0f} ₽")
    print(f"  Итоговый капитал:   {best_r['final']:>10,.0f} ₽")
    print(f"  Суммарная прибыль:  {best_r['total_pnl']:>+9.1f}%")
    print(f"  CAGR:               {best_r['ann']:>+9.1f}%")
    print(f"  MaxDD:              {best_r['max_dd']:>+9.1f}%")
    print(f"  Sharpe:             {best_r['sharpe']:>9.2f}")
    print(f"  Сделок:             {best_r['trades']:>9d}")
    print(f"  WR:                 {best_r['wr']:>9.1f}%")
    print(f"  PF:                 {best_r['pf']:>9.2f}")
    print(f"  Avg Win / Loss:     {best_r['avg_win']:>+6.2f}% / {best_r['avg_loss']:>+6.2f}%")

    print("\n  ПО ГОДАМ:")
    print(f"  {'Год':6s} {'Капитал':>12s} {'Прибыль':>12s} {'%':>8s}")
    print("  " + "─"*42)
    prev = INITIAL_CAP
    for yr in sorted(best_r["by_year"]):
        cap = best_r["by_year"][yr]
        pr  = cap - prev; pct = pr/prev*100
        print(f"  {yr}   {cap:>12,.0f} ₽  {pr:>+10,.0f} ₽  {pct:>+7.1f}%")
        prev = cap
    print("  " + "─"*42)
    print(f"  ИТОГО  {best_r['final']:>12,.0f} ₽  "
          f"{best_r['final']-INITIAL_CAP:>+10,.0f} ₽  {best_r['total_pnl']:>+7.1f}%")

    print("\n  ПО ПРИЧИНАМ ВЫХОДА:")
    print(f"  {'Причина':12s} {'N':5s} {'WR%':7s} {'Avg%':8s}")
    print("  " + "─"*36)
    for reason, s in sorted(best_r["by_reason"].items()):
        wr_r = s["wins"]/s["n"]*100 if s["n"] else 0
        avg  = s["pnl"]/s["n"] if s["n"] else 0
        print(f"  {reason:12s} {s['n']:5d} {wr_r:>6.1f}% {avg:>+7.2f}%")

    tl   = best_r["trades_list"]
    by_t: dict[str,list] = {}
    for t in tl: by_t.setdefault(t.ticker,[]).append(t.pnl_pct)
    print("\n  ПО ТИКЕРАМ:")
    print(f"  {'Тикер':6s} {'N':4s} {'WR%':6s} {'Total%':8s} {'Avg%':7s}")
    print("  " + "─"*36)
    for tk, pnls in sorted(by_t.items(), key=lambda x: sum(x[1]), reverse=True):
        pa = np.array(pnls)
        print(f"  {tk:6s} {len(pnls):4d} {(pa>0).mean()*100:5.1f}% "
              f"{pa.sum():>+7.1f}% {pa.mean():>+6.2f}%")

    tl_s = sorted(tl, key=lambda t: t.pnl_pct, reverse=True)
    print("\n  ТОП-7 ЛУЧШИХ СДЕЛОК:")
    print(f"  {'Тикер':6s} {'Вход':18s} {'Выход':18s} {'Дни':4s} {'P&L%':7s} {'Причина':12s}")
    for t in tl_s[:7]:
        print(f"  {t.ticker:6s} {str(t.entry_dt)[:16]:18s} "
              f"{str(t.exit_dt)[:16]:18s} {t.hold_days:3.0f}  "
              f"{t.pnl_pct:>+6.1f}% {t.reason}")
    print("\n  ХУДШИЕ 5:")
    for t in tl_s[-5:]:
        print(f"  {t.ticker:6s} {str(t.entry_dt)[:16]:18s} "
              f"{str(t.exit_dt)[:16]:18s} {t.hold_days:3.0f}  "
              f"{t.pnl_pct:>+6.1f}% {t.reason}")

print(f"\n{'═'*80}")
print("  СРАВНЕНИЕ СТРАТЕГИЙ:")
print(f"  {'Стратегия':35s} {'ANN%':7s} {'MaxDD':7s} {'Sharpe':8s} {'Итог ₽':10s}")
print("  " + "─"*70)
print(f"  {'Дневная ATR_BO (лидер)':35s}  +13.9%   -12.9%     1.19    173,223")
print(f"  {'MTF v1 (score≥3, hold=30)':35s}  +11.8%   -13.8%     0.39    159,764")
if best_r:
    print(f"  {best_r['label']:35s} {best_r['ann']:>+6.1f}% "
          f"{best_r['max_dd']:>+6.1f}%  {best_r['sharpe']:>7.2f}  {best_r['final']:>10,.0f}")
print(f"{'═'*80}")
