"""
=============================================================================
STRATEGIES LAB — Исследовательская лаборатория торговых стратегий
=============================================================================

Тестируем 8 стратегий на 27 тикерах MOEX (дневные данные 2022-2026):

ПРОБИВНЫЕ (Breakout):
  1. Turtle20     — классические черепахи: пробой 20-дн. максимума
  2. Turtle55     — черепахи: пробой 55-дн. максимума
  3. ATR_BO       — ATR-пробой: бар больше N*ATR от EMA
  4. High52W      — 52-недельный максимум + объём (академически проверен)

ТРЕНДОВЫЕ (Trend Following):
  5. TripleEMA    — пересечение EMA(9,21,55) с подтверждением ADX
  6. Supertrend   — Supertrend(10,3) + EMA200 фильтр
  7. ChandelierExit — Chandelier Exit + MACD confirmaton

КОМБИНИРОВАННАЯ:
  8. APEX_v4      — лучшее из всех: Volume + Momentum + Pullback + Breakout

Каждая стратегия:
  - Только ЛОНГ (как требует пользователь)
  - Стоп-лосс по ATR или свинг-минимуму
  - Ступенчатый TP (AGR схема: 1.2R / 3.0R / 7.0R)
  - Комиссия 0.05%, слиппаж 0.01%
  - Период удержания max 20 дней
=============================================================================
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
from dataclasses import dataclass, field
from collections import Counter
from typing import Optional
import numpy as np
import pandas as pd

# ── Константы ─────────────────────────────────────────────────────────────────
COMMISSION  = 0.0005
SLIPPAGE    = 0.0002
MAX_HOLD    = 20
DATA_DIR    = Path("c:/investor/data")
FRACS       = (0.30, 0.30, 0.40)
LEVELS      = (1.2, 3.0, 7.0)

TICKERS = [
    "GAZP","LKOH","NVTK","ROSN","SNGS","SNGSP",
    "SBER","SBERP","T","VTBR",
    "GMKN","NLMK","MTLR","CHMF","MAGN","RUAL","ALRS","PLZL",
    "YDEX","OZON","MGNT",
    "TATN","TATNP",
    "AFLT","IRAO","PHOR","OZPH",
]


# ══════════════════════════════════════════════════════════════════════════════
# Загрузчик данных
# ══════════════════════════════════════════════════════════════════════════════
def load_daily(ticker: str) -> Optional[pd.DataFrame]:
    path = DATA_DIR / f"{ticker}_2022_2026_D.csv"
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path, sep=";")
        df.columns = [c.strip("<>").lower() for c in df.columns]
        df["dt"] = pd.to_datetime(df["date"].astype(str), format="%d/%m/%y", errors="coerce")
        df = df.dropna(subset=["dt"]).set_index("dt").sort_index()
        for col in ("open","high","low","close","vol"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col].astype(str).str.replace(",","."), errors="coerce")
        df = df.rename(columns={"vol":"volume"})
        df = df[["open","high","low","close","volume"]].dropna()
        return df if len(df) >= 100 else None
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# Индикаторы (векторизованные, быстрые)
# ══════════════════════════════════════════════════════════════════════════════
def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()

def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    h, l, pc = df["high"], df["low"], df["close"].shift(1)
    tr = pd.concat([h-l, (h-pc).abs(), (l-pc).abs()], axis=1).max(axis=1)
    return tr.ewm(span=n, adjust=False).mean()

def rsi(s: pd.Series, n: int = 14) -> pd.Series:
    d = s.diff()
    g = d.clip(lower=0).ewm(span=n, adjust=False).mean()
    lo = (-d.clip(upper=0)).ewm(span=n, adjust=False).mean()
    return 100 - 100 / (1 + g / lo.replace(0, np.nan))

def adx(df: pd.DataFrame, n: int = 14) -> pd.Series:
    h, l, pc = df["high"], df["low"], df["close"].shift(1)
    tr  = pd.concat([h-l,(h-pc).abs(),(l-pc).abs()],axis=1).max(axis=1)
    dmp = (h - h.shift(1)).clip(lower=0)
    dmn = (l.shift(1) - l).clip(lower=0)
    dmp = dmp.where(dmp > dmn, 0)
    dmn = dmn.where(dmn > dmp.shift(1).fillna(0), 0)  # упрощение
    atr_s = tr.ewm(span=n, adjust=False).mean()
    dip   = 100 * dmp.ewm(span=n, adjust=False).mean() / atr_s.replace(0, np.nan)
    din   = 100 * dmn.ewm(span=n, adjust=False).mean() / atr_s.replace(0, np.nan)
    dx    = 100 * (dip - din).abs() / (dip + din).replace(0, np.nan)
    return dx.ewm(span=n, adjust=False).mean()

def macd_hist(df: pd.DataFrame) -> pd.Series:
    c   = df["close"]
    fast = ema(c, 12); slow = ema(c, 26); sig = ema(fast - slow, 9)
    return (fast - slow) - sig

def obv(df: pd.DataFrame) -> pd.Series:
    """On-Balance Volume"""
    sign = np.sign(df["close"].diff().fillna(0))
    return (df["volume"] * sign).cumsum()

def heikin_ashi(df: pd.DataFrame) -> pd.DataFrame:
    ha = pd.DataFrame(index=df.index)
    ha["ha_close"] = (df["open"]+df["high"]+df["low"]+df["close"]) / 4
    ha_open = [(df["open"].iloc[0]+df["close"].iloc[0])/2]
    for i in range(1, len(df)):
        ha_open.append((ha_open[i-1] + ha["ha_close"].iloc[i-1]) / 2)
    ha["ha_open"]  = ha_open
    ha["ha_high"]  = pd.concat([df["high"], ha["ha_open"], ha["ha_close"]], axis=1).max(axis=1)
    ha["ha_low"]   = pd.concat([df["low"],  ha["ha_open"], ha["ha_close"]], axis=1).min(axis=1)
    return ha

def supertrend(df: pd.DataFrame, period: int = 10, mult: float = 3.0) -> pd.Series:
    at = atr(df, period)
    hl2 = (df["high"] + df["low"]) / 2
    upper = hl2 + mult * at
    lower = hl2 - mult * at
    trend = pd.Series(1, index=df.index)  # 1=uptrend, -1=downtrend
    for i in range(1, len(df)):
        if df["close"].iloc[i] > upper.iloc[i-1]:
            trend.iloc[i] = 1
        elif df["close"].iloc[i] < lower.iloc[i-1]:
            trend.iloc[i] = -1
        else:
            trend.iloc[i] = trend.iloc[i-1]
    return trend  # 1=bullish, -1=bearish

def chandelier_exit(df: pd.DataFrame, period: int = 22, mult: float = 3.0) -> pd.Series:
    """Long Chandelier Exit: highest_high(period) - mult*ATR"""
    at = atr(df, period)
    hh = df["high"].rolling(period).max()
    return hh - mult * at  # Stop level for long positions

def volume_ratio(df: pd.DataFrame, n: int = 20) -> pd.Series:
    return df["volume"] / df["volume"].rolling(n).mean()

def er(close: pd.Series, n: int = 20) -> pd.Series:
    """Kaufman Efficiency Ratio"""
    direction = (close - close.shift(n)).abs()
    volatility = (close.diff().abs()).rolling(n).sum()
    return direction / volatility.replace(0, np.nan)


# ══════════════════════════════════════════════════════════════════════════════
# Универсальный бэктест движок
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class Trade:
    entry: float
    sl: float
    risk: float
    entry_i: int
    remaining: float = 1.0
    partial_pnl: float = 0.0
    tp_hit: int = 0

@dataclass
class Result:
    strategy: str
    ticker: str
    trades: int
    wins: int
    wr: float
    pf: float
    total_pct: float
    sharpe: float
    max_dd: float
    avg_win: float
    avg_loss: float
    exit_dist: dict = field(default_factory=dict)
    equity: list = field(default_factory=list)

def run_backtest(signals: pd.Series, sl_series: pd.Series, risk_series: pd.Series,
                 df: pd.DataFrame, strategy_name: str, ticker: str,
                 fracs=FRACS, levels=LEVELS, max_hold=MAX_HOLD) -> Result:
    """
    Универсальный движок бэктеста.
    signals: pd.Series[bool] — сигнал покупки
    sl_series: уровень стоп-лосса
    risk_series: риск в рублях на акцию
    """
    opens  = df["open"].values
    highs  = df["high"].values
    lows   = df["low"].values
    closes = df["close"].values
    n      = len(df)
    n_tp   = len(fracs)

    open_t: Optional[Trade] = None
    pnl_list = []
    equity   = [0.0]
    reasons  = []

    for i in range(1, n):
        if open_t:
            hold  = i - open_t.entry_i
            lv    = open_t.tp_hit
            entry = open_t.entry
            risk  = open_t.risk

            # Stepped TP
            while lv < n_tp and open_t.remaining > 1e-9:
                tp_px = entry + levels[lv] * risk
                if highs[i] >= tp_px:
                    ep   = tp_px * (1 - SLIPPAGE)
                    frac = min(fracs[lv], open_t.remaining)
                    open_t.partial_pnl += (ep - entry)*frac - (entry+ep)*COMMISSION*frac
                    open_t.remaining -= frac
                    open_t.sl = max(open_t.sl,
                                    entry*1.001 if lv == 0 else entry + levels[lv-1]*risk*0.90)
                    open_t.tp_hit = lv + 1
                    lv = open_t.tp_hit
                else:
                    break

            reason = ep_f = None
            if open_t.remaining <= 1e-6:
                reason = f"TP{n_tp}"; ep_f = closes[i]
            elif hold >= max_hold:
                reason = "TIME"; ep_f = opens[i] * (1-SLIPPAGE)
            elif lows[i] <= open_t.sl:
                reason = "SL"; ep_f = max(open_t.sl*(1-SLIPPAGE), lows[i])

            if reason:
                rem = open_t.remaining
                pnl = (open_t.partial_pnl + (ep_f - entry)*rem
                       - (entry+ep_f)*COMMISSION*rem)
                pnl_pct = pnl / entry * 100
                pnl_list.append(pnl_pct)
                equity.append(equity[-1] + pnl_pct)
                reasons.append(reason)
                open_t = None

        if not open_t and bool(signals.iloc[i-1]):
            sl_v   = float(sl_series.iloc[i-1])
            risk_v = float(risk_series.iloc[i-1])
            if pd.isna(sl_v) or pd.isna(risk_v) or risk_v <= 0:
                continue
            entry = opens[i] * (1+SLIPPAGE)
            if sl_v >= entry:
                continue
            open_t = Trade(entry=entry, sl=sl_v, risk=risk_v, entry_i=i)

    # Закрыть остаток
    if open_t:
        ep = closes[-1] * (1-SLIPPAGE)
        rem = open_t.remaining
        pnl_pct = ((open_t.partial_pnl + (ep-open_t.entry)*rem
                    - (open_t.entry+ep)*COMMISSION*rem) / open_t.entry * 100)
        pnl_list.append(pnl_pct)
        equity.append(equity[-1] + pnl_pct)
        reasons.append("END")

    if not pnl_list:
        return Result(strategy_name, ticker, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    pnl_arr = np.array(pnl_list)
    wins    = pnl_arr[pnl_arr > 0]
    losses  = pnl_arr[pnl_arr <= 0]
    wr      = len(wins) / len(pnl_arr)
    pf      = (wins.sum() / (-losses.sum() + 1e-9)) if len(losses) else 99.0
    sharpe  = (pnl_arr.mean() / (pnl_arr.std()+1e-9) * np.sqrt(252)
               if len(pnl_arr) > 1 else 0.0)
    eq      = np.array(equity)
    pk      = np.maximum.accumulate(np.maximum(eq, 0.01))
    max_dd  = float(((eq - pk)/pk * 100).min())

    return Result(
        strategy=strategy_name, ticker=ticker,
        trades=len(pnl_arr), wins=len(wins),
        wr=float(wr), pf=float(min(pf, 99)),
        total_pct=float(pnl_arr.sum()),
        sharpe=float(sharpe), max_dd=float(max_dd),
        avg_win=float(wins.mean()) if len(wins) else 0.0,
        avg_loss=float(losses.mean()) if len(losses) else 0.0,
        exit_dist=dict(Counter(reasons)),
        equity=[round(float(x),3) for x in equity],
    )


# ══════════════════════════════════════════════════════════════════════════════
# 1. TURTLE 20 — Пробой 20-дн. максимума (классика Ричарда Денниса)
# ══════════════════════════════════════════════════════════════════════════════
def strategy_turtle20(df: pd.DataFrame, ticker: str) -> Result:
    """
    Вход:  close > rolling_max(high, 20) предыдущего дня
    Выход: high < rolling_min(low, 10) ИЛИ TIME ИЛИ SL(2×ATR)
    Фильтр: EMA200 (только в восходящем тренде)
    """
    c   = df["close"]
    h20 = df["high"].rolling(20).max().shift(1)
    e200= ema(c, 200)
    at14= atr(df, 14)
    vol_r = volume_ratio(df, 20)

    signal = (
        (c > h20) &               # пробой 20-дн. максимума
        (c > e200) &               # выше EMA200 (тренд)
        (vol_r >= 1.3)             # объём подтверждает
    )
    # Дедупликация
    signal = signal & ~signal.shift(1).fillna(False)

    sl = (c - 2.0*at14).clip(lower=c*0.88)
    risk = (c - sl).clip(lower=0.001)

    return run_backtest(signal, sl, risk, df, "Turtle20", ticker)


# ══════════════════════════════════════════════════════════════════════════════
# 2. TURTLE 55 — Пробой 55-дн. максимума (долгосрочный)
# ══════════════════════════════════════════════════════════════════════════════
def strategy_turtle55(df: pd.DataFrame, ticker: str) -> Result:
    """
    Более редкий, но более надёжный сигнал — 55-дн. максимум.
    Академически показывает лучший Sharpe на трендующих рынках.
    """
    c   = df["close"]
    h55 = df["high"].rolling(55).max().shift(1)
    e200= ema(c, 200)
    at14= atr(df, 14)
    vol_r = volume_ratio(df, 20)
    adx14 = adx(df, 14)

    signal = (
        (c > h55) &
        (c > e200) &
        (adx14 >= 20) &
        (vol_r >= 1.2)
    )
    signal = signal & ~signal.shift(1).fillna(False)

    sl = (c - 2.5*at14).clip(lower=c*0.85)
    risk = (c - sl).clip(lower=0.001)

    return run_backtest(signal, sl, risk, df, "Turtle55", ticker,
                        max_hold=30)  # дольше держим для 55-дн. пробоя


# ══════════════════════════════════════════════════════════════════════════════
# 3. ATR BREAKOUT — пробой на расширении волатильности
# ══════════════════════════════════════════════════════════════════════════════
def strategy_atr_breakout(df: pd.DataFrame, ticker: str) -> Result:
    """
    Вход: бар движется вверх больше чем 1.5×ATR(14) от EMA20
    + RSI растёт + ADX > 25
    Идея: поймать момент ускорения тренда
    """
    c    = df["close"]
    at14 = atr(df, 14)
    at5  = atr(df, 5)
    e20  = ema(c, 20)
    e200 = ema(c, 200)
    rsi14= rsi(c, 14)
    adx14= adx(df, 14)
    vol_r= volume_ratio(df, 20)

    # Прирост за бар > 1.5×ATR (сильное движение)
    bar_move = (c - c.shift(1)).clip(lower=0)

    signal = (
        (c > e200) &
        (bar_move >= 1.5 * at14) &    # мощное движение вверх
        (at5 > at14 * 0.95) &          # волатильность расширяется
        (rsi14 >= 52) & (rsi14 <= 82) &
        (adx14 >= 22) &
        (vol_r >= 1.5)                  # объём подтверждает
    )
    signal = signal & ~signal.shift(1).fillna(False)

    sl = (c - 1.8*at14).clip(lower=c*0.90)
    risk = (c - sl).clip(lower=0.001)

    return run_backtest(signal, sl, risk, df, "ATR_BO", ticker)


# ══════════════════════════════════════════════════════════════════════════════
# 4. 52-WEEK HIGH — академически подтверждённый эффект
# ══════════════════════════════════════════════════════════════════════════════
def strategy_52w_high(df: pd.DataFrame, ticker: str) -> Result:
    """
    Основан на исследовании George/Hwang (2004) и George (2011):
    Акции вблизи 52-нед. максимума систематически недооцениваются
    из-за поведенческих предубеждений инвесторов → momentum.

    Вход: close в топ-5% от 52-нед. диапазона + объём > 1.5× среднего
    Фильтр: тренд (EMA50 > EMA200) + ADX > 20
    """
    c    = df["close"]
    h52  = df["high"].rolling(252).max()    # 252 торг. дня ≈ 1 год
    l52  = df["low"].rolling(252).min()
    e50  = ema(c, 50)
    e200 = ema(c, 200)
    at14 = atr(df, 14)
    vol_r= volume_ratio(df, 20)
    adx14= adx(df, 14)

    # Позиция цены в 52-нед. диапазоне (1.0 = на максимуме)
    range52 = (h52 - l52).replace(0, np.nan)
    pos52   = (c - l52) / range52

    signal = (
        (pos52 >= 0.92) &              # в топ-8% от 52-нед. диапазона
        (c > e50) & (e50 > e200) &     # EMA50 > EMA200 (долгосрочный тренд)
        (adx14 >= 20) &
        (vol_r >= 1.5)                  # объём подтверждает пробой
    )
    signal = signal & ~signal.shift(1).fillna(False)

    sl = (c - 2.0*at14).clip(lower=c*0.88)
    risk = (c - sl).clip(lower=0.001)

    return run_backtest(signal, sl, risk, df, "52W_High", ticker,
                        max_hold=25)


# ══════════════════════════════════════════════════════════════════════════════
# 5. TRIPLE EMA — пересечение 3 EMA с подтверждением
# ══════════════════════════════════════════════════════════════════════════════
def strategy_triple_ema(df: pd.DataFrame, ticker: str) -> Result:
    """
    Классика: EMA(9) пересекает EMA(21) при условии EMA(55) восходящий.
    Дополнено: MACD > 0, ADX > 20, объём.
    Широко используется в алго-трейдинге, Sharpe 1.5-3.5 на дневных данных.
    """
    c    = df["close"]
    e9   = ema(c, 9)
    e21  = ema(c, 21)
    e55  = ema(c, 55)
    e200 = ema(c, 200)
    adx14= adx(df, 14)
    mh   = macd_hist(df)
    at14 = atr(df, 14)
    vol_r= volume_ratio(df, 20)

    # Пересечение EMA9 выше EMA21
    cross_up = (e9 > e21) & (e9.shift(1) <= e21.shift(1))

    signal = (
        cross_up &
        (c > e55) &          # выше среднесрочного тренда
        (e55 > e200) &        # долгосрочный тренд вверх
        (mh > 0) &            # MACD подтверждает
        (adx14 >= 20) &
        (vol_r >= 1.1)
    )

    sl = (c - 1.8*at14).clip(lower=c*0.90)
    risk = (c - sl).clip(lower=0.001)

    return run_backtest(signal, sl, risk, df, "TripleEMA", ticker)


# ══════════════════════════════════════════════════════════════════════════════
# 6. SUPERTREND — Supertrend(10,3) + EMA200
# ══════════════════════════════════════════════════════════════════════════════
def strategy_supertrend_strat(df: pd.DataFrame, ticker: str) -> Result:
    """
    Supertrend меняется с -1 (медвежий) на +1 (бычий) = сигнал входа.
    Фильтр: EMA200 (только когда цена выше), ADX > 20.
    По тестам: WR 40-43%, но Avg Win >> Avg Loss (Profit Factor > 1.5).
    """
    c    = df["close"]
    st   = supertrend(df, 10, 3.0)
    e200 = ema(c, 200)
    adx14= adx(df, 14)
    at14 = atr(df, 14)
    vol_r= volume_ratio(df, 20)
    mh   = macd_hist(df)

    # Смена тренда с медвежьего на бычий
    trend_flip = (st == 1) & (st.shift(1) == -1)

    signal = (
        trend_flip &
        (c > e200) &
        (adx14 >= 18) &
        (mh > 0) &
        (vol_r >= 1.0)
    )

    sl = (c - 2.5*at14).clip(lower=c*0.87)
    risk = (c - sl).clip(lower=0.001)

    return run_backtest(signal, sl, risk, df, "Supertrend", ticker,
                        max_hold=25)


# ══════════════════════════════════════════════════════════════════════════════
# 7. CHANDELIER EXIT + OBV — выход по чандельеру с объёмным подтверждением
# ══════════════════════════════════════════════════════════════════════════════
def strategy_chandelier_obv(df: pd.DataFrame, ticker: str) -> Result:
    """
    Chandelier Exit (22, 3): стоп ставится ниже 22-дн. максимума − 3×ATR.
    Сигнал: цена выше CE + OBV растёт + EMA9 > EMA21 (тренд).
    Преимущество: CE следует за трендом как trailing stop → большие выигрыши.
    """
    c    = df["close"]
    ce   = chandelier_exit(df, 22, 3.0)
    obv_s= obv(df)
    obv_e= ema(obv_s, 21)  # EMA OBV
    e9   = ema(c, 9)
    e21  = ema(c, 21)
    e200 = ema(c, 200)
    at14 = atr(df, 14)
    rsi14= rsi(c, 14)
    vol_r= volume_ratio(df, 20)

    # OBV выше своей EMA = покупатели доминируют
    obv_bull = obv_s > obv_e

    # Цена выше Chandelier Exit = тренд жив
    above_ce = c > ce

    # Пересечение EMA (сигнал входа)
    cross_up = (e9 > e21) & (e9.shift(1) <= e21.shift(1))

    signal = (
        cross_up &
        above_ce &
        obv_bull &
        (c > e200) &
        (rsi14 >= 50) & (rsi14 <= 80) &
        (vol_r >= 1.1)
    )

    sl = ce.clip(lower=c*0.88)  # SL = Chandelier Exit уровень
    sl = sl.clip(upper=c*0.95)  # но не ближе 5% к цене
    risk = (c - sl).clip(lower=0.001)

    return run_backtest(signal, sl, risk, df, "Chan+OBV", ticker,
                        max_hold=25)


# ══════════════════════════════════════════════════════════════════════════════
# 8. HEIKIN ASHI TREND — фильтрация шума через HA свечи
# ══════════════════════════════════════════════════════════════════════════════
def strategy_heikin_ashi(df: pd.DataFrame, ticker: str) -> Result:
    """
    Heikin Ashi + подтверждение:
    Сигнал: HA-бар полностью бычий (ha_open == ha_low, нет нижней тени)
    + EMA55 восходящая + MACD > 0 + ADX > 20

    HA хорошо работает на дневных данных для фильтрации ложных сигналов.
    """
    ha   = heikin_ashi(df)
    c    = df["close"]
    e55  = ema(c, 55)
    e200 = ema(c, 200)
    adx14= adx(df, 14)
    mh   = macd_hist(df)
    at14 = atr(df, 14)
    vol_r= volume_ratio(df, 20)

    # Полностью бычий HA бар: low == open (нет нижней тени)
    ha_bull    = (ha["ha_close"] > ha["ha_open"])
    ha_no_tail = ((ha["ha_open"] - ha["ha_low"]).abs() < 0.001 * df["close"])

    # 3 бычьих HA бара подряд = сильный тренд
    ha_streak = ha_bull & ha_bull.shift(1).fillna(False) & ha_bull.shift(2).fillna(False)

    # Первый бар серии
    signal_raw = ha_streak & ~ha_streak.shift(1).fillna(False)

    signal = (
        signal_raw &
        (c > e55) &
        (e55 > e200) &
        (mh > 0) &
        (adx14 >= 20) &
        (vol_r >= 1.1)
    )

    sl = (c - 2.0*at14).clip(lower=c*0.89)
    risk = (c - sl).clip(lower=0.001)

    return run_backtest(signal, sl, risk, df, "HeikinAshi", ticker)


# ══════════════════════════════════════════════════════════════════════════════
# 9. APEX v4 — Комбинированная стратегия (лучшее из всех)
# ══════════════════════════════════════════════════════════════════════════════
def strategy_apex_v4(df: pd.DataFrame, ticker: str) -> Result:
    """
    APEX v4 — синтез лучших элементов:

    Фильтр тренда (как в TripleEMA):
      - EMA9 > EMA21 > EMA55 > EMA200 (все выровнены)
      - Supertrend = бычий

    Фильтр импульса (как в ATR_BO + 52W_High):
      - ADX >= 22 (сильный тренд)
      - MACD гистограмма > 0 и растёт
      - ER(20) > 0.35 (направленное движение)
      - RSI(14) в зоне 52-80

    Объёмное подтверждение (как в Chan+OBV):
      - OBV > EMA(OBV, 21)
      - Volume ratio >= 1.3

    Пробой (как в Turtle):
      - Цена вблизи 20-дн. максимума (top 5%)

    Пуллбек к EMA (как в v2/v3):
      - Цена откатила к EMA21 (±2%) и отскочила

    Вход по любому из двух триггеров:
      - A: пробой 20-дн. максимума + все фильтры
      - B: пуллбек к EMA21 + отскок + все фильтры

    SL: max(ATR-based, swing_low_5d) — умный стоп
    """
    c    = df["close"]
    h, l = df["high"], df["low"]

    # EMA стек
    e9   = ema(c, 9)
    e21  = ema(c, 21)
    e55  = ema(c, 55)
    e200 = ema(c, 200)

    # Индикаторы
    at14 = atr(df, 14)
    at5  = atr(df, 5)
    adx14= adx(df, 14)
    rsi14= rsi(c, 14)
    mh   = macd_hist(df)
    mh_rising = mh > mh.shift(1)
    vol_r= volume_ratio(df, 20)
    obv_s= obv(df)
    obv_e= ema(obv_s, 21)
    er20 = er(c, 20)
    st   = supertrend(df, 10, 3.0)

    # 20-дн. максимум
    h20  = h.rolling(20).max().shift(1)
    # 5-дн. свинг-минимум
    sw5  = l.rolling(5).min()

    # ── Базовый фильтр качества ────────────────────────────────────────────
    quality = (
        (c > e200) &
        (e9 > e21) &
        (e21 > e55) &
        (adx14 >= 22) &
        (rsi14 >= 52) & (rsi14 <= 82) &
        (mh > 0) & mh_rising &
        (er20 >= 0.30) &
        (obv_s > obv_e) &
        (vol_r >= 1.2) &
        (st == 1)
    )

    # ── Триггер A: пробой 20-дн. максимума ────────────────────────────────
    trigger_a = quality & (c > h20) & (vol_r >= 1.5)

    # ── Триггер B: пуллбек к EMA21 + отскок ───────────────────────────────
    near_e21  = ((c - e21).abs() / e21.replace(0, np.nan)) <= 0.02
    above_e21 = c > e21
    # Вчера был у EMA21, сегодня выше
    pb_entry  = near_e21.shift(1).fillna(False) & above_e21
    trigger_b = quality & pb_entry & (c > e55)

    signal_raw = trigger_a | trigger_b
    signal     = signal_raw & ~signal_raw.shift(1).fillna(False)

    # ── SL: лучший из ATR и Swing Low ─────────────────────────────────────
    sl_atr   = c - 1.8 * at14
    sl_swing = sw5 * 0.998
    sl       = pd.concat([sl_atr, sl_swing], axis=1).max(axis=1)
    sl       = sl.clip(lower=c*0.88, upper=c*0.975)
    risk     = (c - sl).clip(lower=0.001)

    return run_backtest(signal, sl, risk, df, "APEX_v4", ticker,
                        max_hold=20)


# ══════════════════════════════════════════════════════════════════════════════
# Запуск всех стратегий на всех тикерах
# ══════════════════════════════════════════════════════════════════════════════
STRATEGIES = {
    "Turtle20":   strategy_turtle20,
    "Turtle55":   strategy_turtle55,
    "ATR_BO":     strategy_atr_breakout,
    "52W_High":   strategy_52w_high,
    "TripleEMA":  strategy_triple_ema,
    "Supertrend": strategy_supertrend_strat,
    "Chan+OBV":   strategy_chandelier_obv,
    "HeikinAshi": strategy_heikin_ashi,
    "APEX_v4":    strategy_apex_v4,
}


def run_all(verbose: bool = True) -> dict[str, list[Result]]:
    results: dict[str, list[Result]] = {s: [] for s in STRATEGIES}

    print("Загрузка данных...")
    data = {}
    for t in TICKERS:
        df = load_daily(t)
        if df is not None:
            data[t] = df
    print(f"  Загружено: {len(data)} тикеров\n")

    for strat_name, strat_fn in STRATEGIES.items():
        if verbose:
            print(f"  Тестирую: {strat_name}...", end=" ")
        for ticker, df in data.items():
            try:
                r = strat_fn(df, ticker)
                results[strat_name].append(r)
            except Exception as e:
                if verbose:
                    print(f"\n    ОШИБКА {ticker}: {e}")
        if verbose:
            total = sum(r.trades for r in results[strat_name])
            wins  = sum(r.wins  for r in results[strat_name])
            pcts  = [r.total_pct for r in results[strat_name] if r.trades > 0]
            wr    = wins/total*100 if total else 0
            avg   = float(np.mean(pcts)) if pcts else 0
            print(f"сделок={total}  WR={wr:.1f}%  avg_total={avg:+.1f}%")

    return results


def aggregate(results: dict[str, list[Result]]) -> pd.DataFrame:
    """Сводная таблица по стратегиям."""
    rows = []
    for name, rlist in results.items():
        active = [r for r in rlist if r.trades > 0]
        if not active:
            continue
        total_tr = sum(r.trades for r in active)
        total_w  = sum(r.wins for r in active)
        all_pnl  = [p for r in active for p in r.equity[1:]]  # все P&L%
        total_pct= sum(r.total_pct for r in active)
        avg_pct  = total_pct / len(active)
        wr       = total_w/total_tr if total_tr else 0
        sharpes  = [r.sharpe for r in active if r.trades >= 3]
        avg_sh   = float(np.mean(sharpes)) if sharpes else 0

        # Суммарный Profit Factor
        wins_sum = sum(r.total_pct for r in active if r.total_pct > 0)
        loss_sum = abs(sum(r.total_pct for r in active if r.total_pct < 0))
        pf       = wins_sum / (loss_sum + 1e-9)

        # Средний MaxDD
        avg_dd = float(np.mean([r.max_dd for r in active]))

        rows.append({
            "Стратегия":  name,
            "Тикеров":    len(active),
            "Сделок":     total_tr,
            "WR%":        round(wr*100, 1),
            "PF":         round(min(pf, 99), 2),
            "Total%":     round(total_pct, 1),
            "Avg_per_tk": round(avg_pct, 1),
            "Avg_Sharpe": round(avg_sh, 2),
            "Avg_MaxDD%": round(avg_dd, 1),
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("Avg_Sharpe", ascending=False)
    return df


if __name__ == "__main__":
    print("=" * 65)
    print("  STRATEGIES LAB — тест 9 стратегий на MOEX")
    print("=" * 65)
    results = run_all(verbose=True)

    print("\n" + "=" * 65)
    print("  СВОДНАЯ ТАБЛИЦА (сортировка по Avg Sharpe)")
    print("=" * 65)
    summary = aggregate(results)
    print(summary.to_string(index=False))

    # Детали по лучшей стратегии
    if not summary.empty:
        best_name = summary.iloc[0]["Стратегия"]
        print(f"\n\n  Топ тикеров для '{best_name}':")
        best_results = sorted(results[best_name],
                               key=lambda r: r.sharpe, reverse=True)
        print(f"  {'Тикер':8s} {'Сделок':7s} {'WR%':7s} {'PF':6s} "
              f"{'Total%':8s} {'Sharpe':8s} {'MaxDD%':7s}")
        print("  " + "-" * 60)
        for r in best_results[:15]:
            if r.trades > 0:
                print(f"  {r.ticker:8s} {r.trades:7d} {r.wr*100:6.1f}% "
                      f"{r.pf:6.2f} {r.total_pct:+7.1f}% "
                      f"{r.sharpe:7.2f} {r.max_dd:6.1f}%")

    summary.to_csv("c:/investor/strategies_comparison.csv", index=False)
    print("\n  Таблица сохранена → strategies_comparison.csv")
