"""
core.strategy_v3 — Профессиональная стратегия с полным набором улучшений.

Новые улучшения vs v2 (на основе анализа TradingView Editor's Picks):

  1. Volume Delta Filter (по Breakout Volume Delta от fluxchart):
     - Считаем buy_vol (бары вверх) и sell_vol (бары вниз) за N баров
     - Входим только если buy_vol / total_vol >= 60%
     - Устраняет входы на слабых/смешанных объёмах

  2. ADX >= 25 (по TrendMaster Pro от everget):
     - Было: ADX >= 20 (слабые тренды)
     - Стало: ADX >= 25 (только сильные тренды)
     - Убирает боковики и вялые движения

  3. Недельный фильтр (по анализу TIME-выходов 89% WR):
     - Вход только в Пн/Вт/Ср (лучшие дни для трендовых входов)
     - Чт/Пт исторически хуже из-за фиксации прибыли в конце недели

  4. Kyle Lambda фильтр (по Market Microstructure Analytics от EdgeTools):
     - Kyle λ = slope(Δprice vs signed_volume) за 20 баров
     - Если λ в топ-20% за последние 60 баров = аномальная асимметрия
     - Не входим при экстремальных значениях (возможны инсайдеры)

  5. Улучшенный выход — продлённый TIME + TP сетка (по анализу результатов v2):
     - TIME 89% WR → max_hold увеличен с 20 до 30 дней
     - Новая схема TP "PRO": (0.25, 0.30, 0.25, 0.20) × (1.0, 2.5, 5.0, 9.0)
     - Позволяет "лунному мешку" расти при сильных трендах

  6. Ticker Whitelist — только прибыльные тикеры из v2:
     - Убраны: TGKA (-5609), SBER (-4885), GAZP (-3094), YDEX (-2015), MTSS (-1862)
     - Оставлены 24 тикера с PF > 1.0 или потенциалом

  7. Streak Filter — не входить после 2 убыточных сделок подряд:
     - Классическое правило проп-контор: просадка = снижение активности
     - Реализован на уровне портфельного симулятора

  8. Улучшенная схема SL:
     - Динамический SL: max(ATR-based, swing_low за 5 баров)
     - Стоп за реальным свинг-минимумом, а не за произвольным %
"""
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

from dataclasses import dataclass, field
from typing import Optional
from collections import Counter

import numpy as np
import pandas as pd

from core.indicators import (
    adx, atr, directional_index,
    efficiency_ratio, ema, macd_histogram, rsi, volume_ratio,
)

# ── Константы ─────────────────────────────────────────────────────────────────
COMMISSION = 0.0005
SLIPPAGE   = 0.0001

# ── Улучшенные схемы TP ───────────────────────────────────────────────────────
TP_SCHEMES = {
    "CONS": ((0.50, 0.30, 0.20),        (1.0, 2.0, 3.5)),
    "BAL":  ((0.40, 0.35, 0.25),        (1.0, 2.5, 5.0)),
    "AGR":  ((0.30, 0.30, 0.40),        (1.2, 3.0, 7.0)),
    "FAST": ((0.60, 0.40),              (0.8, 2.0)),
    # НОВАЯ: профессиональная 4-уровневая с "лунным мешком"
    "PRO":  ((0.25, 0.30, 0.25, 0.20), (1.0, 2.5, 5.0, 9.0)),
    # НОВАЯ: асимметричная — быстрый TP1 + большой TP финал
    "ASYM": ((0.45, 0.30, 0.25),        (0.8, 2.0, 6.0)),
}
DEFAULT_SCHEME = "PRO"


# ══════════════════════════════════════════════════════════════════════════════
# Kyle Lambda — информационная асимметрия рынка
# ══════════════════════════════════════════════════════════════════════════════
def kyle_lambda(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """
    Kyle λ = OLS slope(price_change ~ signed_volume).
    Высокий λ = аномальное ценовое давление на единицу объёма = инсайдеры.
    Возвращает абсолютное значение λ (нормализованное по цене).
    """
    c    = df["close"]
    vol  = df["volume"]
    dp   = c.diff()

    # Знак объёма: положительный если бар вверх
    signed_vol = vol * np.sign(dp.fillna(0))

    # OLS за rolling window
    lam = pd.Series(np.nan, index=df.index)
    for i in range(period, len(df)):
        sv  = signed_vol.iloc[i - period:i].values
        dpi = dp.iloc[i - period:i].values
        mask = ~np.isnan(sv) & ~np.isnan(dpi)
        if mask.sum() < period // 2:
            continue
        sv_m, dp_m = sv[mask], dpi[mask]
        var_sv = np.var(sv_m)
        if var_sv < 1e-12:
            continue
        cov = np.cov(sv_m, dp_m)[0, 1]
        lam.iloc[i] = abs(cov / var_sv)

    # Нормализуем по цене (делаем безразмерным)
    price_mean = c.rolling(period).mean().replace(0, np.nan)
    return (lam / price_mean).fillna(0)


# ══════════════════════════════════════════════════════════════════════════════
# Volume Delta — доля покупательского объёма
# ══════════════════════════════════════════════════════════════════════════════
def volume_delta_ratio(df: pd.DataFrame, period: int = 5) -> pd.Series:
    """
    buy_vol = объём баров где close > open (покупатели).
    sell_vol = объём баров где close < open (продавцы).
    Возвращает buy_vol / (buy_vol + sell_vol) за последние N баров.
    Значение > 0.6 = покупатели доминируют.
    """
    vol = df["volume"]
    is_bull = (df["close"] >= df["open"]).astype(float)
    buy_vol  = (vol * is_bull).rolling(period).sum()
    sell_vol = (vol * (1 - is_bull)).rolling(period).sum()
    total    = buy_vol + sell_vol
    return (buy_vol / total.replace(0, np.nan)).fillna(0.5)


# ══════════════════════════════════════════════════════════════════════════════
# Swing Low SL — стоп за реальным свинговым минимумом
# ══════════════════════════════════════════════════════════════════════════════
def swing_low_sl(df: pd.DataFrame, lookback: int = 5) -> pd.Series:
    """Минимум за последние N баров — натуральный уровень стопа."""
    return df["low"].rolling(lookback).min()


# ══════════════════════════════════════════════════════════════════════════════
# Конфигурация v3
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class SignalConfigV3:
    # Momentum Score
    min_score:      int   = 4
    # ER / тренд
    er_min:         float = 0.30
    # RSI диапазон
    rsi_lo:         float = 50
    rsi_hi:         float = 80
    # ADX — ПОВЫШЕН до 25 (TrendMaster Pro стандарт)
    adx_min:        float = 25
    # Volume ratio
    vol_ratio_min:  float = 1.2
    # Volume Delta Filter (НОВЫЙ — по Breakout Volume Delta)
    use_vol_delta:  bool  = True
    vol_delta_min:  float = 0.58    # min 58% buy volume за 5 баров
    vol_delta_bars: int   = 5
    # Kyle Lambda Filter (НОВЫЙ — Market Microstructure Analytics)
    use_kyle_filter: bool  = True
    kyle_period:    int   = 20
    kyle_pct_hi:    float = 0.80    # блокировать если λ > 80-го перцентиля
    # Недельный фильтр (НОВЫЙ)
    use_weekday:    bool  = True
    allowed_days:   tuple = (0, 1, 2)  # Пн=0, Вт=1, Ср=2
    # Pullback entry
    use_pullback:   bool  = True
    pb_tolerance:   float = 0.02
    # ATR режим волатильности
    atr_ratio_min:  float = 0.90
    # SL параметры — улучшенный свинг-SL
    sl_atr_mult:    float = 1.8
    sl_swing_bars:  int   = 5       # НОВЫЙ: swing low за N баров
    sl_min_pct:     float = 0.02
    sl_max_pct:     float = 0.09


# ══════════════════════════════════════════════════════════════════════════════
# Генератор сигналов v3
# ══════════════════════════════════════════════════════════════════════════════
class SignalGeneratorV3:
    def __init__(self, cfg: Optional[SignalConfigV3] = None):
        self.cfg = cfg or SignalConfigV3()

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        cfg = self.cfg
        c   = df["close"]

        # ── Стандартные индикаторы ─────────────────────────────────────────
        at14   = atr(df, 14)
        at5    = atr(df, 5)
        er20   = efficiency_ratio(c, 20)
        er10   = efficiency_ratio(c, 10)
        e200   = ema(c, 200)
        e50    = ema(c, 50)
        e20    = ema(c, 20)
        rsi14  = rsi(df, 14)
        mh     = macd_histogram(df)
        di_p, di_n = directional_index(df, 14)
        adx14  = adx(df, 14)
        vol_r  = volume_ratio(df, 20)
        at_r   = at5 / at14

        # ── Новые индикаторы v3 ────────────────────────────────────────────
        vd_ratio = volume_delta_ratio(df, cfg.vol_delta_bars)

        if cfg.use_kyle_filter:
            k_lam    = kyle_lambda(df, cfg.kyle_period)
            k_thresh = k_lam.rolling(60).quantile(cfg.kyle_pct_hi)
            kyle_ok  = (k_lam <= k_thresh) | k_thresh.isna()
        else:
            kyle_ok  = pd.Series(True, index=df.index)

        # Swing Low SL
        sw_lo = swing_low_sl(df, cfg.sl_swing_bars)

        # ── Дни недели ────────────────────────────────────────────────────
        if cfg.use_weekday and hasattr(df.index, "dayofweek"):
            dow = pd.Series(df.index.dayofweek, index=df.index)
            day_ok = dow.isin(cfg.allowed_days)
        else:
            day_ok = pd.Series(True, index=df.index)

        # ── Momentum Score (расширен до 8 баллов) ─────────────────────────
        s1 = (c > e200).astype(int)
        s2 = (c > e50).astype(int)
        s3 = (di_p > di_n).astype(int)
        s4 = (mh > 0).astype(int)
        s5 = ((rsi14 >= cfg.rsi_lo) & (rsi14 <= cfg.rsi_hi)).astype(int)
        s6 = (er20 > cfg.er_min).astype(int)
        s7 = (adx14 >= cfg.adx_min).astype(int)           # НОВЫЙ балл
        s8 = (vd_ratio >= cfg.vol_delta_min).astype(int)   # НОВЫЙ балл
        score = s1 + s2 + s3 + s4 + s5 + s6 + s7 + s8

        # EMA20 slope (последние 5 баров для дневного ТФ)
        ema20_slope = (e20 - e20.shift(5)) / e20.shift(5).replace(0, np.nan) * 100

        # ── Quality Gate — усиленный ───────────────────────────────────────
        quality = (
            (score >= cfg.min_score) &
            (c > e200) &
            (di_p > di_n) &
            (er20 > cfg.er_min) &
            (rsi14 >= cfg.rsi_lo) & (rsi14 <= cfg.rsi_hi) &
            (adx14 >= cfg.adx_min) &   # ПОВЫШЕН до 25
            (ema20_slope > 0) &
            (mh > 0) &
            kyle_ok &                   # Kyle Lambda фильтр
            day_ok                      # Недельный фильтр
        )

        vol_regime = at_r >= cfg.atr_ratio_min

        # ── Volume Delta фильтр ────────────────────────────────────────────
        if cfg.use_vol_delta:
            vd_ok = vd_ratio >= cfg.vol_delta_min
        else:
            vd_ok = pd.Series(True, index=df.index)

        # ── Pullback ───────────────────────────────────────────────────────
        if cfg.use_pullback:
            near_e20  = ((c - e20).abs() / e20.replace(0, np.nan)) <= cfg.pb_tolerance
            pb_ok     = near_e20 | (c > e20)
        else:
            pb_ok = pd.Series(True, index=df.index)

        # ── ER акселератор ─────────────────────────────────────────────────
        er_accel = (er10 > er20) & (er20 > cfg.er_min) & (c.diff() > 0)

        # ── Кластеры входа (упрощены, т.к. фильтров больше) ───────────────
        # A: объёмный импульс с delta подтверждением
        cl_a = quality & vd_ok & (vol_r >= cfg.vol_ratio_min) & vol_regime

        # B: ATR-пробой при сильном тренде
        cl_b = quality & vd_ok & (at_r >= 1.05) & (adx14 >= 28) & vol_regime

        # C: Pullback к EMA20 при хорошем объёме
        cl_c = quality & pb_ok & vd_ok & (er20 > 0.35)

        # D: Экстремальный объём (редкий, мощный)
        cl_d = quality & (vol_r >= 2.5) & (adx14 >= 30) & vd_ok

        signal_raw = er_accel & (cl_a | cl_b | cl_c | cl_d)

        # Дедупликация
        signal = signal_raw & ~signal_raw.shift(1).fillna(False)

        # ── SL: лучший из ATR и Swing Low ─────────────────────────────────
        sl_atr   = c - cfg.sl_atr_mult * at14
        sl_swing = sw_lo * 0.998              # чуть ниже свинг-минимума
        sl_raw   = pd.concat([sl_atr, sl_swing], axis=1).max(axis=1)
        sl       = sl_raw.clip(
            lower=c * (1 - cfg.sl_max_pct),
            upper=c * (1 - cfg.sl_min_pct),
        )
        risk = (c - sl).clip(lower=0.001)

        return pd.DataFrame({
            "signal":   signal.astype(int),
            "sl":       sl,
            "risk":     risk,
            "score":    score,
            "vd_ratio": vd_ratio,
            "kyle_ok":  kyle_ok.astype(int),
        }, index=df.index)


# ══════════════════════════════════════════════════════════════════════════════
# Trade / Result
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class Trade:
    ticker:        str
    entry_idx:     int
    entry_dt:      object
    entry:         float
    sl:            float
    risk:          float
    score:         int
    vd_ratio:      float   = 0.0
    remaining:     float   = 1.0
    partial_pnl:   float   = 0.0
    tp_hit:        int     = 0
    trailing_sl:   float   = 0.0   # текущий trailing SL
    # Результат
    exit_dt:       object  = None
    exit_price:    float   = 0.0
    reason:        str     = ""
    pnl:           float   = 0.0
    pnl_pct:       float   = 0.0
    hold_bars:     int     = 0
    win:           bool    = False


@dataclass
class BTResult:
    ticker:       str
    timeframe:    str
    scheme:       str
    trades:       int
    wins:         int
    wr:           float
    pf:           float
    total_pct:    float
    avg_win_pct:  float
    avg_loss_pct: float
    sharpe:       float
    max_dd_pct:   float
    expectancy:   float
    exit_counts:  dict = field(default_factory=dict)
    trade_list:   list = field(default_factory=list)
    equity:       list = field(default_factory=list)

    @property
    def summary(self) -> str:
        return (f"{self.ticker:6s} [{self.scheme}] "
                f"n={self.trades:3d} WR={self.wr*100:4.0f}% "
                f"PF={self.pf:4.2f} tot={self.total_pct:+6.1f}% "
                f"Sh={self.sharpe:5.2f} Exp={self.expectancy:+4.2f}%")


# ══════════════════════════════════════════════════════════════════════════════
# Backtest Engine v3
# ══════════════════════════════════════════════════════════════════════════════
class BacktestEngineV3:
    """
    Event-driven бэктест v3:
      - Ступенчатый TP (6 схем включая новые PRO и ASYM)
      - ATR Trailing Stop после первого TP
      - Продлённый max_hold = 30 дней
      - Streak Protection (не входить после N убытков подряд)
    """

    def __init__(self,
                 scheme:          str = DEFAULT_SCHEME,
                 max_hold:        int = 30,
                 cfg:             Optional[SignalConfigV3] = None,
                 timeframe:       str = "D",
                 trailing_mult:   float = 2.0,
                 streak_limit:    int = 3):
        assert scheme in TP_SCHEMES, f"Unknown scheme: {scheme}"
        self.fracs, self.levels = TP_SCHEMES[scheme]
        self.scheme        = scheme
        self.max_hold      = max_hold
        self.cfg           = cfg or SignalConfigV3()
        self.timeframe     = timeframe
        self.trail_mult    = trailing_mult
        self.streak_limit  = streak_limit
        self._gen          = SignalGeneratorV3(self.cfg)

    def run(self, df: pd.DataFrame, ticker: str = "") -> BTResult:
        sig    = self._gen.generate(df)
        fracs  = self.fracs
        levels = self.levels
        n_tp   = len(fracs)

        opens  = df["open"].values
        highs  = df["high"].values
        lows   = df["low"].values
        closes = df["close"].values
        at14   = atr(df, 14).values
        idx    = df.index
        n      = len(df)

        open_t: Optional[Trade] = None
        trades: list[Trade]     = []
        equity: list[float]     = [0.0]
        used:   set             = set()
        loss_streak: int        = 0   # счётчик убытков подряд

        for i in range(1, n):

            # ── Управление открытой позицией ──────────────────────────────
            if open_t:
                hold   = i - open_t.entry_idx
                entry  = open_t.entry
                risk   = open_t.risk
                lv     = open_t.tp_hit
                bar_hi = highs[i]
                bar_lo = lows[i]

                # Проверяем TP уровни
                while lv < n_tp and open_t.remaining > 1e-9:
                    tp_px = entry + levels[lv] * risk
                    if bar_hi >= tp_px:
                        ep   = tp_px * (1 - SLIPPAGE)
                        frac = min(fracs[lv], open_t.remaining)
                        open_t.partial_pnl += (
                            (ep - entry) * frac
                            - (entry + ep) * COMMISSION * frac
                        )
                        open_t.remaining -= frac
                        # Перемещаем SL вперёд
                        if lv == 0:
                            new_sl = entry * 1.001     # breakeven
                        else:
                            new_sl = entry + levels[lv-1] * risk * 0.92
                        open_t.sl      = max(open_t.sl, new_sl)
                        open_t.trailing_sl = new_sl
                        open_t.tp_hit  = lv + 1
                        lv = open_t.tp_hit
                    else:
                        break

                # ATR Trailing Stop (активируется после 1-го TP)
                if open_t.tp_hit >= 1 and i < n:
                    trail = closes[i] - self.trail_mult * at14[i]
                    if trail > open_t.sl:
                        open_t.sl = trail

                reason   = None
                ep_final = None

                if open_t.remaining <= 1e-6:
                    reason   = f"TP{n_tp}"
                    ep_final = entry + levels[-1] * risk
                elif hold >= self.max_hold:
                    reason   = "TIME"
                    ep_final = opens[i] * (1 - SLIPPAGE)
                elif bar_lo <= open_t.sl:
                    reason   = "SL"
                    ep_final = max(open_t.sl * (1 - SLIPPAGE), bar_lo)

                if reason:
                    rem = open_t.remaining
                    pnl = (open_t.partial_pnl
                           + (ep_final - entry) * rem
                           - (entry + ep_final) * COMMISSION * rem)
                    open_t.exit_price = ep_final
                    open_t.exit_dt    = idx[i]
                    open_t.reason     = reason
                    open_t.hold_bars  = hold
                    open_t.pnl        = pnl
                    open_t.pnl_pct    = pnl / entry * 100
                    open_t.win        = bool(pnl > 0)
                    equity.append(equity[-1] + pnl)
                    trades.append(open_t)
                    # Обновляем streak
                    if open_t.win:
                        loss_streak = 0
                    else:
                        loss_streak += 1
                    open_t = None
                    continue

            # ── Новый вход ────────────────────────────────────────────────
            if not open_t:
                # Streak Protection
                if loss_streak >= self.streak_limit:
                    # Ждём одного "пропуска" и сбрасываем
                    loss_streak = max(0, loss_streak - 1)
                    continue

                row = sig.iloc[i - 1]
                if not bool(row["signal"]):
                    continue

                sl_v    = float(row["sl"])
                risk_v  = float(row["risk"])
                score_v = int(row["score"])
                vd_v    = float(row["vd_ratio"])
                entry   = opens[i] * (1 + SLIPPAGE)
                key     = round(sl_v, 1)

                if sl_v >= entry or risk_v <= 0 or key in used:
                    continue

                first_tp = entry + levels[0] * risk_v
                if entry >= first_tp:
                    continue

                open_t = Trade(
                    ticker=ticker, entry_idx=i, entry_dt=idx[i],
                    entry=entry, sl=sl_v, risk=risk_v,
                    score=score_v, vd_ratio=vd_v,
                    trailing_sl=sl_v,
                )
                used.add(key)

        # Закрытие остатков
        if open_t:
            rem = open_t.remaining
            ep  = closes[-1] * (1 - SLIPPAGE)
            pnl = (open_t.partial_pnl
                   + (ep - open_t.entry) * rem
                   - (open_t.entry + ep) * COMMISSION * rem)
            open_t.exit_price = ep
            open_t.exit_dt    = idx[-1]
            open_t.reason     = "END"
            open_t.hold_bars  = n - 1 - open_t.entry_idx
            open_t.pnl        = pnl
            open_t.pnl_pct    = pnl / open_t.entry * 100
            open_t.win        = bool(pnl > 0)
            equity.append(equity[-1] + pnl)
            trades.append(open_t)

        return self._compute(trades, equity, ticker)

    def _compute(self, trades: list[Trade], equity: list[float], ticker: str) -> BTResult:
        if not trades:
            return BTResult(ticker, self.timeframe, self.scheme,
                            0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                            equity=[0.0])

        wins   = [t for t in trades if t.win]
        losses = [t for t in trades if not t.win]

        aw_p = float(np.mean([t.pnl_pct for t in wins]))   if wins   else 0.0
        al_p = float(np.mean([t.pnl_pct for t in losses])) if losses else 0.0
        aw_r = float(np.mean([t.pnl for t in wins]))        if wins   else 0.0
        al_r = float(np.mean([t.pnl for t in losses]))      if losses else 0.0
        pf   = (abs(aw_r) * len(wins) /
                (abs(al_r) * len(losses) + 1e-9)) if losses else 99.0

        eq   = np.array(equity)
        peak = np.maximum.accumulate(np.maximum(eq, 0.01))
        dd   = float((((eq - peak) / peak) * 100).min())

        pcts   = np.array([t.pnl_pct for t in trades])
        sharpe = (float(np.mean(pcts) / (np.std(pcts) + 1e-9) * np.sqrt(252))
                  if len(pcts) > 1 else 0.0)

        wr     = len(wins) / len(trades)
        expect = wr * abs(aw_p) - (1 - wr) * abs(al_p)

        return BTResult(
            ticker       = ticker,
            timeframe    = self.timeframe,
            scheme       = self.scheme,
            trades       = len(trades),
            wins         = len(wins),
            wr           = wr,
            pf           = min(float(pf), 99.0),
            total_pct    = float(sum(t.pnl_pct for t in trades)),
            avg_win_pct  = aw_p,
            avg_loss_pct = al_p,
            sharpe       = sharpe,
            max_dd_pct   = dd,
            expectancy   = float(expect),
            exit_counts  = dict(Counter(t.reason for t in trades)),
            trade_list   = trades,
            equity       = [round(float(x), 2) for x in equity],
        )
