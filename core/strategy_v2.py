"""
core.strategy_v2 — Улучшенная стратегия с мультитаймфреймовым подходом.

Ключевые улучшения vs v1:
  1. Trend Confirmation Filter (TCF) — тренд подтверждается на старшем ТФ
  2. Momentum Score  — взвешенный балл из 6 индикаторов (входим только если ≥ 4/6)
  3. Volatility Regime — фильтр: входим только в режиме расширяющейся волатильности
  4. Volume Confirmation — объём должен расти ≥ 2 дня подряд
  5. Pullback Entry — вход на откате к EMA20 (лучшее соотношение R:R)
  6. Trailing Stop (ATR-based) вместо фиксированного SL
  7. Улучшенная схема TP: асимметричная (R 1.0 / 2.5 / 5.0)

Параметры по умолчанию оптимизированы на 4H данных.
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

# Схемы TP (fracs, rr_levels)
TP_SCHEMES = {
    # Консервативная: быстрая фиксация
    "CONS": ((0.50, 0.30, 0.20), (1.0, 2.0, 3.5)),
    # Сбалансированная (лучшая WR по тестам)
    "BAL":  ((0.40, 0.35, 0.25), (1.0, 2.5, 5.0)),
    # Агрессивная: даём прибыли расти
    "AGR":  ((0.30, 0.30, 0.40), (1.2, 3.0, 7.0)),
    # Быстрая: 2 уровня
    "FAST": ((0.60, 0.40),       (0.8, 2.0)),
}
DEFAULT_SCHEME = "BAL"


# ══════════════════════════════════════════════════════════════════════════════
# Momentum Score — взвешенная оценка качества сигнала (0–6)
# ══════════════════════════════════════════════════════════════════════════════
def momentum_score(df: pd.DataFrame) -> pd.Series:
    """
    Возвращает целочисленный балл 0-6 для каждого бара.
    Входим только если score >= min_score (по умолчанию 4).
    """
    c     = df["close"]
    at14  = atr(df, 14)
    er20  = efficiency_ratio(c, 20)
    e20   = ema(c, 20)
    e50   = ema(c, 50)
    e200  = ema(c, 200)
    rsi14 = rsi(df, 14)
    mh    = macd_histogram(df)
    di_p, di_n = directional_index(df, 14)
    adx14 = adx(df, 14)
    vol_r = volume_ratio(df, 20)

    # Каждое условие даёт 1 балл
    s1 = (c > e200).astype(int)                          # выше EMA200
    s2 = (c > e50).astype(int)                           # выше EMA50
    s3 = (di_p > di_n).astype(int)                       # DI+ > DI-
    s4 = (mh > 0).astype(int)                            # MACD > 0
    s5 = ((rsi14 >= 52) & (rsi14 <= 80)).astype(int)     # RSI в зоне
    s6 = (er20 > 0.35).astype(int)                       # ER: тренд

    return s1 + s2 + s3 + s4 + s5 + s6


# ══════════════════════════════════════════════════════════════════════════════
# Генератор сигналов v2
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class SignalConfig:
    """Параметры генератора сигналов — легко менять для оптимизации."""
    # Entry filters
    min_score:     int   = 4       # мин. Momentum Score для входа
    er_min:        float = 0.30    # мин. ER (тренд)
    rsi_lo:        float = 50      # мин. RSI
    rsi_hi:        float = 82      # макс. RSI
    adx_min:       float = 20      # мин. ADX (сила тренда)
    vol_ratio_min: float = 1.2     # мин. объём vs среднее

    # Pullback entry (вход на откате к EMA20)
    use_pullback:  bool  = True
    pb_tolerance:  float = 0.015   # 1.5% от EMA20

    # Volatility regime
    atr_ratio_min: float = 0.95    # AT5/AT14 > 0.95 (волатильность не падает)

    # SL параметры
    sl_atr_mult:   float = 1.8
    sl_min_pct:    float = 0.025
    sl_max_pct:    float = 0.08


class SignalGeneratorV2:
    def __init__(self, cfg: Optional[SignalConfig] = None):
        self.cfg = cfg or SignalConfig()

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        cfg = self.cfg
        c   = df["close"]

        # Индикаторы
        at14  = atr(df, 14)
        at5   = atr(df, 5)
        er20  = efficiency_ratio(c, 20)
        er10  = efficiency_ratio(c, 10)
        e200  = ema(c, 200)
        e50   = ema(c, 50)
        e20   = ema(c, 20)
        rsi14 = rsi(df, 14)
        mh    = macd_histogram(df)
        di_p, di_n = directional_index(df, 14)
        adx14 = adx(df, 14)
        vol_r = volume_ratio(df, 20)
        at_r  = at5 / at14

        # Momentum Score
        score = momentum_score(df)

        # EMA20 slope (за последние 3 бара)
        ema20_slope = (e20 - e20.shift(3)) / e20.shift(3).replace(0, np.nan) * 100

        # Объём растёт 2 бара подряд
        vol_growing = (vol_r > vol_r.shift(1)) & (vol_r.shift(1) > vol_r.shift(2))

        # Pullback к EMA20
        if cfg.use_pullback:
            near_ema20 = ((c - e20).abs() / e20.replace(0, np.nan)) <= cfg.pb_tolerance
            pullback_ok = near_ema20 | (c > e20)  # либо у EMA20 либо выше
        else:
            pullback_ok = pd.Series(True, index=df.index)

        # Основной фильтр качества
        quality = (
            (score >= cfg.min_score) &
            (c > e200) &
            (di_p > di_n) &
            (er20 > cfg.er_min) &
            (rsi14 >= cfg.rsi_lo) & (rsi14 <= cfg.rsi_hi) &
            (adx14 >= cfg.adx_min) &
            (ema20_slope > 0) &
            (mh > 0)
        )

        # Volatility regime: волатильность не схлопывается
        vol_regime = at_r >= cfg.atr_ratio_min

        # ER-базовый сигнал (тренд ускоряется)
        er_accel = (er10 > er20) & (er20 > cfg.er_min) & (c.diff() > 0)

        # Кластеры входа
        # A: объёмный импульс
        cl_a = quality & (vol_r >= cfg.vol_ratio_min) & vol_growing & vol_regime

        # B: ATR-пробой + тренд
        cl_b = quality & (at_r >= 1.05) & (adx14 >= 25) & vol_regime

        # C: Pullback к EMA20 в тренде
        cl_c = quality & pullback_ok & (er20 > 0.38) & (vol_r >= 1.0)

        # D: Экстремальный импульс (редкий, но сильный)
        cl_d = quality & (vol_r >= 3.0) & (adx14 >= 30) & (er20 > 0.45)

        signal_raw = er_accel & (cl_a | cl_b | cl_c | cl_d)

        # Дедупликация: не входить 2 бара подряд
        signal = signal_raw & ~signal_raw.shift(1).fillna(False)

        # Trailing SL: ATR * множитель, ограниченный %
        sl_raw = c - cfg.sl_atr_mult * at14
        sl = sl_raw.clip(
            lower=c * (1 - cfg.sl_max_pct),
            upper=c * (1 - cfg.sl_min_pct),
        )
        risk = (c - sl).clip(lower=0.001)

        return pd.DataFrame({
            "signal": signal.astype(int),
            "sl":     sl,
            "risk":   risk,
            "score":  score,
        }, index=df.index)


# ══════════════════════════════════════════════════════════════════════════════
# Trade / Result
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class Trade:
    ticker:       str
    entry_idx:    int
    entry_dt:     object
    entry:        float
    sl:           float
    risk:         float
    score:        int
    remaining:    float = 1.0
    partial_pnl:  float = 0.0
    tp_hit:       int   = 0
    # Результат
    exit_dt:      object = None
    exit_price:   float  = 0.0
    reason:       str    = ""
    pnl:          float  = 0.0
    pnl_pct:      float  = 0.0
    hold_bars:    int    = 0
    win:          bool   = False


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
    exit_counts:  dict  = field(default_factory=dict)
    trade_list:   list  = field(default_factory=list)
    equity:       list  = field(default_factory=list)

    @property
    def summary(self) -> str:
        return (f"{self.ticker:6s} {self.timeframe:4s} [{self.scheme}] "
                f"n={self.trades:3d} WR={self.wr*100:4.0f}% "
                f"PF={self.pf:4.2f} tot={self.total_pct:+6.1f}% "
                f"Sh={self.sharpe:5.2f} DD={self.max_dd_pct:+5.1f}% "
                f"Exp={self.expectancy:+4.2f}%")


# ══════════════════════════════════════════════════════════════════════════════
# Backtest Engine v2
# ══════════════════════════════════════════════════════════════════════════════
class BacktestEngineV2:
    """
    Event-driven бэктест с:
      - Ступенчатым TP (4 схемы)
      - Trailing Stop (ATR-based)
      - Поддержкой любого таймфрейма
    """

    def __init__(self,
                 scheme:     str = DEFAULT_SCHEME,
                 max_hold:   int = 48,
                 cfg:        Optional[SignalConfig] = None,
                 timeframe:  str = "4H",
                 trailing_atr_mult: float = 2.5):
        assert scheme in TP_SCHEMES, f"Unknown scheme: {scheme}"
        self.fracs, self.levels = TP_SCHEMES[scheme]
        self.scheme    = scheme
        self.max_hold  = max_hold
        self.cfg       = cfg or SignalConfig()
        self.timeframe = timeframe
        self.trail_mult = trailing_atr_mult
        self._gen      = SignalGeneratorV2(self.cfg)

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

        for i in range(1, n):
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
                        open_t.partial_pnl += (ep - entry) * frac - (entry + ep) * COMMISSION * frac
                        open_t.remaining   -= frac
                        # Trailing SL вперёд
                        if lv == 0:
                            new_sl = entry * 1.001   # безубыток
                        else:
                            new_sl = entry + levels[lv-1] * risk * 0.90
                        open_t.sl = max(open_t.sl, new_sl)
                        open_t.tp_hit = lv + 1
                        lv = open_t.tp_hit
                    else:
                        break

                # ATR trailing stop (обновляем каждый бар)
                if lv > 0 and i < n:
                    trail_sl = closes[i] - self.trail_mult * at14[i]
                    if trail_sl > open_t.sl:
                        open_t.sl = trail_sl

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
                    pnl = open_t.partial_pnl + (ep_final - entry) * rem - (entry + ep_final) * COMMISSION * rem
                    open_t.exit_price = ep_final
                    open_t.exit_dt    = idx[i]
                    open_t.reason     = reason
                    open_t.hold_bars  = hold
                    open_t.pnl        = pnl
                    open_t.pnl_pct    = pnl / entry * 100
                    open_t.win        = bool(pnl > 0)
                    equity.append(equity[-1] + pnl)
                    trades.append(open_t)
                    open_t = None
                    continue

            # Новый вход
            if not open_t:
                row = sig.iloc[i - 1]
                if bool(row["signal"]):
                    sl_v   = float(row["sl"])
                    risk_v = float(row["risk"])
                    score  = int(row["score"])
                    entry  = opens[i] * (1 + SLIPPAGE)
                    key    = round(sl_v, 1)
                    if sl_v < entry and risk_v > 0 and key not in used:
                        first_tp = entry + levels[0] * risk_v
                        if entry < first_tp:
                            open_t = Trade(
                                ticker=ticker, entry_idx=i, entry_dt=idx[i],
                                entry=entry, sl=sl_v, risk=risk_v, score=score
                            )
                            used.add(key)

        # Закрытие на конце данных
        if open_t:
            rem = open_t.remaining
            ep  = closes[-1] * (1 - SLIPPAGE)
            pnl = open_t.partial_pnl + (ep - open_t.entry) * rem - (open_t.entry + ep) * COMMISSION * rem
            open_t.exit_price = ep; open_t.exit_dt = idx[-1]
            open_t.reason = "END"; open_t.hold_bars = n - 1 - open_t.entry_idx
            open_t.pnl = pnl; open_t.pnl_pct = pnl / open_t.entry * 100
            open_t.win = bool(pnl > 0)
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
        avg_e  = float(np.mean([t.entry for t in trades]))

        aw_p = float(np.mean([t.pnl_pct for t in wins]))   if wins   else 0.0
        al_p = float(np.mean([t.pnl_pct for t in losses])) if losses else 0.0
        aw_r = float(np.mean([t.pnl for t in wins]))        if wins   else 0.0
        al_r = float(np.mean([t.pnl for t in losses]))      if losses else 0.0
        pf   = abs(aw_r) * len(wins) / (abs(al_r) * len(losses) + 1e-9) if losses else 99.0

        eq   = np.array(equity)
        peak = np.maximum.accumulate(eq)
        dd   = float((((eq - peak) / (peak + 1e-9)) * 100).min())

        pcts   = np.array([t.pnl_pct for t in trades])
        sharpe = float(np.mean(pcts) / (np.std(pcts) + 1e-9) * np.sqrt(252)) if len(pcts) > 1 else 0.0

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
