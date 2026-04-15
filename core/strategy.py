"""
core.strategy — Momentum Filter Strategy с ступенчатым TP.

Архитектура:
  1. SignalGenerator  — генерирует сигналы на основе 3 кластеров
  2. SteppedTPScheme  — параметры схемы частичного закрытия позиции
  3. BacktestEngine   — симулирует исполнение сделок по OHLCV барам
  4. BacktestResult   — результаты с полной статистикой

Кластеры входа (data-driven, найдены на 1128 сделках):
  A. Послеполуденный объёмный всплеск: vol>2.5x AND час 13-22 AND ER>0.43
  B. ATR-импульс начала/конца недели:  ATR5/ATR14>1.08 AND (Пт|Пн) AND ER>0.45
  C. Экстремальный объём + тренд:      vol>3.0x AND ADX>28 AND ER>0.48

Базовые фильтры (всегда обязательны):
  above EMA200 | DI+>DI- | MACD>0 | ER>0.40 | RSI 58-88 | EMA20 slope>0
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import NamedTuple, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from core.indicators import (
    adx, atr, directional_index,
    efficiency_ratio, ema, macd_histogram, rsi, volume_ratio,
)

# ── Constants ─────────────────────────────────────────────────────────────────
COMMISSION = 0.0005   # 0.05% per side
SLIPPAGE   = 0.0001   # 0.01% market impact


# ══════════════════════════════════════════════════════════════════════════════
# Stepped-TP scheme
# ══════════════════════════════════════════════════════════════════════════════
class SteppedTPScheme(NamedTuple):
    """
    Defines how to exit a position in multiple steps.

    fracs:     fraction of position to close at each TP level (must sum to 1.0)
    rr_levels: R-multiple for each TP level (e.g. 0.8 means entry + 0.8×risk)
    label:     human-readable name
    """
    fracs:     tuple
    rr_levels: tuple
    label:     str

    def validate(self):
        assert len(self.fracs) == len(self.rr_levels), "fracs/rr_levels length mismatch"
        assert abs(sum(self.fracs) - 1.0) < 1e-6,     "fracs must sum to 1.0"
        assert all(r > 0 for r in self.rr_levels),    "rr_levels must be positive"
        assert self.rr_levels == tuple(sorted(self.rr_levels)), "rr_levels must be ascending"


# Pre-defined schemes (from backtesting comparison on 5 tickers, 4+ years)
SCHEMES: dict[str, SteppedTPScheme] = {
    "A": SteppedTPScheme((0.30, 0.30, 0.25, 0.15), (1.0, 1.8, 3.0, 5.0), "30/30/25/15 @ R1/1.8/3/5"),
    "B": SteppedTPScheme((0.40, 0.30, 0.20, 0.10), (1.0, 1.8, 3.0, 5.0), "40/30/20/10 @ R1/1.8/3/5"),
    "C": SteppedTPScheme((0.25, 0.25, 0.25, 0.25), (1.0, 1.8, 3.0, 5.0), "25/25/25/25 @ R1/1.8/3/5"),
    "D": SteppedTPScheme((0.50, 0.20, 0.20, 0.10), (1.0, 1.8, 3.0, 5.0), "50/20/20/10 @ R1/1.8/3/5"),
    "E": SteppedTPScheme((0.40, 0.35, 0.25),       (1.0, 2.0, 4.0),       "40/35/25 @ R1/2/4"),
    "F": SteppedTPScheme((0.50, 0.30, 0.20),       (0.8, 1.8, 3.5),       "50/30/20 @ R0.8/1.8/3.5 [BEST WR]"),
    "G": SteppedTPScheme((0.60, 0.40),             (0.8, 2.0),             "60/40 @ R0.8/2.0 [Early]"),
}


# ══════════════════════════════════════════════════════════════════════════════
# Signal generator
# ══════════════════════════════════════════════════════════════════════════════
class SignalGenerator:
    """
    Computes entry signals and SL/risk values for each bar.

    Returns a DataFrame with columns:
        signal  — 1 if entry signal on this bar, 0 otherwise
        sl      — stop-loss price
        risk    — entry_price − sl (computed at signal bar's close)
    """

    # --- Adaptive ER base signal parameters ---
    ER_FAST   = 10
    ER_SLOW   = 20
    ATR_FAST  = 5
    ATR_SLOW  = 14
    VOL_SL    = 1.8     # ATR multiplier for raw SL
    SL_MIN_PCT = 0.012  # minimum SL distance as % of close (1.2%)
    SL_MAX_PCT = 0.035  # maximum SL distance as % of close (3.5%)

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        c = df["close"]

        # ── Indicators ────────────────────────────────────────────────────────
        at14  = atr(df, self.ATR_SLOW)
        at5   = atr(df, self.ATR_FAST)
        er20  = efficiency_ratio(c, self.ER_SLOW)
        er10  = efficiency_ratio(c, self.ER_FAST)
        e200  = ema(c, 200)
        e50   = ema(c, 50)
        e20   = ema(c, 20)
        rsi14 = rsi(df, 14)
        mh    = macd_histogram(df)
        di_p, di_n = directional_index(df, 14)
        adx14 = adx(df, 14)
        vol_r = volume_ratio(df, 20)
        at_r  = at5 / at14

        hour = pd.Series(df.index.hour,       index=df.index)
        dow  = pd.Series(df.index.dayofweek,  index=df.index)

        ema20_slope = (e20 - e20.shift(3)) / e20.shift(3).replace(0, np.nan) * 100

        # ── Adaptive ER base signal (Kaufman regime filter) ───────────────────
        # Buy when: ER rising + price above fast EMA + fast ER > slow ER (trend accelerating)
        er_base = (
            (er20 > 0.35) &
            (er10 > er20) &                   # short-term ER > long-term ER
            (c > e50) &
            (c.diff() > 0)                    # last bar was up
        )

        # ── Quality gate (always required) ───────────────────────────────────
        quality = (
            (c > e200) &
            (di_p > di_n) &
            (mh > 0) &
            (er20 > 0.40) &
            (rsi14 >= 58) & (rsi14 <= 88) &
            (ema20_slope > 0)
        )

        # ── Cluster A: afternoon volume surge ────────────────────────────────
        cl_a = quality & (vol_r > 2.5) & (hour >= 13) & (hour <= 22) & (er20 > 0.43)

        # ── Cluster B: ATR impulse at week edges ─────────────────────────────
        cl_b = quality & (at_r > 1.08) & ((dow == 4) | (dow == 0)) & \
               (er20 > 0.45) & (vol_r > 1.5)

        # ── Cluster C: extreme volume + strong trend ──────────────────────────
        cl_c = quality & (vol_r > 3.0) & (adx14 > 28) & (er20 > 0.48)

        signal_raw = er_base & (cl_a | cl_b | cl_c)

        # Deduplicate consecutive signals
        signal = signal_raw & ~signal_raw.shift(1).fillna(False)

        # ── SL / risk ─────────────────────────────────────────────────────────
        sl_raw = c - self.VOL_SL * at14
        sl = sl_raw.clip(
            lower=c * (1 - self.SL_MAX_PCT),
            upper=c * (1 - self.SL_MIN_PCT),
        )
        risk = (c - sl).clip(lower=0.001)

        return pd.DataFrame({
            "signal": signal.astype(int),
            "sl":     sl,
            "risk":   risk,
        }, index=df.index)


# ══════════════════════════════════════════════════════════════════════════════
# Trade dataclass
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class Trade:
    entry_idx:    int
    entry:        float
    sl:           float
    risk:         float
    entry_dt:     object  = None
    exit_dt:      object  = None
    exit_price:   float   = 0.0
    reason:       str     = ""
    pnl:          float   = 0.0
    pnl_pct:      float   = 0.0
    hold_bars:    int     = 0
    win:          bool    = False
    remaining:    float   = 1.0    # fraction of position still open
    partial_pnl:  float   = 0.0   # P&L from partial exits already done
    tp_levels_hit: int    = 0      # number of TP levels triggered


# ══════════════════════════════════════════════════════════════════════════════
# Backtest result
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class BacktestResult:
    ticker:       str
    scheme_label: str
    trades:       int
    wins:         int
    losses:       int
    wr:           float
    total_pnl:    float
    total_pct:    float
    avg_win:      float
    avg_loss:     float
    profit_factor: float
    max_drawdown: float
    max_dd_pct:   float
    sharpe:       float
    expectancy:   float          # avg P&L per trade as % of avg entry
    exit_counts:  dict = field(default_factory=dict)
    equity:       list = field(default_factory=list)
    trade_list:   list = field(default_factory=list)

    @property
    def summary(self) -> str:
        return (f"{self.ticker}: {self.trades} trades | WR={self.wr*100:.0f}% | "
                f"PF={self.profit_factor:.2f} | Total={self.total_pct:+.1f}% | "
                f"Sharpe={self.sharpe:.2f}")


# ══════════════════════════════════════════════════════════════════════════════
# Backtest engine
# ══════════════════════════════════════════════════════════════════════════════
class BacktestEngine:
    """
    Event-driven backtest with stepped TP support.

    Args:
        scheme:   SteppedTPScheme instance
        max_hold: maximum bars to hold a position (time exit)
    """

    def __init__(self,
                 scheme:   SteppedTPScheme = SCHEMES["F"],
                 max_hold: int = 96):
        self.scheme   = scheme
        self.max_hold = max_hold
        self._sig_gen = SignalGenerator()

    def run(self, df: pd.DataFrame, ticker: str = "") -> BacktestResult:
        """Run backtest on OHLCV DataFrame. Returns BacktestResult."""
        self.scheme.validate()

        sig    = self._sig_gen.generate(df)
        fracs  = self.scheme.fracs
        levels = self.scheme.rr_levels
        n_tp   = len(fracs)

        opens  = df["open"].values
        highs  = df["high"].values
        lows   = df["low"].values
        closes = df["close"].values
        idx    = df.index
        n      = len(df)

        open_t: Optional[Trade] = None
        trades: list[Trade]     = []
        used:   set[float]      = set()   # de-dup by SL price
        equity: list[float]     = [0.0]

        for i in range(1, n):

            # ── Manage open position ──────────────────────────────────────────
            if open_t:
                hold   = i - open_t.entry_idx
                entry  = open_t.entry
                risk   = open_t.risk
                lv     = open_t.tp_levels_hit   # next TP level index
                bar_hi = highs[i]
                bar_lo = lows[i]

                # Check each pending TP level on this bar
                while lv < n_tp and open_t.remaining > 1e-9:
                    tp_px = entry + levels[lv] * risk
                    if bar_hi >= tp_px:
                        ep       = tp_px * (1 - SLIPPAGE)
                        frac     = min(fracs[lv], open_t.remaining)
                        open_t.partial_pnl += (
                            (ep - entry) * frac
                            - (entry + ep) * COMMISSION * frac
                        )
                        open_t.remaining   -= frac
                        # Move SL forward
                        if lv == 0:
                            new_sl = entry * 1.001          # breakeven
                        else:
                            new_sl = entry + levels[lv-1] * risk * 0.95
                        open_t.sl = max(open_t.sl, new_sl)
                        open_t.tp_levels_hit = lv + 1
                        lv = open_t.tp_levels_hit
                    else:
                        break

                # All TP levels reached → close fully
                if open_t.remaining <= 1e-6:
                    self._close(open_t, entry + levels[-1] * risk, f"TP{n_tp}",
                                i, idx, equity, trades)
                    open_t = None
                    continue

                # Time exit
                if hold >= self.max_hold:
                    ep = opens[i] * (1 - SLIPPAGE)
                    self._close_partial(open_t, ep, "TIME", i, idx, equity, trades)
                    open_t = None
                    continue

                # SL hit
                if bar_lo <= open_t.sl:
                    ep = max(open_t.sl * (1 - SLIPPAGE), bar_lo)
                    self._close_partial(open_t, ep, "SL", i, idx, equity, trades)
                    open_t = None
                    continue

            # ── Enter new position ────────────────────────────────────────────
            if not open_t and i > 0:
                row = sig.iloc[i - 1]
                if bool(row["signal"]):
                    sl_v   = float(row["sl"])
                    risk_v = float(row["risk"])
                    entry  = opens[i] * (1 + SLIPPAGE)
                    key    = round(sl_v, 1)
                    first_tp = entry + levels[0] * risk_v
                    if sl_v < entry < first_tp and risk_v > 0 and key not in used:
                        open_t = Trade(i, entry, sl_v, risk_v, entry_dt=idx[i])
                        used.add(key)

        # Close any remaining open trade at end of data
        if open_t:
            ep = closes[-1] * (1 - SLIPPAGE)
            self._close_partial(open_t, ep, "END", n - 1, idx, equity, trades)

        return self._compute_result(trades, equity, ticker)

    # ── Helpers ───────────────────────────────────────────────────────────────
    @staticmethod
    def _close(t: Trade, ep: float, reason: str,
               i: int, idx, equity: list, trades: list):
        """Close a trade where remaining fraction was already tracked by TP hits."""
        ep   = ep * (1 - SLIPPAGE)
        rem  = t.remaining
        pnl  = t.partial_pnl + (ep - t.entry) * rem - (t.entry + ep) * COMMISSION * rem
        t.exit_price = ep; t.exit_dt = idx[i]; t.reason = reason
        t.hold_bars  = i - t.entry_idx
        t.pnl = pnl; t.pnl_pct = pnl / t.entry * 100; t.win = bool(pnl > 0)
        equity.append(equity[-1] + pnl)
        trades.append(t)

    @staticmethod
    def _close_partial(t: Trade, ep: float, reason: str,
                       i: int, idx, equity: list, trades: list):
        """Close remaining position at ep, including accumulated partial_pnl."""
        rem  = t.remaining
        pnl  = t.partial_pnl + (ep - t.entry) * rem - (t.entry + ep) * COMMISSION * rem
        t.exit_price = ep; t.exit_dt = idx[i]; t.reason = reason
        t.hold_bars  = i - t.entry_idx
        t.pnl = pnl; t.pnl_pct = pnl / t.entry * 100; t.win = bool(pnl > 0)
        equity.append(equity[-1] + pnl)
        trades.append(t)

    @staticmethod
    def _compute_result(trades: list[Trade], equity: list[float],
                        ticker: str,
                        scheme_label: str = "") -> BacktestResult:
        if not trades:
            return BacktestResult(ticker, scheme_label,
                                  0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                  equity=[0.0])

        wins   = [t for t in trades if t.win]
        losses = [t for t in trades if not t.win]
        total  = sum(t.pnl for t in trades)
        avg_e  = float(np.mean([t.entry for t in trades]))

        avg_win  = float(np.mean([t.pnl for t in wins]))   if wins   else 0.0
        avg_loss = float(np.mean([t.pnl for t in losses])) if losses else 0.0
        pf       = (abs(avg_win) * len(wins) /
                    (abs(avg_loss) * len(losses) + 1e-9)) if losses else 99.0

        eq_arr   = np.array(equity)
        peak     = np.maximum.accumulate(eq_arr)
        dd_abs   = float((eq_arr - peak).min())
        dd_pct   = dd_abs / (avg_e + 1e-9) * 100

        pnls     = np.array([t.pnl for t in trades])
        sharpe   = (float(np.mean(pnls) / (np.std(pnls) + 1e-9))
                    * np.sqrt(252 * 6.5)) if len(pnls) > 1 else 0.0

        wr       = len(wins) / len(trades)
        expect   = (wr * abs(avg_win) - (1 - wr) * abs(avg_loss)) / (avg_e + 1e-9) * 100

        from collections import Counter
        exit_counts = dict(Counter(t.reason for t in trades))

        return BacktestResult(
            ticker       = ticker,
            scheme_label = scheme_label,
            trades       = len(trades),
            wins         = len(wins),
            losses       = len(losses),
            wr           = wr,
            total_pnl    = float(total),
            total_pct    = float(total / avg_e * 100),
            avg_win      = avg_win,
            avg_loss     = avg_loss,
            profit_factor= min(float(pf), 99.0),
            max_drawdown = dd_abs,
            max_dd_pct   = dd_pct,
            sharpe       = sharpe,
            expectancy   = float(expect),
            exit_counts  = exit_counts,
            equity       = [round(float(x), 2) for x in equity],
            trade_list   = trades,
        )

    def run_scheme_comparison(self, df: pd.DataFrame, ticker: str = "") -> dict[str, BacktestResult]:
        """Run backtest for all 7 predefined schemes and return results dict."""
        results = {}
        for key, scheme in SCHEMES.items():
            engine = BacktestEngine(scheme=scheme, max_hold=self.max_hold)
            r = engine.run(df, ticker)
            r.scheme_label = scheme.label
            results[key] = r
        return results
