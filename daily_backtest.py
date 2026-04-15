"""
Бэктест стратегии на дневном таймфрейме.

Отличия от часового:
  - Кластеры A/B/C адаптированы под дневные данные (убраны hour/dow)
  - max_hold = 20 баров (≈ 1 месяц торговых дней)
  - SL_MIN_PCT = 3%, SL_MAX_PCT = 8% (дневные движения шире)
  - Sharpe масштабируется на sqrt(252) вместо sqrt(252 * 6.5)
  - Схема TP: F (лучшая по WR)
"""
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from core.indicators import (
    adx, atr, directional_index,
    efficiency_ratio, ema, macd_histogram, rsi, volume_ratio,
)

# ── Config ────────────────────────────────────────────────────────────────────
TICKERS = ["GAZP", "LKOH", "MGNT", "MTLR", "NLMK", "NVTK",
           "OZPH", "ROSN", "SBER", "T", "TGKA", "YDEX"]

DATA_DIR = Path("c:/investor/data")
COMMISSION = 0.0005
SLIPPAGE   = 0.0001
MAX_HOLD   = 20       # дней

# ── Адаптированный генератор сигналов для дневного ТФ ─────────────────────────
class DailySignalGenerator:
    ER_FAST    = 10
    ER_SLOW    = 20
    ATR_FAST   = 5
    ATR_SLOW   = 14
    VOL_SL     = 2.0      # шире ATR для дневных баров
    SL_MIN_PCT = 0.03     # мин SL 3%
    SL_MAX_PCT = 0.08     # макс SL 8%

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        c = df["close"]

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

        ema20_slope = (e20 - e20.shift(5)) / e20.shift(5).replace(0, np.nan) * 100

        # Базовый ER-фильтр (без изменений)
        er_base = (
            (er20 > 0.35) &
            (er10 > er20) &
            (c > e50) &
            (c.diff() > 0)
        )

        # Quality gate (ослаблен RSI: 55-85 для дневного)
        quality = (
            (c > e200) &
            (di_p > di_n) &
            (mh > 0) &
            (er20 > 0.38) &
            (rsi14 >= 55) & (rsi14 <= 85) &
            (ema20_slope > 0)
        )

        # Кластер A: объёмный всплеск (без привязки к часу)
        cl_a = quality & (vol_r > 2.0) & (er20 > 0.42)

        # Кластер B: ATR-импульс (без привязки к дню недели)
        cl_b = quality & (at_r > 1.10) & (er20 > 0.43) & (vol_r > 1.3)

        # Кластер C: экстремальный объём + сильный тренд
        cl_c = quality & (vol_r > 2.5) & (adx14 > 25) & (er20 > 0.45)

        signal_raw = er_base & (cl_a | cl_b | cl_c)
        signal = signal_raw & ~signal_raw.shift(1).fillna(False)

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


# ── Trade & Result dataclasses ────────────────────────────────────────────────
@dataclass
class Trade:
    entry_idx:    int
    entry:        float
    sl:           float
    risk:         float
    entry_dt:     object = None
    exit_dt:      object = None
    exit_price:   float  = 0.0
    reason:       str    = ""
    pnl:          float  = 0.0
    pnl_pct:      float  = 0.0
    hold_bars:    int    = 0
    win:          bool   = False
    remaining:    float  = 1.0
    partial_pnl:  float  = 0.0
    tp_levels_hit: int   = 0


@dataclass
class Result:
    ticker:       str
    trades:       int
    wins:         int
    losses:       int
    wr:           float
    pf:           float
    total_pct:    float
    avg_win_pct:  float
    avg_loss_pct: float
    sharpe:       float
    max_dd_pct:   float
    expectancy:   float
    exit_counts:  dict  = field(default_factory=dict)


# ── Backtest engine (дневной) ─────────────────────────────────────────────────
FRACS  = (0.50, 0.30, 0.20)
LEVELS = (0.8,  1.8,  3.5)     # схема F — лучшая WR

def run_backtest(df: pd.DataFrame, ticker: str) -> Result:
    gen = DailySignalGenerator()
    sig = gen.generate(df)

    opens  = df["open"].values
    highs  = df["high"].values
    lows   = df["low"].values
    closes = df["close"].values
    idx    = df.index
    n      = len(df)
    n_tp   = len(FRACS)

    open_t: Optional[Trade] = None
    trades: list[Trade]     = []
    equity: list[float]     = [0.0]
    used:   set             = set()

    for i in range(1, n):
        if open_t:
            hold   = i - open_t.entry_idx
            entry  = open_t.entry
            risk   = open_t.risk
            lv     = open_t.tp_levels_hit
            bar_hi = highs[i]
            bar_lo = lows[i]

            while lv < n_tp and open_t.remaining > 1e-9:
                tp_px = entry + LEVELS[lv] * risk
                if bar_hi >= tp_px:
                    ep   = tp_px * (1 - SLIPPAGE)
                    frac = min(FRACS[lv], open_t.remaining)
                    open_t.partial_pnl += (
                        (ep - entry) * frac - (entry + ep) * COMMISSION * frac
                    )
                    open_t.remaining -= frac
                    open_t.sl = max(open_t.sl,
                                    entry * 1.001 if lv == 0 else
                                    entry + LEVELS[lv-1] * risk * 0.95)
                    open_t.tp_levels_hit = lv + 1
                    lv = open_t.tp_levels_hit
                else:
                    break

            if open_t.remaining <= 1e-6:
                _close(open_t, entry + LEVELS[-1] * risk, f"TP{n_tp}",
                       i, idx, equity, trades)
                open_t = None; continue

            if hold >= MAX_HOLD:
                _close(open_t, opens[i] * (1 - SLIPPAGE), "TIME",
                       i, idx, equity, trades)
                open_t = None; continue

            if bar_lo <= open_t.sl:
                ep = max(open_t.sl * (1 - SLIPPAGE), bar_lo)
                _close(open_t, ep, "SL", i, idx, equity, trades)
                open_t = None; continue

        if not open_t:
            row = sig.iloc[i - 1]
            if bool(row["signal"]):
                sl_v   = float(row["sl"])
                risk_v = float(row["risk"])
                entry  = opens[i] * (1 + SLIPPAGE)
                key    = round(sl_v, 1)
                first_tp = entry + LEVELS[0] * risk_v
                if sl_v < entry < first_tp and risk_v > 0 and key not in used:
                    open_t = Trade(i, entry, sl_v, risk_v, entry_dt=idx[i])
                    used.add(key)

    if open_t:
        _close(open_t, closes[-1] * (1 - SLIPPAGE), "END",
               n - 1, idx, equity, trades)

    return _compute(trades, equity, ticker)


def _close(t: Trade, ep: float, reason: str,
           i: int, idx, equity: list, trades: list):
    rem = t.remaining
    pnl = t.partial_pnl + (ep - t.entry) * rem - (t.entry + ep) * COMMISSION * rem
    t.exit_price = ep; t.exit_dt = idx[i]; t.reason = reason
    t.hold_bars  = i - t.entry_idx
    t.pnl = pnl; t.pnl_pct = pnl / t.entry * 100; t.win = bool(pnl > 0)
    equity.append(equity[-1] + pnl)
    trades.append(t)


def _compute(trades: list[Trade], equity: list[float], ticker: str) -> Result:
    if not trades:
        return Result(ticker, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    wins   = [t for t in trades if t.win]
    losses = [t for t in trades if not t.win]
    avg_e  = float(np.mean([t.entry for t in trades]))

    avg_win_p  = float(np.mean([t.pnl_pct for t in wins]))   if wins   else 0.0
    avg_loss_p = float(np.mean([t.pnl_pct for t in losses])) if losses else 0.0
    avg_win_r  = float(np.mean([t.pnl for t in wins]))        if wins   else 0.0
    avg_loss_r = float(np.mean([t.pnl for t in losses]))      if losses else 0.0
    pf = (abs(avg_win_r) * len(wins) / (abs(avg_loss_r) * len(losses) + 1e-9)) if losses else 99.0

    eq_arr = np.array(equity)
    peak   = np.maximum.accumulate(eq_arr)
    dd_abs = float((eq_arr - peak).min())
    dd_pct = dd_abs / (avg_e + 1e-9) * 100

    pnls   = np.array([t.pnl_pct for t in trades])
    sharpe = float(np.mean(pnls) / (np.std(pnls) + 1e-9) * np.sqrt(252)) if len(pnls) > 1 else 0.0

    wr       = len(wins) / len(trades)
    expect   = wr * abs(avg_win_p) - (1 - wr) * abs(avg_loss_p)

    return Result(
        ticker       = ticker,
        trades       = len(trades),
        wins         = len(wins),
        losses       = len(losses),
        wr           = wr,
        pf           = min(float(pf), 99.0),
        total_pct    = float(sum(t.pnl_pct for t in trades)),
        avg_win_pct  = avg_win_p,
        avg_loss_pct = avg_loss_p,
        sharpe       = sharpe,
        max_dd_pct   = dd_pct,
        expectancy   = float(expect),
        exit_counts  = dict(Counter(t.reason for t in trades)),
    )


# ── CSV loader ────────────────────────────────────────────────────────────────
def load_daily(ticker: str) -> Optional[pd.DataFrame]:
    path = DATA_DIR / f"{ticker}_2022_2026_D.csv"
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path, sep=";", header=0)
        df.columns = [c.strip("<>").lower() for c in df.columns]
        df["datetime"] = pd.to_datetime(
            df["date"].astype(str),
            format="%d/%m/%y", errors="coerce"
        )
        df = df.dropna(subset=["datetime"]).set_index("datetime")
        df = df.rename(columns={"vol": "volume"})[["open","high","low","close","volume"]]
        for col in ["open","high","low","close","volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        return df.dropna()
    except Exception as e:
        print(f"  Ошибка загрузки {ticker}: {e}")
        return None


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 70)
    print("  ДНЕВНОЙ БЭКТЕСТ — Momentum Filter + Stepped TP (схема F)")
    print("  Период: 2022-01 → 2026-03  |  Таймфрейм: Day")
    print("=" * 70)

    results = []
    for ticker in TICKERS:
        df = load_daily(ticker)
        if df is None or len(df) < 250:
            print(f"  {ticker:6s}: нет данных или слишком мало баров")
            continue
        r = run_backtest(df, ticker)
        results.append(r)
        exits_str = "  ".join(f"{k}:{v}" for k, v in sorted(r.exit_counts.items()))
        print(f"  {ticker:6s}  trades={r.trades:3d}  WR={r.wr*100:4.0f}%  "
              f"PF={r.pf:4.2f}  total={r.total_pct:+6.1f}%  "
              f"Sharpe={r.sharpe:5.2f}  MaxDD={r.max_dd_pct:+5.1f}%  "
              f"Exp={r.expectancy:+4.2f}%  [{exits_str}]")

    if not results:
        print("  Нет данных для анализа.")
    else:
        print()
        print("─" * 70)
        total_trades = sum(r.trades for r in results)
        total_wins   = sum(r.wins   for r in results)
        all_wr       = total_wins / total_trades if total_trades else 0
        avg_pf       = float(np.mean([r.pf for r in results if r.trades > 0]))
        avg_total    = float(np.mean([r.total_pct for r in results]))
        avg_sharpe   = float(np.mean([r.sharpe for r in results]))
        avg_exp      = float(np.mean([r.expectancy for r in results]))

        print(f"  ИТОГО   trades={total_trades:3d}  WR={all_wr*100:4.0f}%  "
              f"PF={avg_pf:4.2f}  avg_total={avg_total:+6.1f}%  "
              f"Sharpe={avg_sharpe:5.2f}  Exp={avg_exp:+4.2f}%")
        print()

        # Сравнение часовой vs дневной
        print("=" * 70)
        print("  СРАВНЕНИЕ: ЧАСОВОЙ vs ДНЕВНОЙ таймфрейм")
        print("=" * 70)
        print(f"  {'Параметр':<25} {'Часовой (1H)':>15} {'Дневной (D)':>15}")
        print(f"  {'-'*55}")
        hourly_stats = {
            "Кол-во сделок":    "541",
            "Win Rate":         "54%",
            "Profit Factor":    "0.82",
            "Средняя доходность": "-16.8%",
            "Sharpe Ratio":     "негативный",
        }
        daily_stats = {
            "Кол-во сделок":    str(total_trades),
            "Win Rate":         f"{all_wr*100:.0f}%",
            "Profit Factor":    f"{avg_pf:.2f}",
            "Средняя доходность": f"{avg_total:+.1f}%",
            "Sharpe Ratio":     f"{avg_sharpe:.2f}",
        }
        for key in hourly_stats:
            print(f"  {key:<25} {hourly_stats[key]:>15} {daily_stats[key]:>15}")
        print()
        print("  ► Интерпретация:")
        if avg_total > 0 and avg_pf > 1.0:
            print("  ✓ Дневной таймфрейм ПРИБЫЛЕН — переход оправдан!")
            if all_wr >= 0.55:
                print(f"  ✓ Win Rate {all_wr*100:.0f}% — выше порога безубыточности")
            if avg_exp > 0:
                print(f"  ✓ Положительное математическое ожидание: +{avg_exp:.2f}% на сделку")
        else:
            print("  ✗ Стратегия требует доработки даже на дневном ТФ")
            print("  → Рекомендация: ослабить фильтры или изменить параметры TP")
        print("=" * 70)
