"""
Advanced trading strategies - institutional-grade, not classical indicators.

Strategies (LONG-only, daily/hourly data):
  1. Liquidity Sweep Reversal  (SMC/ICT)
  2. Volume Exhaustion Reversal
  3. Adaptive Regime (Kaufman ER)
  4. Fair Value Gap Retest     (ICT)
  5. Composite Momentum + Mean Reversion (original)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------

def load_csv(path: str | Path) -> pd.DataFrame:
    raw = Path(path).read_text(encoding="utf-8", errors="replace")
    sep = ";" if raw.count(";") > raw.count(",") else ","
    df = pd.read_csv(path, sep=sep)
    df.columns = [c.strip().lower().replace("<", "").replace(">", "") for c in df.columns]
    if "date" in df.columns and "time" in df.columns:
        date_str = df["date"].astype(str).str.strip()
        time_str = df["time"].astype(str).str.zfill(6)
        fmt = "%d/%m/%y %H%M%S" if "/" in date_str.iloc[0] else "%Y%m%d %H%M%S"
        df["datetime"] = pd.to_datetime(date_str + " " + time_str, format=fmt, errors="coerce")
    elif "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    if "vol" in df.columns and "volume" not in df.columns:
        df = df.rename(columns={"vol": "volume"})
    df = df.set_index("datetime").sort_index()
    return df[["open", "high", "low", "close", "volume"]].astype(float).dropna()


# --------------------------------------------------------------------------
# Indicators
# --------------------------------------------------------------------------

def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    h, l, pc = df["high"], df["low"], df["close"].shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()

def _rsi(df: pd.DataFrame, n: int = 14) -> pd.Series:
    d = df["close"].diff()
    g = d.clip(lower=0).ewm(alpha=1/n, min_periods=n, adjust=False).mean()
    l = (-d).clip(lower=0).ewm(alpha=1/n, min_periods=n, adjust=False).mean()
    return 100 - 100 / (1 + g / l.replace(0, np.nan))


# --------------------------------------------------------------------------
# Backtest engine
# --------------------------------------------------------------------------

COMMISSION = 0.0005
SLIPPAGE   = 0.0001

@dataclass
class Trade:
    entry_idx: int
    entry_price: float
    sl: float
    tp: float
    exit_idx: int = 0
    exit_price: float = 0.0
    reason: str = ""

    @property
    def pnl(self):
        return (self.exit_price - self.entry_price
                - (self.entry_price + self.exit_price) * COMMISSION)

    @property
    def win(self): return self.pnl > 0


@dataclass
class Result:
    name: str
    trades: list = field(default_factory=list)
    equity: list = field(default_factory=list)
    initial: float = 100_000.0

    @property
    def n(self): return len(self.trades)
    @property
    def wins(self): return sum(1 for t in self.trades if t.win)
    @property
    def win_rate(self): return self.wins / self.n if self.n else 0.0
    @property
    def total_pnl(self): return sum(t.pnl for t in self.trades)
    @property
    def total_return(self): return self.total_pnl / self.initial
    @property
    def max_dd(self):
        eq = pd.Series(self.equity)
        if len(eq) < 2: return 0.0
        dd = (eq - eq.cummax()) / eq.cummax()
        return float(dd.min())
    @property
    def sharpe(self):
        eq = pd.Series(self.equity)
        r = eq.pct_change().dropna()
        if len(r) < 2 or r.std() == 0: return 0.0
        return float(r.mean() / r.std() * math.sqrt(252))
    @property
    def profit_factor(self):
        g = sum(t.pnl for t in self.trades if t.win)
        l = abs(sum(t.pnl for t in self.trades if not t.win))
        return round(g / l, 3) if l > 0 else float("inf")
    @property
    def avg_win(self):
        w = [t.pnl for t in self.trades if t.win]
        return sum(w)/len(w) if w else 0.0
    @property
    def avg_loss(self):
        l = [t.pnl for t in self.trades if not t.win]
        return sum(l)/len(l) if l else 0.0
    @property
    def expectancy(self):
        return self.win_rate * self.avg_win + (1 - self.win_rate) * self.avg_loss

    def row(self, ticker=""):
        return {
            "Strategy":  self.name,
            "Ticker":    ticker,
            "Trades":    self.n,
            "WR%":       f"{self.win_rate*100:.0f}%",
            "Net P&L":   f"{self.total_pnl:+.0f}",
            "Return%":   f"{self.total_return*100:.1f}%",
            "MaxDD%":    f"{self.max_dd*100:.1f}%",
            "Sharpe":    f"{self.sharpe:.3f}",
            "PF":        f"{self.profit_factor:.2f}" if self.profit_factor != float("inf") else "inf",
            "E/trade":   f"{self.expectancy:+.1f}",
        }


def run_engine(df: pd.DataFrame, signals_fn, name: str,
               initial: float = 100_000.0) -> Result:
    sig = signals_fn(df)
    result = Result(name=name, initial=initial)
    equity = initial
    open_trade: Optional[Trade] = None
    used_entries: set = set()

    closes = df["close"].values
    highs  = df["high"].values
    lows   = df["low"].values
    opens  = df["open"].values
    n = len(df)

    for i in range(1, n):
        result.equity.append(equity)
        if open_trade is not None:
            lo, hi = lows[i], highs[i]
            if lo <= open_trade.sl:
                ep = max(open_trade.sl * (1 - SLIPPAGE), lo)
                open_trade.exit_idx = i; open_trade.exit_price = ep; open_trade.reason = "SL"
                equity += open_trade.pnl
                result.trades.append(open_trade); open_trade = None
            elif hi >= open_trade.tp:
                ep = open_trade.tp * (1 - SLIPPAGE)
                open_trade.exit_idx = i; open_trade.exit_price = ep; open_trade.reason = "TP"
                equity += open_trade.pnl
                result.trades.append(open_trade); open_trade = None

        if open_trade is None:
            prev = i - 1
            if prev >= 0 and bool(sig["signal"].iloc[prev]):
                sl_val = float(sig["sl"].iloc[prev])
                tp_val = float(sig["tp"].iloc[prev])
                entry  = opens[i] * (1 + SLIPPAGE)
                key    = round(sl_val, 2)
                if sl_val < entry < tp_val and key not in used_entries:
                    open_trade = Trade(entry_idx=i, entry_price=entry, sl=sl_val, tp=tp_val)
                    used_entries.add(key)

    if open_trade is not None:
        ep = closes[-1] * (1 - SLIPPAGE)
        open_trade.exit_idx = n - 1; open_trade.exit_price = ep; open_trade.reason = "END"
        equity += open_trade.pnl
        result.trades.append(open_trade)
    return result


# ==========================================================================
# STRATEGY 1: Liquidity Sweep Reversal (SMC/ICT)
# ==========================================================================
def strat_liquidity_sweep(df: pd.DataFrame) -> pd.DataFrame:
    """
    Concept: institutions deliberately push price below retail stop-loss clusters
    (liquidity pools), then reverse sharply to buy cheap.

    Signal logic:
    1. Identify liquidity pool: rolling N-bar low (where stops cluster).
    2. Sweep: current candle low < pool AND close > pool  (bears failed to hold).
    3. Confirmation: next candle closes above sweep-candle close.
    4. Entry: open of candle after confirmation.
    5. SL: below sweep candle low - ATR buffer.
    6. TP: N-bar high (nearest resistance = next liquidity pool).
    """
    N = 20
    at = _atr(df, 14)
    low_pool  = df["low"].rolling(N).min().shift(1)
    high_pool = df["high"].rolling(N).max().shift(1)
    sweep = (df["low"] < low_pool) & (df["close"] > low_pool)

    signals = pd.Series(False, index=df.index)
    sl_arr  = df["low"].copy()
    tp_arr  = df["high"].copy()

    for i in range(N + 1, len(df) - 2):
        if not sweep.iloc[i]:
            continue
        # confirmation candle (i+1): closes above sweep-close
        if df["close"].iloc[i + 1] <= df["close"].iloc[i]:
            continue
        sl  = df["low"].iloc[i] - at.iloc[i] * 0.3
        tp  = float(high_pool.iloc[i])
        ref = df["close"].iloc[i + 1]
        if sl <= 0 or tp <= ref:
            continue
        # signal placed on candle i+1, entry on open of i+2
        signals.iloc[i + 1] = True
        sl_arr.iloc[i + 1]  = sl
        tp_arr.iloc[i + 1]  = tp

    return pd.DataFrame({"signal": signals, "sl": sl_arr, "tp": tp_arr})


# ==========================================================================
# STRATEGY 2: Volume Exhaustion Reversal
# ==========================================================================
def strat_volume_exhaustion(df: pd.DataFrame) -> pd.DataFrame:
    """
    Concept: climax selling (panic) followed by volume exhaustion = institutional
    accumulation. Retail panic out, smart money buys.

    Signal logic:
    1. Find 3+ consecutive bearish candles with escalating volume (selling climax).
    2. Climax volume must be > 1.5x the 20-bar average.
    3. Exhaustion: NEXT candle volume < 60% of climax volume (sellers dried up).
    4. RSI(14) < 45 for oversold context.
    5. SL: below the lowest point of the climax cluster - ATR buffer.
    6. TP: entry + 2.5 x ATR.
    """
    at    = _atr(df, 14)
    rsi14 = _rsi(df, 14)
    close = df["close"].values
    open_ = df["open"].values
    vol   = df["volume"].values
    vol_ma = pd.Series(vol).rolling(20).mean().values
    n = len(df)

    signals = np.zeros(n, dtype=bool)
    sl_arr  = df["low"].values.copy()
    tp_arr  = df["high"].values.copy()

    for i in range(25, n - 2):
        if not all(close[j] < open_[j] for j in range(i-2, i+1)):
            continue
        if not (vol[i-1] > vol[i-2] and vol[i] > vol[i-1]):
            continue
        if vol_ma[i] <= 0 or vol[i] < vol_ma[i] * 1.5:
            continue
        if vol[i+1] > vol[i] * 0.6:
            continue
        if rsi14.iloc[i] > 45:
            continue
        sl  = float(df["low"].iloc[i-2:i+1].min()) - float(at.iloc[i]) * 0.2
        tp  = close[i+1] + 2.5 * float(at.iloc[i])
        signals[i + 1] = True
        sl_arr[i + 1]  = sl
        tp_arr[i + 1]  = tp

    return pd.DataFrame({
        "signal": pd.Series(signals, index=df.index),
        "sl":     pd.Series(sl_arr,  index=df.index),
        "tp":     pd.Series(tp_arr,  index=df.index),
    })


# ==========================================================================
# STRATEGY 3: Adaptive Regime (Kaufman Efficiency Ratio)
# ==========================================================================
def strat_adaptive_regime(df: pd.DataFrame) -> pd.DataFrame:
    """
    Kaufman Efficiency Ratio (ER):
        ER = |price change over N bars| / sum(|daily changes| over N bars)
        ER -> 1.0: clean trend  |  ER -> 0.0: choppy/ranging

    Regime switching:
    - TREND regime  (ER > 0.40): buy N-bar breakout, SL = adaptive MA, TP = 3xATR
    - RANGE regime  (ER < 0.20): buy RSI<35 reversal, SL = low - 0.8xATR, TP = 1.5xATR
    - TRANSITION    (0.20-0.40): skip (uncertainty)

    Key advantage: NO manual parameter tuning per ticker - strategy self-adapts.
    """
    N     = 20
    at    = _atr(df, 14)
    rsi14 = _rsi(df, 14)
    close = df["close"]

    direction  = (close - close.shift(N)).abs()
    volatility = close.diff().abs().rolling(N).sum()
    er = (direction / volatility.replace(0, np.nan)).fillna(0).clip(0, 1)

    # Adaptive Moving Average (AMA / KAMA)
    fast_sc = 2 / (2 + 1)
    slow_sc = 2 / (30 + 1)
    sc  = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    ama = close.copy().astype(float)
    for i in range(1, len(ama)):
        ama.iloc[i] = ama.iloc[i-1] + float(sc.iloc[i]) * (float(close.iloc[i]) - ama.iloc[i-1])

    high_n = df["high"].rolling(N).max().shift(1)

    signals = pd.Series(False, index=df.index)
    sl_arr  = df["low"].copy()
    tp_arr  = df["high"].copy()

    for i in range(N + 5, len(df) - 1):
        regime = float(er.iloc[i])
        c = float(close.iloc[i])

        if regime > 0.40:
            if c > float(high_n.iloc[i]):
                sl = float(ama.iloc[i]) - float(at.iloc[i]) * 0.3
                tp = c + 3.0 * float(at.iloc[i])
                if sl < c < tp:
                    signals.iloc[i] = True
                    sl_arr.iloc[i]  = sl
                    tp_arr.iloc[i]  = tp

        elif regime < 0.20:
            r = float(rsi14.iloc[i])
            r_prev = float(rsi14.iloc[i-1])
            if r < 35 and r > r_prev:
                sl = float(df["low"].iloc[i]) - 0.8 * float(at.iloc[i])
                tp = c + 1.5 * float(at.iloc[i])
                if sl < c < tp:
                    signals.iloc[i] = True
                    sl_arr.iloc[i]  = sl
                    tp_arr.iloc[i]  = tp

    return pd.DataFrame({"signal": signals, "sl": sl_arr, "tp": tp_arr})


# ==========================================================================
# STRATEGY 4: Fair Value Gap (FVG) Retest  — ICT concept
# ==========================================================================
def strat_fvg(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fair Value Gap (bullish): three-candle pattern where
        high[i-2] < low[i]  =>  price jumped creating a gap zone

    These gaps represent areas with no traded volume - institutions use them
    as demand zones on retracements.

    Algorithm:
    1. Detect bullish FVG: high[i-2] < low[i] (upward gap).
    2. Store zone [high[i-2], low[i]] and target = high[i] + ATR.
    3. Wait for retest: price returns to the gap zone.
    4. Signal: candle low touches zone BUT close remains above the gap bottom.
    5. SL: below FVG bottom - ATR*0.2.
    6. TP: high of gap-creating candle + ATR.
    7. FVG invalidated if price closes below gap bottom.
    8. Trend filter: only trade above EMA(200).
    """
    at     = _atr(df, 14)
    ema200 = _ema(df["close"], 200)

    highs  = df["high"].values
    lows   = df["low"].values
    closes = df["close"].values
    n      = len(df)

    signals = np.zeros(n, dtype=bool)
    sl_arr  = lows.copy()
    tp_arr  = highs.copy()

    active_gaps: list = []   # (fvg_bot, fvg_top, tp_target, created_at)

    for i in range(2, n - 1):
        if closes[i] < float(ema200.iloc[i]) * 0.98:
            active_gaps.clear()
            continue

        # New bullish FVG
        if highs[i - 2] < lows[i]:
            fvg_bot = highs[i - 2]
            fvg_top = lows[i]
            tp_tgt  = highs[i] + float(at.iloc[i])
            active_gaps.append((fvg_bot, fvg_top, tp_tgt, i))

        # Check retest of existing gaps
        new_gaps = []
        for (fvg_bot, fvg_top, tp_tgt, created) in active_gaps:
            if i <= created + 1:
                new_gaps.append((fvg_bot, fvg_top, tp_tgt, created))
                continue

            # Retest: candle touches the gap zone
            if lows[i] <= fvg_top and highs[i] >= fvg_bot:
                if closes[i] > fvg_bot:   # held above bottom = bullish
                    sl  = fvg_bot - float(at.iloc[i]) * 0.2
                    tp  = tp_tgt
                    if sl < closes[i] < tp and not signals[i]:
                        signals[i] = True
                        sl_arr[i]  = sl
                        tp_arr[i]  = tp
                    continue   # gap consumed
                else:
                    continue   # gap invalidated

            if closes[i] < fvg_bot:
                continue   # invalidated by close below

            new_gaps.append((fvg_bot, fvg_top, tp_tgt, created))

        active_gaps = new_gaps[-10:]

    return pd.DataFrame({
        "signal": pd.Series(signals, index=df.index),
        "sl":     pd.Series(sl_arr,  index=df.index),
        "tp":     pd.Series(tp_arr,  index=df.index),
    })


# ==========================================================================
# STRATEGY 5: Composite Momentum + Mean Reversion (original)
# ==========================================================================
def strat_composite(df: pd.DataFrame) -> pd.DataFrame:
    """
    Two-phase system combining trend momentum and mean reversion entry.

    PHASE 1 - Momentum filter (are conditions favorable?):
        score = (ROC_20 > 0)       +   slow trend up
                (close > EMA_50)   +   price above trend
                (EMA_20 > EMA_50)  +   trend alignment
                (RSI_14 > 50)          momentum positive
        Trade only if score >= 3 out of 4.

    PHASE 2 - Mean reversion entry (optimal timing within the trend):
        Wait for fast RSI(5) < 30 AND recovering upward = short-term pullback.
        This is the "spring" entry - buying temporary weakness in an uptrend.

    ADAPTIVE SL/TP via volatility percentile:
        vol_pct = percentile rank of ATR5/close over 60 bars
        sl_mult = 1.0 + vol_pct * 1.5   (wider stop when volatile)
        tp_mult = 2.0 + vol_pct * 2.0   (bigger target when volatile)

    Key insight: we align WITH the trend but enter at points of temporary
    weakness - better risk/reward than pure trend-following.
    """
    close = df["close"]
    at14  = _atr(df, 14)
    at5   = _atr(df,  5)
    rsi14 = _rsi(df, 14)
    rsi5  = _rsi(df,  5)
    ema20 = _ema(close, 20)
    ema50 = _ema(close, 50)
    roc20 = (close / close.shift(20) - 1)

    ms = (
        (roc20 > 0).astype(int)
        + (close > ema50).astype(int)
        + (ema20 > ema50).astype(int)
        + (rsi14 > 50).astype(int)
    )

    vol_ratio = (at5 / close).rolling(60).rank(pct=True).fillna(0.5)
    rsi5_rev  = (rsi5 < 30) & (rsi5 > rsi5.shift(1))

    signals = pd.Series(False, index=df.index)
    sl_arr  = df["low"].copy()
    tp_arr  = df["high"].copy()

    for i in range(60, len(df) - 1):
        if ms.iloc[i] < 3:
            continue
        if not bool(rsi5_rev.iloc[i]):
            continue
        vp      = float(vol_ratio.iloc[i])
        c       = float(close.iloc[i])
        sl_mult = 1.0 + vp * 1.5
        tp_mult = 2.0 + vp * 2.0
        sl = c - sl_mult * float(at14.iloc[i])
        tp = c + tp_mult * float(at14.iloc[i])
        if sl < c < tp:
            signals.iloc[i] = True
            sl_arr.iloc[i]  = sl
            tp_arr.iloc[i]  = tp

    return pd.DataFrame({"signal": signals, "sl": sl_arr, "tp": tp_arr})


# ==========================================================================
# RUNNER
# ==========================================================================

STRATEGIES = [
    ("Liquidity Sweep",  strat_liquidity_sweep),
    ("Vol Exhaustion",   strat_volume_exhaustion),
    ("Adaptive Regime",  strat_adaptive_regime),
    ("FVG Retest",       strat_fvg),
    ("Composite MR+Mom", strat_composite),
]

FILES = [
    ("ROSN", "ROSN_220122_260320.csv"),
    ("YDEX", "YDEX_220122_260320.csv"),
    ("LKOH", "LKOH_220122_260320.csv"),
    ("MGNT", "MGNT_220122_260320.csv"),
    ("SBER", "SBER_251222_260320.csv"),
]

COLS  = ["Strategy","Trades","WR%","Net P&L","Return%","MaxDD%","Sharpe","PF","E/trade"]
CWIDTHS = [18, 7, 5, 9, 8, 7, 7, 5, 10]


def print_table(rows: list[dict]):
    header = "  ".join(c.ljust(w) for c, w in zip(COLS, CWIDTHS))
    print(header)
    print("-" * len(header))
    for row in rows:
        print("  ".join(str(row.get(c, "")).ljust(w) for c, w in zip(COLS, CWIDTHS)))


def run_all():
    summary: dict[str, list] = {s[0]: [] for s in STRATEGIES}

    for ticker, fname in FILES:
        path = Path(fname)
        if not path.exists():
            print(f"[!] Not found: {fname}")
            continue
        df = load_csv(path)
        print(f"\n{'='*70}")
        print(f"  {ticker}  |  {len(df)} candles  |  {df.index[0].date()} .. {df.index[-1].date()}")
        print(f"{'='*70}")

        rows = []
        for name, fn in STRATEGIES:
            try:
                r = run_engine(df, fn, name)
                rows.append(r.row(ticker))
                summary[name].append(r)
            except Exception as e:
                rows.append({"Strategy": name, "Trades": "ERR",
                             "WR%": str(e)[:50], **{c: "" for c in COLS[3:]}})
        print_table(rows)

    # ---------- Summary table ----------
    print(f"\n\n{'='*70}")
    print("  SUMMARY -- average across all tickers  (sorted by Sharpe desc)")
    print(f"{'='*70}")

    agg = []
    for name, res_list in summary.items():
        valid = [r for r in res_list if r.n > 0]
        if not valid:
            agg.append({"name": name, "trades": 0, "wr": 0, "ret": 0,
                        "sharpe": -99, "pf": 0, "dd": 0, "exp": 0})
            continue
        pfs = [r.profit_factor for r in valid if r.profit_factor != float("inf")]
        agg.append({
            "name":   name,
            "trades": sum(r.n for r in valid),
            "wr":     sum(r.win_rate for r in valid) / len(valid),
            "ret":    sum(r.total_return for r in valid) / len(valid),
            "sharpe": sum(r.sharpe for r in valid) / len(valid),
            "pf":     sum(pfs) / len(pfs) if pfs else 999.0,
            "dd":     sum(r.max_dd for r in valid) / len(valid),
            "exp":    sum(r.expectancy for r in valid) / len(valid),
        })
    agg.sort(key=lambda x: x["sharpe"], reverse=True)

    hcols = ["Strategy","Trades","Avg WR%","Avg Ret%","Avg Sharpe","Avg PF","Avg DD%","E/trade"]
    hwidths = [18, 7, 8, 9, 10, 7, 8, 10]
    print("  ".join(c.ljust(w) for c, w in zip(hcols, hwidths)))
    print("-" * 85)
    for a in agg:
        if a["trades"] == 0:
            vals = [a["name"], "0"] + ["--"] * 6
        else:
            vals = [
                a["name"], str(a["trades"]),
                f"{a['wr']*100:.0f}%",
                f"{a['ret']*100:.2f}%",
                f"{a['sharpe']:.3f}",
                f"{a['pf']:.2f}",
                f"{a['dd']*100:.1f}%",
                f"{a['exp']:+.1f}",
            ]
        print("  ".join(str(v).ljust(w) for v, w in zip(vals, hwidths)))

    best = next((a for a in agg if a["trades"] > 0), None)
    if best:
        print()
        print(f"  >>> WINNER by Sharpe: {best['name']}")
        print(f"      Sharpe={best['sharpe']:.3f}  WR={best['wr']*100:.0f}%  "
              f"Return={best['ret']*100:.2f}%  PF={best['pf']:.2f}  "
              f"MaxDD={best['dd']*100:.1f}%  E/trade={best['exp']:+.1f}")

    # ---------- vs classical strategies ----------
    print()
    print("  COMPARISON WITH CLASSICAL (from previous test):")
    print("  RSI Reversal (classical):  Sharpe=0.44  WR=58%  Return=+0.29%  PF=1.89")
    print("  Breakout-20  (classical):  Sharpe=0.14  WR=50%  Return=+0.52%  PF=2.15")
    print()
    print("Metrics legend:")
    print("  WR%     - win rate (% of profitable trades)")
    print("  Return% - net return on 100 000 initial capital")
    print("  Sharpe  - annualized Sharpe Ratio (sqrt(252) scale)")
    print("  PF      - Profit Factor: gross profit / gross loss")
    print("  E/trade - mathematical expectancy per trade (price units)")


if __name__ == "__main__":
    run_all()
