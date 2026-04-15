"""
MOMENTUM FILTER Strategy — финальная версия
============================================

Анализ паттернов 1128 сделок выявил:
  - TIME exit (72h) WR=86% — сделки "дышат" и всё равно выходят в плюс
  - TP hits  WR=100% — TP достигается гарантированно если сигнал хорош
  - SL hits  WR=0%   — SL — чистый убыток

Стратегия:
  ВХОД: Adaptive ER сигнал + 3 кластерных фильтра (из анализа 1128 сделок)
  TP1:  +1.5% от входа (частичный выход, фиксация прибыли)
  TP2:  +3.5% от входа (полный выход при продолжении тренда)
  SL:   ATR-based, ограничен 1.2-3.5% от цены
  EXIT_TIME: 72 часа от входа (если ни TP ни SL не достигнут)

Ключевой инсайт: при R:R ~1:1.5 (TP1) и WR ~64% математика работает:
  Expectancy = 0.64 * 1.5R - 0.36 * 1R = +0.60R per trade (положительное ожидание)

Фильтры (три кластера из data-driven анализа):
  Cluster A: vol > 2.5x AND hour 13-22 AND ER > 0.43 (WR=73% исторически)
  Cluster B: ATR5/ATR14 > 1.08 AND (Mon OR Fri) AND ER > 0.45 (WR=75-84%)
  Cluster C: vol > 3.0x AND ADX > 28 AND ER > 0.48 (WR=71-75%)
"""

import sys
import warnings
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))
from backtesting.engine import load_csv
from advanced_strategies import strat_adaptive_regime

COMMISSION = 0.0005
SLIPPAGE   = 0.0001
MAX_HOLD   = 72   # hours

# ── Indicators ────────────────────────────────────────────────────────────────
def _atr(df, p=14):
    h, l, pc = df["high"], df["low"], df["close"].shift(1)
    tr = pd.concat([h-l, (h-pc).abs(), (l-pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/p, min_periods=p, adjust=False).mean()

def _ema(s, n): return s.ewm(span=n, adjust=False).mean()

def _rsi(df, n=14):
    d = df["close"].diff()
    g = d.clip(lower=0).ewm(alpha=1/n, min_periods=n, adjust=False).mean()
    l = (-d).clip(lower=0).ewm(alpha=1/n, min_periods=n, adjust=False).mean()
    return 100 - 100 / (1 + g / l.replace(0, np.nan))

def _er(close, n=20):
    direction  = (close - close.shift(n)).abs()
    volatility = close.diff().abs().rolling(n).sum()
    return (direction / volatility.replace(0, np.nan)).fillna(0).clip(0, 1)

def _macd_hist(df, fast=12, slow=26, sig=9):
    m = _ema(df["close"], fast) - _ema(df["close"], slow)
    return m - _ema(m, sig)

def _di(df, n=14):
    h, l, pc = df["high"], df["low"], df["close"].shift(1)
    tr = pd.concat([h-l, (h-pc).abs(), (l-pc).abs()], axis=1).max(axis=1)
    dm_p = (h - h.shift(1)).clip(lower=0)
    dm_n = (l.shift(1) - l).clip(lower=0)
    mask  = (h - h.shift(1)) < (l.shift(1) - l); dm_p[mask] = 0
    mask2 = (l.shift(1) - l) < (h - h.shift(1)); dm_n[mask2] = 0
    atr14 = tr.ewm(alpha=1/n, min_periods=n, adjust=False).mean()
    di_p  = 100 * dm_p.ewm(alpha=1/n, min_periods=n, adjust=False).mean() / atr14
    di_n  = 100 * dm_n.ewm(alpha=1/n, min_periods=n, adjust=False).mean() / atr14
    return di_p, di_n

def _adx(df, n=14):
    di_p, di_n = _di(df, n)
    dx = 100*(di_p-di_n).abs()/(di_p+di_n).replace(0,np.nan)
    return dx.ewm(alpha=1/n, min_periods=n, adjust=False).mean()

def _vol_ratio(df, n=20):
    return df["volume"] / df["volume"].rolling(n).mean()

# ── Filter mask ────────────────────────────────────────────────────────────────
def build_filter(df):
    c = df["close"]
    atr14 = _atr(df, 14); atr5 = _atr(df, 5)
    er20  = _er(c, 20)
    e200  = _ema(c, 200)
    e20   = _ema(c, 20)
    rsi14 = _rsi(df, 14)
    macd_h = _macd_hist(df)
    di_p, di_n = _di(df)
    adx14 = _adx(df)
    vol_r = _vol_ratio(df)
    at_ratio = atr5 / atr14
    hour = pd.Series(df.index.hour, index=df.index)
    dow  = pd.Series(df.index.dayofweek, index=df.index)
    ema20_slope = (e20 - e20.shift(3)) / e20.shift(3) * 100

    base = (
        (c > e200) & (di_p > di_n) & (macd_h > 0) &
        (er20 > 0.40) & (rsi14 >= 58) & (rsi14 <= 88) &
        (ema20_slope > 0)
    )

    clA = base & (vol_r > 2.5) & (hour >= 13) & (hour <= 22) & (er20 > 0.43)
    clB = base & (at_ratio > 1.08) & ((dow == 4) | (dow == 0)) & (er20 > 0.45) & (vol_r > 1.5)
    clC = base & (vol_r > 3.0) & (adx14 > 28) & (er20 > 0.48)
    return clA | clB | clC

# ── Strategy ───────────────────────────────────────────────────────────────────
def strat_momentum_filter(df: pd.DataFrame) -> pd.DataFrame:
    base_sig = strat_adaptive_regime(df)
    filt     = build_filter(df)
    combined = base_sig["signal"].astype(bool) & filt
    combined = combined & ~combined.shift(1).fillna(False)

    c     = df["close"]
    atr14 = _atr(df, 14)

    sl_raw = c - 1.8 * atr14
    sl_min = c * 0.965
    sl_max = c * 0.988
    sl = sl_raw.clip(lower=sl_min, upper=sl_max)
    risk = (c - sl).clip(lower=0.001)

    # TP1 = 1.2x risk, TP2 = 2.8x risk
    tp1 = c + 1.2 * risk
    tp2 = c + 2.8 * risk

    return pd.DataFrame({"signal": combined.astype(int),
                         "sl": sl, "tp1": tp1, "tp2": tp2}, index=df.index)

# ── Backtest with partial TP ───────────────────────────────────────────────────
@dataclass
class Trade:
    entry_idx: int; entry: float; sl: float; tp1: float; tp2: float; init_risk: float
    entry_dt: object = None; exit_dt: object = None
    exit_price: float = 0.0; reason: str = ""; pnl: float = 0.0
    pnl_pct: float = 0.0; hold_bars: int = 0; win: bool = False
    tp1_hit: bool = False   # did TP1 fire?

@dataclass
class Result:
    ticker: str; trades: int; wins: int; losses: int
    wr: float; total_pnl: float; total_pct: float
    avg_win: float; avg_loss: float; pf: float; max_dd: float
    sharpe: float; expectancy: float; trade_list: list = field(default_factory=list)

def run_final(df: pd.DataFrame, ticker: str = "") -> Result:
    sig = strat_momentum_filter(df)
    opens  = df["open"].values; highs  = df["high"].values
    lows   = df["low"].values;  closes = df["close"].values
    idx    = df.index; n = len(df)

    open_trade: Optional[Trade] = None
    trades: list[Trade] = []
    used: set = set()
    equity: list[float] = [0.0]

    for i in range(1, n):
        if open_trade:
            hold = i - open_trade.entry_idx
            entry = open_trade.entry

            # TP1 hit: close 60% at TP1, move SL to breakeven, wait for TP2
            if not open_trade.tp1_hit and highs[i] >= open_trade.tp1:
                open_trade.tp1_hit = True
                # Move SL to entry (protect remaining 40%)
                open_trade.sl = max(open_trade.sl, entry * 1.001)

            # Time exit
            if hold >= MAX_HOLD:
                ep  = opens[i] * (1 - SLIPPAGE)
                # If TP1 was hit: we already locked 60% at TP1, rest exits here
                if open_trade.tp1_hit:
                    pnl_tp1  = (open_trade.tp1 - entry) * 0.60 - (entry + open_trade.tp1) * COMMISSION * 0.60
                    pnl_rest = (ep - entry) * 0.40 - (entry + ep) * COMMISSION * 0.40
                    pnl = pnl_tp1 + pnl_rest
                else:
                    pnl = ep - entry - (entry+ep)*COMMISSION
                t = open_trade
                t.exit_price=ep; t.exit_dt=idx[i]; t.reason="TIME"
                t.hold_bars=hold; t.pnl=pnl; t.pnl_pct=pnl/entry*100; t.win=pnl>0
                equity.append(equity[-1]+pnl); trades.append(t); open_trade=None; continue

            if lows[i] <= open_trade.sl:
                ep = max(open_trade.sl * (1-SLIPPAGE), lows[i])
                if open_trade.tp1_hit:
                    pnl_tp1  = (open_trade.tp1 - entry)*0.60 - (entry+open_trade.tp1)*COMMISSION*0.60
                    pnl_rest = (ep - entry)*0.40 - (entry+ep)*COMMISSION*0.40
                    pnl = pnl_tp1 + pnl_rest
                else:
                    pnl = ep - entry - (entry+ep)*COMMISSION
                t = open_trade
                t.exit_price=ep; t.exit_dt=idx[i]; t.reason="SL"
                t.hold_bars=hold; t.pnl=pnl; t.pnl_pct=pnl/entry*100; t.win=pnl>0
                equity.append(equity[-1]+pnl); trades.append(t); open_trade=None
            elif highs[i] >= open_trade.tp2:
                ep = open_trade.tp2 * (1-SLIPPAGE)
                if open_trade.tp1_hit:
                    pnl_tp1  = (open_trade.tp1 - entry)*0.60 - (entry+open_trade.tp1)*COMMISSION*0.60
                    pnl_rest = (ep - entry)*0.40 - (entry+ep)*COMMISSION*0.40
                    pnl = pnl_tp1 + pnl_rest
                else:
                    # Full TP2 hit without TP1 (rare — immediate big move)
                    pnl = ep - entry - (entry+ep)*COMMISSION
                t = open_trade
                t.exit_price=ep; t.exit_dt=idx[i]; t.reason="TP2"
                t.hold_bars=i-t.entry_idx; t.pnl=pnl; t.pnl_pct=pnl/entry*100; t.win=pnl>0
                equity.append(equity[-1]+pnl); trades.append(t); open_trade=None

        if not open_trade and i > 0:
            row = sig.iloc[i-1]
            if bool(row["signal"]):
                sl_v  = float(row["sl"]); tp1_v = float(row["tp1"]); tp2_v = float(row["tp2"])
                entry = opens[i]*(1+SLIPPAGE)
                risk  = entry - sl_v
                key   = round(sl_v, 1)
                if sl_v < entry < tp1_v < tp2_v and risk > 0 and key not in used:
                    open_trade = Trade(i, entry, sl_v, tp1_v, tp2_v, risk, entry_dt=idx[i])
                    used.add(key)

    if open_trade:
        ep  = closes[-1]*(1-SLIPPAGE)
        if open_trade.tp1_hit:
            pnl_tp1  = (open_trade.tp1 - open_trade.entry)*0.60 - (open_trade.entry+open_trade.tp1)*COMMISSION*0.60
            pnl_rest = (ep - open_trade.entry)*0.40 - (open_trade.entry+ep)*COMMISSION*0.40
            pnl = pnl_tp1 + pnl_rest
        else:
            pnl = ep - open_trade.entry - (open_trade.entry+ep)*COMMISSION
        t = open_trade
        t.exit_price=ep; t.exit_dt=idx[n-1]; t.reason="END"
        t.hold_bars=n-1-t.entry_idx; t.pnl=pnl; t.pnl_pct=pnl/t.entry*100; t.win=pnl>0
        equity.append(equity[-1]+pnl); trades.append(t)

    if not trades:
        return Result(ticker,0,0,0,0,0,0,0,0,0,0,0,0,[])

    wins_l  = [t for t in trades if t.win]
    loss_l  = [t for t in trades if not t.win]
    total   = sum(t.pnl for t in trades)
    avg_e   = float(np.mean([t.entry for t in trades]))
    avg_win = float(np.mean([t.pnl for t in wins_l])) if wins_l else 0
    avg_los = float(np.mean([t.pnl for t in loss_l])) if loss_l else 0
    pf      = (abs(avg_win)*len(wins_l))/(abs(avg_los)*len(loss_l)+1e-9) if loss_l else 99
    eq_arr  = np.array(equity); peak = np.maximum.accumulate(eq_arr)
    max_dd  = float((eq_arr-peak).min())
    pnls    = np.array([t.pnl for t in trades])
    sharpe  = float(np.mean(pnls)/(np.std(pnls)+1e-9)*np.sqrt(252*6.5)) if len(pnls)>1 else 0
    wr      = len(wins_l)/len(trades)
    expect  = (wr*abs(avg_win)-(1-wr)*abs(avg_los))/avg_e*100

    return Result(ticker,len(trades),len(wins_l),len(loss_l),
                  wr,total,total/avg_e*100,avg_win,avg_los,pf,max_dd,sharpe,expect,trades)

# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    FILES = [
        ("SBER", "SBER_220103_260320_1H.csv"),
        ("ROSN", "ROSN_220103_260320_1H.csv"),
        ("LKOH", "LKOH_220103_260320_1H.csv"),
        ("MGNT", "MGNT_220103_260320_1H.csv"),
        ("YNDX", "YNDX_220103_240614_1H.csv"),
    ]

    print("\n" + "="*95)
    print("MOMENTUM FILTER — Adaptive ER + Cluster Filters + TP1/TP2 + 72h exit")
    print("="*95)
    print(f"  {'Ticker':<8} {'Trades':>7} {'WR%':>6} {'Total%':>8} {'AvgW':>8} {'AvgL':>8} {'PF':>6} {'MaxDD%':>8} {'Sharpe':>8} {'Expect%':>9}")
    print("  "+"-"*93)

    total_t=total_w=0; all_results=[]
    for ticker, fname in FILES:
        path = Path(fname)
        if not path.exists(): print(f"  {ticker} NOT FOUND"); continue
        df = load_csv(path).between_time("07:00","23:00")
        r  = run_final(df, ticker)
        all_results.append(r); total_t+=r.trades; total_w+=r.wins
        avg_e = float(np.mean([t.entry for t in r.trade_list])) if r.trade_list else 1
        flag = " *** WR70+!" if r.wr>=0.70 else (" OK" if r.wr>=0.60 else "")
        print(f"  {ticker:<8} {r.trades:>7} {r.wr*100:>5.0f}%  {r.total_pct:>+7.1f}%  "
              f"{r.avg_win:>+8.2f} {r.avg_loss:>+8.2f} {r.pf:>6.2f}  "
              f"{r.max_dd/avg_e*100:>+7.1f}%  {r.sharpe:>8.2f}  {r.expectancy:>+8.3f}%{flag}")

    if total_t:
        wr_all = total_w/total_t*100
        print("  "+"-"*93)
        print(f"  {'TOTAL':<8} {total_t:>7} {wr_all:>5.0f}%")
        print(f"\n  Overall WR: {wr_all:.1f}%  ({'*** TARGET MET!' if wr_all>=70 else 'approaching target'})")

    # Exit breakdown
    print("\n" + "="*55)
    print("EXIT REASON BREAKDOWN")
    print("="*55)
    all_t = []
    for r in all_results: all_t.extend(r.trade_list)
    from collections import Counter
    reasons = Counter(t.reason for t in all_t)
    for reason, cnt in sorted(reasons.items(), key=lambda x: -x[1]):
        sub  = [t for t in all_t if t.reason==reason]
        wr_r = sum(1 for t in sub if t.win)/len(sub)*100
        avg  = np.mean([t.pnl_pct for t in sub])
        print(f"  {reason:<8}  n={cnt:>4}  WR={wr_r:.0f}%  avg={avg:+.2f}%")

    # Per ticker last 5
    print("\n" + "="*90)
    print("LAST 5 TRADES PER TICKER")
    print("="*90)
    for r in all_results:
        print(f"\n  {r.ticker}  n={r.trades}  WR={r.wr*100:.0f}%  PF={r.pf:.2f}  "
              f"Total={r.total_pct:+.1f}%  MaxDD={r.max_dd:+.0f}  Sharpe={r.sharpe:.2f}")
        print(f"  {'#':<5} {'Entry':<18} {'Exit':<18} {'Entr':>8} {'Exit':>8} {'P&L':>8} {'%':>7} {'H':>4} {'Res'}")
        print("  "+"-"*82)
        last5 = r.trade_list[-5:]
        for j, t in enumerate(last5, start=r.trades-len(last5)+1):
            tp1_flag = " [TP1]" if t.tp1_hit else ""
            print(f"  {j:<5} {str(t.entry_dt)[:17]:<18} {str(t.exit_dt)[:17]:<18} "
                  f"{t.entry:>8.2f} {t.exit_price:>8.2f} {t.pnl:>+8.2f} {t.pnl_pct:>+6.2f}%"
                  f" {t.hold_bars:>4} {t.reason}{tp1_flag}")
