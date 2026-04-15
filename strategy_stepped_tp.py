"""
STEPPED TP Strategy — ступенчатое снятие прибыли
=================================================

Идея: вместо одного TP делаем 4 уровня выхода.
После каждого частичного выхода SL перемещается вперёд (trailing по уровням).

Логика:
  ВХОД: 100% позиции
  TP1 (+1.0 * risk): закрыть 30% → SL переводим в безубыток (entry)
  TP2 (+1.8 * risk): закрыть ещё 30% → SL переводим на TP1
  TP3 (+3.0 * risk): закрыть ещё 25% → SL переводим на TP2
  TP4 (+5.0 * risk): закрыть последние 15% → максимальный выход

Преимущества:
  - Ранняя фиксация снижает количество сделок "вышли в ноль"
  - Оставшаяся часть "бежит" с трендом без риска убытка (SL в прибыли)
  - При сильном движении (TP4) максимизируем прибыль
  - Даже если SL отбивает остаток после TP1 — сделка всё равно прибыльна

Тестируем несколько схем долей:
  A: 30/30/25/15  (стандартная)
  B: 40/30/20/10  (агрессивная фиксация)
  C: 25/25/25/25  (равная)
  D: 50/20/20/10  (быстрая фиксация, "взял и вышел")
"""

import sys
import warnings
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Tuple
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))
from backtesting.engine import load_csv
from strategy_final import build_filter
from advanced_strategies import strat_adaptive_regime

COMMISSION = 0.0005
SLIPPAGE   = 0.0001
MAX_HOLD   = 96    # 4 торговых дня

FILES = [
    ("SBER", "SBER_220103_260320_1H.csv"),
    ("ROSN", "ROSN_220103_260320_1H.csv"),
    ("LKOH", "LKOH_220103_260320_1H.csv"),
    ("MGNT", "MGNT_220103_260320_1H.csv"),
    ("YNDX", "YNDX_220103_240614_1H.csv"),
]

# ── Indicators ────────────────────────────────────────────────────────────────
def _atr(df, p=14):
    h, l, pc = df["high"], df["low"], df["close"].shift(1)
    tr = pd.concat([h-l, (h-pc).abs(), (l-pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/p, min_periods=p, adjust=False).mean()

def _ema(s, n): return s.ewm(span=n, adjust=False).mean()

# ── Signal ─────────────────────────────────────────────────────────────────────
def get_signal(df: pd.DataFrame) -> pd.DataFrame:
    base_sig = strat_adaptive_regime(df)
    filt     = build_filter(df)
    combined = base_sig["signal"].astype(bool) & filt
    combined = combined & ~combined.shift(1).fillna(False)

    c     = df["close"]
    atr14 = _atr(df, 14)

    sl_raw = c - 1.8 * atr14
    sl     = sl_raw.clip(lower=c * 0.965, upper=c * 0.988)
    risk   = (c - sl).clip(lower=0.001)

    return pd.DataFrame({
        "signal": combined.astype(int),
        "sl": sl,
        "risk": risk,
    }, index=df.index)

# ── Backtest engine with stepped TP ───────────────────────────────────────────
@dataclass
class SteppedTrade:
    entry_idx: int
    entry: float
    sl: float
    risk: float
    entry_dt: object = None
    exit_dt: object = None
    exit_price: float = 0.0  # weighted avg exit
    reason: str = ""
    pnl: float = 0.0
    pnl_pct: float = 0.0
    hold_bars: int = 0
    win: bool = False
    # tracking partial exits
    remaining: float = 1.0          # fraction of position still open
    partial_pnl: float = 0.0        # accumulated from partial closes
    tp_levels_hit: int = 0          # how many TP levels triggered

@dataclass
class Result:
    ticker: str; trades: int; wins: int; losses: int
    wr: float; total_pnl: float; total_pct: float
    avg_win: float; avg_loss: float; pf: float
    max_dd: float; sharpe: float; expectancy: float
    tp_distribution: dict = field(default_factory=dict)
    trade_list: list = field(default_factory=list)

def run_stepped(df: pd.DataFrame, ticker: str,
                fracs: Tuple[float,...],
                rr_levels: Tuple[float,...]) -> Result:
    """
    fracs: tuple of fractions to close at each TP level (must sum to 1.0)
    rr_levels: R multiples for each TP (e.g. (1.0, 1.8, 3.0, 5.0))
    """
    assert len(fracs) == len(rr_levels)
    assert abs(sum(fracs) - 1.0) < 0.001

    sig    = get_signal(df)
    opens  = df["open"].values; highs  = df["high"].values
    lows   = df["low"].values;  closes = df["close"].values
    idx    = df.index; n = len(df)

    open_t: Optional[SteppedTrade] = None
    trades: list[SteppedTrade] = []
    used: set = set()
    equity: list[float] = [0.0]

    for i in range(1, n):
        if open_t:
            hold  = i - open_t.entry_idx
            entry = open_t.entry
            risk  = open_t.risk
            level = open_t.tp_levels_hit  # next TP to check

            # ─ Process partial TP hits for this bar ─────────────────────────
            bar_high = highs[i]
            bar_low  = lows[i]

            # Check each remaining TP level
            while level < len(rr_levels) and open_t.remaining > 0:
                tp_price = entry + rr_levels[level] * risk
                if bar_high >= tp_price:
                    ep       = tp_price * (1 - SLIPPAGE)
                    frac     = fracs[level]
                    real_frac= min(frac, open_t.remaining)  # clamp to what's left
                    partial  = (ep - entry) * real_frac - (entry + ep) * COMMISSION * real_frac
                    open_t.partial_pnl += partial
                    open_t.remaining   -= real_frac
                    # Move SL forward: to previous TP level (or breakeven for TP1)
                    if level == 0:
                        new_sl = entry * 1.001  # breakeven + tiny buffer
                    else:
                        new_sl = entry + rr_levels[level-1] * risk * 0.95
                    open_t.sl = max(open_t.sl, new_sl)
                    open_t.tp_levels_hit = level + 1
                    level = open_t.tp_levels_hit
                else:
                    break

            # If all TP levels hit → trade closed
            if open_t.remaining <= 0.001:
                pnl = open_t.partial_pnl
                t   = open_t
                t.exit_dt=idx[i]; t.exit_price=entry+rr_levels[-1]*risk
                t.reason=f"TP{len(rr_levels)}"; t.hold_bars=hold
                t.pnl=pnl; t.pnl_pct=pnl/entry*100; t.win=pnl>0
                equity.append(equity[-1]+pnl); trades.append(t); open_t=None; continue

            # ─ SL hit (on remaining position) ───────────────────────────────
            if open_t and bar_low <= open_t.sl:
                ep       = max(open_t.sl*(1-SLIPPAGE), bar_low)
                rem      = open_t.remaining
                rest_pnl = (ep - entry)*rem - (entry+ep)*COMMISSION*rem
                pnl      = open_t.partial_pnl + rest_pnl
                t        = open_t
                t.exit_dt=idx[i]; t.exit_price=ep; t.reason="SL"
                t.hold_bars=hold; t.pnl=pnl; t.pnl_pct=pnl/entry*100; t.win=pnl>0
                equity.append(equity[-1]+pnl); trades.append(t); open_t=None; continue

            # ─ Time exit ────────────────────────────────────────────────────
            if open_t and hold >= MAX_HOLD:
                ep       = opens[i]*(1-SLIPPAGE)
                rem      = open_t.remaining
                rest_pnl = (ep - entry)*rem - (entry+ep)*COMMISSION*rem
                pnl      = open_t.partial_pnl + rest_pnl
                t        = open_t
                t.exit_dt=idx[i]; t.exit_price=ep; t.reason="TIME"
                t.hold_bars=hold; t.pnl=pnl; t.pnl_pct=pnl/entry*100; t.win=pnl>0
                equity.append(equity[-1]+pnl); trades.append(t); open_t=None

        # ─ New entry ─────────────────────────────────────────────────────────
        if not open_t and i > 0:
            row = sig.iloc[i-1]
            if bool(row["signal"]):
                sl_v   = float(row["sl"]); risk_v = float(row["risk"])
                entry  = opens[i]*(1+SLIPPAGE)
                tp1_v  = entry + rr_levels[0]*risk_v
                key    = round(sl_v, 1)
                if sl_v < entry < tp1_v and risk_v > 0 and key not in used:
                    open_t = SteppedTrade(i, entry, sl_v, risk_v, entry_dt=idx[i])
                    used.add(key)

    if open_t:
        ep  = closes[-1]*(1-SLIPPAGE)
        rem = open_t.remaining
        rest= (ep - open_t.entry)*rem - (open_t.entry+ep)*COMMISSION*rem
        pnl = open_t.partial_pnl + rest
        t   = open_t
        t.exit_dt=idx[n-1]; t.exit_price=ep; t.reason="END"
        t.hold_bars=n-1-t.entry_idx; t.pnl=pnl; t.pnl_pct=pnl/t.entry*100; t.win=pnl>0
        equity.append(equity[-1]+pnl); trades.append(t)

    if not trades:
        return Result(ticker,0,0,0,0,0,0,0,0,0,0,0,0,{})

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

    # TP distribution
    tp_dist = {}
    for t in trades:
        tp_dist[t.reason] = tp_dist.get(t.reason, 0) + 1

    return Result(ticker,len(trades),len(wins_l),len(loss_l),
                  wr,total,total/avg_e*100,avg_win,avg_los,pf,max_dd,sharpe,expect,
                  tp_dist, trades)

# ── Run all schemes ────────────────────────────────────────────────────────────
SCHEMES = {
    "A 30/30/25/15": {
        "fracs":     (0.30, 0.30, 0.25, 0.15),
        "rr_levels": (1.0,  1.8,  3.0,  5.0),
    },
    "B 40/30/20/10": {
        "fracs":     (0.40, 0.30, 0.20, 0.10),
        "rr_levels": (1.0,  1.8,  3.0,  5.0),
    },
    "C 25/25/25/25": {
        "fracs":     (0.25, 0.25, 0.25, 0.25),
        "rr_levels": (1.0,  1.8,  3.0,  5.0),
    },
    "D 50/20/20/10": {
        "fracs":     (0.50, 0.20, 0.20, 0.10),
        "rr_levels": (1.0,  1.8,  3.0,  5.0),
    },
    "E 3 levels 40/35/25": {
        "fracs":     (0.40, 0.35, 0.25),
        "rr_levels": (1.0,  2.0,  4.0),
    },
    "F 3 levels 50/30/20": {
        "fracs":     (0.50, 0.30, 0.20),
        "rr_levels": (0.8,  1.8,  3.5),
    },
    "G early 60/40":       {
        "fracs":     (0.60, 0.40),
        "rr_levels": (0.8,  2.0),
    },
}

if __name__ == "__main__":
    print("\n" + "="*100)
    print("STEPPED TP — сравнение схем частичного закрытия позиции")
    print("="*100)

    scheme_summaries = []

    for scheme_name, params in SCHEMES.items():
        results = []
        total_t = total_w = 0

        for ticker, fname in FILES:
            path = Path(fname)
            if not path.exists(): continue
            df = load_csv(path).between_time("07:00","23:00")
            r  = run_stepped(df, ticker, params["fracs"], params["rr_levels"])
            results.append(r); total_t += r.trades; total_w += r.wins

        wr_all = total_w/total_t*100 if total_t else 0
        total_pct = np.mean([r.total_pct for r in results])
        avg_pf    = np.mean([r.pf for r in results])
        avg_sh    = np.mean([r.sharpe for r in results])

        scheme_summaries.append({
            "name": scheme_name, "wr": wr_all,
            "total_pct": total_pct, "pf": avg_pf,
            "sharpe": avg_sh, "trades": total_t,
            "results": results
        })

    # Sort by WR
    scheme_summaries.sort(key=lambda x: -x["wr"])

    print(f"\n  {'Схема':<22} {'Trades':>7} {'WR%':>6} {'Avg Total%':>11} {'Avg PF':>8} {'Avg Sharpe':>11}")
    print("  "+"-"*70)
    for s in scheme_summaries:
        flag = " *** BEST!" if s["wr"] >= 70 else (" OK" if s["wr"] >= 60 else "")
        print(f"  {s['name']:<22} {s['trades']:>7} {s['wr']:>5.0f}%  {s['total_pct']:>+10.1f}%  "
              f"{s['pf']:>8.2f}  {s['sharpe']:>10.2f}{flag}")

    # Detailed results for top scheme
    best = scheme_summaries[0]
    print(f"\n{'='*100}")
    print(f"ДЕТАЛЬНЫЕ РЕЗУЛЬТАТЫ — лучшая схема: {best['name']}")
    print(f"{'='*100}")
    rr_str = "/".join(str(x) for x in SCHEMES[best["name"]]["rr_levels"])
    frac_str = "/".join(f"{int(x*100)}%" for x in SCHEMES[best["name"]]["fracs"])
    print(f"  TP уровни (x Risk): {rr_str}  |  Доли закрытия: {frac_str}")
    print(f"\n  {'Ticker':<8} {'Trades':>7} {'WR%':>6} {'Total%':>8} {'AvgWin':>8} {'AvgLoss':>9} {'PF':>6} {'MaxDD%':>9} {'Sharpe':>8}")
    print("  "+"-"*78)

    for r in best["results"]:
        avg_e = float(np.mean([t.entry for t in r.trade_list])) if r.trade_list else 1
        print(f"  {r.ticker:<8} {r.trades:>7} {r.wr*100:>5.0f}%  {r.total_pct:>+7.1f}%  "
              f"{r.avg_win:>+8.2f} {r.avg_loss:>+9.2f} {r.pf:>6.2f}  "
              f"{r.max_dd/avg_e*100:>+8.1f}%  {r.sharpe:>8.2f}")

    # Exit distribution for best scheme
    print(f"\n  Распределение выходов ({best['name']}):")
    all_trades_best = []
    for r in best["results"]: all_trades_best.extend(r.trade_list)
    from collections import Counter
    dist = Counter(t.reason for t in all_trades_best)
    for reason, cnt in sorted(dist.items(), key=lambda x: -x[1]):
        sub = [t for t in all_trades_best if t.reason==reason]
        wr_r = sum(1 for t in sub if t.win)/len(sub)*100
        avg  = float(np.mean([t.pnl_pct for t in sub]))
        print(f"    {reason:<8}  n={cnt:>4}  WR={wr_r:.0f}%  avg={avg:+.2f}%")

    # ── Full per-ticker detail for top 2 schemes ─────────────────────────────
    print(f"\n{'='*100}")
    print("ПОДРОБНО: ТОП-2 схемы по каждому тикеру")
    print(f"{'='*100}")

    for s in scheme_summaries[:2]:
        print(f"\n  Схема: {s['name']}  (WR={s['wr']:.0f}%)")
        print(f"  {'Ticker':<8} {'WR%':>6} {'Total%':>8} {'PF':>6} {'Sharpe':>8}  Выходы")
        print("  "+"-"*70)
        for r in s["results"]:
            dist_str = "  ".join(f"{k}={v}" for k,v in sorted(r.tp_distribution.items()))
            print(f"  {r.ticker:<8} {r.wr*100:>5.0f}%  {r.total_pct:>+7.1f}%  {r.pf:>6.2f}  {r.sharpe:>8.2f}  {dist_str}")

    # ── Last 5 trades for best scheme ────────────────────────────────────────
    print(f"\n{'='*100}")
    print(f"5 ПОСЛЕДНИХ СДЕЛОК — {best['name']}")
    print(f"{'='*100}")
    for r in best["results"]:
        print(f"\n  {r.ticker}  n={r.trades}  WR={r.wr*100:.0f}%")
        print(f"  {'#':<5} {'Entry':<18} {'Exit':<18} {'Entr':>8} {'P&L':>8} {'%':>7} {'H':>4} {'Res':<6} {'TPs hit':>8}")
        print("  "+"-"*80)
        last5 = r.trade_list[-5:]
        for j, t in enumerate(last5, start=r.trades-len(last5)+1):
            print(f"  {j:<5} {str(t.entry_dt)[:17]:<18} {str(t.exit_dt)[:17]:<18} "
                  f"{t.entry:>8.2f} {t.pnl:>+8.2f} {t.pnl_pct:>+6.2f}%"
                  f" {t.hold_bars:>4} {t.reason:<6} {t.tp_levels_hit:>8}")
