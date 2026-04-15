"""
Full strategy comparison on fresh hourly (1H) MOEX data.
Runs both classical and advanced strategies on all 5 tickers.
"""
import sys
import math
import warnings
from pathlib import Path
from dataclasses import dataclass, field
import pandas as pd
warnings.filterwarnings("ignore")

# ── Re-use load_csv from engine ──────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from backtesting.engine import load_csv

# ── Re-import all strategy functions ─────────────────────────────────────────
from strategy_compare    import (strat_ema_cross, strat_rsi_reversal,
                                  strat_breakout, strat_macd, strat_donchian)
from advanced_strategies import (strat_liquidity_sweep, strat_volume_exhaustion,
                                  strat_adaptive_regime, strat_fvg, strat_composite)

COMMISSION = 0.0005
SLIPPAGE   = 0.0001

# ── Engine (same as advanced_strategies.py) ───────────────────────────────────
@dataclass
class Trade:
    entry_idx: int; entry_price: float; sl: float; tp: float
    exit_idx: int = 0; exit_price: float = 0.0; reason: str = ""
    @property
    def pnl(self): return (self.exit_price - self.entry_price
                           - (self.entry_price + self.exit_price) * COMMISSION)
    @property
    def win(self): return self.pnl > 0

@dataclass
class Result:
    name: str; trades: list = field(default_factory=list)
    equity: list = field(default_factory=list); initial: float = 100_000.0
    @property
    def n(self): return len(self.trades)
    @property
    def win_rate(self): return sum(1 for t in self.trades if t.win) / self.n if self.n else 0
    @property
    def total_pnl(self): return sum(t.pnl for t in self.trades)
    @property
    def total_return(self): return self.total_pnl / self.initial
    @property
    def max_dd(self):
        eq = pd.Series(self.equity)
        if len(eq) < 2: return 0.0
        return float(((eq - eq.cummax()) / eq.cummax()).min())
    @property
    def sharpe(self):
        eq = pd.Series(self.equity)
        r = eq.pct_change().dropna()
        # hourly data: annualise with sqrt(252*15) ~ sqrt(3780)
        ann = math.sqrt(252 * 15)
        return float(r.mean() / r.std() * ann) if len(r) > 1 and r.std() > 0 else 0.0
    @property
    def profit_factor(self):
        g = sum(t.pnl for t in self.trades if t.win)
        l = abs(sum(t.pnl for t in self.trades if not t.win))
        return round(g / l, 3) if l > 0 else float("inf")
    @property
    def expectancy(self):
        wr = self.win_rate
        aw = sum(t.pnl for t in self.trades if t.win) / max(1, sum(1 for t in self.trades if t.win))
        al = sum(t.pnl for t in self.trades if not t.win) / max(1, sum(1 for t in self.trades if not t.win))
        return wr * aw + (1 - wr) * al

def run_engine(df, signals_fn, name, initial=100_000.0):
    sig = signals_fn(df)
    result = Result(name=name, initial=initial)
    equity = initial; open_trade = None; used: set = set()
    closes = df["close"].values; highs = df["high"].values
    lows   = df["low"].values;   opens = df["open"].values
    n = len(df)
    for i in range(1, n):
        result.equity.append(equity)
        if open_trade:
            if lows[i] <= open_trade.sl:
                ep = max(open_trade.sl*(1-SLIPPAGE), lows[i])
                open_trade.exit_idx=i; open_trade.exit_price=ep; open_trade.reason="SL"
                equity += open_trade.pnl; result.trades.append(open_trade); open_trade=None
            elif highs[i] >= open_trade.tp:
                ep = open_trade.tp*(1-SLIPPAGE)
                open_trade.exit_idx=i; open_trade.exit_price=ep; open_trade.reason="TP"
                equity += open_trade.pnl; result.trades.append(open_trade); open_trade=None
        if not open_trade and i > 0 and bool(sig["signal"].iloc[i-1]):
            sl_v=float(sig["sl"].iloc[i-1]); tp_v=float(sig["tp"].iloc[i-1])
            entry=opens[i]*(1+SLIPPAGE); key=round(sl_v,2)
            if sl_v < entry < tp_v and key not in used:
                open_trade = Trade(i, entry, sl_v, tp_v); used.add(key)
    if open_trade:
        ep=closes[-1]*(1-SLIPPAGE); open_trade.exit_idx=n-1
        open_trade.exit_price=ep; open_trade.reason="END"
        equity+=open_trade.pnl; result.trades.append(open_trade)
    return result

# ── Strategy registry ─────────────────────────────────────────────────────────
CLASSICAL = [
    ("EMA Cross",      strat_ema_cross),
    ("RSI Reversal",   strat_rsi_reversal),
    ("Breakout-20",    strat_breakout),
    ("MACD",           strat_macd),
    ("Donchian-20",    strat_donchian),
]
ADVANCED = [
    ("Liq.Sweep",      strat_liquidity_sweep),
    ("Vol.Exhaust",    strat_volume_exhaustion),
    ("Adaptive ER",    strat_adaptive_regime),
    ("FVG Retest",     strat_fvg),
    ("Composite",      strat_composite),
]
ALL_STRATS = CLASSICAL + ADVANCED

# hourly files (downloaded from MOEX ISS)
FILES_1H = [
    ("SBER", "SBER_220103_260320_1H.csv"),
    ("ROSN", "ROSN_220103_260320_1H.csv"),
    ("LKOH", "LKOH_220103_260320_1H.csv"),
    ("MGNT", "MGNT_220103_260320_1H.csv"),
    ("YNDX", "YNDX_220103_240614_1H.csv"),
]

COLS   = ["Strategy","Trades","WR%","Return%","MaxDD%","Sharpe","PF","E/trade"]
WIDTHS = [15, 7, 6, 9, 7, 8, 6, 9]

def fmt_row(r: Result):
    pf = f"{r.profit_factor:.2f}" if r.profit_factor != float("inf") else "inf"
    return {
        "Strategy": r.name, "Trades": r.n,
        "WR%":     f"{r.win_rate*100:.0f}%",
        "Return%": f"{r.total_return*100:.2f}%",
        "MaxDD%":  f"{r.max_dd*100:.1f}%",
        "Sharpe":  f"{r.sharpe:.3f}",
        "PF":      pf,
        "E/trade": f"{r.expectancy:+.1f}",
    }

def print_table(rows):
    hdr = "  ".join(c.ljust(w) for c, w in zip(COLS, WIDTHS))
    print(hdr); print("-"*len(hdr))
    for row in rows:
        print("  ".join(str(row.get(c,"")).ljust(w) for c, w in zip(COLS, WIDTHS)))

def run_all():
    summary: dict[str, list] = {s[0]: [] for s in ALL_STRATS}

    for ticker, fname in FILES_1H:
        path = Path(fname)
        if not path.exists():
            print(f"[!] Missing: {fname}"); continue
        df = load_csv(path)
        # Keep only trading hours 07:00-23:00
        df = df.between_time("07:00", "23:00")
        print(f"\n{'='*68}")
        print(f"  {ticker}  |  {len(df)} hourly candles  "
              f"|  {df.index[0].date()} .. {df.index[-1].date()}")
        print(f"{'='*68}")

        rows = []
        for name, fn in ALL_STRATS:
            try:
                r = run_engine(df, fn, name)
                rows.append(fmt_row(r))
                summary[name].append(r)
            except Exception as e:
                rows.append({"Strategy": name, "Trades": "ERR",
                             **{c: str(e)[:20] for c in COLS[2:]}})
        print_table(rows)

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n\n{'='*68}")
    print("  SUMMARY (hourly 1H data, 2022-2026, sorted by Sharpe)")
    print(f"{'='*68}")

    agg = []
    for name, res_list in summary.items():
        valid = [r for r in res_list if r.n > 0]
        if not valid:
            agg.append({"name": name, "trades": 0, "sharpe": -99,
                        "wr": 0, "ret": 0, "pf": 0, "dd": 0, "exp": 0,
                        "group": "CLS" if name in [s[0] for s in CLASSICAL] else "ADV"})
            continue
        pfs = [r.profit_factor for r in valid if r.profit_factor != float("inf")]
        group = "CLS" if name in [s[0] for s in CLASSICAL] else "ADV"
        agg.append({
            "name":   name,
            "group":  group,
            "trades": sum(r.n for r in valid),
            "wr":     sum(r.win_rate for r in valid) / len(valid),
            "ret":    sum(r.total_return for r in valid) / len(valid),
            "sharpe": sum(r.sharpe for r in valid) / len(valid),
            "pf":     sum(pfs)/len(pfs) if pfs else 999.0,
            "dd":     sum(r.max_dd for r in valid) / len(valid),
            "exp":    sum(r.expectancy for r in valid) / len(valid),
        })
    agg.sort(key=lambda x: x["sharpe"], reverse=True)

    hcols  = ["#","Group","Strategy","Trades","Avg WR%","Avg Ret%","Sharpe","Avg PF","Avg DD%","E/trade"]
    hwidths = [3, 5, 15, 7, 8, 9, 8, 7, 8, 9]
    print("  ".join(c.ljust(w) for c,w in zip(hcols, hwidths)))
    print("-"*90)
    for rank, a in enumerate(agg, 1):
        pf_s = f"{a['pf']:.2f}" if a['pf'] != 999.0 else "inf"
        if a["trades"] == 0:
            vals = [str(rank), a["group"], a["name"], "0"] + ["--"]*6
        else:
            vals = [str(rank), a["group"], a["name"], str(a["trades"]),
                    f"{a['wr']*100:.0f}%", f"{a['ret']*100:.2f}%",
                    f"{a['sharpe']:.3f}", pf_s,
                    f"{a['dd']*100:.1f}%", f"{a['exp']:+.1f}"]
        print("  ".join(str(v).ljust(w) for v,w in zip(vals, hwidths)))

    top = [a for a in agg if a["trades"] > 0]
    if top:
        best = top[0]
        print(f"\n  WINNER: {best['name']} ({best['group']})")
        print(f"  Sharpe={best['sharpe']:.3f}  WR={best['wr']*100:.0f}%  "
              f"Return={best['ret']*100:.2f}%  PF={best['pf']:.2f}  "
              f"MaxDD={best['dd']*100:.1f}%  E/trade={best['exp']:+.1f}")

        best_adv = next((a for a in top if a["group"]=="ADV"), None)
        best_cls = next((a for a in top if a["group"]=="CLS"), None)
        print()
        if best_cls: print(f"  Best CLASSICAL : {best_cls['name']:<15} Sharpe={best_cls['sharpe']:.3f}")
        if best_adv: print(f"  Best ADVANCED  : {best_adv['name']:<15} Sharpe={best_adv['sharpe']:.3f}")

if __name__ == "__main__":
    run_all()
