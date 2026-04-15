"""
Сравнение таймфреймов 4H / 8H / 12H / D на стратегии v2.
Шаг 1: быстрый тест лучших параметров на каждом ТФ
Шаг 2: оптимизация на лучшем ТФ
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import pandas as pd
import json
from core.strategy_v2 import BacktestEngineV2, SignalConfig

DATA_DIR = Path("c:/investor/data")

TICKERS = [
    "GAZP","LKOH","NVTK","ROSN","SNGS","SNGSP",
    "SBER","SBERP","T","VTBR",
    "GMKN","NLMK","MTLR","CHMF","MAGN","RUAL","ALRS","PLZL",
    "YDEX","OZON","MGNT",
    "TATN","TATNP",
    "AFLT","TGKA","IRAO","MTSS","PHOR","OZPH",
]

TIMEFRAMES = {
    "4H":  ("4H",  36),
    "8H":  ("8H",  20),
    "12H": ("12H", 14),
    "D":   ("D",   20),
}

def load_df(ticker: str, tf: str) -> pd.DataFrame | None:
    path = DATA_DIR / f"{ticker}_2022_2026_{tf}.csv"
    if not path.exists(): return None
    try:
        df = pd.read_csv(path, sep=";")
        df.columns = [c.strip("<>").lower() for c in df.columns]
        if tf == "D":
            df["dt"] = pd.to_datetime(df["date"].astype(str), format="%d/%m/%y", errors="coerce")
        else:
            df["dt"] = pd.to_datetime(
                df["date"].astype(str) + " " + df["time"].astype(str).str.zfill(6),
                format="%d/%m/%y %H%M%S", errors="coerce")
        df = (df.dropna(subset=["dt"]).set_index("dt")
                .rename(columns={"vol":"volume"})
               [["open","high","low","close","volume"]])
        for c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna()
        return df if len(df) >= 100 else None
    except: return None

def run_portfolio(tf: str, max_hold: int, scheme: str, cfg: SignalConfig) -> dict:
    engine = BacktestEngineV2(scheme=scheme, max_hold=max_hold, cfg=cfg, timeframe=tf)
    all_trades, all_wins, pct_list, eq = 0, 0, [], [0.0]
    for ticker in TICKERS:
        df = load_df(ticker, tf)
        if df is None: continue
        r = engine.run(df, ticker)
        all_trades += r.trades; all_wins += r.wins
        pct_list.extend([t.pnl_pct for t in r.trade_list])
        for t in r.trade_list: eq.append(eq[-1] + t.pnl)
    if all_trades < 5: return {}
    wr   = all_wins / all_trades
    pcts = np.array(pct_list)
    sh   = float(np.mean(pcts)/(np.std(pcts)+1e-9)*np.sqrt(252)) if len(pcts)>1 else 0
    wp   = [p for p in pct_list if p>0]; lp = [p for p in pct_list if p<=0]
    aw   = float(np.mean(wp)) if wp else 0
    al   = float(np.mean(lp)) if lp else 0
    pf   = abs(aw)*len(wp)/(abs(al)*len(lp)+1e-9) if lp else 99
    exp  = wr*abs(aw)-(1-wr)*abs(al)
    eq_a = np.array(eq); pk = np.maximum.accumulate(eq_a)
    dd   = float((((eq_a-pk)/(pk+1e-9))*100).min())
    score = wr*0.25 + min(pf/5,1)*0.25 + min(sh/3,1)*0.25 + min(max(exp,0)/3,1)*0.25
    return {"tf":tf,"scheme":scheme,"trades":all_trades,"wr":wr,
            "pf":min(pf,99),"sharpe":sh,"dd":dd,"expect":exp,"score":score}

# ─────────────────────────────────────────────────────────
# ШАГ 1: быстрое сравнение ТФ с базовыми параметрами
# ─────────────────────────────────────────────────────────
print("="*68)
print("  ШАГ 1: Сравнение таймфреймов (базовые параметры)")
print("="*68)

base_cfg = SignalConfig(min_score=4, er_min=0.30, adx_min=20,
                        vol_ratio_min=1.2, sl_atr_mult=1.8, use_pullback=True)
tf_results = []
for tf, (_, max_hold) in TIMEFRAMES.items():
    r = run_portfolio(tf, max_hold, "BAL", base_cfg)
    if r:
        tf_results.append(r)
        print(f"  {tf:4s}  n={r['trades']:4d}  WR={r['wr']*100:4.0f}%  "
              f"PF={r['pf']:4.2f}  Sh={r['sharpe']:5.2f}  "
              f"DD={r['dd']:+5.1f}%  Exp={r['expect']:+4.2f}%  Score={r['score']:.3f}")

tf_results.sort(key=lambda x: x["score"], reverse=True)
best_tf = tf_results[0]["tf"]
best_max_hold = TIMEFRAMES[best_tf][1]
print(f"\n  ► Лучший ТФ: {best_tf}")

# ─────────────────────────────────────────────────────────
# ШАГ 2: оптимизация на лучшем ТФ
# ─────────────────────────────────────────────────────────
print(f"\n{'='*68}")
print(f"  ШАГ 2: Оптимизация параметров на {best_tf}")
print(f"{'='*68}")

PARAM_GRID = [
    # (min_score, er_min, adx_min, vol_min, sl_mult, use_pb, scheme)
    (4, 0.30, 20, 1.2, 1.8, True,  "BAL"),
    (4, 0.30, 20, 1.0, 1.8, False, "BAL"),
    (4, 0.28, 18, 1.0, 1.6, True,  "BAL"),
    (3, 0.28, 18, 1.0, 1.6, True,  "AGR"),
    (4, 0.30, 20, 1.2, 1.8, True,  "AGR"),
    (4, 0.32, 22, 1.3, 2.0, True,  "BAL"),
    (4, 0.30, 20, 1.2, 1.5, True,  "CONS"),
    (3, 0.25, 15, 1.0, 1.5, True,  "BAL"),
    (4, 0.30, 20, 1.2, 1.8, True,  "FAST"),
    (4, 0.28, 20, 1.0, 1.8, False, "AGR"),
    (5, 0.35, 25, 1.5, 1.8, True,  "BAL"),
    (4, 0.30, 18, 1.0, 1.6, True,  "AGR"),
]

opt_results = []
for i, (ms, er, adx, vol, sl, pb, scheme) in enumerate(PARAM_GRID, 1):
    cfg = SignalConfig(min_score=ms, er_min=er, adx_min=adx,
                      vol_ratio_min=vol, sl_atr_mult=sl, use_pullback=pb)
    r = run_portfolio(best_tf, best_max_hold, scheme, cfg)
    if not r: continue
    r["params"] = (ms, er, adx, vol, sl, pb, scheme)
    opt_results.append(r)
    print(f"  [{i:2d}] sc{ms} er{er:.2f} adx{adx} vol{vol:.1f} sl{sl:.1f} "
          f"pb={pb} {scheme:5s} → "
          f"n={r['trades']:4d} WR={r['wr']*100:4.0f}% PF={r['pf']:4.2f} "
          f"Sh={r['sharpe']:5.2f} Score={r['score']:.3f}")

opt_results.sort(key=lambda x: x["score"], reverse=True)
best = opt_results[0]
ms, er, adx, vol, sl, pb, scheme = best["params"]

print(f"\n{'='*68}")
print(f"  ПОБЕДИТЕЛЬ: {best_tf} / {scheme}")
print(f"  Параметры: min_score={ms} er_min={er} adx_min={adx} "
      f"vol_min={vol} sl_mult={sl} pullback={pb}")
print(f"  n={best['trades']} WR={best['wr']*100:.0f}% PF={best['pf']:.2f} "
      f"Sharpe={best['sharpe']:.2f} DD={best['dd']:+.1f}% Score={best['score']:.3f}")
print(f"{'='*68}")

cfg_out = {"tf": best_tf, "max_hold": best_max_hold, "scheme": scheme,
           "min_score": ms, "er_min": er, "adx_min": adx,
           "vol_ratio_min": vol, "sl_atr_mult": sl, "use_pullback": pb}
with open(DATA_DIR / "best_config.json", "w") as f:
    json.dump(cfg_out, f, indent=2)
print("  Конфиг сохранён → best_config.json")
