"""
Оптимизация параметров стратегии v2 на дневном таймфрейме.
Перебирает 15 наборов параметров × 4 схемы TP.
"""
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

# Кэшируем данные
print("Загружаю данные...", end=" ", flush=True)
DATA = {}
for t in TICKERS:
    p = DATA_DIR / f"{t}_2022_2026_D.csv"
    if not p.exists():
        continue
    try:
        df = pd.read_csv(p, sep=";")
        df.columns = [c.strip("<>").lower() for c in df.columns]
        df["dt"] = pd.to_datetime(df["date"].astype(str), format="%d/%m/%y", errors="coerce")
        df = (df.dropna(subset=["dt"]).set_index("dt")
                .rename(columns={"vol":"volume"})
               [["open","high","low","close","volume"]])
        for c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna()
        if len(df) >= 100:
            DATA[t] = df
    except Exception as e:
        print(f"\n  ОШИБКА {t}: {e}")
print(f"OK ({len(DATA)} тикеров)")


def run_grid(ms, er, adx, vol, sl, pb, scheme, mh):
    cfg = SignalConfig(
        min_score=ms, er_min=er, adx_min=adx, vol_ratio_min=vol,
        sl_atr_mult=sl, use_pullback=pb, sl_min_pct=0.025, sl_max_pct=0.08,
    )
    eng = BacktestEngineV2(scheme=scheme, max_hold=mh, cfg=cfg, timeframe="D")
    nt, nw, pcts, eq = 0, 0, [], [0.0]
    for t, df in DATA.items():
        r = eng.run(df, t)
        nt += r.trades
        nw += r.wins
        pcts.extend([x.pnl_pct for x in r.trade_list])
        for x in r.trade_list:
            eq.append(eq[-1] + x.pnl)
    if nt < 5:
        return None
    wr   = nw / nt
    pa   = np.array(pcts)
    sh   = float(np.mean(pa) / (np.std(pa) + 1e-9) * np.sqrt(252)) if len(pa) > 1 else 0
    wp   = [p for p in pcts if p > 0]
    lp   = [p for p in pcts if p <= 0]
    aw   = float(np.mean(wp)) if wp else 0
    al   = float(np.mean(lp)) if lp else 0
    pf   = abs(aw)*len(wp) / (abs(al)*len(lp) + 1e-9) if lp else 99.0
    exp  = wr * abs(aw) - (1 - wr) * abs(al)
    eq_a = np.array(eq)
    pk   = np.maximum.accumulate(eq_a)
    dd   = float((((eq_a - pk) / (pk + 1e-9)) * 100).min())
    sc   = wr*0.25 + min(pf/5, 1)*0.25 + min(sh/3, 1)*0.25 + min(max(exp, 0)/3, 1)*0.25
    return dict(ms=ms, er=er, adx=adx, vol=vol, sl=sl, pb=pb, scheme=scheme, mh=mh,
                trades=nt, wr=wr, pf=min(float(pf), 99.0),
                sharpe=sh, dd=dd, expect=float(exp), score=float(sc))


GRID = [
    # ms   er     adx  vol   sl    pb     scheme  mh
    (4, 0.30, 20, 1.2, 1.8, True,  "BAL", 20),
    (4, 0.28, 18, 1.0, 1.6, True,  "BAL", 25),
    (4, 0.28, 18, 1.0, 1.6, True,  "AGR", 25),
    (3, 0.25, 15, 1.0, 1.5, True,  "AGR", 30),
    (3, 0.25, 15, 1.0, 1.5, False, "BAL", 25),
    (4, 0.30, 20, 1.2, 1.8, True,  "AGR", 20),
    (4, 0.30, 20, 1.0, 1.8, False, "BAL", 20),
    (4, 0.25, 15, 1.0, 1.5, True,  "BAL", 30),
    (3, 0.22, 12, 0.8, 1.5, True,  "AGR", 30),
    (3, 0.25, 15, 1.0, 1.8, True,  "BAL", 25),
    (4, 0.30, 20, 1.2, 2.0, True,  "CONS",20),
    (4, 0.28, 18, 1.0, 1.8, False, "AGR", 25),
    (3, 0.22, 12, 0.8, 1.5, False, "BAL", 35),
    (4, 0.25, 15, 1.0, 1.5, False, "AGR", 25),
    (2, 0.20, 10, 0.8, 1.5, True,  "BAL", 30),
]

print()
print("=" * 82)
print("  ОПТИМИЗАЦИЯ: 15 наборов параметров × дневной ТФ")
print("=" * 82)
print(f"  {'#':>2} {'ms':>3} {'er':>5} {'adx':>4} {'vol':>4} {'sl':>4} {'pb':>5} "
      f"{'sch':>5} {'n':>4} {'WR%':>5} {'PF':>5} {'Sh':>6} {'DD%':>6} {'Exp%':>5} {'Score':>6}")
print("  " + "-"*80)

results = []
for i, (ms, er, adx, vol, sl, pb, sch, mh) in enumerate(GRID, 1):
    r = run_grid(ms, er, adx, vol, sl, pb, sch, mh)
    if not r:
        print(f"  {i:2d}  нет сделок")
        continue
    results.append(r)
    print(f"  {i:2d} {ms:3d} {er:5.2f} {adx:4d} {vol:4.1f} {sl:4.1f} "
          f"{str(pb):>5s} {sch:>5s} {r['trades']:>4d} {r['wr']*100:>4.0f}% "
          f"{r['pf']:>5.2f} {r['sharpe']:>6.2f} {r['dd']:>+5.1f}% "
          f"{r['expect']:>+4.2f}% {r['score']:>6.3f}")

if not results:
    print("Нет результатов.")
    exit()

results.sort(key=lambda x: x["score"], reverse=True)
best = results[0]

print()
print("=" * 82)
print("  ТОП-5:")
for r in results[:5]:
    print(f"  ms={r['ms']} er={r['er']:.2f} adx={r['adx']} vol={r['vol']:.1f} "
          f"sl={r['sl']:.1f} pb={r['pb']} {r['scheme']:5s} mh={r['mh']} → "
          f"WR={r['wr']*100:.0f}% PF={r['pf']:.2f} Sh={r['sharpe']:.2f} "
          f"Score={r['score']:.3f}")

print()
print("=" * 82)
print(f"  ПОБЕДИТЕЛЬ: scheme={best['scheme']} ms={best['ms']} er={best['er']:.2f} "
      f"adx={best['adx']} vol={best['vol']:.1f} sl={best['sl']:.1f} "
      f"pb={best['pb']} mh={best['mh']}")
print(f"  n={best['trades']}  WR={best['wr']*100:.0f}%  PF={best['pf']:.2f}  "
      f"Sharpe={best['sharpe']:.2f}  DD={best['dd']:+.1f}%  "
      f"Exp={best['expect']:+.2f}%  Score={best['score']:.3f}")
print("=" * 82)

out = {k: v for k, v in best.items() if k != "score"}
out["tf"] = "D"
with open(DATA_DIR / "best_config.json", "w") as f:
    json.dump(out, f, indent=2)
print("  Сохранено → best_config.json")
