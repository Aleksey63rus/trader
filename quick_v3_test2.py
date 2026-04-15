"""
Тест 2: Ищем лучший выход/SL при входе v2-style (ADX20, без weekday/kyle).
Фокус на: SL множители, схемы TP, max_hold.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
from core.strategy_v3 import BacktestEngineV3, SignalConfigV3
from pro_portfolio_v3 import load_daily

SAMPLE = ["LKOH", "NVTK", "PLZL", "MGNT", "NLMK", "CHMF", "TATN", "ROSN"]
data = {}
for t in SAMPLE:
    df = load_daily(t)
    if df is not None:
        data[t] = df

BASE_CFG = dict(adx_min=20, use_vol_delta=False, use_weekday=False, use_kyle_filter=False)

TESTS = [
    # name,                    sl_mult, sl_min, sl_max, swing, scheme, hold, trail
    ("BAL,sl1.8,sw5,hold20",   1.8, 0.025, 0.08, 5, "BAL", 20, 2.0),
    ("BAL,sl1.5,sw5,hold20",   1.5, 0.020, 0.07, 5, "BAL", 20, 2.0),
    ("BAL,sl2.0,sw5,hold20",   2.0, 0.025, 0.09, 5, "BAL", 20, 2.0),
    ("BAL,sl1.8,sw3,hold20",   1.8, 0.025, 0.08, 3, "BAL", 20, 2.0),
    ("PRO,sl1.8,sw5,hold25",   1.8, 0.025, 0.08, 5, "PRO", 25, 2.0),
    ("PRO,sl1.8,sw5,hold30",   1.8, 0.025, 0.08, 5, "PRO", 30, 2.0),
    ("BAL,sl1.8,sw5,hold25",   1.8, 0.025, 0.08, 5, "BAL", 25, 2.0),
    ("BAL,sl1.8,sw5,hold30",   1.8, 0.025, 0.08, 5, "BAL", 30, 2.0),
    ("ASYM,sl1.8,sw5,hold25",  1.8, 0.025, 0.08, 5, "ASYM", 25, 1.5),
    ("ASYM,sl1.8,sw5,hold30",  1.8, 0.025, 0.08, 5, "ASYM", 30, 1.5),
    ("AGR,sl2.0,sw5,hold25",   2.0, 0.025, 0.09, 5, "AGR", 25, 2.5),
    ("AGR,sl1.8,sw5,hold25",   1.8, 0.025, 0.08, 5, "AGR", 25, 2.0),
    ("CONS,sl1.5,sw3,hold20",  1.5, 0.020, 0.07, 3, "CONS", 20, 1.5),
]

print(f"{'Config':28s}  {'Tr':4s}  {'WR':6s}  {'Total%':8s}  {'SL_tr%':6s}  {'TM_tr%':6s}  {'Sharpe':6s}")
print("-" * 80)
best_score = -999; best = None
for name, sl_mult, sl_min, sl_max, sw, scheme, hold, trail in TESTS:
    cfg = SignalConfigV3(sl_atr_mult=sl_mult, sl_min_pct=sl_min, sl_max_pct=sl_max,
                         sl_swing_bars=sw, **BASE_CFG)
    total_tr = 0; total_w = 0; total_pct = 0; sharpes = []
    sl_cnt = 0; tm_cnt = 0
    for t, df in data.items():
        eng = BacktestEngineV3(scheme=scheme, max_hold=hold, cfg=cfg,
                               streak_limit=99, trailing_mult=trail)  # streak off
        r   = eng.run(df, t)
        total_tr  += r.trades
        total_w   += r.wins
        total_pct += r.total_pct
        sharpes.append(r.sharpe)
        sl_cnt += r.exit_counts.get("SL", 0)
        tm_cnt += r.exit_counts.get("TIME", 0)
    wr  = total_w / total_tr * 100 if total_tr else 0
    sl_p = sl_cnt / total_tr * 100 if total_tr else 0
    tm_p = tm_cnt / total_tr * 100 if total_tr else 0
    sh  = float(np.mean(sharpes)) if sharpes else 0
    score = wr * sh
    marker = " <--" if score > best_score and total_tr >= 20 else ""
    if score > best_score and total_tr >= 20:
        best_score = score; best = name
    print(f"{name:28s}  {total_tr:4d}  {wr:5.1f}%  {total_pct:+8.1f}%  {sl_p:5.0f}%  {tm_p:5.0f}%  {sh:6.2f}{marker}")

print(f"\nЛучший: {best}")
