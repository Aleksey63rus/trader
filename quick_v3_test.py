"""
Быстрый тест v3 на 5 тикерах — проверяем ключевые улучшения по отдельности.
Kyle Lambda убран из теста (слишком медленный на Python loops).
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
print(f"Тестируем на {len(data)} тикерах\n")

TESTS = [
    # Название,              adx, vol_delta, weekday, scheme, hold
    ("v2-style(ADX20,BAL,20)", 20,  False, False, "BAL", 20),
    ("ADX20+vd, BAL, 20",      20,  True,  False, "BAL", 20),
    ("ADX20+vd+wd, BAL, 20",   20,  True,  True,  "BAL", 20),
    ("ADX22+vd+wd, BAL, 20",   22,  True,  True,  "BAL", 20),
    ("ADX22+vd+wd, PRO, 25",   22,  True,  True,  "PRO", 25),
    ("ADX22+vd+wd, PRO, 30",   22,  True,  True,  "PRO", 30),
    ("ADX20+vd+wd, PRO, 30",   20,  True,  True,  "PRO", 30),
    ("ADX20+vd+wd, AGR, 30",   20,  True,  True,  "AGR", 30),
    ("ADX22+vd, PRO, 30",      22,  True,  False, "PRO", 30),
    ("ADX20+vd, PRO, 30",      20,  True,  False, "PRO", 30),
    ("ADX20+vd, ASYM, 30",     20,  True,  False, "ASYM", 30),
    ("ADX20+vd, AGR, 25",      20,  True,  False, "AGR", 25),
]

print(f"{'Config':30s}  {'Tr':4s}  {'WR':6s}  {'Total%':8s}  {'SL%':5s}  {'Sharpe':6s}")
print("-" * 70)
best = None; best_score = -999
for name, adx, vd, wd, scheme, hold in TESTS:
    cfg = SignalConfigV3(adx_min=adx, use_vol_delta=vd, use_weekday=wd,
                         use_kyle_filter=False)  # kyle выключен для скорости
    total_tr = 0; total_w = 0; total_pct = 0; sharpes = []; sl_cnt = 0
    for t, df in data.items():
        eng = BacktestEngineV3(scheme=scheme, max_hold=hold, cfg=cfg, streak_limit=3)
        r   = eng.run(df, t)
        total_tr  += r.trades
        total_w   += r.wins
        total_pct += r.total_pct
        sharpes.append(r.sharpe)
        sl_cnt += r.exit_counts.get("SL", 0)
    wr  = total_w / total_tr * 100 if total_tr else 0
    sl_pct = sl_cnt / total_tr * 100 if total_tr else 0
    sh  = float(np.mean(sharpes)) if sharpes else 0
    score = wr * sh  # WR × Sharpe
    marker = " <-- BEST" if score > best_score and total_tr >= 20 else ""
    if score > best_score and total_tr >= 20:
        best_score = score
        best = (name, adx, vd, wd, scheme, hold)
    print(f"{name:30s}  {total_tr:4d}  {wr:5.1f}%  {total_pct:+8.1f}%  {sl_pct:4.0f}%  {sh:6.2f}{marker}")

print(f"\nЛучший: {best}")
