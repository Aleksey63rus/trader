"""
Оптимизация параметров strategy_v3.
Находим лучшие комбинации ADX / kyle / weekday / TP scheme.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
from core.strategy_v3 import BacktestEngineV3, SignalConfigV3
from pro_portfolio_v3 import load_daily, TICKERS

data = {}
for t in TICKERS:
    df = load_daily(t)
    if df is not None:
        data[t] = df
print(f"Загружено: {len(data)} тикеров\n")

CONFIGS = [
    ("ADX25+K+WD",  dict(adx_min=25, use_kyle_filter=True,  use_weekday=True)),
    ("ADX22+K+WD",  dict(adx_min=22, use_kyle_filter=True,  use_weekday=True)),
    ("ADX20+K+WD",  dict(adx_min=20, use_kyle_filter=True,  use_weekday=True)),
    ("ADX22+WD",    dict(adx_min=22, use_kyle_filter=False, use_weekday=True)),
    ("ADX20+WD",    dict(adx_min=20, use_kyle_filter=False, use_weekday=True)),
    ("ADX22",       dict(adx_min=22, use_kyle_filter=False, use_weekday=False)),
    ("ADX20",       dict(adx_min=20, use_kyle_filter=False, use_weekday=False)),
    ("ADX18",       dict(adx_min=18, use_kyle_filter=False, use_weekday=False)),
]

SCHEMES = ["PRO", "BAL", "AGR", "CONS"]
HOLDS   = [20, 25, 30]

best_score = -9999
best_combo = None
results = []

for cfg_name, kw in CONFIGS:
    for scheme in SCHEMES:
        for mh in HOLDS:
            cfg = SignalConfigV3(**kw)
            total_pct = 0; total_tr = 0; total_w = 0; sharpes = []
            wins_sum = 0; losses_sum = 0
            for t, df in data.items():
                eng = BacktestEngineV3(scheme=scheme, max_hold=mh, cfg=cfg, streak_limit=3)
                r   = eng.run(df, t)
                total_pct += r.total_pct
                total_tr  += r.trades
                total_w   += r.wins
                sharpes.append(r.sharpe)
            wr  = total_w / total_tr * 100 if total_tr else 0
            sh  = float(np.mean(sharpes)) if sharpes else 0
            # Score = WR * PF * Sharpe (жадный индекс)
            score = (wr / 100) * (total_pct / 100 if total_pct > 0 else -1) * sh
            results.append((score, cfg_name, scheme, mh, total_tr, wr, total_pct, sh))
            if score > best_score:
                best_score = score
                best_combo = (cfg_name, scheme, mh, kw)

results.sort(reverse=True)
print(f"{'Config':15s}  {'Scheme':5s}  {'Hold':4s}  {'Sigs':5s}  {'WR':6s}  {'Total%':8s}  {'Sharpe':6s}  {'Score':8s}")
print("-" * 75)
for row in results[:20]:
    score, cfg_name, scheme, mh, tr, wr, tp, sh = row
    print(f"{cfg_name:15s}  {scheme:5s}  {mh:4d}  {tr:5d}  {wr:5.1f}%  {tp:+8.1f}%  {sh:6.2f}  {score:8.4f}")

print(f"\nЛучшая комбинация: {best_combo}")
