"""
Оптимизация лучшей стратегии ATR_BO.
Тестируем разные параметры на всех тикерах, находим лучшую конфигурацию.
Затем комбинируем ATR_BO + 52W_High + TripleEMA в один портфель.
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import pandas as pd

from strategies_lab import (
    load_daily, TICKERS, ema, atr, adx, rsi, macd_hist, volume_ratio, obv, supertrend,
    run_backtest,
)
from portfolio_lab import portfolio_sim, print_detailed

DATA_DIR = Path("c:/investor/data")

print("Загрузка данных...")
data = {}
for t in TICKERS:
    df = load_daily(t)
    if df is not None:
        data[t] = df
print(f"  Загружено: {len(data)} тикеров\n")


# ══════════════════════════════════════════════════════════════════════════════
# Шаг 1: Оптимизация параметров ATR_BO
# ══════════════════════════════════════════════════════════════════════════════
print("ШАГОВАЯ ОПТИМИЗАЦИЯ ATR_BO")
print("─" * 60)

GRID = [
    # bar_mult, atr_ratio, rsi_lo, rsi_hi, adx_min, vol_min
    (1.2, 0.90, 50, 82, 20, 1.3),
    (1.5, 0.95, 52, 82, 22, 1.5),  # base
    (1.5, 0.90, 50, 80, 20, 1.3),
    (1.5, 0.95, 52, 82, 18, 1.3),
    (1.8, 0.90, 50, 82, 18, 1.3),
    (1.8, 0.95, 52, 80, 22, 1.5),
    (2.0, 0.90, 50, 82, 20, 1.2),
    (2.0, 0.95, 55, 80, 25, 1.5),
    (1.5, 0.90, 50, 80, 20, 1.0),  # без vol фильтра
    (1.5, 0.85, 50, 85, 18, 1.2),  # более мягкий
    (1.3, 0.90, 52, 78, 22, 1.5),  # умеренный
    (1.0, 0.85, 50, 82, 18, 1.0),  # минимальные требования
]

best_score = -999
best_params = None

print(f"{'bm':4s} {'ar':4s} {'rlo':4s} {'rhi':4s} {'adx':4s} {'vol':4s} "
      f"{'Сд':5s} {'WR%':6s} {'PF':5s} {'Total%':8s} {'Sharpe':7s}")
print("─" * 60)

for bm, ar, rlo, rhi, adx_min, vol_min in GRID:
    total_tr = 0; total_w = 0; total_pct = 0; sharpes = []

    for ticker, df in data.items():
        c     = df["close"]
        at14  = atr(df, 14)
        at5   = atr(df, 5)
        e200  = ema(c, 200)
        rsi14 = rsi(c, 14)
        adx14 = adx(df, 14)
        vol_r = volume_ratio(df, 20)
        bar_mv= (c - c.shift(1)).clip(lower=0)

        sig = ((c > e200) &
               (bar_mv >= bm * at14) &
               (at5 > at14 * ar) &
               (rsi14 >= rlo) & (rsi14 <= rhi) &
               (adx14 >= adx_min) &
               (vol_r >= vol_min))
        sig = sig & ~sig.shift(1).fillna(False)
        sl  = (c - 1.8*at14).clip(lower=c*0.90)
        risk= (c - sl).clip(lower=0.001)

        try:
            r = run_backtest(sig, sl, risk, df, "ATR_BO", ticker)
            total_tr  += r.trades
            total_w   += r.wins
            total_pct += r.total_pct
            sharpes.append(r.sharpe)
        except Exception:
            pass

    wr  = total_w/total_tr*100 if total_tr else 0
    sh  = float(np.mean(sharpes)) if sharpes else 0
    score = wr * sh / 100
    mk = " <--" if score > best_score and total_tr >= 30 else ""
    if score > best_score and total_tr >= 30:
        best_score = score
        best_params = (bm, ar, rlo, rhi, adx_min, vol_min)
    print(f"{bm:4.1f} {ar:4.2f} {rlo:4d} {rhi:4d} {adx_min:4d} {vol_min:4.1f} "
          f"{total_tr:5d} {wr:5.1f}% {total_pct/total_tr*100 if total_tr else 0:4.2f} "
          f"{total_pct:+7.1f}% {sh:6.2f}{mk}")

print(f"\nЛучшие параметры ATR_BO: {best_params}")


# ══════════════════════════════════════════════════════════════════════════════
# Шаг 2: Финальная стратегия FUSION = ATR_BO + 52W_High + TripleEMA
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*60)
print("  FUSION STRATEGY — объединение 3 лучших стратегий")
print("═"*60)

def generate_fusion_signals(df: pd.DataFrame, bm=1.5, ar=0.95,
                             rlo=52, rhi=82, adx_min=22, vol_min=1.5) -> pd.DataFrame:
    """
    Сигнал от любой из 3 стратегий:
      A. ATR_BO  — пробой на объёме (WR 60.8%)
      B. 52W_High — у годового максимума (академически проверен)
      C. TripleEMA — пересечение EMA (Sharpe 4.34 в одиночном тесте)
    + Общий фильтр: EMA200 + ADX + RSI
    Ранжируем по "силе" сигнала (score)
    """
    c    = df["close"]
    h    = df["high"]
    l    = df["low"]
    at14 = atr(df, 14)
    at5  = atr(df, 5)
    e9   = ema(c, 9)
    e21  = ema(c, 21)
    e55  = ema(c, 55)
    e200 = ema(c, 200)
    adx14= adx(df, 14)
    rsi14= rsi(c, 14)
    mh   = macd_hist(df)
    vol_r= volume_ratio(df, 20)
    obv_s= obv(df)
    obv_e= ema(obv_s, 21)
    st   = supertrend(df, 10, 3.0)

    # Базовый фильтр качества (обязателен для всех)
    quality = (
        (c > e200) &
        (adx14 >= 18) &
        (rsi14 >= 48) & (rsi14 <= 85) &
        (mh > 0)
    )

    # A: ATR Breakout
    bar_mv = (c - c.shift(1)).clip(lower=0)
    atr_bo = (quality &
               (bar_mv >= bm * at14) &
               (at5 > at14 * ar) &
               (rsi14 >= rlo) & (rsi14 <= rhi) &
               (adx14 >= adx_min) &
               (vol_r >= vol_min))

    # B: 52-Week High
    h52   = h.rolling(252).max()
    l52   = l.rolling(252).min()
    range52 = (h52-l52).replace(0, np.nan)
    pos52 = (c - l52) / range52
    w52   = (quality &
              (pos52 >= 0.92) &
              (c > e55) & (e55 > e200) &
              (vol_r >= 1.4) & (st == 1))

    # C: Triple EMA crossover
    cross_up = (e9 > e21) & (e9.shift(1) <= e21.shift(1))
    tema  = (quality &
              cross_up &
              (c > e55) & (e55 > e200) &
              (obv_s > obv_e) &
              (vol_r >= 1.1))

    # Оцениваем силу сигнала (0-7 баллов)
    score = (atr_bo.astype(int) * 3 +
             w52.astype(int)   * 2 +
             tema.astype(int)  * 2 +
             (obv_s > obv_e).astype(int) +
             (vol_r >= 1.5).astype(int) +
             (st == 1).astype(int))

    # Сигнал = хотя бы одна из стратегий ИЛИ score >= 2
    sig_raw = (atr_bo | w52 | tema) & (score >= 2)
    sig     = sig_raw & ~sig_raw.shift(1).fillna(False)

    # SL: ATR-based
    sl   = (c - 1.8*at14).clip(lower=c*0.90)
    risk = (c - sl).clip(lower=0.001)

    return pd.DataFrame({
        "signal": sig.astype(int),
        "sl":     sl,
        "risk":   risk,
        "score":  score,
    }, index=df.index)


# Тест Fusion на отдельных тикерах
print("\nТест FUSION на тикерах:")
total_tr = 0; total_w = 0; total_pct = 0; sharpes = []

for ticker, df in data.items():
    try:
        sdf = generate_fusion_signals(df, *best_params if best_params else (1.5,0.95,52,82,22,1.5))
        sig  = sdf["signal"].astype(bool)
        sl   = sdf["sl"]
        risk = sdf["risk"]
        r = run_backtest(sig, sl, risk, df, "Fusion", ticker)
        total_tr  += r.trades
        total_w   += r.wins
        total_pct += r.total_pct
        sharpes.append(r.sharpe)
    except Exception:
        pass

wr_f = total_w/total_tr*100 if total_tr else 0
sh_f = float(np.mean(sharpes)) if sharpes else 0
print(f"  Сделок: {total_tr}, WR: {wr_f:.1f}%, Total%: {total_pct:+.1f}%, Sharpe: {sh_f:.2f}")


# ══════════════════════════════════════════════════════════════════════════════
# Шаг 3: Портфельная симуляция Fusion
# ══════════════════════════════════════════════════════════════════════════════
print("\nПортфельная симуляция FUSION...")

bp = best_params if best_params else (1.5, 0.95, 52, 82, 22, 1.5)
fusion_sigs = {}
for ticker, df in data.items():
    try:
        s = generate_fusion_signals(df, *bp)
        fusion_sigs[ticker] = s
    except Exception:
        pass

res_fusion = portfolio_sim("FUSION", fusion_sigs, data)
print(f"  Сделок={res_fusion['trades']}  WR={res_fusion['wr']:.1f}%  "
      f"P&L={res_fusion['total_rub']:+,.0f}₽  Ann={res_fusion['ann_ret']:+.1f}%  "
      f"DD={res_fusion['max_dd']:.1f}%  PF={res_fusion['pf']:.2f}")

print_detailed(res_fusion)

# Сравнение с v3c
print("\n" + "═"*60)
print("  СРАВНЕНИЕ ИТОГОВ")
print("═"*60)
print(f"  {'Стратегия':15s} {'P&L ₽':10s} {'Ann%':7s} {'WR%':7s} {'PF':5s} {'DD%':7s}")
print("  " + "─"*54)
baseline = [
    ("v3c (текущая)", 22198,   5.0, 49.1, 1.25, -9.6),
    ("ATR_BO",        21549,   4.9, 60.8, 1.61, -5.0),
    ("52W_High",      22352,   5.0, 47.9, None, -16.5),
    ("TripleEMA",     13878,   3.2, 53.1, 1.59, -11.3),
    ("Turtle55",      11846,   2.8, 46.8, None, -14.2),
]
for name, pnl, ann, wr, pf, dd in baseline:
    pf_s = f"{pf:.2f}" if pf else "N/A"
    print(f"  {name:15s} {pnl:>+9,.0f} {ann:>+6.1f}% {wr:>6.1f}% {pf_s:5s} {dd:>6.1f}%")

fusion_row = ("FUSION (новая)", res_fusion['total_rub'], res_fusion['ann_ret'],
              res_fusion['wr'], res_fusion['pf'], res_fusion['max_dd'])
print(f"  {'FUSION (новая)':15s} {fusion_row[1]:>+9,.0f} {fusion_row[2]:>+6.1f}% "
      f"{fusion_row[3]:>6.1f}% {min(fusion_row[4],99):.2f} {fusion_row[5]:>6.1f}%")
print("═"*60)

# Сохранение сигналов
print("\nСохраняем лучшую стратегию (FUSION)...")
import json
params = {
    "name": "FUSION",
    "bar_mult": bp[0], "atr_ratio": bp[1],
    "rsi_lo": bp[2], "rsi_hi": bp[3],
    "adx_min": bp[4], "vol_min": bp[5],
    "max_hold": 20, "scheme": "AGR",
    "sl_atr_mult": 1.8,
}
with open("c:/investor/fusion_config.json", "w") as f:
    json.dump(params, f, indent=2)
print("  → fusion_config.json")
