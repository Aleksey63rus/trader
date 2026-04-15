"""
=============================================================================
ГЛУБОКИЙ АНАЛИЗ SL и TP — Поиск оптимальных параметров выхода
=============================================================================

Вопрос: Почему SL выходы убыточны на 60-70%?

ГИПОТЕЗЫ:
  H1: SL слишком близко — рынок "вытряхивает" позицию перед движением
  H2: TP слишком мало — стратегия закрывает позицию до пика прибыли
  H3: Лучше работать БЕЗ SL (только TIME+TP) на дневных данных
  H4: Trailing Stop лучше фиксированного SL

МЕТОДОЛОГИЯ:
  1. Анализируем что происходит с ценой ПОСЛЕ срабатывания SL
  2. Тестируем 20+ комбинаций SL% и TP уровней
  3. Сравниваем: фиксированный SL vs широкий SL vs без SL vs trailing

=============================================================================
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd

from strategies_lab import (
    load_daily, TICKERS, COMMISSION, SLIPPAGE,
    ema, atr, adx, rsi, volume_ratio,
)

DATA_DIR = Path("c:/investor/data")

# ── Загрузка данных ────────────────────────────────────────────────────────────
print("Загрузка данных...")
DATA = {}
for t in TICKERS:
    df = load_daily(t)
    if df is not None:
        DATA[t] = df
print(f"  Загружено: {len(DATA)} тикеров\n")


# ══════════════════════════════════════════════════════════════════════════════
# Генератор сигналов ATR_BO (лучшая стратегия)
# ══════════════════════════════════════════════════════════════════════════════
def get_atr_bo_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Сигналы стратегии ATR_BO с SL/risk на каждый бар."""
    c     = df["close"]
    at14  = atr(df, 14)
    at5   = atr(df, 5)
    e200  = ema(c, 200)
    rsi14 = rsi(c, 14)
    adx14 = adx(df, 14)
    vol_r = volume_ratio(df, 20)
    bm    = (c - c.shift(1)).clip(lower=0)

    sig = ((c > e200) & (bm >= 1.5*at14) & (at5 > at14*0.95) &
           (rsi14 >= 52) & (rsi14 <= 82) & (adx14 >= 22) & (vol_r >= 1.5))
    sig = sig & ~sig.shift(1).fillna(False)

    return pd.DataFrame({
        "signal": sig.astype(int),
        "at14":   at14,
    }, index=df.index)


# ══════════════════════════════════════════════════════════════════════════════
# АНАЛИЗ 1: Что происходит после срабатывания SL
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 65)
print("АНАЛИЗ 1: Что происходит с ценой ПОСЛЕ срабатывания SL?")
print("=" * 65)

sl_pct = 0.05  # базовый SL = 5%

total_sl_trades = 0
price_after = defaultdict(list)  # {N_дней: [изменение_цены%]}

for ticker, df in DATA.items():
    sigs = get_atr_bo_signals(df)
    c    = df["close"].values
    h    = df["high"].values
    l    = df["low"].values
    n    = len(df)

    for i in range(1, n-30):
        if not sigs["signal"].iloc[i-1]:
            continue
        entry = df["open"].iloc[i] * (1 + SLIPPAGE)
        sl    = entry * (1 - sl_pct)

        # Ищем первый бар где SL срабатывает
        sl_hit = False
        for j in range(i, min(i+20, n)):
            if l[j] <= sl:
                sl_hit = True
                sl_day = j
                break

        if sl_hit and sl_day + 30 < n:
            total_sl_trades += 1
            # Что происходит с ценой через 1,3,5,10,20,30 дней после SL
            sl_price = sl
            for days in [1, 3, 5, 10, 15, 20, 30]:
                if sl_day + days < n:
                    future_close = c[sl_day + days]
                    chg = (future_close - sl_price) / sl_price * 100
                    price_after[days].append(chg)

print(f"\n  Всего SL-выходов проанализировано: {total_sl_trades}")
print("\n  Среднее изменение цены ПОСЛЕ срабатывания SL:")
print(f"  {'Дней после SL':15s} {'Avg%':8s} {'Медиана%':10s} {'% выше SL':12s}")
print("  " + "─" * 50)
for days in [1, 3, 5, 10, 15, 20, 30]:
    if price_after[days]:
        arr = np.array(price_after[days])
        avg  = arr.mean()
        med  = np.median(arr)
        pct_above = (arr > 0).mean() * 100
        print(f"  {days:15d} {avg:>+8.2f}% {med:>+10.2f}% {pct_above:>10.1f}%")

print("\n  ВЫВОД: Если через N дней >50% выше SL → SL срабатывает ПРЕЖДЕВРЕМЕННО")


# ══════════════════════════════════════════════════════════════════════════════
# АНАЛИЗ 2: Как далеко уходит цена до SL (минимальная просадка внутри позиции)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("АНАЛИЗ 2: Максимальная просадка внутри ВЫИГРЫШНЫХ сделок")
print("=" * 65)
print("(Насколько глубоко уходит цена до того как пойти в +)")

max_dd_in_winners = []

for ticker, df in DATA.items():
    sigs = get_atr_bo_signals(df)
    c    = df["close"].values
    h    = df["high"].values
    l    = df["low"].values
    op   = df["open"].values
    n    = len(df)
    at14v= sigs["at14"].values

    for i in range(1, n-21):
        if not sigs["signal"].iloc[i-1]:
            continue
        entry = op[i] * (1 + SLIPPAGE)
        # Смотрим 20 баров вперёд
        future_h = max(h[i:min(i+20,n)])
        future_l = min(l[i:min(i+20,n)])
        final_c  = c[min(i+20,n-1)]

        total_move = (future_h - entry) / entry * 100
        max_drawdown_from_entry = (future_l - entry) / entry * 100

        # Выигрышная сделка = цена выросла >3% за 20 дней
        if total_move >= 3.0:
            max_dd_in_winners.append(max_drawdown_from_entry)

if max_dd_in_winners:
    arr = np.array(max_dd_in_winners)
    print(f"\n  Анализ {len(arr)} выигрышных паттернов (рост >3% за 20 дней):")
    print(f"  Средняя просадка до роста:  {arr.mean():+.2f}%")
    print(f"  Медиана просадки:           {np.median(arr):+.2f}%")
    print(f"  90% просадки ≤:             {np.percentile(arr, 90):+.2f}%")
    print(f"  95% просадок ≤:             {np.percentile(arr, 95):+.2f}%")
    print(f"  99% просадок ≤:             {np.percentile(arr, 99):+.2f}%")
    print(f"\n  % случаев просадка > 5%:   {(arr < -5).mean()*100:.1f}%")
    print(f"  % случаев просадка > 8%:   {(arr < -8).mean()*100:.1f}%")
    print(f"  % случаев просадка > 10%:  {(arr < -10).mean()*100:.1f}%")
    print(f"  % случаев просадка > 12%:  {(arr < -12).mean()*100:.1f}%")
    print(f"\n  → Оптимальный SL должен быть ≥ {abs(np.percentile(arr,95)):.1f}% от входа")


# ══════════════════════════════════════════════════════════════════════════════
# АНАЛИЗ 3: Насколько большой максимальный выход достигается
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("АНАЛИЗ 3: Максимальный потенциал роста за N дней после сигнала")
print("=" * 65)

for ticker, df in list(DATA.items())[:5]:
    sigs = get_atr_bo_signals(df)

potential = defaultdict(list)
for ticker, df in DATA.items():
    sigs = get_atr_bo_signals(df)
    c    = df["close"].values
    h    = df["high"].values
    op   = df["open"].values
    n    = len(df)
    for i in range(1, n-35):
        if not sigs["signal"].iloc[i-1]:
            continue
        entry = op[i] * (1 + SLIPPAGE)
        for days in [5, 10, 15, 20, 25, 30, 40, 60]:
            end = min(i+days, n)
            max_h = max(h[i:end])
            pct   = (max_h - entry) / entry * 100
            potential[days].append(pct)

print("\n  Максимальный рост (high) за N дней от точки входа:")
print(f"  {'Дней':8s} {'Avg max%':10s} {'Медиана%':10s} {'75%':8s} {'90%':8s} {'95%':8s}")
print("  " + "─" * 56)
for days in [5, 10, 15, 20, 25, 30, 40, 60]:
    if potential[days]:
        arr = np.array(potential[days])
        print(f"  {days:8d} {arr.mean():>+10.2f}% {np.median(arr):>+10.2f}% "
              f"{np.percentile(arr,75):>+8.2f}% {np.percentile(arr,90):>+8.2f}% "
              f"{np.percentile(arr,95):>+8.2f}%")

print("\n  → Медиана показывает реальный потенциал, 90%ile - лучшие случаи")


# ══════════════════════════════════════════════════════════════════════════════
# УНИВЕРСАЛЬНЫЙ ДВИЖОК ДЛЯ ТЕСТА РАЗНЫХ SL/TP
# ══════════════════════════════════════════════════════════════════════════════
def test_exit_config(sl_type: str, sl_param: float,
                     tp_type: str, tp_params: tuple,
                     max_hold: int = 25,
                     trailing_mult: float = 0.0) -> dict:
    """
    sl_type: 'pct' | 'atr' | 'none' | 'trailing'
    sl_param: для pct=процент (0.05=5%), для atr=множитель
    tp_type: 'stepped' | 'single' | 'none'
    tp_params: для stepped=(fracs, levels), для single=(level,)
    trailing_mult: ATR множитель для trailing stop (0=выключен)
    """
    total_tr = 0; total_w = 0; total_pct = 0.0
    sl_exits = 0; time_exits = 0; tp_exits = 0
    all_pnl = []

    for ticker, df in DATA.items():
        sigs = get_atr_bo_signals(df)
        c    = df["close"].values
        h    = df["high"].values
        l    = df["low"].values
        op   = df["open"].values
        at14v= sigs["at14"].values
        n    = len(df)

        open_entry = None
        open_sl    = None
        open_entry_i = None
        partial_pnl = 0.0
        remaining   = 1.0
        tp_hit      = 0

        for i in range(1, n):
            # Обновление позиции
            if open_entry is not None:
                hold  = i - open_entry_i
                entry = open_entry
                hi    = h[i]
                lo    = l[i]
                curr_sl = open_sl

                # Trailing stop update
                if trailing_mult > 0 and tp_hit >= 1:
                    trail = c[i] - trailing_mult * at14v[i]
                    if trail > curr_sl:
                        open_sl = trail
                        curr_sl = trail

                # TP проверки
                if tp_type == 'stepped':
                    fracs, levels = tp_params
                    while tp_hit < len(levels) and remaining > 1e-9:
                        tp_px = entry + levels[tp_hit] * (entry * sl_param if sl_type == 'pct' else at14v[open_entry_i])
                        if hi >= tp_px:
                            frac = min(fracs[tp_hit], remaining)
                            ep   = tp_px * (1 - SLIPPAGE)
                            partial_pnl += (ep - entry)*frac - (entry+ep)*COMMISSION*frac
                            remaining -= frac
                            # Сдвигаем SL на BEP или выше
                            if sl_type != 'none':
                                if tp_hit == 0:
                                    open_sl = max(open_sl, entry * 1.001)
                                else:
                                    prev_tp = entry + levels[tp_hit-1] * (entry*sl_param if sl_type=='pct' else at14v[open_entry_i])
                                    open_sl = max(open_sl, prev_tp * 0.95)
                                curr_sl = open_sl
                            tp_hit += 1
                        else:
                            break

                elif tp_type == 'single' and len(tp_params) > 0:
                    tp_px = entry + tp_params[0] * (entry * sl_param if sl_type == 'pct' else at14v[open_entry_i] * 3)
                    if hi >= tp_px:
                        ep = tp_px * (1 - SLIPPAGE)
                        partial_pnl += (ep - entry)*remaining - (entry+ep)*COMMISSION*remaining
                        remaining = 0.0
                        tp_hit = 1

                reason = ep_f = None

                if remaining <= 1e-6:
                    reason = "TP"; ep_f = entry
                elif hold >= max_hold:
                    reason = "TIME"; ep_f = op[i] * (1-SLIPPAGE)
                elif sl_type != 'none' and lo <= curr_sl:
                    reason = "SL"
                    ep_f = max(curr_sl * (1-SLIPPAGE), lo)

                if reason:
                    rem = remaining
                    pnl = (partial_pnl + (ep_f - entry)*rem
                           - (entry+ep_f)*COMMISSION*rem)
                    pnl_pct = pnl / entry * 100
                    total_pct += pnl_pct
                    all_pnl.append(pnl_pct)
                    if pnl > 0: total_w += 1
                    total_tr += 1
                    if reason == "SL": sl_exits += 1
                    elif reason == "TIME": time_exits += 1
                    else: tp_exits += 1
                    # Сброс
                    open_entry = open_sl = open_entry_i = None
                    partial_pnl = 0.0; remaining = 1.0; tp_hit = 0
                    continue

            # Новый вход
            if open_entry is None and bool(sigs["signal"].iloc[i-1]):
                at14_i = float(sigs["at14"].iloc[i-1])
                entry  = op[i] * (1 + SLIPPAGE)
                if sl_type == 'pct':
                    sl_v = entry * (1 - sl_param)
                elif sl_type == 'atr':
                    sl_v = entry - sl_param * at14_i
                elif sl_type == 'none' or sl_type == 'trailing':
                    sl_v = entry * 0.50  # очень далёкий SL (фактически нет)
                else:
                    sl_v = entry * 0.95
                open_entry   = entry
                open_sl      = sl_v
                open_entry_i = i
                partial_pnl  = 0.0
                remaining    = 1.0
                tp_hit       = 0

        # Принудительное закрытие
        if open_entry is not None:
            ep = c[-1] * (1-SLIPPAGE)
            pnl = (partial_pnl + (ep-open_entry)*remaining
                   - (open_entry+ep)*COMMISSION*remaining)
            pnl_pct = pnl / open_entry * 100
            total_pct += pnl_pct
            all_pnl.append(pnl_pct)
            if pnl > 0: total_w += 1
            total_tr += 1; time_exits += 1

    if not all_pnl:
        return {"trades": 0, "wr": 0, "total_pct": 0, "sharpe": 0,
                "sl_exits": 0, "time_exits": 0, "tp_exits": 0}

    arr    = np.array(all_pnl)
    wr     = total_w / total_tr * 100 if total_tr else 0
    sharpe = (arr.mean() / (arr.std()+1e-9) * np.sqrt(252)
              if len(arr) > 1 else 0.0)
    wins   = arr[arr > 0]
    losses = arr[arr <= 0]
    pf     = wins.sum() / (-losses.sum()+1e-9) if len(losses) else 99.0

    return {
        "trades": total_tr, "wr": wr, "total_pct": total_pct,
        "avg_pct": total_pct/total_tr if total_tr else 0,
        "sharpe": sharpe, "pf": min(pf, 99),
        "sl_exits": sl_exits, "time_exits": time_exits, "tp_exits": tp_exits,
        "avg_win": float(wins.mean()) if len(wins) else 0,
        "avg_loss": float(losses.mean()) if len(losses) else 0,
    }


# ══════════════════════════════════════════════════════════════════════════════
# ТЕСТ A: Разные уровни SL (фиксированный %)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("ТЕСТ A: Влияние размера SL на результат")
print("(TP фиксированный: 30%/30%/40% × 1.2R/3R/7R)")
print("=" * 65)

BASE_FRACS  = (0.30, 0.30, 0.40)
BASE_LEVELS_R = (1.2, 3.0, 7.0)

def make_stepped_pct(sl_pct, fracs, rr_levels):
    """Конвертируем R-уровни в % от цены."""
    # levels в % = rr * sl_pct
    levels = tuple(r * sl_pct for r in rr_levels)
    return (fracs, levels)

SL_TESTS = [
    ("БЕЗ SL",      'none',  0.05,  0.0),
    ("SL 3%",       'pct',   0.03,  0.0),
    ("SL 4%",       'pct',   0.04,  0.0),
    ("SL 5%",       'pct',   0.05,  0.0),
    ("SL 6%",       'pct',   0.06,  0.0),
    ("SL 7%",       'pct',   0.07,  0.0),
    ("SL 8%",       'pct',   0.08,  0.0),
    ("SL 10%",      'pct',   0.10,  0.0),
    ("SL 12%",      'pct',   0.12,  0.0),
    ("SL 15%",      'pct',   0.15,  0.0),
    ("SL 20%",      'pct',   0.20,  0.0),
    ("SL 1.5×ATR",  'atr',   1.5,   0.0),
    ("SL 2×ATR",    'atr',   2.0,   0.0),
    ("SL 2.5×ATR",  'atr',   2.5,   0.0),
    ("SL 3×ATR",    'atr',   3.0,   0.0),
    ("SL 4×ATR",    'atr',   4.0,   0.0),
]

print(f"\n  {'Конфиг':14s} {'Сд':5s} {'WR%':6s} {'PF':5s} "
      f"{'Total%':8s} {'SL/TM/TP':14s} {'Sharpe':7s}")
print("  " + "─" * 65)

best_sl_score = -999; best_sl_cfg = None
for name, sl_type, sl_param, trail_mult in SL_TESTS:
    # TP уровни в % от цены
    if sl_type == 'pct':
        fracs = BASE_FRACS
        levels= tuple(r * sl_param for r in BASE_LEVELS_R)
    elif sl_type == 'atr':
        fracs = BASE_FRACS
        levels= (1.2, 3.0, 7.0)  # будет умножено на ATR в движке
        # Перейдём на ATR-based TP
        levels= (0.04, 0.10, 0.25)  # ~средние % уровни
    else:
        fracs = BASE_FRACS
        levels= (0.06, 0.15, 0.30)

    r = test_exit_config(sl_type, sl_param, 'stepped', (fracs, levels), max_hold=25)
    score = r['wr'] * r['pf'] / 100 if r['trades'] > 0 else -99
    mk = " ◄" if score > best_sl_score and r['trades'] >= 30 else ""
    if score > best_sl_score and r['trades'] >= 30:
        best_sl_score = score
        best_sl_cfg   = (sl_type, sl_param, fracs, levels)

    dist = f"{r['sl_exits']}sl/{r['time_exits']}tm/{r['tp_exits']}tp"
    print(f"  {name:14s} {r['trades']:5d} {r['wr']:5.1f}% {r['pf']:5.2f} "
          f"{r['total_pct']:>+7.1f}% {dist:14s} {r['sharpe']:6.2f}{mk}")


# ══════════════════════════════════════════════════════════════════════════════
# ТЕСТ B: Разные TP уровни (R-мультипликаторы)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("ТЕСТ B: Влияние уровней TP на результат")
print("(SL фиксированный 5%)")
print("=" * 65)

SL_FIXED = 0.05

TP_TESTS = [
    # name, fracs, % уровни (уже в долях цены)
    ("Single 2R",    (1.0,),          (2*SL_FIXED,)),
    ("Single 3R",    (1.0,),          (3*SL_FIXED,)),
    ("Single 5R",    (1.0,),          (5*SL_FIXED,)),
    ("Single 8R",    (1.0,),          (8*SL_FIXED,)),
    ("Single 10R",   (1.0,),          (10*SL_FIXED,)),
    ("Single 15R",   (1.0,),          (15*SL_FIXED,)),
    ("2-уровня 1/2R",(0.50,0.50),     (1*SL_FIXED, 2*SL_FIXED)),
    ("2-уровня 1/3R",(0.50,0.50),     (1*SL_FIXED, 3*SL_FIXED)),
    ("2-уровня 1/5R",(0.60,0.40),     (1*SL_FIXED, 5*SL_FIXED)),
    ("3-ур 1/3/7R",  (0.30,0.30,0.40),(1*SL_FIXED, 3*SL_FIXED, 7*SL_FIXED)),
    ("3-ур 1/4/10R", (0.30,0.30,0.40),(1*SL_FIXED, 4*SL_FIXED,10*SL_FIXED)),
    ("3-ур 1/5/12R", (0.30,0.30,0.40),(1*SL_FIXED, 5*SL_FIXED,12*SL_FIXED)),
    ("3-ур 2/5/15R", (0.30,0.30,0.40),(2*SL_FIXED, 5*SL_FIXED,15*SL_FIXED)),
    ("3-ур 2/6/20R", (0.25,0.35,0.40),(2*SL_FIXED, 6*SL_FIXED,20*SL_FIXED)),
    ("4-ур 1/3/7/15R",(0.25,0.25,0.25,0.25),(SL_FIXED,3*SL_FIXED,7*SL_FIXED,15*SL_FIXED)),
    ("4-ур 1/3/8/20R",(0.30,0.25,0.25,0.20),(SL_FIXED,3*SL_FIXED,8*SL_FIXED,20*SL_FIXED)),
    ("БЕЗ TP (TIME)", (1.0,),         (9.9,)),  # только TIME — TP > 9.9x = никогда
]

print(f"\n  {'Конфиг':18s} {'Сд':5s} {'WR%':6s} {'PF':5s} "
      f"{'Total%':8s} {'SL/TM/TP':12s} {'Avg_win':8s} {'Sharpe':7s}")
print("  " + "─" * 75)

best_tp_score = -999; best_tp_cfg = None
for name, fracs, levels in TP_TESTS:
    # Все конфигурации идут как 'stepped'
    r = test_exit_config('pct', SL_FIXED, 'stepped', (fracs, levels), max_hold=25)
    score = r['wr'] * r['pf'] / 100 if r['trades'] > 0 else -99
    mk = " ◄" if score > best_tp_score and r['trades'] >= 20 else ""
    if score > best_tp_score and r['trades'] >= 20:
        best_tp_score = score
        best_tp_cfg   = (fracs, levels)

    dist = f"{r['sl_exits']}sl/{r['time_exits']}tm/{r['tp_exits']}tp"
    print(f"  {name:18s} {r['trades']:5d} {r['wr']:5.1f}% {r['pf']:5.2f} "
          f"{r['total_pct']:>+7.1f}% {dist:12s} {r['avg_win']:>+7.2f}% "
          f"{r['sharpe']:6.2f}{mk}")


# ══════════════════════════════════════════════════════════════════════════════
# ТЕСТ C: Trailing Stop вместо фиксированного SL
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("ТЕСТ C: Trailing Stop — позволяем прибыли расти")
print("=" * 65)

TRAILING_TESTS = [
    ("Trail 1×ATR,SL5%",     'pct', 0.05, 1.0, 25),
    ("Trail 1.5×ATR,SL5%",   'pct', 0.05, 1.5, 25),
    ("Trail 2×ATR,SL5%",     'pct', 0.05, 2.0, 25),
    ("Trail 2.5×ATR,SL5%",   'pct', 0.05, 2.5, 25),
    ("Trail 1×ATR,SL8%",     'pct', 0.08, 1.0, 25),
    ("Trail 2×ATR,SL8%",     'pct', 0.08, 2.0, 30),
    ("Trail 2×ATR,SL10%",    'pct', 0.10, 2.0, 35),
    ("Trail 1.5×ATR,noSL",   'none',0.05, 1.5, 30),
    ("Trail 2×ATR,noSL",     'none',0.05, 2.0, 35),
    ("Trail 2.5×ATR,noSL",   'none',0.05, 2.5, 40),
    ("Trail 3×ATR,noSL",     'none',0.05, 3.0, 45),
]

print(f"\n  {'Конфиг':22s} {'Сд':5s} {'WR%':6s} {'PF':5s} "
      f"{'Total%':8s} {'SL/TM':8s} {'Avg_win':8s} {'Sharpe':7s}")
print("  " + "─" * 70)

best_tr_score = -999; best_trail_cfg = None
for name, sl_type, sl_param, trail_mult, mh in TRAILING_TESTS:
    # Без TP — только trailing + TIME выход
    fracs  = (1.0,)
    levels = (9.9,)  # TP никогда не достигнет (9.9 = 990% от SL%)
    r = test_exit_config(sl_type, sl_param, 'stepped', (fracs, levels),
                         max_hold=mh, trailing_mult=trail_mult)
    score = r['wr'] * r['pf'] / 100 if r['trades'] > 0 else -99
    mk = " ◄" if score > best_tr_score and r['trades'] >= 20 else ""
    if score > best_tr_score and r['trades'] >= 20:
        best_tr_score  = score
        best_trail_cfg = (sl_type, sl_param, trail_mult, mh)

    dist = f"{r['sl_exits']}sl/{r['time_exits']}tm"
    print(f"  {name:22s} {r['trades']:5d} {r['wr']:5.1f}% {r['pf']:5.2f} "
          f"{r['total_pct']:>+7.1f}% {dist:8s} {r['avg_win']:>+7.2f}% "
          f"{r['sharpe']:6.2f}{mk}")


# ══════════════════════════════════════════════════════════════════════════════
# ТЕСТ D: Гибридный режим — Trail + Stepped TP
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("ТЕСТ D: ГИБРИД — Stepped TP + Trailing Stop после 1-го TP")
print("(Берём прибыль ступенчато И даём остатку расти)")
print("=" * 65)

HYBRID_TESTS = [
    # name,  SL%,  fracs,              R×SL levels,   trail, hold
    ("50%@2R+trail2×",   0.05, (0.50,0.50), (0.10,0.80), 2.0, 30),
    ("50%@3R+trail2×",   0.05, (0.50,0.50), (0.15,0.80), 2.0, 35),
    ("40%@2R+trail2×",   0.05, (0.40,0.60), (0.10,0.80), 2.0, 30),
    ("33%@1R+trail2×",   0.05, (0.33,0.67), (0.05,0.80), 2.0, 30),
    ("25%@1R+25%@3R+trail",0.05,(0.25,0.25,0.50),(0.05,0.15,0.80),2.0,35),
    ("30%@2R+70%trail",  0.07, (0.30,0.70), (0.14,0.80), 2.0, 35),
    ("40%@2R+trail1.5",  0.07, (0.40,0.60), (0.14,0.80), 1.5, 30),
    ("50%@3R+trail1.5",  0.07, (0.50,0.50), (0.21,0.80), 1.5, 35),
    ("SL8%,50%@3R+trail",0.08, (0.50,0.50), (0.24,0.80), 2.0, 35),
    ("SL10%,40%@5R+trail",0.10,(0.40,0.60), (0.50,0.80), 2.0, 40),
]

print(f"\n  {'Конфиг':24s} {'Сд':5s} {'WR%':6s} {'PF':5s} "
      f"{'Total%':8s} {'Avg_win':8s} {'Sharpe':7s}")
print("  " + "─" * 68)

best_hyb_score = -999; best_hybrid_cfg = None
for name, sl_pct, fracs, rel_levels, trail, mh in HYBRID_TESTS:
    levels = tuple(v for v in rel_levels)
    r = test_exit_config('pct', sl_pct, 'stepped', (fracs, levels),
                         max_hold=mh, trailing_mult=trail)
    score = r['wr'] * r['pf'] / 100 if r['trades'] > 0 else -99
    mk = " ◄" if score > best_hyb_score and r['trades'] >= 20 else ""
    if score > best_hyb_score and r['trades'] >= 20:
        best_hyb_score  = score
        best_hybrid_cfg = (sl_pct, fracs, rel_levels, trail, mh)

    print(f"  {name:24s} {r['trades']:5d} {r['wr']:5.1f}% {r['pf']:5.2f} "
          f"{r['total_pct']:>+7.1f}% {r['avg_win']:>+7.2f}% "
          f"{r['sharpe']:6.2f}{mk}")


# ══════════════════════════════════════════════════════════════════════════════
# ИТОГОВЫЙ СВОДНЫЙ ОТЧЁТ
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "═" * 65)
print("  ИТОГОВЫЙ ОТЧЁТ: ЛУЧШИЕ КОНФИГУРАЦИИ ВЫХОДА")
print("═" * 65)
print(f"\n  Лучший SL:      {best_sl_cfg}")
print(f"  Лучший TP:      {best_tp_cfg}")
print(f"  Лучший Trail:   {best_trail_cfg}")
print(f"  Лучший Гибрид:  {best_hybrid_cfg}")
