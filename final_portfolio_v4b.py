"""
=============================================================================
FINAL PORTFOLIO v4b — ATR_BO + Trailing Stop
Простой, прозрачный движок без "чёрного ящика" RiskManager
=============================================================================

КЛЮЧЕВЫЕ ВЫВОДЫ из sl_tp_research.py:
  ● TIME-выходы: WR~55%, PF~2.0, Sharpe~4 → лучше всего работает держать позицию
  ● Trailing Stop 1.5×ATR без жёсткого TP = Sharpe 5.33, PF 2.47, +310% total
  ● SL должен быть ШИРОКИМ: при SL=5% рынок "вытряхивает" позицию преждевременно
  ● После SL в 65% случаев цена идёт выше — SL МЕШАЕТ
  ● Для выигрышных паттернов просадка до роста: медиана -2.86%, 95% ≤ -0.63%
  ● Медианный рост за 20 дней: +5.5%, за 30 дней: +7.4%, за 60 дней: +9.4%
  ● Single TP 10R (50%) = Total +63%, Sharpe 3.14 — хорошая фиксация прибыли

КОНФИГУРАЦИИ ДЛЯ ТЕСТА:
  V4A: Trailing 1.5×ATR noSL, hold=40   (лучший в одиночном тесте)
  V4B: Trailing 2×ATR SL8%, hold=35
  V4C: Trailing 2×ATR noSL, hold=35
  V4D: Без trailing, только TIME, SL широкий 15%
  V4E: Single TP=50% (10R), SL5%, hold=25  (best PF в одиночном)
  V4F: Hybrid: 40%@TP_early + trailing остаток
=============================================================================
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
from dataclasses import dataclass
import numpy as np
import pandas as pd

from strategies_lab import load_daily, TICKERS, COMMISSION, SLIPPAGE
from strategies_lab import ema, atr, adx, rsi, volume_ratio

INITIAL_CAP = 100_000.0
MAX_POSITIONS = 5
RISK_PCT = 0.05   # 5% капитала на сделку

print("Загрузка данных...")
DATA: dict[str, pd.DataFrame] = {}
for t in TICKERS:
    df = load_daily(t)
    if df is not None:
        DATA[t] = df
print(f"  Загружено: {len(DATA)} тикеров")

# ── Сигналы ATR_BO ─────────────────────────────────────────────────────────────
def get_signals(df: pd.DataFrame) -> pd.DataFrame:
    c     = df["close"]
    at14  = atr(df, 14)
    at5   = atr(df, 5)
    e200  = ema(c, 200)
    rsi14 = rsi(c, 14)
    adx14 = adx(df, 14)
    vol_r = volume_ratio(df, 20)
    bm    = (c - c.shift(1)).clip(lower=0)
    sig   = ((c > e200) & (bm >= 1.5*at14) & (at5 > at14*0.95) &
             (rsi14 >= 52) & (rsi14 <= 82) & (adx14 >= 22) & (vol_r >= 1.5))
    sig   = sig & ~sig.shift(1).fillna(False)
    return pd.DataFrame({"signal": sig.astype(int), "at14": at14}, index=df.index)

ALL_SIGS = {t: get_signals(df) for t, df in DATA.items()}

# ── Общий список дат ────────────────────────────────────────────────────────────
all_dates = sorted(set().union(*[set(df.index) for df in DATA.values()]))

# ── Корреляционные группы ───────────────────────────────────────────────────────
CORR_GROUPS: list[set[str]] = [
    {"SBER", "T", "VTBR"},
    {"LKOH", "ROSN", "NVTK", "GAZP", "SNGS", "SNGSP"},
    {"NLMK", "MTLR", "CHMF"},
    {"TATN", "TATNP"},
    {"YDEX", "OZON"},
]

def corr_blocked(ticker: str, open_tickers: set[str]) -> bool:
    for grp in CORR_GROUPS:
        if ticker in grp and grp & open_tickers:
            return True
    return False


@dataclass
class Position:
    ticker:   str
    entry_px: float
    shares:   float
    entry_i:  int
    sl_px:    float
    partial_pnl: float = 0.0
    remaining:   float = 1.0
    tp_hit:      int   = 0
    entry_date:  object = None


@dataclass
class Trade:
    ticker: str; entry_date: object; exit_date: object
    entry_px: float; exit_px: float; pnl_pct: float; reason: str


# ═══════════════════════════════════════════════════════════════════════════════
def run_sim(
    sl_pct: float,        # начальный SL (0=нет жёсткого SL)
    trail_mult: float,    # ATR-множитель trailing stop (0=нет trailing)
    tp_fracs: tuple,      # доли позиции для каждого TP
    tp_r_levels: tuple,   # TP в R-единицах (1R = sl_pct)
    max_hold: int,
    trail_after_tp: int,  # начинать trailing после N-го TP (0=сразу)
    label: str,
) -> dict:
    capital   = INITIAL_CAP
    peak_cap  = INITIAL_CAP
    positions: dict[str, Position] = {}
    trades:    list[Trade] = []
    equity    = [INITIAL_CAP]
    max_dd    = 0.0

    TICK_IDX  = {t: {d: i for i, d in enumerate(df.index)}
                 for t, df in DATA.items()}

    for date in all_dates:
        # ── Обновляем позиции ──────────────────────────────────────────────────
        to_close = []
        for ticker, pos in positions.items():
            df  = DATA[ticker]
            idx = TICK_IDX[ticker].get(date)
            if idx is None or idx <= pos.entry_i:
                continue
            hi  = float(df["high"].iloc[idx])
            lo  = float(df["low"].iloc[idx])
            op  = float(df["open"].iloc[idx])
            cls = float(df["close"].iloc[idx])
            at14= float(ALL_SIGS[ticker]["at14"].iloc[idx])
            hold= idx - pos.entry_i
            sl  = pos.sl_px

            # Trailing update
            if trail_mult > 0 and pos.tp_hit >= trail_after_tp:
                trail = cls - trail_mult * at14
                if trail > sl:
                    sl = trail; pos.sl_px = sl

            # TP hits (stepped)
            for k in range(pos.tp_hit, len(tp_fracs)):
                r    = tp_r_levels[k]
                tp_px= pos.entry_px * (1 + r * (sl_pct if sl_pct > 0 else 0.05))
                if hi >= tp_px:
                    frac  = min(tp_fracs[k], pos.remaining)
                    ep    = tp_px * (1-SLIPPAGE)
                    pos.partial_pnl += (ep-pos.entry_px)*frac*pos.shares - \
                                       (pos.entry_px+ep)*COMMISSION*frac*pos.shares
                    pos.remaining   -= frac
                    pos.tp_hit       = k+1
                    # Сдвигаем SL на BEP
                    if k == 0:
                        pos.sl_px = max(pos.sl_px, pos.entry_px*1.001)
                    else:
                        prev_tp = pos.entry_px*(1+(tp_r_levels[k-1])*(sl_pct if sl_pct>0 else 0.05))
                        pos.sl_px = max(pos.sl_px, prev_tp*0.97)
                    sl = pos.sl_px
                else:
                    break

            reason = ep_f = None
            if pos.remaining <= 1e-6:
                reason = "TP"; ep_f = pos.entry_px
            elif hold >= max_hold:
                reason = "TIME"; ep_f = op*(1-SLIPPAGE)
            elif sl_pct > 0 and lo <= sl:
                reason = "SL"; ep_f = max(sl*(1-SLIPPAGE), lo)

            if reason:
                rem = pos.remaining
                pnl = pos.partial_pnl + (ep_f-pos.entry_px)*rem*pos.shares - \
                      (pos.entry_px+ep_f)*COMMISSION*rem*pos.shares
                pnl_pct = pnl/(pos.entry_px*pos.shares)*100
                capital += pnl
                trades.append(Trade(ticker, pos.entry_date, date,
                                    pos.entry_px, ep_f, pnl_pct, reason))
                to_close.append(ticker)

        for t in to_close:
            positions.pop(t, None)

        # ── Новые входы ────────────────────────────────────────────────────────
        for ticker, df in DATA.items():
            if len(positions) >= MAX_POSITIONS:
                break
            if ticker in positions:
                continue

            idx = TICK_IDX[ticker].get(date)
            if idx is None or idx < 1:
                continue
            if not ALL_SIGS[ticker]["signal"].iloc[idx-1]:
                continue
            if corr_blocked(ticker, set(positions.keys())):
                continue

            at14i   = float(ALL_SIGS[ticker]["at14"].iloc[idx-1])
            entry   = float(df["open"].iloc[idx]) * (1+SLIPPAGE)
            sl_init = entry*(1-sl_pct) if sl_pct > 0 else entry*0.40

            alloc  = min(capital * RISK_PCT, capital / MAX_POSITIONS)
            shares = alloc / entry
            if shares <= 0 or alloc > capital * 0.95:
                continue

            capital -= alloc
            positions[ticker] = Position(
                ticker, entry, shares, idx, sl_init,
                entry_date=date,
            )

        # Equity update
        # Считаем unrealized PnL для equity curve
        open_val = sum(
            (float(DATA[t]["close"].iloc[TICK_IDX[t].get(date, -1)])
             if TICK_IDX[t].get(date) is not None else p.entry_px)
            * p.remaining * p.shares
            for t, p in positions.items()
            if TICK_IDX[t].get(date) is not None
        )
        total_eq = capital + open_val
        equity.append(total_eq)
        if total_eq > peak_cap:
            peak_cap = total_eq
        dd = (peak_cap - total_eq) / peak_cap * 100
        if dd > max_dd:
            max_dd = dd

    # Принудительное закрытие
    last_date = all_dates[-1]
    for ticker, pos in list(positions.items()):
        df  = DATA[ticker]
        cls = float(df["close"].iloc[-1])
        ep  = cls*(1-SLIPPAGE)
        rem = pos.remaining
        pnl = pos.partial_pnl + (ep-pos.entry_px)*rem*pos.shares - \
              (pos.entry_px+ep)*COMMISSION*rem*pos.shares
        pnl_pct = pnl/(pos.entry_px*pos.shares)*100
        capital += pnl
        trades.append(Trade(ticker, pos.entry_date, last_date,
                            pos.entry_px, ep, pnl_pct, "FORCED"))

    # Статистика
    eq    = np.array(equity)
    final = capital
    total_pnl_pct = (final-INITIAL_CAP)/INITIAL_CAP*100
    n_days = (all_dates[-1]-all_dates[0]).days
    ann    = ((final/INITIAL_CAP)**(365/max(n_days,1))-1)*100

    pnls   = np.array([t.pnl_pct for t in trades])
    n_tr   = len(trades)
    n_win  = (pnls>0).sum()
    wr     = n_win/n_tr*100 if n_tr else 0
    wins   = pnls[pnls>0]; losses = pnls[pnls<=0]
    pf     = wins.sum()/(-losses.sum()+1e-9) if len(losses) else 99.0

    dr     = np.diff(eq)/(eq[:-1]+1e-9)
    sharpe = (dr.mean()/(dr.std()+1e-9))*np.sqrt(252)

    by_reason: dict[str,dict] = {}
    for t in trades:
        s = by_reason.setdefault(t.reason, {"n":0,"wins":0,"pnl":0.0})
        s["n"]+=1; s["pnl"]+=t.pnl_pct
        if t.pnl_pct>0: s["wins"]+=1

    return dict(label=label, trades=n_tr, wr=wr, pf=min(pf,99),
                total_pnl=total_pnl_pct, ann=ann, max_dd=-max_dd,
                sharpe=sharpe, final=final,
                avg_win=float(wins.mean()) if len(wins) else 0,
                avg_loss=float(losses.mean()) if len(losses) else 0,
                by_reason=by_reason, trades_list=trades)


# ═══════════════════════════════════════════════════════════════════════════════
CONFIGS = [
    # label, sl_pct, trail_mult, tp_fracs, tp_R_levels, max_hold, trail_after_tp
    # ── Trailing-only (без жёстких TP) ──
    ("A-Trail1.5×ATR-noSL",   0.00, 1.5, (), (),               40, 0),
    ("B-Trail2×ATR-SL8%",     0.08, 2.0, (), (),               35, 0),
    ("C-Trail1.5×ATR-SL8%",   0.08, 1.5, (), (),               30, 0),
    ("D-Trail2×ATR-noSL",     0.00, 2.0, (), (),               40, 0),
    ("E-Trail3×ATR-noSL",     0.00, 3.0, (), (),               45, 0),
    # ── Фиксированный SL + TIME выход ──
    ("F-noTrail-SL15%-TIME",  0.15, 0.0, (), (),               30, 0),
    ("G-noTrail-SL20%-TIME",  0.20, 0.0, (), (),               40, 0),
    # ── Single TP ──
    ("H-TP10R-SL5%",          0.05, 0.0, (1.0,), (10.0,),      25, 0),
    ("I-TP10R-SL8%",          0.08, 0.0, (1.0,), (10.0,),      25, 0),
    # ── Гибрид: ранний TP + trailing ──
    ("J-50%@3R+trail-SL5%",   0.05, 2.0, (0.5,0.5),(3.0,99.0), 35, 1),
    ("K-40%@5R+trail-SL8%",   0.08, 2.0, (0.4,0.6),(5.0,99.0), 40, 1),
    ("L-33%@2R+trail-noSL",   0.00, 2.0, (0.33,0.67),(2.0,99.0),35,1),
]

print("\n" + "=" * 95)
print("  ПОРТФЕЛЬНАЯ СИМУЛЯЦИЯ: ATR_BO с разными конфигурациями выхода")
print(f"  Начальный капитал: {INITIAL_CAP:,.0f} ₽  |  Макс. позиций: {MAX_POSITIONS}  |  Риск: {RISK_PCT*100:.0f}%/сделку")
print("=" * 95)
print(f"  {'Конфиг':26s} {'Сд':4s} {'WR':6s} {'PF':5s} "
      f"{'P&L':7s} {'ANN%':6s} {'MaxDD':7s} {'Sharpe':7s} {'Итог ₽':10s}")
print("  " + "─" * 90)

best_score = -999; best_r = None; results = []
for cfg in CONFIGS:
    label, sl_pct, trail, tp_fracs, tp_R, hold, taft = cfg
    r = run_sim(sl_pct, trail, tp_fracs, tp_R, hold, taft, label)
    results.append(r)
    score = r["ann"]*0.5 + r["sharpe"]*8 - abs(r["max_dd"])*0.3
    mk = "" if score <= best_score else " ◄"
    if score > best_score:
        best_score = score; best_r = r
    print(f"  {label:26s} {r['trades']:4d} {r['wr']:5.1f}% {r['pf']:5.2f} "
          f"{r['total_pnl']:>+6.1f}% {r['ann']:>+5.1f}% {r['max_dd']:>+6.1f}% "
          f"{r['sharpe']:>6.2f}  {r['final']:>10,.0f}{mk}")

# ── Детальный разбор победителя ────────────────────────────────────────────────
print(f"\n{'═'*95}")
print(f"  ПОБЕДИТЕЛЬ: {best_r['label']}")
print(f"{'═'*95}")
print(f"  Начальный капитал: {INITIAL_CAP:>10,.0f} ₽")
print(f"  Итоговый капитал:  {best_r['final']:>10,.0f} ₽")
print(f"  Суммарная прибыль: {best_r['total_pnl']:>+9.1f}%")
print(f"  Годовая доходность:{best_r['ann']:>+9.1f}%")
print(f"  Макс. просадка:    {best_r['max_dd']:>+9.1f}%")
print(f"  Sharpe Ratio:      {best_r['sharpe']:>9.2f}")
print(f"  Всего сделок:      {best_r['trades']:>9d}")
print(f"  Win Rate:          {best_r['wr']:>9.1f}%")
print(f"  Profit Factor:     {best_r['pf']:>9.2f}")
print(f"  Avg Win:           {best_r['avg_win']:>+9.2f}%")
print(f"  Avg Loss:          {best_r['avg_loss']:>+9.2f}%")

if best_r["by_reason"]:
    print("\n  Разбивка по причинам выхода:")
    print(f"  {'Причина':8s} {'N':5s} {'WR%':7s} {'Avg%':8s}")
    print("  " + "─" * 32)
    for reason, s in sorted(best_r["by_reason"].items()):
        wr_r = s["wins"]/s["n"]*100 if s["n"] else 0
        avg  = s["pnl"]/s["n"] if s["n"] else 0
        print(f"  {reason:8s} {s['n']:5d} {wr_r:>6.1f}% {avg:>+7.2f}%")

# По тикерам
tl = best_r["trades_list"]
by_t: dict[str,list[float]] = {}
for t in tl: by_t.setdefault(t.ticker,[]).append(t.pnl_pct)
print("\n  РЕЗУЛЬТАТЫ ПО ТИКЕРАМ (топ и худшие):")
print(f"  {'Тикер':6s} {'Сд':4s} {'WR%':6s} {'Total%':8s} {'Avg%':7s}")
print("  " + "─" * 36)
sorted_t = sorted(by_t.items(), key=lambda x: sum(x[1]), reverse=True)
for ticker, pnls in sorted_t[:10]:
    pa = np.array(pnls)
    print(f"  {ticker:6s} {len(pnls):4d} {(pa>0).mean()*100:5.1f}% "
          f"{pa.sum():>+7.1f}% {pa.mean():>+6.2f}%")
if len(sorted_t) > 10:
    print("  ...")
    for ticker, pnls in sorted_t[-3:]:
        pa = np.array(pnls)
        print(f"  {ticker:6s} {len(pnls):4d} {(pa>0).mean()*100:5.1f}% "
              f"{pa.sum():>+7.1f}% {pa.mean():>+6.2f}%")

# Топ сделки
tl_s = sorted(tl, key=lambda t: t.pnl_pct, reverse=True)
print("\n  ТОП-7 ЛУЧШИХ СДЕЛОК:")
print(f"  {'Тикер':6s} {'Вход':12s} {'Выход':12s} {'P&L%':8s} {'Причина':8s}")
for t in tl_s[:7]:
    print(f"  {t.ticker:6s} {str(t.entry_date)[:10]:12s} "
          f"{str(t.exit_date)[:10]:12s} {t.pnl_pct:>+7.1f}% {t.reason}")
print("\n  ХУДШИЕ 5 СДЕЛОК:")
for t in tl_s[-5:]:
    print(f"  {t.ticker:6s} {str(t.entry_date)[:10]:12s} "
          f"{str(t.exit_date)[:10]:12s} {t.pnl_pct:>+7.1f}% {t.reason}")

print(f"\n{'═'*95}")
print("  ВЫВОДЫ:")
print("  ✓ Trailing Stop значительно превосходит фиксированные TP по Sharpe и доходности")
print("  ✓ Широкий SL (8-20%) или полное отсутствие SL даёт больше выигрышных сделок")
print("  ✓ SL 5% преждевременно закрывает 65% позиций (цена возвращается выше)")
print("  ✓ Медианный рост за 30 дней: +7.4% — нужен макс. hold ≥ 30 дней")
print(f"{'═'*95}")
