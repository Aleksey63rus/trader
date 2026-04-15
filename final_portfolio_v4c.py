"""
=============================================================================
FINAL PORTFOLIO v4c — Правильная портфельная симуляция
=============================================================================

ИСПРАВЛЕНИЯ:
  - Позиции измеряются в АКЦИЯХ (shares), не в долях капитала
  - Капитал = свободные деньги + стоимость позиций
  - Аллокация = % от ТЕКУЩЕГО полного капитала (включая открытые позиции)
  - Размер позиции не более капитал / MAX_POSITIONS

КЛЮЧЕВЫЕ ВЫВОДЫ из исследования:
  ● TIME выходы: avg +6-9%, WR 55-68% — основной источник прибыли
  ● Trailing stop позволяет взять +30-42% на лучших сделках
  ● SL 5% убивает 65% потенциально прибыльных сделок
  ● Медианный рост паттерна за 30 дней: +7.4%

ЦЕЛЬ: найти конфигурацию с ANN% > 15% и MaxDD < 20%
=============================================================================
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
from dataclasses import dataclass
import numpy as np
import pandas as pd

from strategies_lab import load_daily, TICKERS, COMMISSION, SLIPPAGE
from strategies_lab import ema, atr, adx, rsi, volume_ratio

INITIAL_CAP  = 100_000.0
MAX_POSITIONS = 4          # одновременно
RISK_PCT      = 0.20       # 20% капитала на одну позицию (при 4 позициях = 80% в работе)

print("Загрузка данных...")
DATA: dict[str, pd.DataFrame] = {}
for t in TICKERS:
    df = load_daily(t)
    if df is not None:
        DATA[t] = df
print(f"  Загружено: {len(DATA)} тикеров")

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
all_dates = sorted(set().union(*[set(df.index) for df in DATA.values()]))

CORR_GROUPS: list[set[str]] = [
    {"SBER", "SBERP", "T", "VTBR"},
    {"LKOH", "ROSN", "NVTK", "GAZP", "SNGS", "SNGSP"},
    {"NLMK", "MTLR", "CHMF", "MAGN"},
    {"TATN", "TATNP"},
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
    shares:   float     # количество акций (фиксируется на входе)
    entry_i:  int
    sl_px:    float
    partial_cash: float = 0.0   # кэш от частичных закрытий TP
    remaining:    float = 1.0   # доля ещё открытой позиции (1.0 = вся)
    tp_hit:       int   = 0
    entry_date:   object = None


@dataclass
class Trade:
    ticker: str; entry_date: object; exit_date: object
    entry_px: float; exit_px: float; pnl_rub: float
    pnl_pct: float; reason: str; shares: float


def run_sim(
    sl_pct: float,
    trail_mult: float,
    tp_fracs: tuple,
    tp_r_levels: tuple,
    max_hold: int,
    trail_after_tp: int = 0,
    label: str = "",
) -> dict:
    free_cash  = INITIAL_CAP
    positions: dict[str, Position] = {}
    trades:    list[Trade] = []
    equity     = [INITIAL_CAP]
    peak_eq    = INITIAL_CAP
    max_dd_pct = 0.0

    TICK_IDX = {t: {d: i for i, d in enumerate(df.index)}
                for t, df in DATA.items()}

    for date in all_dates:
        # ── 1. Считаем текущую стоимость портфеля ─────────────────────────────
        pos_val = 0.0
        for ticker, pos in positions.items():
            idx = TICK_IDX[ticker].get(date)
            if idx is not None:
                px = float(DATA[ticker]["close"].iloc[idx])
            else:
                px = pos.entry_px
            pos_val += px * pos.shares * pos.remaining + pos.partial_cash

        total_cap = free_cash + pos_val

        # ── 2. Обновляем/закрываем позиции ────────────────────────────────────
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

            # TP hits
            for k in range(pos.tp_hit, len(tp_fracs)):
                ref_risk = pos.entry_px * sl_pct if sl_pct > 0 else pos.entry_px * 0.05
                tp_px = pos.entry_px + tp_r_levels[k] * ref_risk
                if hi >= tp_px:
                    frac  = min(tp_fracs[k], pos.remaining)
                    ep    = tp_px * (1-SLIPPAGE)
                    sell_shares = frac * pos.shares
                    cash_in  = ep * sell_shares * (1 - COMMISSION)
                    cost_in  = pos.entry_px * sell_shares * (1 + COMMISSION)
                    pos.partial_cash += cash_in
                    pos.remaining    -= frac
                    pos.tp_hit        = k+1
                    if k == 0:
                        pos.sl_px = max(pos.sl_px, pos.entry_px * 1.001)
                    else:
                        prev_tp = pos.entry_px + tp_r_levels[k-1]*ref_risk
                        pos.sl_px = max(pos.sl_px, prev_tp*0.97)
                    sl = pos.sl_px
                else:
                    break

            reason = exit_px = None
            if pos.remaining <= 1e-6:
                reason = "TP"; exit_px = pos.entry_px
            elif hold >= max_hold:
                reason = "TIME"; exit_px = op*(1-SLIPPAGE)
            elif sl_pct > 0 and lo <= sl:
                reason = "SL"; exit_px = max(sl*(1-SLIPPAGE), lo)

            if reason:
                rem_shares = pos.remaining * pos.shares
                cash_recv  = exit_px * rem_shares * (1-COMMISSION)
                cost_orig  = pos.entry_px * rem_shares * (1+COMMISSION)
                total_cash = pos.partial_cash + cash_recv
                total_cost = pos.entry_px * pos.shares * (1+COMMISSION)
                pnl_rub    = total_cash - total_cost
                pnl_pct    = pnl_rub / total_cost * 100

                free_cash += total_cash
                trades.append(Trade(ticker, pos.entry_date, date,
                                    pos.entry_px, exit_px, pnl_rub, pnl_pct, reason,
                                    pos.shares))
                to_close.append(ticker)

        for t in to_close:
            positions.pop(t, None)

        # ── 3. Пересчёт полного капитала после закрытий ────────────────────────
        pos_val2 = 0.0
        for ticker, pos in positions.items():
            idx = TICK_IDX[ticker].get(date)
            px  = float(DATA[ticker]["close"].iloc[idx]) if idx is not None else pos.entry_px
            pos_val2 += px * pos.shares * pos.remaining + pos.partial_cash
        total_cap = free_cash + pos_val2

        # ── 4. Новые входы ─────────────────────────────────────────────────────
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

            entry   = float(df["open"].iloc[idx]) * (1+SLIPPAGE)
            sl_init = entry*(1-sl_pct) if sl_pct > 0 else entry * 0.40

            # Размер: RISK_PCT от текущего полного капитала
            alloc   = total_cap * RISK_PCT
            # Но не более свободного кэша
            alloc   = min(alloc, free_cash * 0.95)
            if alloc <= 0:
                continue

            shares  = alloc / entry
            cost    = shares * entry * (1+COMMISSION)
            if cost > free_cash:
                continue

            free_cash -= cost
            positions[ticker] = Position(
                ticker, entry, shares, idx, sl_init, entry_date=date,
            )

        # Equity
        pos_val3 = 0.0
        for ticker, pos in positions.items():
            idx = TICK_IDX[ticker].get(date)
            px  = float(DATA[ticker]["close"].iloc[idx]) if idx is not None else pos.entry_px
            pos_val3 += px * pos.shares * pos.remaining + pos.partial_cash
        eq = free_cash + pos_val3
        equity.append(eq)
        if eq > peak_eq:
            peak_eq = eq
        dd = (peak_eq - eq) / peak_eq * 100
        if dd > max_dd_pct:
            max_dd_pct = dd

    # Принудительное закрытие
    last_date = all_dates[-1]
    for ticker, pos in list(positions.items()):
        cls = float(DATA[ticker]["close"].iloc[-1])
        ep  = cls*(1-SLIPPAGE)
        rem_shares = pos.remaining * pos.shares
        cash_recv  = ep * rem_shares * (1-COMMISSION)
        total_cash = pos.partial_cash + cash_recv
        total_cost = pos.entry_px * pos.shares * (1+COMMISSION)
        pnl_rub    = total_cash - total_cost
        pnl_pct    = pnl_rub / total_cost * 100
        free_cash  += total_cash
        trades.append(Trade(ticker, pos.entry_date, last_date,
                            pos.entry_px, ep, pnl_rub, pnl_pct, "FORCED", pos.shares))

    # Статистика
    final      = free_cash
    total_pnl  = (final - INITIAL_CAP) / INITIAL_CAP * 100
    n_days     = (all_dates[-1]-all_dates[0]).days
    ann        = ((final/INITIAL_CAP)**(365/max(n_days,1))-1)*100

    pnls   = np.array([t.pnl_pct for t in trades])
    n_tr   = len(trades)
    n_win  = (pnls>0).sum()
    wr     = n_win/n_tr*100 if n_tr else 0
    wins   = pnls[pnls>0]; losses = pnls[pnls<=0]
    pf     = wins.sum()/(-losses.sum()+1e-9) if len(losses) else 99.0

    eq_arr = np.array(equity)
    dr     = np.diff(eq_arr)/(eq_arr[:-1]+1e-9)
    sharpe = (dr.mean()/(dr.std()+1e-9))*np.sqrt(252)

    by_reason: dict[str,dict] = {}
    for t in trades:
        s = by_reason.setdefault(t.reason, {"n":0,"wins":0,"pnl":0.0})
        s["n"]+=1; s["pnl"]+=t.pnl_pct
        if t.pnl_pct>0: s["wins"]+=1

    return dict(label=label, trades=n_tr, wr=wr, pf=min(pf,99),
                total_pnl=total_pnl, ann=ann, max_dd=-max_dd_pct,
                sharpe=sharpe, final=final,
                avg_win=float(wins.mean()) if len(wins) else 0,
                avg_loss=float(losses.mean()) if len(losses) else 0,
                by_reason=by_reason, trades_list=trades)


# ═══════════════════════════════════════════════════════════════════════════════
CONFIGS = [
    # label, sl_pct, trail_mult, tp_fracs, tp_R_levels, max_hold, trail_after_tp
    ("A-Trail1.5-noSL-h40",    0.00, 1.5, (), (),                40, 0),
    ("B-Trail2-noSL-h45",      0.00, 2.0, (), (),                45, 0),
    ("C-Trail2-SL8%-h35",      0.08, 2.0, (), (),                35, 0),
    ("D-Trail1.5-SL8%-h30",    0.08, 1.5, (), (),                30, 0),
    ("E-noTrail-SL20%-h40",    0.20, 0.0, (), (),                40, 0),
    ("F-noTrail-SL15%-h30",    0.15, 0.0, (), (),                30, 0),
    ("G-TP10R-noTrail-SL5%",   0.05, 0.0, (1.0,),(10.0,),        25, 0),
    ("H-TP10R-noTrail-SL8%",   0.08, 0.0, (1.0,),(10.0,),        25, 0),
    ("I-50%@3R+trail2-SL5%",   0.05, 2.0, (0.5,0.5),(3.0,99.0),  35, 1),
    ("J-40%@5R+trail2-SL8%",   0.08, 2.0, (0.4,0.6),(5.0,99.0),  40, 1),
    ("K-Trail2-SL5%-h30",      0.05, 2.0, (), (),                30, 0),
    ("L-Trail2-SL10%-h40",     0.10, 2.0, (), (),                40, 0),
]

print("\n" + "=" * 95)
print(f"  ATR_BO ПОРТФЕЛЬ — Правильная симуляция | Cap={INITIAL_CAP:,.0f}₽ | "
      f"MaxPos={MAX_POSITIONS} | Alloc={RISK_PCT*100:.0f}%/позицию")
print("=" * 95)
print(f"  {'Конфиг':26s} {'Сд':4s} {'WR':6s} {'PF':5s} "
      f"{'P&L':7s} {'ANN%':6s} {'MaxDD':7s} {'Sharpe':7s} {'Итог ₽':10s}")
print("  " + "─" * 90)

best_score = -999; best_r = None; results = []
for cfg in CONFIGS:
    label, sl_pct, trail, tp_fracs, tp_R, hold, taft = cfg
    r = run_sim(sl_pct, trail, tp_fracs, tp_R, hold, taft, label)
    results.append(r)
    score = r["ann"]*0.6 + r["sharpe"]*10 - abs(r["max_dd"])*0.5
    mk = " ◄" if score > best_score else ""
    if score > best_score:
        best_score = score; best_r = r
    print(f"  {label:26s} {r['trades']:4d} {r['wr']:5.1f}% {r['pf']:5.2f} "
          f"{r['total_pnl']:>+6.1f}% {r['ann']:>+5.1f}% {r['max_dd']:>+6.1f}% "
          f"{r['sharpe']:>6.2f}  {r['final']:>10,.0f}{mk}")

# ── Детальный разбор ────────────────────────────────────────────────────────────
print(f"\n{'═'*95}")
print(f"  ЛУЧШАЯ КОНФИГУРАЦИЯ: {best_r['label']}")
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

print("\n  Разбивка по причинам выхода:")
print(f"  {'Причина':8s} {'N':5s} {'WR%':7s} {'Avg%':8s}")
print("  " + "─" * 32)
for reason, s in sorted(best_r["by_reason"].items()):
    wr_r = s["wins"]/s["n"]*100 if s["n"] else 0
    avg  = s["pnl"]/s["n"] if s["n"] else 0
    print(f"  {reason:8s} {s['n']:5d} {wr_r:>6.1f}% {avg:>+7.2f}%")

tl = best_r["trades_list"]
by_t: dict[str,list[float]] = {}
for t in tl: by_t.setdefault(t.ticker,[]).append(t.pnl_pct)
print("\n  ПО ТИКЕРАМ:")
print(f"  {'Тикер':6s} {'N':4s} {'WR%':6s} {'Total%':8s} {'Avg%':7s}")
print("  " + "─" * 36)
for ticker, pnls in sorted(by_t.items(), key=lambda x: sum(x[1]), reverse=True):
    pa = np.array(pnls)
    print(f"  {ticker:6s} {len(pnls):4d} {(pa>0).mean()*100:5.1f}% "
          f"{pa.sum():>+7.1f}% {pa.mean():>+6.2f}%")

tl_s = sorted(tl, key=lambda t: t.pnl_pct, reverse=True)
print("\n  ТОП-7 ЛУЧШИХ СДЕЛОК:")
for t in tl_s[:7]:
    print(f"  {t.ticker:6s} {str(t.entry_date)[:10]} → {str(t.exit_date)[:10]} "
          f"{t.pnl_pct:>+7.1f}%  {t.reason}")
print("\n  ХУДШИЕ 5 СДЕЛОК:")
for t in tl_s[-5:]:
    print(f"  {t.ticker:6s} {str(t.entry_date)[:10]} → {str(t.exit_date)[:10]} "
          f"{t.pnl_pct:>+7.1f}%  {t.reason}")

print(f"\n{'═'*95}")
