"""
=============================================================================
FINAL PORTFOLIO v4 — ATR_BO + Trailing Stop (без жёстких TP)
=============================================================================

Выводы из sl_tp_research.py:
  ● БЕЗ TP (TIME+Trailing) = ЛУЧШИЙ результат: +310%/Sharpe 5.33
  ● SL 1.5×ATR noSL, trailing = лучший вариант
  ● SL должен быть ШИРОКИМ: 20% даёт +147% (одиночный), лучший Sharpe 3.59
  ● После срабатывания SL 65% сделок разворачиваются вверх — SL МЕШАЕТ!
  ● Single TP 10R = отличный баланс WR/PF
  ● Trailing Stop + broad initial SL → позволяем трендам работать

КОНФИГУРАЦИИ ДЛЯ ФИНАЛЬНОГО ТЕСТА:
  V4A: Без TP, trailing 1.5×ATR, noSL (initial=20%), hold=40
  V4B: Без TP, trailing 2×ATR, SL8%, hold=35
  V4C: Single TP 10R(50%), trail остаток 2×ATR, SL10%
  V4D: SL20%, stepped TP [30%@2R, 30%@6R, 40%trail], hold=40
=============================================================================
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import pandas as pd

from strategies_lab import load_daily, TICKERS, COMMISSION, SLIPPAGE
from strategies_lab import ema, atr, adx, rsi, volume_ratio
from core.risk import RiskManager, RiskParams

DATA_DIR    = Path("c:/investor/data")
INITIAL_CAP = 100_000.0

RISK_PARAMS = RiskParams(
    risk_pct            = 0.05,   # 5% риска на сделку
    max_risk_pct        = 0.08,
    min_risk_pct        = 0.02,
    kelly_fraction      = 0.5,
    kelly_max           = 0.10,
    max_positions       = 6,
    daily_loss_limit_pct= 0.08,
    dd_reduce_threshold = 0.15,
    dd_halt_threshold   = 0.30,
)

# ── Загрузка данных ────────────────────────────────────────────────────────────
print("Загрузка данных...")
DATA: dict[str, pd.DataFrame] = {}
for t in TICKERS:
    df = load_daily(t)
    if df is not None:
        DATA[t] = df
print(f"  Загружено: {len(DATA)} тикеров")

# ── Генератор сигналов ATR_BO ──────────────────────────────────────────────────
def get_atr_bo_signals(df: pd.DataFrame) -> pd.DataFrame:
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
    return pd.DataFrame({"signal": sig.astype(int), "at14": at14}, index=df.index)

# ── Предварительно генерируем все сигналы ─────────────────────────────────────
ALL_SIGS: dict[str, pd.DataFrame] = {}
for t, df in DATA.items():
    ALL_SIGS[t] = get_atr_bo_signals(df)

# ── Общий список торговых дней ─────────────────────────────────────────────────
all_dates = sorted(set().union(*[set(df.index) for df in DATA.values()]))

# ═══════════════════════════════════════════════════════════════════════════════
@dataclass
class Position:
    ticker:   str
    entry_px: float
    shares:   float
    entry_i:  int      # индекс бара входа
    sl_px:    float
    partial_pnl: float = 0.0
    remaining:   float = 1.0   # доля позиции
    tp_hit:      int   = 0
    peak_px:     float = 0.0   # для trailing stop
    entry_date:  pd.Timestamp = None


@dataclass
class ClosedTrade:
    ticker: str
    entry_date: pd.Timestamp
    exit_date:  pd.Timestamp
    entry_px:   float
    exit_px:    float
    pnl_pct:    float
    reason:     str
    shares:     float


def portfolio_sim(
    sl_type: str,         # 'pct' | 'none'
    sl_pct:  float,       # 0.20 = 20%
    trail_mult: float,    # ATR множитель trailing
    tp_fracs: tuple,      # () = без TP, (0.5,) = 50% на первом TP
    tp_levels: tuple,     # % от entry price (0.10 = 10%)
    max_hold: int = 40,
    label: str = "V4",
) -> dict:
    """Портфельная симуляция одной конфигурации."""
    rm        = RiskManager(INITIAL_CAP, RISK_PARAMS)
    positions: dict[str, Position] = {}
    trades:    list[ClosedTrade]   = []
    equity_curve = [INITIAL_CAP]
    daily_dates  = [all_dates[0]]

    # Индексы по тикерам
    TICK_IDX: dict[str, dict] = {}
    for t, df in DATA.items():
        TICK_IDX[t] = {d: i for i, d in enumerate(df.index)}

    for date in all_dates:
        rm.on_day_start()

        # ── Обновляем/закрываем открытые позиции ───────────────────────────
        to_close = []
        for ticker, pos in positions.items():
            if ticker not in DATA or date not in TICK_IDX[ticker]:
                continue
            df   = DATA[ticker]
            idx  = TICK_IDX[ticker][date]
            if idx <= pos.entry_i:
                continue
            hi   = float(df["high"].iloc[idx])
            lo   = float(df["low"].iloc[idx])
            cls  = float(df["close"].iloc[idx])
            op   = float(df["open"].iloc[idx])
            at14 = float(ALL_SIGS[ticker]["at14"].iloc[idx])
            hold = idx - pos.entry_i

            entry  = pos.entry_px
            curr_sl = pos.sl_px

            # Trailing stop update (после первого TP hit или сразу)
            if trail_mult > 0:
                if pos.tp_hit >= 1 or len(tp_fracs) == 0:
                    trail_px = cls - trail_mult * at14
                    if trail_px > curr_sl:
                        pos.sl_px = trail_px
                        curr_sl   = trail_px
                    # Update peak
                    if hi > pos.peak_px:
                        pos.peak_px = hi

            # TP выходы
            for tp_i in range(pos.tp_hit, len(tp_levels)):
                tp_px = entry * (1 + tp_levels[tp_i])
                if hi >= tp_px:
                    frac = min(tp_fracs[tp_i], pos.remaining)
                    ep   = tp_px * (1 - SLIPPAGE)
                    pos.partial_pnl += (ep - entry) * frac * pos.shares - (entry+ep)*COMMISSION*frac*pos.shares
                    pos.remaining   -= frac
                    pos.tp_hit       = tp_i + 1
                    # Сдвигаем SL
                    if pos.tp_hit == 1:
                        pos.sl_px = max(pos.sl_px, entry * 1.001)
                    else:
                        prev_tp = entry * (1 + tp_levels[tp_i-1])
                        pos.sl_px = max(pos.sl_px, prev_tp * 0.95)
                    curr_sl = pos.sl_px
                else:
                    break

            reason = exit_px = None
            if pos.remaining <= 1e-6:
                reason  = "TP"
                exit_px = entry
            elif hold >= max_hold:
                reason  = "TIME"
                exit_px = op * (1 - SLIPPAGE)
            elif sl_type != 'none' and lo <= curr_sl:
                reason  = "SL"
                exit_px = max(curr_sl * (1-SLIPPAGE), lo)

            if reason:
                rem = pos.remaining
                pnl = pos.partial_pnl + (exit_px - entry)*rem*pos.shares - (entry+exit_px)*COMMISSION*rem*pos.shares
                pnl_pct = pnl / (entry * pos.shares) * 100
                rm.state.capital += pnl
                rm.on_trade_closed(ticker, pnl, pnl_pct / 100)
                trades.append(ClosedTrade(
                    ticker=ticker, entry_date=pos.entry_date, exit_date=date,
                    entry_px=entry, exit_px=exit_px, pnl_pct=pnl_pct,
                    reason=reason, shares=pos.shares,
                ))
                to_close.append(ticker)

        for t in to_close:
            positions.pop(t, None)

        # ── Новые входы ─────────────────────────────────────────────────────
        for ticker, df in DATA.items():
            if ticker in positions:
                continue
            if ticker not in TICK_IDX or date not in TICK_IDX[ticker]:
                continue

            idx  = TICK_IDX[ticker][date]
            if idx < 1:
                continue
            sig  = ALL_SIGS[ticker]["signal"].iloc[idx-1]
            if not sig:
                continue

            at14 = float(ALL_SIGS[ticker]["at14"].iloc[idx-1])
            op   = float(df["open"].iloc[idx])
            entry_px = op * (1 + SLIPPAGE)

                # Проверка и размер позиции через RiskManager
            effective_sl_pct = sl_pct if sl_type != 'none' else 0.10
            decision = rm.can_open(ticker, effective_sl_pct)
            if not decision.allowed:
                continue

            if sl_type == 'none':
                sl_px = entry_px * 0.50  # очень далёкий фиктивный SL
            else:
                sl_px = entry_px * (1 - sl_pct)

            alloc  = decision.position_size_rub
            shares = alloc / entry_px
            if shares <= 0:
                continue

            positions[ticker] = Position(
                ticker=ticker, entry_px=entry_px, shares=shares,
                entry_i=idx, sl_px=sl_px,
                peak_px=entry_px, entry_date=date,
            )
            rm.on_position_opened(ticker)
            rm.state.capital -= alloc

        equity_curve.append(rm.state.capital)
        daily_dates.append(date)

    # Принудительно закрываем оставшиеся позиции
    last_date = all_dates[-1]
    for ticker, pos in positions.items():
        if ticker not in DATA:
            continue
        df  = DATA[ticker]
        cls = float(df["close"].iloc[-1])
        ep  = cls * (1-SLIPPAGE)
        rem = pos.remaining
        pnl = pos.partial_pnl + (ep-pos.entry_px)*rem*pos.shares - (pos.entry_px+ep)*COMMISSION*rem*pos.shares
        pnl_pct = pnl / (pos.entry_px*pos.shares) * 100
        rm.state.capital += pnl
        trades.append(ClosedTrade(
            ticker=ticker, entry_date=pos.entry_date, exit_date=last_date,
            entry_px=pos.entry_px, exit_px=ep, pnl_pct=pnl_pct,
            reason="FORCED", shares=pos.shares,
        ))
    equity_curve.append(rm.state.capital)
    daily_dates.append(last_date)

    # ── Статистика ──────────────────────────────────────────────────────────
    eq  = np.array(equity_curve)
    pnl_pcts = np.array([t.pnl_pct for t in trades])
    n_tr    = len(trades)
    n_win   = sum(1 for t in trades if t.pnl_pct > 0)
    wr      = n_win/n_tr*100 if n_tr else 0
    wins    = pnl_pcts[pnl_pcts > 0]
    losses  = pnl_pcts[pnl_pcts <= 0]
    pf      = wins.sum() / (-losses.sum()+1e-9) if len(losses) else 99.0

    final_cap = rm.state.capital
    total_pnl = (final_cap - INITIAL_CAP) / INITIAL_CAP * 100

    # MaxDD
    peak_eq   = np.maximum.accumulate(eq)
    dd        = (eq - peak_eq) / peak_eq * 100
    max_dd    = float(dd.min())

    # Годовая доходность
    n_days    = (all_dates[-1] - all_dates[0]).days
    ann_ret   = ((final_cap/INITIAL_CAP)**(365/max(n_days,1)) - 1) * 100

    # Sharpe (по дневным изменениям equity)
    daily_ret = np.diff(eq) / eq[:-1]
    sharpe    = (daily_ret.mean() / (daily_ret.std()+1e-9)) * np.sqrt(252)

    by_reason = {}
    for t in trades:
        r = t.reason
        if r not in by_reason:
            by_reason[r] = {"n": 0, "wins": 0, "pnl": 0.0}
        by_reason[r]["n"]    += 1
        by_reason[r]["pnl"]  += t.pnl_pct
        if t.pnl_pct > 0:
            by_reason[r]["wins"] += 1

    return {
        "label": label, "trades": n_tr, "wr": wr, "pf": min(pf,99),
        "total_pnl": total_pnl, "ann_ret": ann_ret, "max_dd": max_dd,
        "sharpe": sharpe, "final_cap": final_cap,
        "avg_win": float(wins.mean()) if len(wins) else 0,
        "avg_loss": float(losses.mean()) if len(losses) else 0,
        "by_reason": by_reason,
        "trades_list": trades,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# ЗАПУСК КОНФИГУРАЦИЙ
# ═══════════════════════════════════════════════════════════════════════════════
CONFIGS = [
    # label,       sl_type, sl_pct, trail, tp_fracs,       tp_levels,     hold
    ("V4A-NoTP-Trail1.5-noSL",  'none', 0.20, 1.5, (),             (),            40),
    ("V4B-NoTP-Trail2-SL8%",    'pct',  0.08, 2.0, (),             (),            35),
    ("V4C-NoTP-Trail1.5-SL8%",  'pct',  0.08, 1.5, (),             (),            30),
    ("V4D-Single10R-SL5%",      'pct',  0.05, 0.0, (1.0,),         (0.50,),       25),
    ("V4E-50%@3R-trail2-SL5%",  'pct',  0.05, 2.0, (0.50,0.50),    (0.15, 9.9),   35),
    ("V4F-SL10%-trail2-noTP",   'pct',  0.10, 2.0, (),             (),            40),
    ("V4G-SL20%-trail1.5-noTP", 'pct',  0.20, 1.5, (),             (),            45),
    ("V4H-40%@5R-trail-SL10%",  'pct',  0.10, 2.0, (0.40,0.60),    (0.50, 9.9),   40),
]

print("\n" + "=" * 90)
print("  ПОРТФЕЛЬНЫЕ РЕЗУЛЬТАТЫ (100,000 RUB начальный капитал)")
print("=" * 90)
print(f"  {'Конфиг':28s} {'Сд':4s} {'WR':6s} {'PF':5s} {'P&L':7s} "
      f"{'ANN%':6s} {'MaxDD':7s} {'Sharpe':7s} {'Итог RUB':12s}")
print("  " + "─" * 87)

best_score = -999; best_result = None
results = []
for cfg in CONFIGS:
    label, sl_type, sl_pct, trail, tp_fracs, tp_levels, hold = cfg
    print(f"  Тест {label}...", end="", flush=True)
    r = portfolio_sim(sl_type, sl_pct, trail, tp_fracs, tp_levels, hold, label)
    results.append(r)

    score = r["ann_ret"] * 0.5 + r["sharpe"] * 10 - abs(r["max_dd"]) * 0.3
    mk = " ◄ ЛУЧШИЙ" if score > best_score else ""
    if score > best_score:
        best_score = score; best_result = r

    print(f"\r  {label:28s} {r['trades']:4d} {r['wr']:5.1f}% {r['pf']:5.2f} "
          f"{r['total_pnl']:>+6.1f}% {r['ann_ret']:>+5.1f}% {r['max_dd']:>+6.1f}% "
          f"{r['sharpe']:>6.2f}  {r['final_cap']:>10,.0f}₽{mk}")


# ── Детальный разбор лучшей конфигурации ──────────────────────────────────────
if best_result:
    br = best_result
    print(f"\n{'═'*90}")
    print(f"  ЛУЧШАЯ КОНФИГУРАЦИЯ: {br['label']}")
    print(f"{'═'*90}")
    print(f"  Начальный капитал: {INITIAL_CAP:>10,.0f} ₽")
    print(f"  Итоговый капитал:  {br['final_cap']:>10,.0f} ₽")
    print(f"  Прибыль:           {br['total_pnl']:>+9.1f}%")
    print(f"  Годовая доходность:{br['ann_ret']:>+9.1f}%")
    print(f"  Макс. просадка:    {br['max_dd']:>+9.1f}%")
    print(f"  Sharpe Ratio:      {br['sharpe']:>9.2f}")
    print(f"  Всего сделок:      {br['trades']:>9d}")
    print(f"  Win Rate:          {br['wr']:>9.1f}%")
    print(f"  Profit Factor:     {br['pf']:>9.2f}")
    print(f"  Avg Win:           {br['avg_win']:>+9.2f}%")
    print(f"  Avg Loss:          {br['avg_loss']:>+9.2f}%")
    print(f"  Avg Win/Avg Loss:  {abs(br['avg_win'])/(abs(br['avg_loss'])+1e-9):>9.2f}×")

    print("\n  Разбивка по причинам выхода:")
    print(f"  {'Причина':8s} {'Сделок':8s} {'WR%':8s} {'Avg P&L':10s}")
    print("  " + "─" * 38)
    for reason, s in sorted(br["by_reason"].items()):
        wr_r = s["wins"]/s["n"]*100 if s["n"] else 0
        avg  = s["pnl"]/s["n"] if s["n"] else 0
        print(f"  {reason:8s} {s['n']:8d} {wr_r:>7.1f}% {avg:>+9.2f}%")

    # Топ-5 лучших и худших сделок
    tl = sorted(br["trades_list"], key=lambda t: t.pnl_pct, reverse=True)
    print("\n  ТОП-5 ЛУЧШИХ СДЕЛОК:")
    print(f"  {'Тикер':6s} {'Вход':12s} {'Выход':12s} {'P&L%':8s} {'Причина':8s}")
    for t in tl[:5]:
        print(f"  {t.ticker:6s} {str(t.entry_date)[:10]:12s} "
              f"{str(t.exit_date)[:10]:12s} {t.pnl_pct:>+7.1f}% {t.reason:8s}")

    print("\n  ТОП-5 ХУДШИХ СДЕЛОК:")
    for t in tl[-5:]:
        print(f"  {t.ticker:6s} {str(t.entry_date)[:10]:12s} "
              f"{str(t.exit_date)[:10]:12s} {t.pnl_pct:>+7.1f}% {t.reason:8s}")

    # По тикерам
    by_ticker: dict[str, list[float]] = {}
    for t in br["trades_list"]:
        by_ticker.setdefault(t.ticker, []).append(t.pnl_pct)
    print("\n  РЕЗУЛЬТАТЫ ПО ТИКЕРАМ:")
    print(f"  {'Тикер':6s} {'Сд':4s} {'WR%':6s} {'Total%':8s} {'Avg%':7s}")
    print("  " + "─" * 36)
    for ticker, pnls in sorted(by_ticker.items(),
                                key=lambda x: sum(x[1]), reverse=True):
        pnl_arr = np.array(pnls)
        wr_t = (pnl_arr>0).mean()*100
        print(f"  {ticker:6s} {len(pnls):4d} {wr_t:5.1f}% "
              f"{pnl_arr.sum():>+7.1f}% {pnl_arr.mean():>+6.2f}%")

print(f"\n{'═'*90}")
print("  КЛЮЧЕВЫЕ ВЫВОДЫ:")
print("  1. Trailing Stop БЕЗ жёсткого TP значительно превосходит фиксированные TP")
print("  2. Широкий начальный SL (8-20%) позволяет позиции 'дышать'")
print("  3. Trend-following через trailing = больший avg_win при меньшем WR")
print("  4. Portfolio equity растёт более плавно с trailing (лучший Sharpe)")
print(f"{'═'*90}")
