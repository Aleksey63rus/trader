"""
Профессиональная портфельная симуляция с полным риск-менеджментом.

Уровни защиты капитала (как в проп-конторах):
  1. Position Sizing  — риск 1% капитала на сделку, лот рассчитывается автоматически
  2. Kelly Criterion  — оптимальный % риска, адаптируется по мере накопления статистики
  3. Daily Loss Limit — 3% потерь за день = стоп торговли до следующего дня
  4. Max DD Guard     — просадка 10% = лот × 0.5; просадка 15% = заморозка торговли
  5. Correlation Filter — нельзя открывать >1 позиции из одной корреляционной группы

Стратегия: Momentum Filter + Stepped TP (схема F) на дневном таймфрейме.
Начальный капитал: 100 000 руб.
"""
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict
from typing import Optional
import numpy as np
import pandas as pd

from core.indicators import (
    adx, atr, directional_index,
    efficiency_ratio, ema, macd_histogram, rsi, volume_ratio,
)
from core.risk import RiskManager, RiskParams

# ── Конфиг ────────────────────────────────────────────────────────────────────
INITIAL_CAPITAL = 100_000.0
COMMISSION      = 0.0005     # 0.05% per side
SLIPPAGE        = 0.0001     # 0.01%
MAX_HOLD        = 20         # дней

# Схема TP: F (лучшая WR по бэктесту)
FRACS  = (0.50, 0.30, 0.20)
LEVELS = (0.8,  1.8,  3.5)

TICKERS = [
    # Нефть и газ
    "GAZP", "LKOH", "NVTK", "ROSN", "SNGS", "SNGSP",
    # Банки
    "SBER", "SBERP", "T", "VTBR",
    # Металлургия и добыча
    "GMKN", "NLMK", "MTLR", "CHMF", "MAGN", "RUAL", "ALRS", "PLZL",
    # Технологии и ритейл
    "YDEX", "OZON", "MGNT", "X5",
    # Татнефть
    "TATN", "TATNP",
    # Прочие
    "AFLT", "TGKA", "IRAO", "MTSS", "PHOR", "OZPH",
]

DATA_DIR = Path("c:/investor/data")


# ── Генератор сигналов (дневной ТФ) ──────────────────────────────────────────
def build_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Возвращает DataFrame с колонками: signal, sl, risk."""
    c = df["close"]

    at14  = atr(df, 14);  at5 = atr(df, 5)
    er20  = efficiency_ratio(c, 20)
    er10  = efficiency_ratio(c, 10)
    e200  = ema(c, 200);  e50 = ema(c, 50);  e20 = ema(c, 20)
    rsi14 = rsi(df, 14)
    mh    = macd_histogram(df)
    di_p, di_n = directional_index(df, 14)
    adx14 = adx(df, 14)
    vol_r = volume_ratio(df, 20)
    at_r  = at5 / at14

    ema20_slope = (e20 - e20.shift(5)) / e20.shift(5).replace(0, np.nan) * 100

    er_base = (er20 > 0.35) & (er10 > er20) & (c > e50) & (c.diff() > 0)

    quality = (
        (c > e200) & (di_p > di_n) & (mh > 0) &
        (er20 > 0.38) & (rsi14 >= 55) & (rsi14 <= 85) & (ema20_slope > 0)
    )

    cl_a = quality & (vol_r > 2.0) & (er20 > 0.42)
    cl_b = quality & (at_r > 1.10) & (er20 > 0.43) & (vol_r > 1.3)
    cl_c = quality & (vol_r > 2.5) & (adx14 > 25) & (er20 > 0.45)

    signal_raw = er_base & (cl_a | cl_b | cl_c)
    signal     = signal_raw & ~signal_raw.shift(1).fillna(False)

    sl_raw = c - 2.0 * at14
    sl = sl_raw.clip(lower=c * 0.92, upper=c * 0.97)   # SL: 3-8% от цены
    risk = (c - sl).clip(lower=0.001)

    return pd.DataFrame({"signal": signal.astype(int), "sl": sl, "risk": risk},
                        index=df.index)


# ── Загрузчик данных ──────────────────────────────────────────────────────────
def load_daily(ticker: str) -> Optional[pd.DataFrame]:
    path = DATA_DIR / f"{ticker}_2022_2026_D.csv"
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path, sep=";", header=0)
        df.columns = [c.strip("<>").lower() for c in df.columns]
        df["datetime"] = pd.to_datetime(df["date"].astype(str),
                                        format="%d/%m/%y", errors="coerce")
        df = (df.dropna(subset=["datetime"])
                .set_index("datetime")
                .rename(columns={"vol": "volume"})
               [["open", "high", "low", "close", "volume"]])
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        return df.dropna()
    except Exception as e:
        print(f"  Ошибка {ticker}: {e}")
        return None


# ── Dataclasses ───────────────────────────────────────────────────────────────
@dataclass
class Position:
    ticker:     str
    entry_idx:  int
    entry_dt:   object
    entry:      float
    sl:         float
    risk:       float
    pos_size:   float        # размер позиции в рублях
    shares:     float        # кол-во акций (pos_size / entry)
    remaining:  float = 1.0
    partial_pnl: float = 0.0
    tp_levels_hit: int = 0
    risk_pct_used: float = 0.0


@dataclass
class ClosedTrade:
    ticker:     str
    entry_dt:   object
    exit_dt:    object
    entry:      float
    exit_price: float
    reason:     str
    pos_size:   float
    pnl_rub:    float
    pnl_pct:    float
    hold_days:  int
    win:        bool
    tp_levels:  int
    risk_pct:   float
    capital_after: float


# ── Портфельный симулятор ─────────────────────────────────────────────────────
class ProPortfolioSimulator:
    def __init__(self):
        self.rm = RiskManager(
            initial_capital=INITIAL_CAPITAL,
            params=RiskParams(
                risk_pct              = 0.01,   # 1% на сделку
                max_risk_pct          = 0.02,
                min_risk_pct          = 0.005,
                max_positions         = 3,
                daily_loss_limit_pct  = 0.03,   # 3% дневной лимит
                dd_reduce_threshold   = 0.10,   # при DD 10% → лот × 0.5
                dd_halt_threshold     = 0.15,   # при DD 15% → стоп
                dd_lot_multiplier     = 0.5,
                kelly_fraction        = 0.25,   # 25% от Kelly
                kelly_max             = 0.02,
            )
        )
        self.data:    dict[str, pd.DataFrame] = {}
        self.signals: dict[str, pd.DataFrame] = {}
        self.trades:  list[ClosedTrade] = []
        self.equity:  list[tuple] = [(pd.Timestamp("2022-01-01"),
                                      INITIAL_CAPITAL)]
        self.blocked_log: list[dict] = []   # лог заблокированных сигналов

    # ── Загрузка данных ───────────────────────────────────────────────────────
    def load(self):
        print("  Загрузка данных...")
        for t in TICKERS:
            df = load_daily(t)
            if df is not None and len(df) >= 250:
                self.data[t]    = df
                self.signals[t] = build_signals(df)
                print(f"    {t:6s}: {len(df)} баров, "
                      f"сигналов={int(self.signals[t]['signal'].sum())}")
            else:
                print(f"    {t:6s}: пропуск (нет данных)")

    # ── Главный цикл симуляции ────────────────────────────────────────────────
    def run(self):
        # Строим единый список всех дат (торговых дней)
        all_dates = sorted(set(
            d for df in self.data.values() for d in df.index
        ))

        open_positions: dict[str, Position] = {}

        for date in all_dates:
            self.rm.on_day_start()

            # 1. Управляем открытыми позициями
            to_close = []
            for ticker, pos in open_positions.items():
                if ticker not in self.data:
                    continue
                df = self.data[ticker]
                if date not in df.index:
                    continue

                bar   = df.loc[date]
                hi, lo = float(bar["high"]), float(bar["low"])
                i_now = df.index.get_loc(date)
                hold  = i_now - pos.entry_idx
                lv    = pos.tp_levels_hit
                n_tp  = len(FRACS)

                # Проверяем TP уровни
                while lv < n_tp and pos.remaining > 1e-9:
                    tp_px = pos.entry + LEVELS[lv] * pos.risk
                    if hi >= tp_px:
                        ep   = tp_px * (1 - SLIPPAGE)
                        frac = min(FRACS[lv], pos.remaining)
                        pos.partial_pnl += ((ep - pos.entry) * frac * pos.shares
                                            - (pos.entry + ep) * COMMISSION * frac * pos.shares)
                        pos.remaining -= frac
                        pos.sl = max(pos.sl,
                                     pos.entry * 1.001 if lv == 0 else
                                     pos.entry + LEVELS[lv-1] * pos.risk * 0.95)
                        pos.tp_levels_hit = lv + 1
                        lv = pos.tp_levels_hit
                    else:
                        break

                reason = None
                ep_final = None

                if pos.remaining <= 1e-6:
                    reason   = f"TP{n_tp}"
                    ep_final = pos.entry + LEVELS[-1] * pos.risk
                elif hold >= MAX_HOLD:
                    reason   = "TIME"
                    ep_final = float(bar["open"]) * (1 - SLIPPAGE)
                elif lo <= pos.sl:
                    reason   = "SL"
                    ep_final = max(pos.sl * (1 - SLIPPAGE), lo)

                if reason:
                    rem  = pos.remaining
                    pnl  = (pos.partial_pnl
                            + (ep_final - pos.entry) * rem * pos.shares
                            - (pos.entry + ep_final) * COMMISSION * rem * pos.shares)
                    pnl_pct = pnl / pos.pos_size * 100

                    trade = ClosedTrade(
                        ticker    = ticker,
                        entry_dt  = pos.entry_dt,
                        exit_dt   = date,
                        entry     = pos.entry,
                        exit_price= ep_final,
                        reason    = reason,
                        pos_size  = pos.pos_size,
                        pnl_rub   = round(pnl, 2),
                        pnl_pct   = round(pnl_pct, 3),
                        hold_days = hold,
                        win       = bool(pnl > 0),
                        tp_levels = pos.tp_levels_hit,
                        risk_pct  = pos.risk_pct_used,
                        capital_after = self.rm.state.capital + pnl,
                    )
                    self.trades.append(trade)
                    self.rm.on_trade_closed(ticker, pnl, pnl_pct)
                    self.equity.append((date, self.rm.state.capital))
                    to_close.append(ticker)

            for t in to_close:
                del open_positions[t]

            # 2. Ищем новые сигналы
            for ticker in TICKERS:
                if ticker in open_positions:
                    continue
                if ticker not in self.data:
                    continue
                df  = self.data[ticker]
                sig = self.signals[ticker]
                if date not in df.index:
                    continue

                i_now = df.index.get_loc(date)
                if i_now < 1:
                    continue

                # Сигнал формируется на предыдущем баре, входим на текущем open
                prev_row = sig.iloc[i_now - 1]
                if not bool(prev_row["signal"]):
                    continue

                sl_v   = float(prev_row["sl"])
                risk_v = float(prev_row["risk"])
                entry  = float(df.iloc[i_now]["open"]) * (1 + SLIPPAGE)

                if sl_v >= entry or risk_v <= 0:
                    continue

                sl_pct = (entry - sl_v) / entry

                # Спрашиваем риск-менеджер
                decision = self.rm.can_open(ticker, sl_pct)
                if not decision.allowed:
                    self.blocked_log.append({
                        "date": date, "ticker": ticker,
                        "reason": decision.reason,
                        "capital": self.rm.state.capital,
                    })
                    continue

                pos_size = decision.position_size_rub
                shares   = pos_size / entry

                pos = Position(
                    ticker        = ticker,
                    entry_idx     = i_now,
                    entry_dt      = date,
                    entry         = entry,
                    sl            = sl_v,
                    risk          = risk_v,
                    pos_size      = pos_size,
                    shares        = shares,
                    risk_pct_used = decision.risk_pct_used,
                )
                open_positions[ticker] = pos
                self.rm.on_position_opened(ticker)

        # Закрываем всё что осталось на последнем баре
        for ticker, pos in open_positions.items():
            df = self.data[ticker]
            last_close = float(df["close"].iloc[-1]) * (1 - SLIPPAGE)
            rem  = pos.remaining
            pnl  = (pos.partial_pnl
                    + (last_close - pos.entry) * rem * pos.shares
                    - (pos.entry + last_close) * COMMISSION * rem * pos.shares)
            pnl_pct = pnl / pos.pos_size * 100
            trade = ClosedTrade(
                ticker    = ticker,
                entry_dt  = pos.entry_dt,
                exit_dt   = df.index[-1],
                entry     = pos.entry,
                exit_price= last_close,
                reason    = "END",
                pos_size  = pos.pos_size,
                pnl_rub   = round(pnl, 2),
                pnl_pct   = round(pnl_pct, 3),
                hold_days = len(df) - pos.entry_idx,
                win       = bool(pnl > 0),
                tp_levels = pos.tp_levels_hit,
                risk_pct  = pos.risk_pct_used,
                capital_after = self.rm.state.capital + pnl,
            )
            self.trades.append(trade)
            self.rm.on_trade_closed(ticker, pnl, pnl_pct)
            self.equity.append((df.index[-1], self.rm.state.capital))


# ── Отчёт ─────────────────────────────────────────────────────────────────────
def print_report(sim: ProPortfolioSimulator):
    trades = sim.trades
    rm     = sim.rm

    if not trades:
        print("  Нет сделок.")
        return

    wins   = [t for t in trades if t.win]
    losses = [t for t in trades if not t.win]
    total_pnl = sum(t.pnl_rub for t in trades)
    final_cap = INITIAL_CAPITAL + total_pnl
    total_ret = total_pnl / INITIAL_CAPITAL * 100

    # Годовая доходность
    first_dt = min(t.entry_dt for t in trades)
    last_dt  = max(t.exit_dt  for t in trades)
    years    = max((last_dt - first_dt).days / 365.25, 0.01)
    annual_ret = ((final_cap / INITIAL_CAPITAL) ** (1 / years) - 1) * 100

    # Sharpe
    pnl_series = pd.Series([t.pnl_pct for t in trades])
    sharpe = (pnl_series.mean() / (pnl_series.std() + 1e-9) * np.sqrt(252)
              if len(pnl_series) > 1 else 0.0)

    # Max Drawdown
    eq_vals = np.array([e[1] for e in sim.equity])
    peak    = np.maximum.accumulate(eq_vals)
    dd_arr     = (eq_vals - peak) / (peak + 1e-9) * 100
    max_dd_pct = float(dd_arr.min())

    # PF
    gross_win  = sum(t.pnl_rub for t in wins)
    gross_loss = abs(sum(t.pnl_rub for t in losses))
    pf = gross_win / (gross_loss + 1e-9)

    # W/L ratio
    avg_win_r  = float(np.mean([t.pnl_rub for t in wins]))   if wins   else 0.0
    avg_loss_r = float(np.mean([t.pnl_rub for t in losses])) if losses else 0.0
    wl_ratio   = abs(avg_win_r / (avg_loss_r + 1e-9))

    # Среднее удержание
    avg_hold = float(np.mean([t.hold_days for t in trades]))

    print()
    print("═" * 65)
    print("  ПРОФЕССИОНАЛЬНАЯ ПОРТФЕЛЬНАЯ СИМУЛЯЦИЯ — ИТОГОВЫЙ ОТЧЁТ")
    print("═" * 65)
    print(f"  Период:            {first_dt.date()} → {last_dt.date()} ({years:.1f} лет)")
    print(f"  Начальный капитал: {INITIAL_CAPITAL:>12,.0f} руб.")
    print(f"  Финальный капитал: {final_cap:>12,.0f} руб.")
    print(f"  P&L:               {total_pnl:>+12,.0f} руб.  ({total_ret:+.1f}%)")
    print(f"  Годовая доходность:{annual_ret:>12.1f}%")
    print()
    print("  ── Качество стратегии ──────────────────────────────────")
    print(f"  Всего сделок:      {len(trades):>12d}")
    print(f"  Прибыльных:        {len(wins):>12d}  ({len(wins)/len(trades)*100:.0f}%)")
    print(f"  Убыточных:         {len(losses):>12d}  ({len(losses)/len(trades)*100:.0f}%)")
    print(f"  Win Rate:          {len(wins)/len(trades)*100:>11.0f}%")
    print(f"  Profit Factor:     {pf:>12.2f}")
    print(f"  Sharpe Ratio:      {sharpe:>12.2f}")
    print(f"  Avg Win/Loss:      {wl_ratio:>12.2f}x")
    print(f"  Avg Win:           {avg_win_r:>+12,.0f} руб.")
    print(f"  Avg Loss:          {avg_loss_r:>+12,.0f} руб.")
    print(f"  Avg Hold:          {avg_hold:>11.1f} дн.")
    print()
    print("  ── Риск-менеджмент ─────────────────────────────────────")
    print(f"  Max Drawdown:      {max_dd_pct:>11.1f}%")
    print(f"  DD-стопов:         {rm.state.dd_halt_count:>12d}")
    print(f"  Дн. стопов:        {rm.state.daily_halt_count:>12d}")
    print(f"  Kelly риск (итог): {rm._kelly_risk_pct()*100:>11.2f}%")
    print(f"  Заблок. сигналов:  {len(sim.blocked_log):>12d}")

    # Причины блокировки
    from collections import Counter
    block_reasons = Counter(b["reason"].split(" ")[0] for b in sim.blocked_log)
    if block_reasons:
        print("    └ по причинам: " +
              "  ".join(f"{k}:{v}" for k, v in block_reasons.most_common()))

    print()
    print("  ── По тикерам ──────────────────────────────────────────")
    print(f"  {'Тикер':<6} {'Сдел':>4} {'WR%':>5} {'PF':>5} "
          f"{'P&L руб':>10} {'AvgHold':>7}")
    print(f"  {'-'*50}")
    by_ticker = defaultdict(list)
    for t in trades:
        by_ticker[t.ticker].append(t)
    for tkr in sorted(by_ticker):
        tt     = by_ticker[tkr]
        tw     = [x for x in tt if x.win]
        wr_t   = len(tw) / len(tt) * 100 if tt else 0
        pnl_t  = sum(x.pnl_rub for x in tt)
        ww     = [x.pnl_rub for x in tw]
        ll     = [x.pnl_rub for x in tt if not x.win]
        pf_t   = (sum(ww) / (abs(sum(ll)) + 1e-9)) if ll else 99.0
        hold_t = np.mean([x.hold_days for x in tt])
        print(f"  {tkr:<6} {len(tt):>4d} {wr_t:>4.0f}% {min(pf_t,99):>5.2f} "
              f"{pnl_t:>+10,.0f} {hold_t:>6.1f}д")

    print()
    print("  ── Выходы ──────────────────────────────────────────────")
    from collections import Counter
    exits = Counter(t.reason for t in trades)
    for reason, cnt in sorted(exits.items()):
        wins_r = sum(1 for t in trades if t.reason == reason and t.win)
        pnl_r  = sum(t.pnl_rub for t in trades if t.reason == reason)
        print(f"  {reason:<6}  n={cnt:>3d}  WR={wins_r/cnt*100:>4.0f}%  "
              f"P&L={pnl_r:>+9,.0f} руб.")

    print()
    print("  ── Годовой разрез ──────────────────────────────────────")
    years_dict = defaultdict(list)
    for t in trades:
        years_dict[t.exit_dt.year].append(t)
    for yr in sorted(years_dict):
        yt     = years_dict[yr]
        yw     = [x for x in yt if x.win]
        pnl_yr = sum(x.pnl_rub for x in yt)
        wr_yr  = len(yw) / len(yt) * 100 if yt else 0
        print(f"  {yr}  сделок={len(yt):>3d}  WR={wr_yr:>4.0f}%  "
              f"P&L={pnl_yr:>+9,.0f} руб.")

    print()
    print("  ── Интерпретация (проп-стандарты) ──────────────────────")
    ok = []
    warn = []
    if annual_ret >= 20:
        ok.append(f"✓ Годовая доходность {annual_ret:.1f}% ≥ 20% — цель достигнута!")
    else:
        warn.append(f"✗ Годовая доходность {annual_ret:.1f}% < 20% — ниже цели")
    if len(wins)/len(trades) >= 0.60:
        ok.append(f"✓ WR {len(wins)/len(trades)*100:.0f}% — хороший показатель")
    else:
        warn.append(f"⚠ WR {len(wins)/len(trades)*100:.0f}% — нужно выше 60%")
    if pf >= 1.5:
        ok.append(f"✓ PF {pf:.2f} — стратегия прибыльна")
    else:
        warn.append(f"✗ PF {pf:.2f} < 1.5 — слабая прибыльность")
    if max_dd_pct >= -15:
        ok.append(f"✓ MaxDD {max_dd_pct:.1f}% — в пределах нормы")
    else:
        warn.append(f"⚠ MaxDD {max_dd_pct:.1f}% — большая просадка")
    if sharpe >= 1.0:
        ok.append(f"✓ Sharpe {sharpe:.2f} ≥ 1.0 — хорошее соотношение риск/доходность")
    else:
        warn.append(f"⚠ Sharpe {sharpe:.2f} < 1.0 — соотношение нужно улучшить")

    for line in ok:
        print(f"  {line}")
    for line in warn:
        print(f"  {line}")

    print("═" * 65)


# ── Запуск ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("═" * 65)
    print("  ЗАПУСК ПРОФЕССИОНАЛЬНОЙ СИМУЛЯЦИИ")
    print("  Капитал: 100 000 руб. | ТФ: Daily | Риск: 1%/сделку")
    print("═" * 65)

    sim = ProPortfolioSimulator()
    sim.load()
    print()
    print("  Симуляция торговли...")
    sim.run()
    print_report(sim)

    # Сохраняем лог сделок
    if sim.trades:
        df_out = pd.DataFrame([
            {
                "ticker":      t.ticker,
                "entry_dt":    t.entry_dt,
                "exit_dt":     t.exit_dt,
                "entry":       t.entry,
                "exit":        t.exit_price,
                "reason":      t.reason,
                "pos_size_rub":t.pos_size,
                "pnl_rub":     t.pnl_rub,
                "pnl_pct":     t.pnl_pct,
                "hold_days":   t.hold_days,
                "win":         t.win,
                "tp_levels":   t.tp_levels,
                "risk_pct":    t.risk_pct,
                "capital_after":t.capital_after,
            }
            for t in sim.trades
        ])
        out_path = Path("c:/investor/pro_trades.csv")
        df_out.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"\n  Лог сделок сохранён: {out_path}")
