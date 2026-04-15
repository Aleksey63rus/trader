"""
Портфельный бэктест — реальная торговля на 100 000 рублей.

Правила:
  - Стартовый капитал: 100 000 руб.
  - Торгуем на ВЕСЬ доступный кэш (максимальные позиции кратно лотам)
  - Одна позиция на тикер одновременно
  - До 3 открытых позиций одновременно (диверсификация)
  - Комиссия: 0.05% от оборота (БКС)
  - Проскальзывание: 0.01%
  - Стратегия: Momentum Filter + Stepped TP (схема F)
  - Если сигналы на нескольких тикерах — выбираем по наибольшему ER(20)
  - Лот MOEX = 1 акция (для простоты, реальные лоты учтены через lot_size)
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from core.indicators import atr, ema, rsi, efficiency_ratio, macd_histogram
from core.indicators import directional_index, adx, volume_ratio
from core.data_loader import load_csv

# ── Constants ─────────────────────────────────────────────────────────────────
INITIAL_CAPITAL = 100_000.0   # рублей
COMMISSION      = 0.0005       # 0.05% per side
SLIPPAGE        = 0.0001       # market impact
MAX_POSITIONS   = 3            # макс открытых позиций одновременно
RISK_PER_TRADE  = 0.33         # от кэша на одну позицию (до 33%)
MAX_HOLD        = 96           # часов (4 торговых дня)

# Stepped TP схема F: 50/30/20 @ R0.8/1.8/3.5
FRACS  = (0.50, 0.30, 0.20)
LEVELS = (0.8,  1.8,  3.5)

# Минимальные лоты MOEX (шт. акций в 1 лоте)
LOT_SIZES = {
    "GAZP": 10, "LKOH": 1,  "MGNT": 1,  "MTLR": 10,
    "NLMK": 10, "NVTK": 1,  "ROSN": 1,  "SBER": 10,
    "T":    1,  "TGKA": 1000,"YDEX": 1,  "OZPH": 1,
}

TICKERS_1H = {
    "GAZP": "GAZP_2022_2026_1H.csv",
    "LKOH": "LKOH_2022_2026_1H.csv",
    "MGNT": "MGNT_2022_2026_1H.csv",
    "MTLR": "MTLR_2022_2026_1H.csv",
    "NLMK": "NLMK_2022_2026_1H.csv",
    "NVTK": "NVTK_2022_2026_1H.csv",
    "ROSN": "ROSN_2022_2026_1H.csv",
    "SBER": "SBER_2022_2026_1H.csv",
    "T":    "T_2022_2026_1H.csv",
    "TGKA": "TGKA_2022_2026_1H.csv",
    "YDEX": "YDEX_2022_2026_1H.csv",
    "OZPH": "OZPH_2022_2026_1H.csv",
}


# ══════════════════════════════════════════════════════════════════════════════
# Signal generator (копия из core.strategy, адаптирована под портфель)
# ══════════════════════════════════════════════════════════════════════════════
def build_signals(df: pd.DataFrame) -> pd.DataFrame:
    c     = df["close"]
    at14  = atr(df, 14)
    at5   = atr(df, 5)
    er20  = efficiency_ratio(c, 20)
    er10  = efficiency_ratio(c, 10)
    e200  = ema(c, 200)
    e50   = ema(c, 50)
    e20   = ema(c, 20)
    rsi14 = rsi(df, 14)
    mh    = macd_histogram(df)
    di_p, di_n = directional_index(df, 14)
    adx14 = adx(df, 14)
    vol_r = volume_ratio(df, 20)
    at_r  = at5 / at14
    hour  = pd.Series(df.index.hour,      index=df.index)
    dow   = pd.Series(df.index.dayofweek, index=df.index)
    slope = (e20 - e20.shift(3)) / e20.shift(3).replace(0, np.nan) * 100

    er_base = (er20 > 0.35) & (er10 > er20) & (c > e50) & (c.diff() > 0)

    quality = (
        (c > e200) & (di_p > di_n) & (mh > 0) &
        (er20 > 0.40) & (rsi14 >= 58) & (rsi14 <= 88) & (slope > 0)
    )

    cl_a = quality & (vol_r > 2.5) & (hour >= 13) & (hour <= 22) & (er20 > 0.43)
    cl_b = quality & (at_r > 1.08) & ((dow == 4) | (dow == 0)) & (er20 > 0.45) & (vol_r > 1.5)
    cl_c = quality & (vol_r > 3.0) & (adx14 > 28) & (er20 > 0.48)

    sig_raw = er_base & (cl_a | cl_b | cl_c)
    signal  = sig_raw & ~sig_raw.shift(1).fillna(False)

    sl_raw = c - 1.8 * at14
    sl     = sl_raw.clip(lower=c * 0.965, upper=c * 0.988)
    risk   = (c - sl).clip(lower=0.001)

    return pd.DataFrame({
        "signal": signal.astype(int),
        "sl":     sl,
        "risk":   risk,
        "er20":   er20,
    }, index=df.index)


# ══════════════════════════════════════════════════════════════════════════════
# Position & trade record
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class Position:
    ticker:      str
    entry_dt:    datetime
    entry_price: float
    shares:      int          # количество акций
    cost:        float        # полная стоимость входа (с комиссией)
    sl:          float
    risk:        float
    entry_idx:   int
    remaining:   float = 1.0          # доля позиции ещё открытой
    partial_pnl: float = 0.0          # накопленный P&L от частичных выходов
    tp_hit:      int   = 0
    sl_current:  float = 0.0          # текущий trailing SL
    exit_dt:     Optional[datetime] = None
    exit_price:  float = 0.0
    reason:      str   = ""
    pnl:         float = 0.0
    pnl_pct:     float = 0.0
    hold_bars:   int   = 0

    def __post_init__(self):
        self.sl_current = self.sl


@dataclass
class TradeRecord:
    num:         int
    ticker:      str
    entry_dt:    str
    exit_dt:     str
    entry_price: float
    exit_price:  float
    shares:      int
    pnl:         float          # чистая прибыль/убыток в рублях
    pnl_pct:     float          # % от вложенной суммы
    reason:      str
    tp_hit:      int
    hold_bars:   int
    capital_after: float


# ══════════════════════════════════════════════════════════════════════════════
# Portfolio simulator
# ══════════════════════════════════════════════════════════════════════════════
class PortfolioSimulator:

    def __init__(self):
        self.capital    = INITIAL_CAPITAL
        self.positions:  dict[str, Position] = {}
        self.trade_log:  list[TradeRecord]   = []
        self.equity_ts:  list[tuple]         = []  # (datetime, capital)
        self.trade_num   = 0

    # ── Load all data ─────────────────────────────────────────────────────────
    def load_data(self) -> dict[str, pd.DataFrame]:
        dfs = {}
        for tk, fname in TICKERS_1H.items():
            p = Path(fname)
            if not p.exists():
                print(f"  ! {fname} не найден, пропускаем {tk}")
                continue
            df = load_csv(p).between_time("07:00", "23:00")
            sig = build_signals(df)
            dfs[tk] = pd.concat([df, sig], axis=1)
            print(f"  {tk}: {len(df):,} баров")
        return dfs

    # ── Close position ─────────────────────────────────────────────────────────
    def _close_position(self, pos: Position, price: float, dt: datetime,
                        reason: str, bar_idx: int):
        rem  = pos.remaining
        ep   = price * (1 - SLIPPAGE)
        pnl  = (pos.partial_pnl
                + (ep - pos.entry_price) * pos.shares * rem
                - pos.entry_price * pos.shares * rem * COMMISSION
                - ep * pos.shares * rem * COMMISSION)
        pos.exit_dt    = dt
        pos.exit_price = ep
        pos.reason     = reason
        pos.hold_bars  = bar_idx - pos.entry_idx
        pos.pnl        = pnl
        pos.pnl_pct    = pnl / pos.cost * 100
        self.capital  += pos.cost + pnl   # возвращаем вложенное + прибыль
        self.trade_num += 1

        rec = TradeRecord(
            num          = self.trade_num,
            ticker       = pos.ticker,
            entry_dt     = str(pos.entry_dt)[:16],
            exit_dt      = str(dt)[:16],
            entry_price  = round(pos.entry_price, 2),
            exit_price   = round(ep, 2),
            shares       = pos.shares,
            pnl          = round(pnl, 2),
            pnl_pct      = round(pos.pnl_pct, 2),
            reason       = reason,
            tp_hit       = pos.tp_hit,
            hold_bars    = pos.hold_bars,
            capital_after= round(self.capital, 2),
        )
        self.trade_log.append(rec)
        self.equity_ts.append((dt, round(self.capital, 2)))

    # ── Update open positions on new bar ──────────────────────────────────────
    def _update_position(self, pos: Position, hi: float, lo: float,
                         open_: float, dt: datetime, bar_idx: int) -> bool:
        """Returns True if position was fully closed."""
        entry = pos.entry_price
        risk  = pos.risk

        # Check TP levels
        while pos.tp_hit < len(FRACS) and pos.remaining > 1e-9:
            tp_px = entry + LEVELS[pos.tp_hit] * risk
            if hi >= tp_px:
                frac   = min(FRACS[pos.tp_hit], pos.remaining)
                ep     = tp_px * (1 - SLIPPAGE)
                partial = (
                    (ep - entry) * pos.shares * frac
                    - entry * pos.shares * frac * COMMISSION
                    - ep   * pos.shares * frac * COMMISSION
                )
                pos.partial_pnl += partial
                pos.remaining   -= frac
                # Move SL
                if pos.tp_hit == 0:
                    pos.sl_current = entry * 1.001
                else:
                    prev_tp = entry + LEVELS[pos.tp_hit - 1] * risk
                    pos.sl_current = max(pos.sl_current, prev_tp * 0.99)
                pos.tp_hit += 1
            else:
                break

        # All TPs hit
        if pos.remaining <= 1e-6:
            close_px = entry + LEVELS[-1] * risk
            self._close_position(pos, close_px, dt, f"TP{len(FRACS)}", bar_idx)
            return True

        # Time exit
        if (bar_idx - pos.entry_idx) >= MAX_HOLD:
            self._close_position(pos, open_, dt, "TIME", bar_idx)
            return True

        # SL hit
        if lo <= pos.sl_current:
            hit_px = max(pos.sl_current * (1 - SLIPPAGE), lo)
            self._close_position(pos, hit_px, dt, "SL", bar_idx)
            return True

        return False

    # ── Main simulation ────────────────────────────────────────────────────────
    def run(self, dfs: dict[str, pd.DataFrame]):
        # Build unified timeline
        all_times = sorted(
            set().union(*(df.index.tolist() for df in dfs.values()))
        )

        print(f"\nСимуляция: {all_times[0].date()} → {all_times[-1].date()}")
        print(f"Тикеров: {len(dfs)}, таймстемпов: {len(all_times):,}")

        # Cache arrays per ticker for speed
        data: dict[str, dict] = {}
        for tk, df in dfs.items():
            data[tk] = {
                "idx":    {dt: i for i, dt in enumerate(df.index)},
                "open":   df["open"].values,
                "high":   df["high"].values,
                "low":    df["low"].values,
                "close":  df["close"].values,
                "signal": df["signal"].values,
                "sl":     df["sl"].values,
                "risk":   df["risk"].values,
                "er20":   df["er20"].values,
                "index":  df.index,
            }

        bar_counters: dict[str, int] = {tk: 0 for tk in dfs}

        for global_i, dt in enumerate(all_times):

            # ── 1. Update open positions ───────────────────────────────────────
            closed = []
            for tk, pos in self.positions.items():
                d = data[tk]
                if dt not in d["idx"]:
                    continue
                i = d["idx"][dt]
                if self._update_position(
                    pos, d["high"][i], d["low"][i], d["open"][i], dt, i
                ):
                    closed.append(tk)

            for tk in closed:
                del self.positions[tk]

            # ── 2. Look for new signals ────────────────────────────────────────
            n_open = len(self.positions)
            if n_open < MAX_POSITIONS and self.capital > 5_000:

                # Collect candidates: signal on previous bar
                candidates = []
                for tk, d in data.items():
                    if tk in self.positions:
                        continue
                    if dt not in d["idx"]:
                        continue
                    i = d["idx"][dt]
                    if i < 1:
                        continue
                    prev_sig = d["signal"][i - 1]
                    if prev_sig != 1:
                        continue
                    er = float(d["er20"][i - 1])
                    sl = float(d["sl"][i - 1])
                    rk = float(d["risk"][i - 1])
                    entry_px = d["open"][i] * (1 + SLIPPAGE)
                    if sl <= 0 or rk <= 0 or np.isnan(sl) or np.isnan(rk):
                        continue
                    if sl >= entry_px:
                        continue
                    candidates.append((tk, er, entry_px, sl, rk, i))

                # Sort by ER20 desc → pick top (MAX_POSITIONS - n_open)
                candidates.sort(key=lambda x: -x[1])
                slots = MAX_POSITIONS - n_open

                for tk, er, entry_px, sl, rk, i in candidates[:slots]:
                    lot  = LOT_SIZES.get(tk, 1)
                    # Allocate RISK_PER_TRADE of available cash
                    alloc = min(self.capital * RISK_PER_TRADE,
                                self.capital / max(1, slots))
                    # Number of lots we can afford
                    n_lots  = max(1, int(alloc / (entry_px * lot)))
                    n_lots  = min(n_lots, int(self.capital / (entry_px * lot)))
                    if n_lots < 1:
                        continue
                    shares = n_lots * lot
                    cost   = entry_px * shares * (1 + COMMISSION)
                    if cost > self.capital:
                        shares = int(self.capital / (entry_px * (1 + COMMISSION)))
                        shares = (shares // lot) * lot
                        if shares < lot:
                            continue
                        cost = entry_px * shares * (1 + COMMISSION)

                    self.capital -= cost
                    pos = Position(
                        ticker      = tk,
                        entry_dt    = dt,
                        entry_price = entry_px,
                        shares      = shares,
                        cost        = cost,
                        sl          = sl,
                        risk        = rk,
                        entry_idx   = i,
                    )
                    self.positions[tk] = pos

            # Track equity every ~24 bars (daily snapshot)
            if global_i % 24 == 0:
                # Mark-to-market of open positions
                mtm = 0.0
                for tk, pos in self.positions.items():
                    d = data[tk]
                    if dt in d["idx"]:
                        mtm += d["close"][d["idx"][dt]] * pos.shares * pos.remaining

                self.equity_ts.append((dt, round(self.capital + mtm, 2)))

        # Close remaining positions at last available price
        print(f"  Закрываем {len(self.positions)} незакрытых позиций...")
        for tk, pos in list(self.positions.items()):
            d    = data[tk]
            last_dt = d["index"][-1]
            last_px = float(d["close"][-1])
            last_i  = len(d["index"]) - 1
            self._close_position(pos, last_px, last_dt, "END", last_i)


# ══════════════════════════════════════════════════════════════════════════════
# Report builder
# ══════════════════════════════════════════════════════════════════════════════
def print_report(sim: PortfolioSimulator):
    trades = sim.trade_log
    if not trades:
        print("Нет сделок.")
        return

    df = pd.DataFrame([t.__dict__ for t in trades])
    wins   = df[df["pnl"] > 0]
    losses = df[df["pnl"] <= 0]

    final_cap  = sim.capital
    total_pnl  = final_cap - INITIAL_CAPITAL
    total_pct  = total_pnl / INITIAL_CAPITAL * 100
    wr         = len(wins) / len(df) * 100 if len(df) else 0
    avg_win    = wins["pnl"].mean()    if len(wins)   else 0
    avg_loss   = losses["pnl"].mean()  if len(losses) else 0
    pf         = (abs(wins["pnl"].sum()) / abs(losses["pnl"].sum())
                  if len(losses) > 0 else 99.0)

    # Max drawdown
    eq  = [e for _, e in sim.equity_ts]
    pk  = np.maximum.accumulate(eq)
    dd  = np.array(eq) - pk
    mdd = float(dd.min())
    mdd_pct = mdd / INITIAL_CAPITAL * 100

    # Per-ticker stats
    ticker_stats = df.groupby("ticker").agg(
        trades=("pnl","count"),
        wr_pct=("pnl", lambda x: (x>0).mean()*100),
        total_pnl=("pnl","sum"),
    ).reset_index().sort_values("total_pnl", ascending=False)

    # Exit distribution
    exits = df.groupby("reason").size().reset_index(name="n")

    print("\n" + "="*70)
    print("  MOMENTUM FILTER TRADER — ПОРТФЕЛЬНЫЙ ОТЧЁТ")
    print("="*70)
    print(f"  Период:          {df['entry_dt'].min()[:10]}  →  {df['exit_dt'].max()[:10]}")
    print(f"  Стартовый кап.:  {INITIAL_CAPITAL:,.0f} руб.")
    print(f"  Финальный кап.:  {final_cap:,.0f} руб.")
    print(f"  Итоговая P&L:    {total_pnl:+,.0f} руб.  ({total_pct:+.1f}%)")
    print(f"  Кол-во сделок:   {len(df)}")
    print(f"  Win Rate:        {wr:.1f}%")
    print(f"  Profit Factor:   {pf:.2f}")
    print(f"  Ср. выигрыш:     {avg_win:+.0f} руб.")
    print(f"  Ср. проигрыш:    {avg_loss:+.0f} руб.")
    print(f"  Max Drawdown:    {mdd:,.0f} руб.  ({mdd_pct:.1f}%)")
    print()

    print("  ПО ТИКЕРАМ:")
    print(f"  {'Тикер':<8} {'Сделок':>7} {'WR%':>7} {'P&L руб':>12}")
    print("  " + "-"*37)
    for _, row in ticker_stats.iterrows():
        sign = "+" if row["total_pnl"] >= 0 else ""
        print(f"  {row['ticker']:<8} {int(row['trades']):>7} "
              f"{row['wr_pct']:>6.1f}%  {sign}{row['total_pnl']:>10,.0f}")

    print()
    print("  ВЫХОДЫ:")
    for _, row in exits.iterrows():
        wr_r = df[df["reason"]==row["reason"]]["pnl"].apply(lambda x: x>0).mean()*100
        print(f"  {row['reason']:<8}: {int(row['n']):>4} сделок  WR={wr_r:.0f}%")

    print()
    print("  ЖУРНАЛ СДЕЛОК (все {0}):".format(len(df)))
    print(f"  {'№':>4} {'Тикер':<6} {'Вход':>16} {'Выход':>16} "
          f"{'Акций':>6} {'Цена вх':>9} {'Цена вых':>9} "
          f"{'P&L руб':>10} {'P&L%':>7} {'Выход':>6} {'TP':>3} {'Кап. после':>12}")
    print("  " + "-"*120)
    for t in trades:
        sign = "+" if t.pnl >= 0 else ""
        pct  = f"{sign}{t.pnl_pct:.2f}%"
        pnl  = f"{sign}{t.pnl:,.0f}"
        print(f"  {t.num:>4} {t.ticker:<6} {t.entry_dt:>16} {t.exit_dt:>16} "
              f"{t.shares:>6,} {t.entry_price:>9.2f} {t.exit_price:>9.2f} "
              f"{pnl:>10} {pct:>8} {t.reason:>6} {t.tp_hit:>3} "
              f"{t.capital_after:>12,.0f}")

    # Save to CSV
    df.to_csv("portfolio_trades.csv", index=False, encoding="utf-8-sig")
    print("\n  Сохранено: portfolio_trades.csv")


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Загружаем данные...")
    sim  = PortfolioSimulator()
    dfs  = sim.load_data()

    if not dfs:
        print("Нет данных для симуляции!")
        sys.exit(1)

    sim.run(dfs)
    print_report(sim)
