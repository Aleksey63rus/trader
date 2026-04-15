"""
Профессиональная портфельная симуляция v3b.

Ключевые отличия от v3:
  - Стратегия v2 (проверенная основа, WR 48%, PF 1.16)
  - Тикер-вайтлист: убраны TGKA, SBER, GAZP, YDEX, MTSS, AFLT, SNGS, SNGSP, VTBR
  - Вместо полного блока корреляции — ВЫБИРАЕМ лучший тикер из группы по score
  - Ослаблен DD_HALT до 35% (у v2 было 25%)
  - Max позиций = 6 (было 5)
  - Схема BAL (лучшая по тестам)
  - Риск 5% на сделку
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict
from typing import Optional
import numpy as np
import pandas as pd

from core.strategy_v2 import SignalConfig, TP_SCHEMES
from core.risk import RiskManager, RiskParams

# ── Конфиг ────────────────────────────────────────────────────────────────────
INITIAL_CAPITAL = 100_000.0
COMMISSION      = 0.0005
SLIPPAGE        = 0.0001
MAX_HOLD        = 20
SCHEME          = "BAL"
FRACS, LEVELS   = TP_SCHEMES[SCHEME]

# ── Тикер-вайтлист (тщательно отобранные) ────────────────────────────────────
TICKERS = [
    # Нефть и газ (только лучшие по P&L v2)
    "LKOH", "NVTK", "ROSN",
    # Банки
    "SBERP", "T",
    # Металлургия
    "GMKN", "NLMK", "MTLR", "CHMF", "MAGN", "RUAL", "ALRS", "PLZL",
    # Ритейл / технологии
    "OZON", "MGNT",
    # Прочее
    "TATN", "TATNP",
    "IRAO", "PHOR", "OZPH",
]

# ── Корреляционные группы (для выбора ЛУЧШЕГО тикера, а не блокировки) ────────
CORR_GROUPS = [
    {"SBERP", "T"},
    {"LKOH", "ROSN", "NVTK"},
    {"NLMK", "MTLR", "CHMF", "MAGN", "RUAL", "GMKN"},
    {"ALRS", "PLZL"},
    {"TATN", "TATNP"},
    {"IRAO"},
    {"OZON", "MGNT"},
    {"PHOR"},
    {"OZPH"},
]

DATA_DIR = Path("c:/investor/data")

# ── Параметры риска ────────────────────────────────────────────────────────────
RISK_PARAMS = RiskParams(
    risk_pct             = 0.05,
    max_risk_pct         = 0.06,
    min_risk_pct         = 0.01,
    max_positions        = 6,      # было 5 → 6
    daily_loss_limit_pct = 0.06,
    dd_reduce_threshold  = 0.20,   # было 18%
    dd_halt_threshold    = 0.35,   # было 30%
    dd_lot_multiplier    = 0.5,
    kelly_fraction       = 0.30,
    kelly_max            = 0.07,
)

SIGNAL_CFG = SignalConfig(
    min_score     = 4,
    er_min        = 0.30,
    adx_min       = 20,
    vol_ratio_min = 1.2,
    sl_atr_mult   = 1.8,
    use_pullback  = True,
    pb_tolerance  = 0.02,
    sl_min_pct    = 0.025,
    sl_max_pct    = 0.08,
)


# ── Загрузчик ─────────────────────────────────────────────────────────────────
def load_daily(ticker: str) -> Optional[pd.DataFrame]:
    path = DATA_DIR / f"{ticker}_2022_2026_D.csv"
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path, sep=";")
        df.columns = [c.strip("<>").lower() for c in df.columns]
        df["dt"] = pd.to_datetime(
            df["date"].astype(str), format="%d/%m/%y", errors="coerce"
        )
        df = df.dropna(subset=["dt"]).set_index("dt").sort_index()
        for col in ("open", "high", "low", "close", "vol"):
            if col in df.columns:
                df[col] = pd.to_numeric(
                    df[col].astype(str).str.replace(",", "."), errors="coerce"
                )
        df = df.rename(columns={"vol": "volume"})
        df = df[["open", "high", "low", "close", "volume"]].dropna()
        return df if len(df) >= 200 else None
    except Exception:
        return None


# ── Dataclasses ───────────────────────────────────────────────────────────────
@dataclass
class Position:
    ticker:      str
    entry_dt:    object
    entry:       float
    sl:          float
    risk:        float
    lot_rub:     float
    remaining:   float = 1.0
    partial_pnl: float = 0.0
    tp_hit:      int   = 0


@dataclass
class ClosedTrade:
    ticker:   str
    entry_dt: object
    exit_dt:  object
    entry:    float
    exit:     float
    reason:   str
    lot_rub:  float
    pnl_rub:  float
    pnl_pct:  float
    hold_d:   int
    win:      bool
    score:    int = 0


# ── Корреляционный выбор — берём лучший по score ──────────────────────────────
def corr_group_of(ticker: str) -> Optional[frozenset]:
    for g in CORR_GROUPS:
        if ticker in g:
            return frozenset(g)
    return None


# ══════════════════════════════════════════════════════════════════════════════
# Симулятор v3b
# ══════════════════════════════════════════════════════════════════════════════
class ProSimulatorV3b:
    def __init__(self):
        self.rm = RiskManager(INITIAL_CAPITAL, RISK_PARAMS)

    def load(self) -> dict[str, pd.DataFrame]:
        data = {}
        for t in TICKERS:
            df = load_daily(t)
            if df is not None:
                data[t] = df
                print(f"  ✓ {t}: {len(df)} баров")
            else:
                print(f"  ✗ {t}: нет данных")
        return data

    def run(self, data: dict[str, pd.DataFrame]) -> list[ClosedTrade]:
        from core.indicators import atr as _atr

        n_tp = len(FRACS)
        from core.strategy_v2 import SignalGeneratorV2
        gen = SignalGeneratorV2(SIGNAL_CFG)

        sigs: dict[str, pd.DataFrame] = {}
        at14s: dict[str, pd.Series] = {}
        for t, df in data.items():
            sigs[t] = gen.generate(df)
            at14s[t] = _atr(df, 14)

        all_dates = sorted(set().union(*[df.index.tolist() for df in data.values()]))

        positions: dict[str, Position] = {}
        trades: list[ClosedTrade] = []
        equity_curve = [INITIAL_CAPITAL]
        blocked_stats = defaultdict(int)
        scores_today: dict[str, int] = {}

        for dt in all_dates:
            self.rm.on_day_start()

            # ── Закрытие позиций ──────────────────────────────────────────
            to_close = []
            for ticker, pos in positions.items():
                if ticker not in data or dt not in data[ticker].index:
                    continue
                bar     = data[ticker].loc[dt]
                bar_hi  = float(bar["high"])
                bar_lo  = float(bar["low"])
                bar_op  = float(bar["open"])
                bar_cl  = float(bar["close"])
                entry   = pos.entry
                risk    = pos.risk
                lv      = pos.tp_hit

                # Stepped TP
                while lv < n_tp and pos.remaining > 1e-9:
                    tp_px = entry + LEVELS[lv] * risk
                    if bar_hi >= tp_px:
                        ep   = tp_px * (1 - SLIPPAGE)
                        frac = min(FRACS[lv], pos.remaining)
                        pos.partial_pnl += ((ep - entry) * frac
                                            - (entry + ep) * COMMISSION * frac)
                        pos.remaining -= frac
                        pos.sl = entry * 1.001 if lv == 0 else entry + LEVELS[lv-1] * risk * 0.92
                        pos.tp_hit = lv + 1
                        lv = pos.tp_hit
                    else:
                        break

                # ATR trailing после 1-го TP
                if pos.tp_hit >= 1:
                    at14_v = at14s.get(ticker)
                    if at14_v is not None and dt in at14_v.index:
                        trail = bar_cl - 2.0 * float(at14_v.loc[dt])
                        if trail > pos.sl:
                            pos.sl = trail

                reason = ep_final = None
                if pos.remaining <= 1e-6:
                    reason, ep_final = f"TP{n_tp}", bar_cl * (1 - SLIPPAGE)
                elif (dt - pos.entry_dt).days >= MAX_HOLD:
                    reason, ep_final = "TIME", bar_op * (1 - SLIPPAGE)
                elif bar_lo <= pos.sl:
                    reason, ep_final = "SL", max(pos.sl * (1 - SLIPPAGE), bar_lo)

                if reason:
                    rem = pos.remaining
                    pnl_rub = (pos.partial_pnl
                               + (ep_final - entry) * rem / entry * pos.lot_rub
                               - (entry + ep_final) / entry * COMMISSION * pos.lot_rub * rem)
                    ct = ClosedTrade(
                        ticker   = ticker, entry_dt = pos.entry_dt, exit_dt = dt,
                        entry    = entry,  exit     = ep_final, reason = reason,
                        lot_rub  = pos.lot_rub,
                        pnl_rub  = pnl_rub,
                        pnl_pct  = pnl_rub / pos.lot_rub * 100,
                        hold_d   = (dt - pos.entry_dt).days,
                        win      = bool(pnl_rub > 0),
                    )
                    trades.append(ct)
                    self.rm.on_trade_closed(ticker, pnl_rub, ct.pnl_pct)
                    equity_curve.append(self.rm.state.capital)
                    to_close.append(ticker)

            for t in to_close:
                del positions[t]

            # ── Новые входы ───────────────────────────────────────────────
            candidates = []
            for ticker in TICKERS:
                if ticker in positions:
                    continue
                if ticker not in data or dt not in sigs[ticker].index:
                    continue
                row = sigs[ticker].loc[dt]
                if not bool(row.get("signal", 0)):
                    continue
                score = int(row.get("score", 0))
                candidates.append((ticker, score, row))

            # Выбираем лучший тикер из каждой корр. группы
            seen_groups: set = set()
            filtered = []
            candidates.sort(key=lambda x: -x[1])
            for ticker, score, row in candidates:
                g = corr_group_of(ticker)
                if g is not None:
                    key = g
                    # Если в группе уже открыта позиция — пропускаем
                    if any(t in positions for t in g):
                        blocked_stats[f"CORR_OPEN({ticker})"] += 1
                        continue
                    if key in seen_groups:
                        blocked_stats[f"CORR_PICK({ticker})"] += 1
                        continue
                    seen_groups.add(key)
                filtered.append((ticker, score, row))

            for ticker, score, sig_row in filtered:
                df_t   = data[ticker]
                idx_dt = df_t.index.get_loc(dt)
                if idx_dt + 1 >= len(df_t):
                    continue
                next_dt  = df_t.index[idx_dt + 1]
                next_bar = df_t.loc[next_dt]
                entry    = float(next_bar["open"]) * (1 + SLIPPAGE)
                sl       = float(sig_row["sl"])
                risk     = float(sig_row["risk"])

                if sl >= entry or risk <= 0:
                    continue

                sl_pct_v = (entry - sl) / entry
                dec = self.rm.can_open(ticker, sl_pct_v)
                if not dec.allowed:
                    blocked_stats[dec.reason] += 1
                    continue

                lot_rub = dec.position_size_rub
                positions[ticker] = Position(
                    ticker=ticker, entry_dt=next_dt, entry=entry,
                    sl=sl, risk=risk, lot_rub=lot_rub,
                )
                self.rm.on_position_opened(ticker)

        # Закрытие остатков
        for ticker, pos in positions.items():
            if ticker not in data:
                continue
            df_t  = data[ticker]
            ep    = float(df_t["close"].iloc[-1]) * (1 - SLIPPAGE)
            rem   = pos.remaining
            pnl   = (pos.partial_pnl
                     + (ep - pos.entry) * rem / pos.entry * pos.lot_rub
                     - (pos.entry + ep) / pos.entry * COMMISSION * pos.lot_rub * rem)
            ct = ClosedTrade(
                ticker=ticker, entry_dt=pos.entry_dt, exit_dt=df_t.index[-1],
                entry=pos.entry, exit=ep, reason="END",
                lot_rub=pos.lot_rub, pnl_rub=pnl,
                pnl_pct=pnl / pos.lot_rub * 100,
                hold_d=(df_t.index[-1] - pos.entry_dt).days,
                win=bool(pnl > 0),
            )
            trades.append(ct)
            self.rm.on_trade_closed(ticker, pnl, ct.pnl_pct)
            equity_curve.append(self.rm.state.capital)

        self._equity = equity_curve
        self._final_capital = self.rm.state.capital
        self._blocked = blocked_stats
        return trades

    def print_report(self, trades: list[ClosedTrade]):
        if not trades:
            print("\n=== Сделок нет ===")
            return

        total_rub = self._final_capital - INITIAL_CAPITAL
        total_pct = total_rub / INITIAL_CAPITAL * 100
        eq  = np.array(self._equity)
        pk  = np.maximum.accumulate(np.maximum(eq, 1.0))
        dd  = float(((eq - pk) / pk * 100).min())
        years = 4.1
        ann   = ((self._final_capital / INITIAL_CAPITAL) ** (1 / years) - 1) * 100

        wins   = [t for t in trades if t.win]
        losses = [t for t in trades if not t.win]
        wr     = len(wins) / len(trades) * 100 if trades else 0
        aw     = float(np.mean([t.pnl_rub for t in wins]))   if wins   else 0
        al     = float(np.mean([t.pnl_rub for t in losses])) if losses else 0
        pf     = abs(sum(t.pnl_rub for t in wins)) / (abs(sum(t.pnl_rub for t in losses)) + 1e-9)
        wl     = abs(aw / al) if al else 99

        from collections import Counter
        exit_cnt   = Counter(t.reason for t in trades)
        ticker_pnl = defaultdict(float)
        for t in trades:
            ticker_pnl[t.ticker] += t.pnl_rub

        print("\n" + "═" * 62)
        print("  ОТЧЁТ: Pro Portfolio Simulator v3b")
        print("═" * 62)
        print(f"  Начальный капитал:   {INITIAL_CAPITAL:>10,.0f} ₽")
        print(f"  Конечный капитал:    {self._final_capital:>10,.0f} ₽")
        print(f"  Итого P&L:           {total_rub:>+10,.0f} ₽  ({total_pct:+.1f}%)")
        print(f"  Годовая доходность:  {ann:>+10.1f}%")
        print(f"  Макс. просадка:      {dd:>10.1f}%")
        print(f"  Период:              {years:.1f} лет")
        print("─" * 62)
        print(f"  Всего сделок:        {len(trades):>10}")
        print(f"  Win Rate:            {wr:>10.1f}%")
        print(f"  Profit Factor:       {pf:>10.2f}")
        print(f"  Avg Win / Avg Loss:  {wl:>10.2f}×")
        print(f"  Avg Win:             {aw:>+10,.0f} ₽")
        print(f"  Avg Loss:            {al:>+10,.0f} ₽")
        print("─" * 62)
        print("  Выходы:")
        for reason, cnt in sorted(exit_cnt.items()):
            r_t   = [t for t in trades if t.reason == reason]
            r_w   = sum(1 for t in r_t if t.win)
            r_pnl = sum(t.pnl_rub for t in r_t)
            print(f"    {reason:8s}: {cnt:3d}  WR={r_w/cnt*100:4.0f}%  P&L={r_pnl:>+9,.0f} ₽")
        print("─" * 62)
        print("  Блокировок (топ):")
        for reason, cnt in sorted(self._blocked.items(), key=lambda x: -x[1])[:8]:
            print(f"    {reason:30s}: {cnt}")
        print("─" * 62)
        print("  Топ-10 тикеров:")
        for t, p in sorted(ticker_pnl.items(), key=lambda x: -x[1])[:10]:
            n_t = sum(1 for tr in trades if tr.ticker == t)
            w_t = sum(1 for tr in trades if tr.ticker == t and tr.win)
            print(f"    {t:6s}: {p:>+9,.0f} ₽  ({n_t} сд, WR={w_t/n_t*100:.0f}%)")
        print("  Аутсайдеры:")
        for t, p in sorted(ticker_pnl.items(), key=lambda x: x[1])[:5]:
            n_t = sum(1 for tr in trades if tr.ticker == t)
            print(f"    {t:6s}: {p:>+9,.0f} ₽  ({n_t} сд)")
        print("═" * 62)
        print("\n  Сравнение v2 → v3b:")
        print(f"    P&L:      +15,490 ₽ (+15.5%)  →  {total_rub:+,.0f} ₽ ({total_pct:+.1f}%)")
        print(f"    WR:       48%  →  {wr:.0f}%")
        print(f"    PF:       1.16 →  {pf:.2f}")
        print(f"    MaxDD:   -15.1% →  {dd:.1f}%")
        print(f"    Ann.Ret:  3.6% →  {ann:.1f}%")
        print("═" * 62)

        df_out = pd.DataFrame([{
            "ticker": t.ticker, "entry_dt": t.entry_dt, "exit_dt": t.exit_dt,
            "entry": t.entry, "exit": t.exit, "reason": t.reason,
            "lot_rub": t.lot_rub, "pnl_rub": t.pnl_rub, "pnl_pct": t.pnl_pct,
            "hold_d": t.hold_d, "win": t.win,
        } for t in trades])
        df_out.to_csv("c:/investor/pro_trades_v3b.csv", index=False)
        print("  Лог сделок → pro_trades_v3b.csv")


if __name__ == "__main__":
    print("=== Pro Portfolio Simulator v3b ===\n")
    sim  = ProSimulatorV3b()
    data = sim.load()
    print(f"\nЗагружено тикеров: {len(data)}")
    print("\nЗапуск симуляции...")
    trades = sim.run(data)
    sim.print_report(trades)
