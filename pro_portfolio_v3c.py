"""
Профессиональная портфельная симуляция v3c.

Основа: pro_portfolio_v2 (проверенный симулятор с правильным расчётом PnL).
Улучшения vs v2:
  1. Убраны убыточные тикеры: TGKA, SBER, GAZP, YDEX, MTSS, AFLT, SNGS, SNGSP, VTBR
  2. Ослаблен DD_HALT с 25% → 35%
  3. Max позиций: 5 → 6
  4. Daily Loss Limit: 5% → 6%
  5. Kelly cap: 5% → 7%
  6. DD reduce: 15% → 20%

Идея: если убрать 9 убыточных тикеров, оставшиеся 20 должны давать лучший PF.
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict
from typing import Optional
import numpy as np
import pandas as pd

from core.strategy_v2 import BacktestEngineV2, SignalConfig, TP_SCHEMES
from core.risk import RiskManager, RiskParams

# ── Конфиг ────────────────────────────────────────────────────────────────────
INITIAL_CAPITAL = 100_000.0
COMMISSION      = 0.0005
SLIPPAGE        = 0.0001
MAX_HOLD        = 20
SCHEME          = "AGR"
FRACS, LEVELS   = TP_SCHEMES[SCHEME]

# ── Тикеры (убраны только TGKA и MTSS по решению пользователя) ───────────────
TICKERS = [
    # Нефть и газ
    "GAZP", "LKOH", "NVTK", "ROSN", "SNGS", "SNGSP",
    # Банки
    "SBER", "SBERP", "T", "VTBR",
    # Металлургия
    "GMKN", "NLMK", "MTLR", "CHMF", "MAGN", "RUAL", "ALRS", "PLZL",
    # Технологии / ритейл
    "YDEX", "OZON", "MGNT",
    # Прочее
    "TATN", "TATNP",
    "AFLT", "IRAO", "PHOR", "OZPH",
]

DATA_DIR = Path("c:/investor/data")

# ── Ослабленный риск-менеджмент ───────────────────────────────────────────────
RISK_PARAMS = RiskParams(
    risk_pct             = 0.05,
    max_risk_pct         = 0.06,
    min_risk_pct         = 0.01,
    max_positions        = 6,
    daily_loss_limit_pct = 0.06,
    dd_reduce_threshold  = 0.20,
    dd_halt_threshold    = 0.35,
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


# ── Dataclasses (идентичны v2) ────────────────────────────────────────────────
@dataclass
class Position:
    ticker:      str
    entry_idx:   int
    entry_dt:    object
    entry:       float
    sl:          float
    risk:        float
    pos_rub:     float
    shares:      float
    remaining:   float = 1.0
    partial_pnl: float = 0.0
    tp_hit:      int   = 0
    risk_pct:    float = 0.0


@dataclass
class ClosedTrade:
    ticker:       str
    entry_dt:     object
    exit_dt:      object
    entry:        float
    exit_price:   float
    reason:       str
    pos_rub:      float
    pnl_rub:      float
    pnl_pct:      float
    hold_days:    int
    win:          bool
    tp_levels:    int
    risk_pct:     float
    capital_after: float


# ── Симулятор v3c ─────────────────────────────────────────────────────────────
class ProSimulatorV3c:
    def __init__(self):
        self.rm      = RiskManager(INITIAL_CAPITAL, RISK_PARAMS)
        self.data:   dict[str, pd.DataFrame] = {}
        self.signals:dict[str, pd.DataFrame] = {}
        self.trades: list[ClosedTrade]       = []
        self.equity: list[tuple]             = [(pd.Timestamp("2022-01-01"), INITIAL_CAPITAL)]
        self.blocked:list[dict]              = []

    def load(self):
        print("  Загрузка данных...")
        gen = BacktestEngineV2(SCHEME, MAX_HOLD, SIGNAL_CFG, "D")
        for t in TICKERS:
            df = load_daily(t)
            if df is None or len(df) < 250:
                print(f"    {t:8s}: нет данных")
                continue
            self.data[t]    = df
            self.signals[t] = gen._gen.generate(df)
            sigs = int(self.signals[t]["signal"].sum())
            print(f"    {t:8s}: {len(df)} баров  сигналов={sigs}")

    def run(self):
        all_dates = sorted({d for df in self.data.values() for d in df.index})
        open_pos: dict[str, Position] = {}
        n_tp = len(FRACS)

        for date in all_dates:
            self.rm.on_day_start()

            # ── Закрытие позиций ──────────────────────────────────────────
            to_close = []
            for ticker, pos in open_pos.items():
                df = self.data.get(ticker)
                if df is None or date not in df.index:
                    continue
                bar  = df.loc[date]
                hi   = float(bar["high"])
                lo   = float(bar["low"])
                op   = float(bar["open"])
                i_now = df.index.get_loc(date)
                hold  = i_now - pos.entry_idx
                lv    = pos.tp_hit

                # Stepped TP
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
                                     pos.entry + LEVELS[lv-1] * pos.risk * 0.90)
                        pos.tp_hit = lv + 1
                        lv = pos.tp_hit
                    else:
                        break

                reason = ep_f = None
                if pos.remaining <= 1e-6:
                    reason = f"TP{n_tp}"; ep_f = pos.entry + LEVELS[-1] * pos.risk
                elif hold >= MAX_HOLD:
                    reason = "TIME"; ep_f = op * (1 - SLIPPAGE)
                elif lo <= pos.sl:
                    reason = "SL"; ep_f = max(pos.sl * (1 - SLIPPAGE), lo)

                if reason:
                    rem = pos.remaining
                    pnl = (pos.partial_pnl + (ep_f - pos.entry) * rem * pos.shares
                           - (pos.entry + ep_f) * COMMISSION * rem * pos.shares)
                    pnl_pct = pnl / pos.pos_rub * 100
                    ct = ClosedTrade(
                        ticker=ticker, entry_dt=pos.entry_dt, exit_dt=date,
                        entry=pos.entry, exit_price=ep_f, reason=reason,
                        pos_rub=pos.pos_rub, pnl_rub=round(pnl, 2),
                        pnl_pct=round(pnl_pct, 3), hold_days=hold,
                        win=bool(pnl > 0), tp_levels=pos.tp_hit,
                        risk_pct=pos.risk_pct,
                        capital_after=self.rm.state.capital + pnl,
                    )
                    self.trades.append(ct)
                    self.rm.on_trade_closed(ticker, pnl, pnl_pct)
                    self.equity.append((date, self.rm.state.capital))
                    to_close.append(ticker)

            for t in to_close:
                del open_pos[t]

            # ── Новые сигналы ─────────────────────────────────────────────
            for ticker in TICKERS:
                if ticker in open_pos:
                    continue
                df  = self.data.get(ticker)
                sig = self.signals.get(ticker)
                if df is None or sig is None or date not in df.index:
                    continue
                i_now = df.index.get_loc(date)
                if i_now < 1:
                    continue
                prev = sig.iloc[i_now - 1]
                if not bool(prev["signal"]):
                    continue

                sl_v   = float(prev["sl"])
                risk_v = float(prev["risk"])
                entry  = float(df.iloc[i_now]["open"]) * (1 + SLIPPAGE)
                if sl_v >= entry or risk_v <= 0:
                    continue

                sl_pct = (entry - sl_v) / entry
                dec = self.rm.can_open(ticker, sl_pct)
                if not dec.allowed:
                    self.blocked.append({"date": date, "ticker": ticker, "reason": dec.reason})
                    continue

                pos_rub = dec.position_size_rub
                shares  = pos_rub / entry
                open_pos[ticker] = Position(
                    ticker=ticker, entry_idx=i_now, entry_dt=date,
                    entry=entry, sl=sl_v, risk=risk_v,
                    pos_rub=pos_rub, shares=shares, risk_pct=dec.risk_pct_used,
                )
                self.rm.on_position_opened(ticker)

        # Закрытие остатков
        for ticker, pos in open_pos.items():
            df  = self.data[ticker]
            ep  = float(df["close"].iloc[-1]) * (1 - SLIPPAGE)
            rem = pos.remaining
            pnl = (pos.partial_pnl + (ep - pos.entry) * rem * pos.shares
                   - (pos.entry + ep) * COMMISSION * rem * pos.shares)
            pnl_pct = pnl / pos.pos_rub * 100
            ct = ClosedTrade(
                ticker=ticker, entry_dt=pos.entry_dt, exit_dt=df.index[-1],
                entry=pos.entry, exit_price=ep, reason="END",
                pos_rub=pos.pos_rub, pnl_rub=round(pnl, 2),
                pnl_pct=round(pnl_pct, 3), hold_days=len(df) - 1 - pos.entry_idx,
                win=bool(pnl > 0), tp_levels=pos.tp_hit,
                risk_pct=pos.risk_pct,
                capital_after=self.rm.state.capital + pnl,
            )
            self.trades.append(ct)
            self.rm.on_trade_closed(ticker, pnl, pnl_pct)
            self.equity.append((df.index[-1], self.rm.state.capital))


def print_report(sim: ProSimulatorV3c):
    trades = sim.trades
    if not trades:
        print("Сделок нет.")
        return

    cap_final = sim.rm.state.capital
    total_rub = cap_final - INITIAL_CAPITAL
    total_pct = total_rub / INITIAL_CAPITAL * 100

    eq_vals   = np.array([e[1] for e in sim.equity])
    peak      = np.maximum.accumulate(np.maximum(eq_vals, 1.0))
    dd_arr    = (eq_vals - peak) / peak * 100
    max_dd    = float(dd_arr.min())

    years = 4.1
    ann   = ((cap_final / INITIAL_CAPITAL) ** (1 / years) - 1) * 100

    wins   = [t for t in trades if t.win]
    losses = [t for t in trades if not t.win]
    wr     = len(wins) / len(trades) * 100
    aw_rub = float(np.mean([t.pnl_rub for t in wins]))   if wins   else 0
    al_rub = float(np.mean([t.pnl_rub for t in losses])) if losses else 0
    pf     = abs(sum(t.pnl_rub for t in wins)) / (abs(sum(t.pnl_rub for t in losses)) + 1e-9)
    wl     = abs(aw_rub / al_rub) if al_rub else 99

    from collections import Counter
    exit_cnt   = Counter(t.reason for t in trades)
    ticker_pnl = defaultdict(float)
    for t in trades:
        ticker_pnl[t.ticker] += t.pnl_rub

    blocked_reasons = Counter(b["reason"] for b in sim.blocked)

    print("\n" + "═" * 62)
    print("  ОТЧЁТ: Pro Portfolio Simulator v3c  [без TGKA и MTSS]")
    print("═" * 62)
    print(f"  Начальный капитал:   {INITIAL_CAPITAL:>10,.0f} ₽")
    print(f"  Конечный капитал:    {cap_final:>10,.0f} ₽")
    print(f"  Итого P&L:           {total_rub:>+10,.0f} ₽  ({total_pct:+.1f}%)")
    print(f"  Годовая доходность:  {ann:>+10.1f}%")
    print(f"  Макс. просадка:      {max_dd:>10.1f}%")
    print(f"  Период:              {years:.1f} лет")
    print("─" * 62)
    print(f"  Всего сделок:        {len(trades):>10}")
    print(f"  Win Rate:            {wr:>10.1f}%")
    print(f"  Profit Factor:       {pf:>10.2f}")
    print(f"  Avg Win / Avg Loss:  {wl:>10.2f}×")
    print(f"  Avg Win:             {aw_rub:>+10,.0f} ₽")
    print(f"  Avg Loss:            {al_rub:>+10,.0f} ₽")
    print("─" * 62)
    print("  Выходы:")
    for reason, cnt in sorted(exit_cnt.items()):
        r_t   = [t for t in trades if t.reason == reason]
        r_w   = sum(1 for t in r_t if t.win)
        r_pnl = sum(t.pnl_rub for t in r_t)
        print(f"    {reason:8s}: {cnt:3d}  WR={r_w/cnt*100:4.0f}%  P&L={r_pnl:>+9,.0f} ₽")
    print("─" * 62)
    print("  Блокировок:")
    for reason, cnt in blocked_reasons.most_common(8):
        print(f"    {reason:35s}: {cnt}")
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
    print("\n  Сравнение v2 → v3c:")
    print(f"    P&L:      +15,490 ₽ (+15.5%)  →  {total_rub:+,.0f} ₽ ({total_pct:+.1f}%)")
    print(f"    WR:       48%  →  {wr:.0f}%")
    print(f"    PF:       1.16 →  {pf:.2f}")
    print(f"    MaxDD:   -15.1% →  {max_dd:.1f}%")
    print(f"    Ann.Ret:  3.6% →  {ann:.1f}%")
    print("═" * 62)

    df_out = pd.DataFrame([{
        "ticker": t.ticker, "entry_dt": t.entry_dt, "exit_dt": t.exit_dt,
        "entry": t.entry, "exit_price": t.exit_price, "reason": t.reason,
        "pos_rub": t.pos_rub, "pnl_rub": t.pnl_rub, "pnl_pct": t.pnl_pct,
        "hold_days": t.hold_days, "win": t.win,
    } for t in trades])
    df_out.to_csv("c:/investor/pro_trades_v3c.csv", index=False)
    print("  Лог сделок → pro_trades_v3c.csv")


if __name__ == "__main__":
    print("=== Pro Portfolio Simulator v3c ===\n")
    sim = ProSimulatorV3c()
    sim.load()
    print(f"\nЗагружено тикеров: {len(sim.data)}")
    print("Запуск симуляции...")
    sim.run()
    print_report(sim)
