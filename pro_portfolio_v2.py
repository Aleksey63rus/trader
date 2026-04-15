"""
Профессиональная портфельная симуляция v2.

Изменения vs v1:
  - Стратегия v2 (улучшенные сигналы: Momentum Score + Pullback + Trailing SL)
  - Риск 5% на сделку (было 1%)
  - Max 5 позиций (было 3)
  - Daily Loss Limit 5% (было 3%)
  - DD reduce при 15% (было 10%), halt при 25% (было 15%)
  - Kelly cap 5% (было 2%)
  - Схема TP: AGR (агрессивная, больший потенциал)
  - Таймфрейм: Daily
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

# Лучшие параметры из оптимизации
SCHEME = "AGR"   # (0.30, 0.30, 0.40) × (1.2, 3.0, 7.0)
FRACS, LEVELS = TP_SCHEMES[SCHEME]

TICKERS = [
    "GAZP","LKOH","NVTK","ROSN","SNGS","SNGSP",
    "SBER","SBERP","T","VTBR",
    "GMKN","NLMK","MTLR","CHMF","MAGN","RUAL","ALRS","PLZL",
    "YDEX","OZON","MGNT",
    "TATN","TATNP",
    "AFLT","TGKA","IRAO","MTSS","PHOR","OZPH",
]

DATA_DIR = Path("c:/investor/data")

# Параметры риска
RISK_PARAMS = RiskParams(
    risk_pct             = 0.05,   # 5% на сделку
    max_risk_pct         = 0.05,
    min_risk_pct         = 0.01,
    max_positions        = 5,      # до 5 позиций
    daily_loss_limit_pct = 0.05,   # 5% дневной лимит
    dd_reduce_threshold  = 0.15,   # при DD 15% → лот × 0.5
    dd_halt_threshold    = 0.25,   # при DD 25% → стоп
    dd_lot_multiplier    = 0.5,
    kelly_fraction       = 0.25,
    kelly_max            = 0.05,   # Kelly cap 5%
)

# Параметры стратегии
SIGNAL_CFG = SignalConfig(
    min_score     = 4,
    er_min        = 0.30,
    adx_min       = 20,
    vol_ratio_min = 1.2,
    sl_atr_mult   = 1.8,
    use_pullback  = True,
    sl_min_pct    = 0.025,
    sl_max_pct    = 0.08,
)


# ── Загрузчик ─────────────────────────────────────────────────────────────────
def load_daily(ticker: str) -> Optional[pd.DataFrame]:
    path = DATA_DIR / f"{ticker}_2022_2026_D.csv"
    if not path.exists(): return None
    try:
        df = pd.read_csv(path, sep=";")
        df.columns = [c.strip("<>").lower() for c in df.columns]
        df["dt"] = pd.to_datetime(df["date"].astype(str), format="%d/%m/%y", errors="coerce")
        df = (df.dropna(subset=["dt"]).set_index("dt")
                .rename(columns={"vol":"volume"})
               [["open","high","low","close","volume"]])
        for c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna()
        return df if len(df) >= 100 else None
    except: return None


# ── Dataclasses ───────────────────────────────────────────────────────────────
@dataclass
class Position:
    ticker:      str
    entry_idx:   int
    entry_dt:    object
    entry:       float
    sl:          float
    risk:        float
    pos_rub:     float      # сумма позиции в рублях
    shares:      float      # кол-во акций
    remaining:   float = 1.0
    partial_pnl: float = 0.0
    tp_hit:      int   = 0
    risk_pct:    float = 0.0


@dataclass
class ClosedTrade:
    ticker:     str
    entry_dt:   object
    exit_dt:    object
    entry:      float
    exit_price: float
    reason:     str
    pos_rub:    float
    pnl_rub:    float
    pnl_pct:    float
    hold_days:  int
    win:        bool
    tp_levels:  int
    risk_pct:   float
    capital_after: float


# ── Симулятор ─────────────────────────────────────────────────────────────────
class ProSimulatorV2:
    def __init__(self):
        self.rm = RiskManager(INITIAL_CAPITAL, RISK_PARAMS)
        self.data:    dict[str, pd.DataFrame] = {}
        self.signals: dict[str, pd.DataFrame] = {}
        self.trades:  list[ClosedTrade]       = []
        self.equity:  list[tuple]             = [(pd.Timestamp("2022-01-01"), INITIAL_CAPITAL)]
        self.blocked: list[dict]              = []

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

            # Управление открытыми позициями
            to_close = []
            for ticker, pos in open_pos.items():
                df = self.data.get(ticker)
                if df is None or date not in df.index: continue
                bar    = df.loc[date]
                hi, lo = float(bar["high"]), float(bar["low"])
                op     = float(bar["open"])
                i_now  = df.index.get_loc(date)
                hold   = i_now - pos.entry_idx
                lv     = pos.tp_hit

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
                        pos.tp_hit = lv + 1; lv = pos.tp_hit
                    else: break

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

            for t in to_close: del open_pos[t]

            # Поиск новых сигналов
            for ticker in TICKERS:
                if ticker in open_pos: continue
                df  = self.data.get(ticker)
                sig = self.signals.get(ticker)
                if df is None or sig is None or date not in df.index: continue
                i_now = df.index.get_loc(date)
                if i_now < 1: continue
                prev = sig.iloc[i_now - 1]
                if not bool(prev["signal"]): continue

                sl_v   = float(prev["sl"])
                risk_v = float(prev["risk"])
                entry  = float(df.iloc[i_now]["open"]) * (1 + SLIPPAGE)
                if sl_v >= entry or risk_v <= 0: continue

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
                pnl_pct=round(pnl_pct, 3), hold_days=len(df)-pos.entry_idx,
                win=bool(pnl > 0), tp_levels=pos.tp_hit, risk_pct=pos.risk_pct,
                capital_after=self.rm.state.capital + pnl,
            )
            self.trades.append(ct)
            self.rm.on_trade_closed(ticker, pnl, pnl_pct)
            self.equity.append((df.index[-1], self.rm.state.capital))


# ── Отчёт ─────────────────────────────────────────────────────────────────────
def print_report(sim: ProSimulatorV2):
    trades = sim.trades
    rm     = sim.rm
    if not trades:
        print("  Нет сделок."); return

    wins   = [t for t in trades if t.win]
    losses = [t for t in trades if not t.win]
    total_pnl = sum(t.pnl_rub for t in trades)
    final_cap = INITIAL_CAPITAL + total_pnl
    total_ret = total_pnl / INITIAL_CAPITAL * 100

    first_dt = min(t.entry_dt for t in trades)
    last_dt  = max(t.exit_dt  for t in trades)
    years    = max((last_dt - first_dt).days / 365.25, 0.01)
    ann_ret  = ((final_cap / INITIAL_CAPITAL) ** (1 / years) - 1) * 100

    pnl_s  = pd.Series([t.pnl_pct for t in trades])
    sharpe = float(pnl_s.mean() / (pnl_s.std() + 1e-9) * np.sqrt(252)) if len(pnl_s) > 1 else 0

    eq_v  = np.array([e[1] for e in sim.equity])
    peak  = np.maximum.accumulate(np.maximum(eq_v, 1.0))
    dd    = float(((eq_v - peak) / peak * 100).min())

    gw = sum(t.pnl_rub for t in wins)
    gl = abs(sum(t.pnl_rub for t in losses))
    pf = gw / (gl + 1e-9)

    aw = float(np.mean([t.pnl_rub for t in wins]))   if wins   else 0
    al = float(np.mean([t.pnl_rub for t in losses])) if losses else 0
    wl = abs(aw / (al + 1e-9))

    print()
    print("═"*65)
    print("  ПРОФЕССИОНАЛЬНАЯ СИМУЛЯЦИЯ v2 — ИТОГОВЫЙ ОТЧЁТ")
    print("  (Риск 5%/сделку | Стратегия v2 | Схема AGR | Daily ТФ)")
    print("═"*65)
    print(f"  Период:            {first_dt.date()} → {last_dt.date()} ({years:.1f} лет)")
    print(f"  Нач. капитал:       {INITIAL_CAPITAL:>12,.0f} руб.")
    print(f"  Фин. капитал:       {final_cap:>12,.0f} руб.")
    print(f"  P&L:                {total_pnl:>+12,.0f} руб.  ({total_ret:+.1f}%)")
    print(f"  Годовая доходность: {ann_ret:>12.1f}%")
    print()
    print("  ── Качество ────────────────────────────────────────────")
    print(f"  Сделок:             {len(trades):>12d}")
    print(f"  Win Rate:           {len(wins)/len(trades)*100:>11.0f}%")
    print(f"  Profit Factor:      {pf:>12.2f}")
    print(f"  Sharpe Ratio:       {sharpe:>12.2f}")
    print(f"  Avg Win/Loss:       {wl:>12.2f}x")
    print(f"  Avg Win:            {aw:>+12,.0f} руб.")
    print(f"  Avg Loss:           {al:>+12,.0f} руб.")
    print()
    print("  ── Риск ────────────────────────────────────────────────")
    print(f"  Max Drawdown:       {dd:>11.1f}%")
    print(f"  DD-стопов:          {rm.state.dd_halt_count:>12d}")
    print(f"  Дн. стопов:         {rm.state.daily_halt_count:>12d}")
    print(f"  Kelly риск (итог):  {rm._kelly_risk_pct()*100:>11.2f}%")
    print(f"  Заблок. сигналов:   {len(sim.blocked):>12d}")
    from collections import Counter
    br = Counter(b["reason"].split(" ")[0] for b in sim.blocked)
    if br: print("    └ " + "  ".join(f"{k}:{v}" for k,v in br.most_common()))

    print()
    print("  ── По тикерам ──────────────────────────────────────────")
    print(f"  {'Тикер':<7} {'n':>4} {'WR%':>5} {'PF':>5} {'P&L':>10} {'Hold':>6}")
    print(f"  {'-'*45}")
    by_t = defaultdict(list)
    for t in trades: by_t[t.ticker].append(t)
    for tkr in sorted(by_t):
        tt  = by_t[tkr]
        tw  = [x for x in tt if x.win]
        pnl_t = sum(x.pnl_rub for x in tt)
        ww  = [x.pnl_rub for x in tw]
        ll  = [x.pnl_rub for x in tt if not x.win]
        pf_t = sum(ww)/(abs(sum(ll))+1e-9) if ll else 99
        hold_t = float(np.mean([x.hold_days for x in tt]))
        wr_t = len(tw)/len(tt)*100 if tt else 0
        print(f"  {tkr:<7} {len(tt):>4d} {wr_t:>4.0f}% {min(pf_t,99):>5.2f} "
              f"{pnl_t:>+10,.0f} {hold_t:>5.1f}д")

    print()
    print("  ── Выходы ──────────────────────────────────────────────")
    from collections import Counter
    exits = Counter(t.reason for t in trades)
    for reason, cnt in sorted(exits.items()):
        wr_r = sum(1 for t in trades if t.reason==reason and t.win)
        pnl_r = sum(t.pnl_rub for t in trades if t.reason==reason)
        print(f"  {reason:<6} n={cnt:>3d}  WR={wr_r/cnt*100:>4.0f}%  P&L={pnl_r:>+10,.0f} руб.")

    print()
    print("  ── Годовой разрез ──────────────────────────────────────")
    years_d = defaultdict(list)
    for t in trades: years_d[t.exit_dt.year].append(t)
    for yr in sorted(years_d):
        yt = years_d[yr]
        yw = [x for x in yt if x.win]
        pnl_yr = sum(x.pnl_rub for x in yt)
        print(f"  {yr}  n={len(yt):>3d}  WR={len(yw)/len(yt)*100:>4.0f}%  "
              f"P&L={pnl_yr:>+10,.0f} руб.")

    print()
    print("  ── Оценка (проп-стандарты) ─────────────────────────────")
    checks = [
        (ann_ret >= 20,   f"Годовая доход. {ann_ret:.1f}% {'≥' if ann_ret>=20 else '<'} 20%"),
        (len(wins)/len(trades) >= 0.55, f"WR {len(wins)/len(trades)*100:.0f}% {'≥' if len(wins)/len(trades)>=0.55 else '<'} 55%"),
        (pf >= 1.5,       f"PF {pf:.2f} {'≥' if pf>=1.5 else '<'} 1.5"),
        (dd >= -20,       f"MaxDD {dd:.1f}% {'≥' if dd>=-20 else '<'} -20%"),
        (sharpe >= 1.0,   f"Sharpe {sharpe:.2f} {'≥' if sharpe>=1.0 else '<'} 1.0"),
    ]
    for ok, msg in checks:
        print(f"  {'✓' if ok else '✗'} {msg}")
    print("═"*65)


if __name__ == "__main__":
    print("═"*65)
    print("  ПРОФЕССИОНАЛЬНАЯ СИМУЛЯЦИЯ v2")
    print("  Риск: 5%/сделку | ТФ: Daily | Схема: AGR | Позиций: 5")
    print("═"*65)
    sim = ProSimulatorV2()
    sim.load()
    print()
    print("  Симуляция...")
    sim.run()
    print_report(sim)

    if sim.trades:
        df_out = pd.DataFrame([{
            "ticker": t.ticker, "entry_dt": t.entry_dt, "exit_dt": t.exit_dt,
            "entry": t.entry, "exit": t.exit_price, "reason": t.reason,
            "pos_rub": t.pos_rub, "pnl_rub": t.pnl_rub, "pnl_pct": t.pnl_pct,
            "hold_days": t.hold_days, "win": t.win, "tp_levels": t.tp_levels,
            "risk_pct": t.risk_pct, "capital_after": t.capital_after,
        } for t in sim.trades])
        out = DATA_DIR / "pro_trades_v2.csv"
        df_out.to_csv(out, index=False, encoding="utf-8-sig")
        print(f"\n  Лог сохранён: {out}")
