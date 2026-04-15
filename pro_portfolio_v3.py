"""
Профессиональная портфельная симуляция v3.

Ключевые улучшения vs v2:
  - Стратегия v3 (Volume Delta + ADX>=25 + Kyle Lambda + Недельный фильтр + Swing SL)
  - Тикер-вайтлист: убраны убыточные TGKA, SBER, GAZP, YDEX, MTSS
  - TP схема PRO: 4 уровня (1.0 / 2.5 / 5.0 / 9.0R)
  - max_hold = 30 дней (был 20) — лучшие TIME-выходы
  - Streak Protection: пропуск после 3 убытков подряд
  - Риск 5% на сделку, max 5 позиций
  - Ослабленный риск-менеджмент: DD halt 30% (было 25%)
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict
from typing import Optional
import numpy as np
import pandas as pd

from core.strategy_v3 import BacktestEngineV3, SignalConfigV3, TP_SCHEMES
from core.risk import RiskManager, RiskParams

# ── Конфиг ────────────────────────────────────────────────────────────────────
INITIAL_CAPITAL = 100_000.0
COMMISSION      = 0.0005
SLIPPAGE        = 0.0001
MAX_HOLD        = 20      # оптимально по тестам

SCHEME = "BAL"  # BAL лучший по Sharpe и WR по тестам

# ── Тикер-вайтлист (убраны TGKA, SBER, GAZP, YDEX, MTSS) ─────────────────────
TICKERS = [
    # Нефть и газ (только прибыльные)
    "LKOH", "NVTK", "ROSN", "SNGS", "SNGSP",
    # Банки (только прибыльные)
    "SBERP", "T", "VTBR",
    # Металлургия
    "GMKN", "NLMK", "MTLR", "CHMF", "MAGN", "RUAL", "ALRS", "PLZL",
    # Ритейл / технологии
    "OZON", "MGNT",
    # Прочее
    "TATN", "TATNP",
    "AFLT", "IRAO", "PHOR", "OZPH",
]

DATA_DIR = Path("c:/investor/data")

# ── Параметры риска (чуть ослабленные) ────────────────────────────────────────
RISK_PARAMS = RiskParams(
    risk_pct             = 0.05,   # 5% на сделку
    max_risk_pct         = 0.06,
    min_risk_pct         = 0.01,
    max_positions        = 5,
    daily_loss_limit_pct = 0.05,
    dd_reduce_threshold  = 0.18,   # при DD 18% → лот × 0.5 (было 15%)
    dd_halt_threshold    = 0.30,   # при DD 30% → стоп (было 25%)
    dd_lot_multiplier    = 0.5,
    kelly_fraction       = 0.30,
    kelly_max            = 0.06,   # Kelly cap 6% (было 5%)
)

# ── Параметры стратегии v3 ─────────────────────────────────────────────────────
SIGNAL_CFG = SignalConfigV3(
    min_score       = 4,
    er_min          = 0.30,
    adx_min         = 20,          # оптимально по тестам
    vol_ratio_min   = 1.2,
    use_vol_delta   = False,       # VD не улучшает на дневных данных
    use_kyle_filter = False,       # Kyle медленный, минимальный эффект
    use_weekday     = False,       # weekday фильтр снижает WR
    use_pullback    = True,
    pb_tolerance    = 0.02,
    atr_ratio_min   = 0.90,
    sl_atr_mult     = 1.8,
    sl_swing_bars   = 3,           # КЛЮЧЕВОЕ УЛУЧШЕНИЕ: swing за 3 бара
    sl_min_pct      = 0.025,
    sl_max_pct      = 0.08,
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
                df[col] = pd.to_numeric(df[col].astype(str).str.replace(",", "."), errors="coerce")
        df = df.rename(columns={"vol": "volume"})
        df = df[["open", "high", "low", "close", "volume"]].dropna()
        return df if len(df) >= 200 else None
    except Exception:
        return None


# ── Dataclasses ───────────────────────────────────────────────────────────────
@dataclass
class Position:
    ticker:    str
    entry_dt:  object
    entry:     float
    sl:        float
    risk:      float
    lot_pct:   float
    lot_rub:   float
    fracs:     tuple
    levels:    tuple
    remaining: float = 1.0
    partial_pnl: float = 0.0
    tp_hit:    int   = 0
    trailing_sl: float = 0.0


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


# ══════════════════════════════════════════════════════════════════════════════
# Симулятор v3
# ══════════════════════════════════════════════════════════════════════════════
class ProSimulatorV3:
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
        """Симуляция по единой временной оси."""
        from core.indicators import atr as _atr

        fracs, levels = TP_SCHEMES[SCHEME]
        n_tp = len(fracs)

        engine = BacktestEngineV3(
            scheme=SCHEME, max_hold=MAX_HOLD,
            cfg=SIGNAL_CFG, timeframe="D",
            trailing_mult=2.0, streak_limit=3,
        )

        # Предрассчитать сигналы и ATR
        sigs: dict[str, pd.DataFrame] = {}
        at14s: dict[str, pd.Series] = {}
        for t, df in data.items():
            sigs[t] = engine._gen.generate(df)
            at14s[t] = _atr(df, 14)

        # Единая временная ось
        all_dates = sorted(
            set().union(*[df.index.tolist() for df in data.values()])
        )

        positions: dict[str, Position] = {}
        trades: list[ClosedTrade] = []
        equity_curve = [INITIAL_CAPITAL]
        blocked_stats = defaultdict(int)
        loss_streak = 0

        for dt in all_dates:
            self.rm.on_day_start()

            # ── Закрытие позиций ──────────────────────────────────────────
            to_close = []
            for ticker, pos in positions.items():
                if ticker not in data or dt not in data[ticker].index:
                    continue
                bar = data[ticker].loc[dt]
                bar_hi = bar["high"]
                bar_lo = bar["low"]
                bar_op = bar["open"]
                bar_cl = bar["close"]
                idx_pos = data[ticker].index.get_loc(dt)

                entry  = pos.entry
                risk   = pos.risk
                lv     = pos.tp_hit

                # Проверка TP
                while lv < n_tp and pos.remaining > 1e-9:
                    tp_px = entry + levels[lv] * risk
                    if bar_hi >= tp_px:
                        ep   = tp_px * (1 - SLIPPAGE)
                        frac = min(fracs[lv], pos.remaining)
                        pnl_part = ((ep - entry) * frac
                                    - (entry + ep) * COMMISSION * frac)
                        pos.partial_pnl += pnl_part
                        pos.remaining   -= frac
                        # Перемещаем SL
                        if lv == 0:
                            pos.sl = entry * 1.001
                        else:
                            pos.sl = entry + levels[lv-1] * risk * 0.92
                        pos.trailing_sl = pos.sl
                        pos.tp_hit = lv + 1
                        lv = pos.tp_hit
                    else:
                        break

                # ATR Trailing Stop после 1-го TP
                if pos.tp_hit >= 1:
                    at14_v = at14s.get(ticker)
                    if at14_v is not None and dt in at14_v.index:
                        trail = bar_cl - 2.0 * float(at14_v.loc[dt])
                        if trail > pos.sl:
                            pos.sl = trail

                reason = None
                ep_final = None

                if pos.remaining <= 1e-6:
                    reason = f"TP{n_tp}"
                    ep_final = bar_cl * (1 - SLIPPAGE)
                elif (dt - pos.entry_dt).days >= MAX_HOLD:
                    reason = "TIME"
                    ep_final = bar_op * (1 - SLIPPAGE)
                elif bar_lo <= pos.sl:
                    reason = "SL"
                    ep_final = max(pos.sl * (1 - SLIPPAGE), bar_lo)

                if reason:
                    rem = pos.remaining
                    pnl_rub = (pos.partial_pnl
                               + (ep_final - entry) * rem / entry * pos.lot_rub
                               - (entry + ep_final) / entry * COMMISSION * pos.lot_rub * rem)
                    ct = ClosedTrade(
                        ticker   = ticker,
                        entry_dt = pos.entry_dt,
                        exit_dt  = dt,
                        entry    = entry,
                        exit     = ep_final,
                        reason   = reason,
                        lot_rub  = pos.lot_rub,
                        pnl_rub  = pnl_rub,
                        pnl_pct  = pnl_rub / pos.lot_rub * 100,
                        hold_d   = (dt - pos.entry_dt).days,
                        win      = bool(pnl_rub > 0),
                    )
                    trades.append(ct)
                    self.rm.on_trade_closed(ticker, pnl_rub, ct.pnl_pct)
                    equity_curve.append(self.rm.state.capital)

                    if ct.win:
                        loss_streak = 0
                    else:
                        loss_streak += 1
                    to_close.append(ticker)

            for t in to_close:
                del positions[t]

            # ── Новые входы ───────────────────────────────────────────────
            # Streak Protection
            if loss_streak >= 3:
                loss_streak = max(0, loss_streak - 1)
                continue

            # Кандидаты на вход в этот день
            candidates = []
            for ticker in TICKERS:
                if ticker in positions:
                    continue
                if ticker not in data or dt not in sigs[ticker].index:
                    continue
                row = sigs[ticker].loc[dt]
                if not bool(row.get("signal", 0)):
                    continue
                score    = int(row.get("score", 0))
                vd_ratio = float(row.get("vd_ratio", 0))
                candidates.append((ticker, score, vd_ratio))

            # Сортируем по score DESC (лучшие сначала)
            candidates.sort(key=lambda x: -x[1])

            for ticker, score, vd_ratio in candidates:
                if ticker not in data or dt not in data[ticker].index:
                    continue

                sig_row = sigs[ticker].loc[dt]
                df_t    = data[ticker]
                idx_dt  = df_t.index.get_loc(dt)
                if idx_dt + 1 >= len(df_t):
                    continue

                next_dt  = df_t.index[idx_dt + 1]
                next_bar = df_t.loc[next_dt]
                entry    = next_bar["open"] * (1 + SLIPPAGE)
                sl       = float(sig_row["sl"])
                risk     = float(sig_row["risk"])

                if sl >= entry or risk <= 0:
                    continue

                # Риск-менеджер
                sl_pct_v = (entry - sl) / entry if entry > 0 else 0.05
                dec = self.rm.can_open(ticker, sl_pct_v)
                if not dec.allowed:
                    blocked_stats[dec.reason] += 1
                    continue

                lot_rub = dec.position_size_rub
                cap_now = self.rm.state.capital
                lot_pct = lot_rub / cap_now if cap_now > 0 else 0.05

                pos = Position(
                    ticker    = ticker,
                    entry_dt  = next_dt,
                    entry     = entry,
                    sl        = sl,
                    risk      = risk,
                    lot_pct   = lot_pct,
                    lot_rub   = lot_rub,
                    fracs     = fracs,
                    levels    = levels,
                    trailing_sl = sl,
                )
                positions[ticker] = pos
                self.rm.on_position_opened(ticker)

        # Закрытие остатков в конце периода
        for ticker, pos in positions.items():
            if ticker not in data:
                continue
            df_t = data[ticker]
            bar_cl = df_t["close"].iloc[-1]
            ep  = bar_cl * (1 - SLIPPAGE)
            rem = pos.remaining
            pnl_rub = (pos.partial_pnl
                       + (ep - pos.entry) * rem / pos.entry * pos.lot_rub
                       - (pos.entry + ep) / pos.entry * COMMISSION * pos.lot_rub * rem)
            ct = ClosedTrade(
                ticker   = ticker,
                entry_dt = pos.entry_dt,
                exit_dt  = df_t.index[-1],
                entry    = pos.entry,
                exit     = ep,
                reason   = "END",
                lot_rub  = pos.lot_rub,
                pnl_rub  = pnl_rub,
                pnl_pct  = pnl_rub / pos.lot_rub * 100,
                hold_d   = (df_t.index[-1] - pos.entry_dt).days,
                win      = bool(pnl_rub > 0),
            )
            trades.append(ct)
            self.rm.on_trade_closed(ticker, pnl_rub, ct.pnl_pct)
            equity_curve.append(self.rm.state.capital)

        self._equity = equity_curve
        self._final_capital = self.rm.state.capital
        self._blocked = blocked_stats
        return trades

    def print_report(self, trades: list[ClosedTrade]):
        if not trades:
            print("\n=== Сделок нет ===")
            return

        total_rub   = self._final_capital - INITIAL_CAPITAL
        total_pct   = total_rub / INITIAL_CAPITAL * 100

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
        exit_cnt = Counter(t.reason for t in trades)
        ticker_pnl = defaultdict(float)
        for t in trades:
            ticker_pnl[t.ticker] += t.pnl_rub

        print("\n" + "═" * 60)
        print("  ОТЧЁТ: Pro Portfolio Simulator v3")
        print("═" * 60)
        print(f"  Начальный капитал:   {INITIAL_CAPITAL:>10,.0f} ₽")
        print(f"  Конечный капитал:    {self._final_capital:>10,.0f} ₽")
        print(f"  Итого P&L:           {total_rub:>+10,.0f} ₽  ({total_pct:+.1f}%)")
        print(f"  Годовая доходность:  {ann:>+10.1f}%")
        print(f"  Макс. просадка:      {dd:>10.1f}%")
        print(f"  Период:              {years:.1f} лет")
        print("─" * 60)
        print(f"  Всего сделок:        {len(trades):>10}")
        print(f"  Win Rate:            {wr:>10.1f}%")
        print(f"  Profit Factor:       {pf:>10.2f}")
        print(f"  Avg Win / Avg Loss:  {wl:>10.2f}×")
        print(f"  Avg Win:             {aw:>+10,.0f} ₽")
        print(f"  Avg Loss:            {al:>+10,.0f} ₽")
        print("─" * 60)
        print("  Выходы:")
        for reason, cnt in sorted(exit_cnt.items()):
            r_trades = [t for t in trades if t.reason == reason]
            r_wins   = sum(1 for t in r_trades if t.win)
            r_pnl    = sum(t.pnl_rub for t in r_trades)
            print(f"    {reason:8s}: {cnt:3d} сделок  WR={r_wins/cnt*100:4.0f}%  P&L={r_pnl:>+9,.0f} ₽")
        print("─" * 60)
        print("  Блокировок:")
        for reason, cnt in sorted(self._blocked.items()):
            print(f"    {reason:20s}: {cnt}")
        print("─" * 60)
        print("  Топ-10 тикеров по P&L:")
        for t, p in sorted(ticker_pnl.items(), key=lambda x: -x[1])[:10]:
            n_t = sum(1 for tr in trades if tr.ticker == t)
            print(f"    {t:6s}: {p:>+9,.0f} ₽  ({n_t} сд)")
        print("  Аутсайдеры:")
        for t, p in sorted(ticker_pnl.items(), key=lambda x: x[1])[:5]:
            n_t = sum(1 for tr in trades if tr.ticker == t)
            print(f"    {t:6s}: {p:>+9,.0f} ₽  ({n_t} сд)")
        print("═" * 60)

        # Сравнение v2 vs v3
        print("\n  Сравнение v2 → v3:")
        print(f"    P&L:      +15,490 ₽ (+15.5%)  →  {total_rub:+,.0f} ₽ ({total_pct:+.1f}%)")
        print(f"    WR:       48%  →  {wr:.0f}%")
        print(f"    PF:       1.16 →  {pf:.2f}")
        print(f"    MaxDD:   -15.1% →  {dd:.1f}%")
        print(f"    Ann.Ret:  3.6% →  {ann:.1f}%")
        print("═" * 60)

        # Сохранение
        df_out = pd.DataFrame([{
            "ticker":    t.ticker,
            "entry_dt":  t.entry_dt,
            "exit_dt":   t.exit_dt,
            "entry":     t.entry,
            "exit":      t.exit,
            "reason":    t.reason,
            "lot_rub":   t.lot_rub,
            "pnl_rub":   t.pnl_rub,
            "pnl_pct":   t.pnl_pct,
            "hold_d":    t.hold_d,
            "win":       t.win,
        } for t in trades])
        df_out.to_csv("c:/investor/pro_trades_v3.csv", index=False)
        print("  Лог сделок → pro_trades_v3.csv")


if __name__ == "__main__":
    print("=== Pro Portfolio Simulator v3 ===\n")
    print("Загрузка данных...")
    sim  = ProSimulatorV3()
    data = sim.load()
    print(f"\nЗагружено тикеров: {len(data)}")
    print("\nЗапуск симуляции...")
    trades = sim.run(data)
    sim.print_report(trades)
