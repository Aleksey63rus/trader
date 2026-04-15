"""
Стратегия 1: ATR Breakout Daily (ATR_BO)
=========================================
Результат с реальными комиссиями БКС «Инвестор» (покупка 0%, продажа 0.30%):
  +2.0% годовых, MaxDD -57.4%, Sharpe 0.25, WR 41.2%, капитал 108 642 ₽
  (без комиссий: +4.0% годовых, 117 789 ₽ — разница 9 147 ₽ за 4 года)

Доходность по годам (БКС реал.):
  2023: +7.5% / 2024: +10.6% / 2025: −5.2% / 2026: −3.6%

Период теста: 2022–2026, 22 тикера MOEX, капитал 100 000 ₽, 68 сделок

Комиссии БКС «Инвестор»: покупка 0%, продажа 0.30% (раздельно)
  slippage 0.10% → потери 9 147 ₽ за 4 года (~134 ₽/сделку)

Логика:
  - Таймфрейм: Дневной (D)
  - Вход: бар с сильным бычьим движением (≥1.5×ATR14) + EMA200 + RSI + ADX + Volume
  - Выход: Trailing Stop 2×ATR(D), удержание до 45 дней
  - Капитал: 4 позиции × 25% = 100% в работе
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Any
import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from core.indicators import ema, atr, rsi, adx, volume_ratio

STRATEGY_ID   = "atr_bo_daily"
STRATEGY_NAME = "ATR Breakout Daily"
STRATEGY_DESC = (
    "Торгует сильные дневные пробои с подтверждением тренда. "
    "Вход: свеча-пробой ≥1.5×ATR с фильтрами EMA200/RSI/ADX/Volume. "
    "Выход: Trailing Stop 2×ATR, максимальное удержание 45 дней. "
    "Рекомендуется для среднесрочных позиций на MOEX."
)
VERSION = "1.0.0"


@dataclass
class ATRBOConfig:
    """Настраиваемые параметры стратегии."""
    # Фильтры входа
    atr_period: int = 14
    atr_bo_mult: float = 1.5        # множитель ATR для пробоя
    ema_period: int = 200            # период EMA-тренда
    rsi_period: int = 14
    rsi_min: float = 52.0
    rsi_max: float = 82.0
    adx_period: int = 14
    adx_min: float = 22.0
    vol_ratio_period: int = 20
    vol_ratio_min: float = 1.5       # объём выше среднего в N раз

    # Параметры выхода
    trail_mult: float = 2.0          # ATR-множитель trailing stop
    max_hold_days: int = 45          # максимальное удержание (дней)
    sl_pct: float = 0.0              # 0 = без жёсткого SL (только trailing)

    # Портфельные параметры
    # 4 позиции × 25% = 100% капитала — полное использование без простоя кэша
    max_positions: int = 4
    risk_pct: float = 0.25           # доля капитала на 1 позицию (4×25%=100%)
    # БКС «Инвестор»: покупка 0%, продажа 0.30% (по тарифному плану клиента)
    # Проскальзывание MOEX для ликвидных акций на дневном ТФ: ~0.10%
    commission_buy:  float = 0.0000  # покупка: 0% (БКС Инвестор)
    commission_sell: float = 0.0030  # продажа: 0.30% (БКС Инвестор)
    slippage: float        = 0.0010  # 0.10% проскальзывание MOEX
    # Совместимость: commission = среднее для отображения в UI
    commission: float      = 0.0015  # среднее (0+0.30)/2 — только для отображения
    reinvest: bool = True            # реинвестировать прибыль

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ATRBOConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def generate_signals(df: pd.DataFrame, cfg: ATRBOConfig) -> pd.DataFrame:
    """
    Генерирует сигналы на покупку.
    Возвращает DataFrame с колонкой 'signal' (bool) и 'at14' (ATR).
    """
    c    = df["close"]
    at14 = atr(df, cfg.atr_period)
    at5  = atr(df, 5)
    e200 = ema(c, cfg.ema_period)
    r    = rsi(c, cfg.rsi_period)
    a    = adx(df, cfg.adx_period)
    vr   = volume_ratio(df, cfg.vol_ratio_period)
    bm   = (c - c.shift(1)).clip(lower=0)

    sig = (
        (c > e200) &
        (bm >= cfg.atr_bo_mult * at14) &
        (at5 > at14 * 0.95) &
        (r >= cfg.rsi_min) & (r <= cfg.rsi_max) &
        (a >= cfg.adx_min) &
        (vr >= cfg.vol_ratio_min)
    )
    # Не входим подряд — ждём нового сигнала
    sig = sig & ~sig.shift(1).fillna(False)

    return pd.DataFrame({
        "signal": sig,
        "at14":   at14,
        "ema200": e200,
        "rsi":    r,
        "adx":    a,
        "vol_ratio": vr,
    }, index=df.index)


@dataclass
class BacktestPosition:
    ticker:    str
    entry_date: pd.Timestamp
    entry_px:  float
    shares:    float
    trail_sl:  float


@dataclass
class BacktestTrade:
    ticker:     str
    entry_date: pd.Timestamp
    exit_date:  pd.Timestamp
    entry_px:   float
    exit_px:    float
    shares:     float
    pnl_rub:    float
    pnl_pct:    float
    reason:     str          # TIME | SL | FORCED
    hold_days:  float

    def to_dict(self) -> dict:
        return {
            "ticker":     self.ticker,
            "entry_date": str(self.entry_date)[:10],
            "exit_date":  str(self.exit_date)[:10],
            "entry_px":   round(self.entry_px, 2),
            "exit_px":    round(self.exit_px, 2),
            "shares":     round(self.shares, 4),
            "pnl_rub":    round(self.pnl_rub, 2),
            "pnl_pct":    round(self.pnl_pct, 2),
            "reason":     self.reason,
            "hold_days":  round(self.hold_days, 1),
        }


def run_backtest(
    data: dict[str, pd.DataFrame],
    cfg: ATRBOConfig,
    initial_capital: float = 100_000.0,
) -> dict[str, Any]:
    """
    Портфельный бэктест стратегии ATR_BO.

    Args:
        data: {ticker: DataFrame с OHLCV данными}
        cfg:  конфигурация стратегии
        initial_capital: начальный капитал в рублях

    Returns:
        dict с полными результатами бэктеста
    """
    COMM_BUY  = getattr(cfg, 'commission_buy',  cfg.commission)
    COMM_SELL = getattr(cfg, 'commission_sell', cfg.commission)
    SLIP      = cfg.slippage

    # Генерируем сигналы для всех тикеров
    all_sigs: dict[str, pd.DataFrame] = {}
    for t, df in data.items():
        if len(df) >= 210:
            all_sigs[t] = generate_signals(df, cfg)

    if not all_sigs:
        return {"error": "Недостаточно данных"}

    # Общий список торговых дней
    all_dates = sorted(set().union(*[set(df.index) for df in data.values()]))

    # Корреляционные группы (запрет одновременного входа)
    CORR = [
        {"SBER", "SBERP", "T", "VTBR"},
        {"LKOH", "ROSN", "NVTK", "GAZP", "SNGS", "SNGSP"},
        {"NLMK", "MTLR", "CHMF", "MAGN"},
        {"TATN", "TATNP"},
    ]

    def corr_blocked(ticker, open_tickers):
        return any(ticker in g and g & open_tickers for g in CORR)

    TICK_IDX = {t: {d: i for i, d in enumerate(df.index)} for t, df in data.items()}

    free_cash  = initial_capital
    positions: dict[str, BacktestPosition] = {}
    trades:    list[BacktestTrade] = []
    equity     = [initial_capital]
    equity_dates = [all_dates[0]]
    peak_eq    = initial_capital
    max_dd     = 0.0

    for date in all_dates:
        # ── Закрытие позиций ──────────────────────────────────────────────────
        to_close = []
        for ticker, pos in positions.items():
            if ticker not in data or date not in TICK_IDX[ticker]:
                continue
            df  = data[ticker]
            idx = TICK_IDX[ticker][date]
            if idx <= TICK_IDX[ticker].get(pos.entry_date, -1):
                continue

            hi   = float(df["high"].iloc[idx])
            lo   = float(df["low"].iloc[idx])
            op   = float(df["open"].iloc[idx])
            cls_ = float(df["close"].iloc[idx])
            at14 = float(all_sigs[ticker]["at14"].iloc[idx]) if ticker in all_sigs else 0.0
            hold = (date - pos.entry_date).days

            # Trailing stop update
            if at14 > 0:
                trail = cls_ - cfg.trail_mult * at14
                if trail > pos.trail_sl:
                    pos.trail_sl = trail

            reason = exit_px = None
            if cfg.sl_pct > 0 and lo <= pos.trail_sl:
                reason  = "SL"
                exit_px = max(pos.trail_sl * (1 - SLIP), lo)
            elif cfg.sl_pct == 0 and lo <= pos.trail_sl:
                reason  = "SL"
                exit_px = max(pos.trail_sl * (1 - SLIP), lo)
            elif hold >= cfg.max_hold_days:
                reason  = "TIME"
                exit_px = op * (1 - SLIP)

            if reason:
                cash = exit_px * pos.shares * (1 - COMM_SELL)  # продажа: комиссия 0.30%
                cost = pos.entry_px * pos.shares * (1 + COMM_BUY)  # покупка: комиссия 0%
                pnl  = cash - cost
                pnl_pct = pnl / cost * 100
                free_cash += cash
                trades.append(BacktestTrade(
                    ticker=ticker, entry_date=pos.entry_date, exit_date=date,
                    entry_px=pos.entry_px, exit_px=exit_px, shares=pos.shares,
                    pnl_rub=pnl, pnl_pct=pnl_pct, reason=reason,
                    hold_days=(date - pos.entry_date).total_seconds() / 86400,
                ))
                to_close.append(ticker)
        for t in to_close:
            positions.pop(t, None)

        # ── Новые входы ────────────────────────────────────────────────────────
        for ticker, sig_df in all_sigs.items():
            if len(positions) >= cfg.max_positions:
                break
            if ticker in positions:
                continue
            if date not in TICK_IDX.get(ticker, {}):
                continue

            idx = TICK_IDX[ticker][date]
            if idx < 1:
                continue
            if not bool(sig_df["signal"].iloc[idx - 1]):
                continue
            if corr_blocked(ticker, set(positions.keys())):
                continue

            df     = data[ticker]
            entry  = float(df["open"].iloc[idx]) * (1 + SLIP)
            at14_i = float(sig_df["at14"].iloc[idx - 1])

            # Размер позиции
            pos_val = sum(
                float(data[t]["close"].iloc[TICK_IDX[t].get(date, -1)]) * p.shares
                for t, p in positions.items()
                if TICK_IDX[t].get(date) is not None
            )
            if cfg.reinvest:
                total_cap = free_cash + pos_val
            else:
                total_cap = initial_capital

            alloc  = min(total_cap * cfg.risk_pct, free_cash * 0.98)
            shares = alloc / entry
            cost   = shares * entry * (1 + COMM_BUY)   # покупка: 0%
            if cost > free_cash or shares <= 0:
                continue

            trail_init = entry - cfg.trail_mult * at14_i
            free_cash -= cost
            positions[ticker] = BacktestPosition(
                ticker=ticker, entry_date=date, entry_px=entry,
                shares=shares, trail_sl=trail_init,
            )

        # Equity
        pos_val = sum(
            float(data[t]["close"].iloc[TICK_IDX[t].get(date, -1)]) * p.shares
            for t, p in positions.items()
            if TICK_IDX[t].get(date) is not None
        )
        eq = free_cash + pos_val
        equity.append(eq)
        equity_dates.append(date)
        if eq > peak_eq:
            peak_eq = eq
        dd = (peak_eq - eq) / peak_eq * 100
        if dd > max_dd:
            max_dd = dd

    # Принудительное закрытие
    last_date = all_dates[-1]
    for ticker, pos in list(positions.items()):
        if ticker not in data:
            continue
        cls_ = float(data[ticker]["close"].iloc[-1])
        ep   = cls_ * (1 - SLIP)
        cash = ep * pos.shares * (1 - COMM_SELL)   # продажа: 0.30%
        cost = pos.entry_px * pos.shares * (1 + COMM_BUY)   # покупка: 0%
        free_cash += cash
        trades.append(BacktestTrade(
            ticker=ticker, entry_date=pos.entry_date, exit_date=last_date,
            entry_px=pos.entry_px, exit_px=ep, shares=pos.shares,
            pnl_rub=cash - cost, pnl_pct=(cash - cost) / cost * 100,
            reason="FORCED",
            hold_days=(last_date - pos.entry_date).total_seconds() / 86400,
        ))

    # ── Статистика ─────────────────────────────────────────────────────────────
    final     = free_cash
    total_pnl = (final - initial_capital) / initial_capital * 100
    n_days    = (all_dates[-1] - all_dates[0]).days
    ann_ret   = ((final / initial_capital) ** (365 / max(n_days, 1)) - 1) * 100

    pnls    = np.array([t.pnl_pct for t in trades])
    n_tr    = len(trades)
    n_win   = (pnls > 0).sum()
    wr      = n_win / n_tr * 100 if n_tr else 0
    wins    = pnls[pnls > 0]
    losses  = pnls[pnls <= 0]
    pf      = wins.sum() / (-losses.sum() + 1e-9) if len(losses) else 99.0

    eq_arr  = np.array(equity)
    dr      = np.diff(eq_arr) / (eq_arr[:-1] + 1e-9)
    sharpe  = (dr.mean() / (dr.std() + 1e-9)) * np.sqrt(252)

    by_year: dict[str, dict] = {}
    running = initial_capital
    for t in sorted(trades, key=lambda x: x.exit_date):
        yr = str(t.exit_date)[:4]
        running += t.pnl_rub
        if yr not in by_year:
            by_year[yr] = {"start": running - t.pnl_rub, "end": running}
        else:
            by_year[yr]["end"] = running

    by_ticker: dict[str, dict] = {}
    for t in trades:
        d = by_ticker.setdefault(t.ticker, {"n": 0, "wins": 0, "pnl": 0.0})
        d["n"] += 1; d["pnl"] += t.pnl_pct
        if t.pnl_pct > 0: d["wins"] += 1

    by_reason: dict[str, dict] = {}
    for t in trades:
        d = by_reason.setdefault(t.reason, {"n": 0, "wins": 0, "pnl": 0.0})
        d["n"] += 1; d["pnl"] += t.pnl_pct
        if t.pnl_pct > 0: d["wins"] += 1

    return {
        "strategy_id":   STRATEGY_ID,
        "strategy_name": STRATEGY_NAME,
        "initial_capital": initial_capital,
        "final_capital": round(final, 2),
        "total_pnl_pct": round(total_pnl, 2),
        "annual_return": round(ann_ret, 2),
        "max_drawdown":  round(-max_dd, 2),
        "sharpe":        round(sharpe, 3),
        "n_trades":      n_tr,
        "win_rate":      round(wr, 1),
        "profit_factor": round(min(pf, 99), 2),
        "avg_win_pct":   round(float(wins.mean()), 2) if len(wins) else 0,
        "avg_loss_pct":  round(float(losses.mean()), 2) if len(losses) else 0,
        "n_days":        n_days,
        "config":        cfg.to_dict(),
        "equity_curve":  [round(e, 2) for e in equity],
        "equity_dates":  [str(d)[:10] for d in equity_dates],
        "trades":        [t.to_dict() for t in trades],
        "by_year":       {
            yr: {
                "start": round(v["start"], 2),
                "end":   round(v["end"], 2),
                "pnl":   round(v["end"] - v["start"], 2),
                "pnl_pct": round((v["end"] - v["start"]) / v["start"] * 100, 1),
            }
            for yr, v in by_year.items()
        },
        "by_ticker": {
            t: {
                "n": v["n"],
                "wr": round(v["wins"] / v["n"] * 100, 1) if v["n"] else 0,
                "total_pnl": round(v["pnl"], 2),
                "avg_pnl":   round(v["pnl"] / v["n"], 2) if v["n"] else 0,
            }
            for t, v in sorted(by_ticker.items(), key=lambda x: -x[1]["pnl"])
        },
        "by_reason": {
            r: {
                "n": v["n"],
                "wr": round(v["wins"] / v["n"] * 100, 1) if v["n"] else 0,
                "avg_pnl": round(v["pnl"] / v["n"], 2) if v["n"] else 0,
            }
            for r, v in by_reason.items()
        },
    }
