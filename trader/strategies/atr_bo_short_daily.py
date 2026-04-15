"""
ATR Breakout — SHORT (зеркальные медвежьи сигналы).
Учёт: открытие шорта = продажа (commission_sell), закрытие = покупка (commission_buy).
Equity = free_cash + Σ (entry − mark) * shares по открытым шортам.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.indicators import ema, atr, rsi, adx, volume_ratio
from strategies.atr_bo_daily import ATRBOConfig, BacktestTrade


def generate_signals_short(df: pd.DataFrame, cfg: ATRBOConfig) -> pd.DataFrame:
    c = df["close"]
    at14 = atr(df, cfg.atr_period)
    at5 = atr(df, 5)
    e200 = ema(c, cfg.ema_period)
    r = rsi(c, cfg.rsi_period)
    a = adx(df, cfg.adx_period)
    vr = volume_ratio(df, cfg.vol_ratio_period)
    down = (c.shift(1) - c).clip(lower=0)
    sig = (
        (c < e200)
        & (down >= cfg.atr_bo_mult * at14)
        & (at5 > at14 * 0.95)
        & (r >= (100 - cfg.rsi_max))
        & (r <= (100 - cfg.rsi_min))
        & (a >= cfg.adx_min)
        & (vr >= cfg.vol_ratio_min)
    )
    sig = sig & ~sig.shift(1).fillna(False)
    return pd.DataFrame({"signal": sig, "at14": at14}, index=df.index)


@dataclass
class SPos:
    ticker: str
    entry_date: pd.Timestamp
    entry_px: float
    shares: float
    trail_sl: float


def run_short_backtest(
    data: dict[str, pd.DataFrame],
    cfg: ATRBOConfig,
    initial_capital: float = 100_000.0,
) -> dict[str, Any]:
    COMM_BUY = getattr(cfg, "commission_buy", cfg.commission)
    COMM_SELL = getattr(cfg, "commission_sell", cfg.commission)
    SLIP = cfg.slippage

    all_sigs: dict[str, pd.DataFrame] = {}
    for t, df in data.items():
        if len(df) >= 210:
            all_sigs[t] = generate_signals_short(df, cfg)

    if not all_sigs:
        return {"error": "Нет данных"}

    all_dates = sorted(set().union(*[set(df.index) for df in data.values()]))
    CORR = [
        {"SBER", "SBERP", "T", "VTBR"},
        {"LKOH", "ROSN", "NVTK", "GAZP", "SNGS", "SNGSP"},
        {"NLMK", "MTLR", "CHMF", "MAGN"},
        {"TATN", "TATNP"},
    ]

    def corr_blocked(ticker, open_tickers):
        return any(ticker in g and g & open_tickers for g in CORR)

    TICK_IDX = {t: {d: i for i, d in enumerate(df.index)} for t, df in data.items()}

    free_cash = initial_capital
    positions: dict[str, SPos] = {}
    trades: list[BacktestTrade] = []
    equity = [initial_capital]
    peak_eq = initial_capital
    max_dd = 0.0

    def mark_short_pnl(d):
        s = 0.0
        for tk, p in positions.items():
            idx = TICK_IDX[tk].get(d)
            if idx is None:
                continue
            cls_ = float(data[tk]["close"].iloc[idx])
            s += (p.entry_px - cls_) * p.shares
        return s

    for date in all_dates:
        to_close = []
        for ticker, pos in positions.items():
            if ticker not in data or date not in TICK_IDX[ticker]:
                continue
            df = data[ticker]
            idx = TICK_IDX[ticker][date]
            if idx <= TICK_IDX[ticker].get(pos.entry_date, -1):
                continue
            hi = float(df["high"].iloc[idx])
            op = float(df["open"].iloc[idx])
            cls_ = float(df["close"].iloc[idx])
            at14 = float(all_sigs[ticker]["at14"].iloc[idx]) if ticker in all_sigs else 0.0
            hold = (date - pos.entry_date).days

            if at14 > 0:
                trail = cls_ + cfg.trail_mult * at14
                if trail < pos.trail_sl:
                    pos.trail_sl = trail

            reason = exit_px = None
            if hi >= pos.trail_sl:
                reason = "SL"
                exit_px = min(pos.trail_sl * (1 + SLIP), hi)
            elif hold >= cfg.max_hold_days:
                reason = "TIME"
                exit_px = op * (1 + SLIP)

            if reason:
                cover = exit_px * pos.shares * (1 + COMM_BUY)
                proceeds = pos.entry_px * pos.shares * (1 - SLIP) * (1 - COMM_SELL)
                margin = pos.entry_px * pos.shares
                # при открытии: cash += proceeds − margin; при закрытии: cash += margin − cover
                free_cash = free_cash - cover + margin
                pnl = proceeds - cover
                cost_basis = pos.entry_px * pos.shares * (1 + COMM_BUY)
                pnl_pct = pnl / max(cost_basis, 1e-9) * 100
                trades.append(
                    BacktestTrade(
                        ticker, pos.entry_date, date, pos.entry_px, exit_px,
                        pos.shares, pnl, pnl_pct, reason,
                        (date - pos.entry_date).total_seconds() / 86400,
                    )
                )
                to_close.append(ticker)
        for t in to_close:
            positions.pop(t, None)

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

            df = data[ticker]
            entry_raw = float(df["open"].iloc[idx])
            entry = entry_raw * (1 - SLIP)
            at14_i = float(sig_df["at14"].iloc[idx - 1])

            mtm_short = sum(
                (p.entry_px - float(data[tk]["close"].iloc[TICK_IDX[tk].get(date, -1)])) * p.shares
                for tk, p in positions.items()
                if TICK_IDX[tk].get(date) is not None
            )
            total_cap = free_cash + mtm_short if cfg.reinvest else initial_capital
            alloc = min(total_cap * cfg.risk_pct, free_cash * 0.98)
            shares = alloc / max(entry, 1e-9)
            margin = entry_raw * shares
            proceeds = entry_raw * shares * (1 - SLIP) * (1 - COMM_SELL)
            if margin > free_cash + 1e-6 or shares <= 0:
                continue
            free_cash = free_cash - margin + proceeds
            trail_init = entry_raw + cfg.trail_mult * at14_i
            positions[ticker] = SPos(ticker, date, entry_raw, shares, trail_init)

        pos_val = mark_short_pnl(date)
        eq = free_cash + pos_val
        equity.append(eq)
        if eq > peak_eq:
            peak_eq = eq
        dd = (peak_eq - eq) / peak_eq * 100 if peak_eq else 0
        if dd > max_dd:
            max_dd = dd

    last_date = all_dates[-1]
    for ticker, pos in list(positions.items()):
        if ticker not in data:
            continue
        cls_ = float(data[ticker]["close"].iloc[-1])
        ep = cls_ * (1 + SLIP)
        cover = ep * pos.shares * (1 + COMM_BUY)
        proceeds = pos.entry_px * pos.shares * (1 - SLIP) * (1 - COMM_SELL)
        margin = pos.entry_px * pos.shares
        free_cash = free_cash - cover + margin
        pnl = proceeds - cover
        trades.append(
            BacktestTrade(
                ticker, pos.entry_date, last_date, pos.entry_px, ep, pos.shares,
                pnl, pnl / max(pos.entry_px * pos.shares, 1e-9) * 100, "FORCED",
                (last_date - pos.entry_date).total_seconds() / 86400,
            )
        )

    final = free_cash
    n_days = (all_dates[-1] - all_dates[0]).days
    ann = ((final / initial_capital) ** (365 / max(n_days, 1)) - 1) * 100
    total_pnl = (final - initial_capital) / initial_capital * 100
    pnls = np.array([t.pnl_pct for t in trades]) if trades else np.array([])
    n_tr = len(trades)
    wr = float((pnls > 0).sum() / n_tr * 100) if n_tr else 0.0
    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]
    pf = float(wins.sum() / (-losses.sum() + 1e-9)) if len(losses) else 99.0
    eq_arr = np.array(equity)
    dr = np.diff(eq_arr) / (eq_arr[:-1] + 1e-9)
    sharpe = float((dr.mean() / (dr.std() + 1e-9)) * np.sqrt(252))

    return {
        "strategy_id": "atr_bo_short_daily",
        "final_capital": round(final, 2),
        "annual_return": round(ann, 2),
        "total_pnl_pct": round(total_pnl, 2),
        "max_drawdown": round(-max_dd, 2),
        "sharpe": round(sharpe, 3),
        "n_trades": n_tr,
        "win_rate": round(wr, 1),
        "profit_factor": round(min(pf, 99), 2),
        "n_days": n_days,
        "trades": [t.to_dict() for t in trades],
    }
