"""Метрики бэктеста и отчёты."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

import constants as C


@dataclass
class PerformanceReport:
    total_return_pct: float
    max_drawdown_pct: float
    sharpe_ratio: float | None
    win_rate_pct: float
    profit_factor: float | None
    total_trades: int
    avg_win: float
    avg_loss: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_return_pct": self.total_return_pct,
            "max_drawdown_pct": self.max_drawdown_pct,
            "sharpe_ratio": self.sharpe_ratio,
            "win_rate_pct": self.win_rate_pct,
            "profit_factor": self.profit_factor,
            "total_trades": self.total_trades,
            "avg_win": self.avg_win,
            "avg_loss": self.avg_loss,
        }


def _max_drawdown(equity: np.ndarray) -> float:
    if equity.size == 0:
        return 0.0
    peak = np.maximum.accumulate(equity)
    dd = (peak - equity) / np.where(peak > 0, peak, 1.0)
    return float(np.max(dd) * 100.0)


def compute_metrics(
    trades: Iterable[Any],
    initial_capital: float,
    equity_curve: list[tuple[int, float]] | None = None,
) -> PerformanceReport:
    """trades: объекты с атрибутом pnl (ClosedTrade)."""
    tlist = list(trades)
    if not tlist:
        return PerformanceReport(0.0, 0.0, None, 0.0, None, 0, 0.0, 0.0)

    pnls = np.array([float(t.pnl) for t in tlist], dtype=float)
    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]
    win_rate = float(len(wins) / len(pnls) * 100.0) if len(pnls) else 0.0
    gross_profit = float(wins.sum()) if wins.size else 0.0
    gross_loss = float(-losses.sum()) if losses.size else 0.0
    pf = (gross_profit / gross_loss) if gross_loss > 1e-9 else None

    final = initial_capital + float(pnls.sum())
    total_ret = ((final / initial_capital) - 1.0) * 100.0 if initial_capital > 0 else 0.0

    if equity_curve and len(equity_curve) > 1:
        eq = np.array([e[1] for e in equity_curve], dtype=float)
        mdd = _max_drawdown(eq)
        rets = np.diff(eq) / np.where(eq[:-1] > 0, eq[:-1], 1.0)
        rets = rets[np.isfinite(rets)]
        if rets.size > 1 and float(np.std(rets)) > 1e-12:
            rf_per_bar = (1.0 + C.RISK_FREE_RATE_ANNUAL) ** (1.0 / C.TRADING_DAYS_PER_YEAR) - 1.0
            excess = rets.mean() - rf_per_bar
            sharpe = float(excess / np.std(rets) * np.sqrt(min(rets.size, C.TRADING_DAYS_PER_YEAR)))
        else:
            sharpe = None
    else:
        mdd = 0.0
        sharpe = None

    return PerformanceReport(
        total_return_pct=total_ret,
        max_drawdown_pct=mdd,
        sharpe_ratio=sharpe,
        win_rate_pct=win_rate,
        profit_factor=pf,
        total_trades=len(tlist),
        avg_win=float(wins.mean()) if wins.size else 0.0,
        avg_loss=float(losses.mean()) if losses.size else 0.0,
    )
