"""API запуска бэктеста на демо-данных."""
from __future__ import annotations

from fastapi import APIRouter

from backtesting.engine import BacktestEngine, demo_ohlc_frame
from reporting.reporter import compute_metrics

router = APIRouter(prefix="/api/backtest", tags=["backtest"])


@router.post("/demo")
async def run_demo(symbol: str = "DEMO", capital: float = 100_000.0, lot_size: int = 10) -> dict:
    df = demo_ohlc_frame(500)
    eng = BacktestEngine(lot_size=lot_size)
    res = eng.run(df, symbol=symbol, initial_capital=capital)
    rep = compute_metrics(res.trades, capital, res.equity_curve)
    return {
        "symbol": symbol,
        "metrics": rep.to_dict(),
        "n_trades": len(res.trades),
        "n_signals_logged": len(res.signals_log),
        "final_capital": res.final_capital,
        "circuit_breaker_open": eng.risk.circuit.is_open,
    }
