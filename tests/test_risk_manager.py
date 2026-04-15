"""Тесты риск-менеджера."""
from __future__ import annotations

from risk.manager import RiskManager


def test_position_size_basic() -> None:
    rm = RiskManager()
    lots = rm.calculate_position_lots(
        capital=100_000.0,
        entry_price=300.0,
        stop_loss=285.0,
        lot_size=10,
        min_price_step=0.01,
        open_positions=0,
    )
    assert lots >= 1
    assert lots <= 66  # грубая верхняя граница при 2% риска


def test_circuit_breaker() -> None:
    rm = RiskManager()
    rm.circuit.on_trade_closed(-100)
    rm.circuit.on_trade_closed(-100)
    rm.circuit.on_trade_closed(-100)
    assert rm.circuit.is_open
    assert rm.calculate_position_lots(100_000, 100, 90, 10, 0.01, 0) == 0
    rm.circuit.reset()
    assert not rm.circuit.is_open
