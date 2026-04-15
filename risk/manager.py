"""Риск-менеджмент: размер позиции и circuit breaker."""
from __future__ import annotations

from dataclasses import dataclass, field

import constants as C


@dataclass
class CircuitBreaker:
    """Останавливает торговлю после серии убыточных сделок."""

    threshold: int = C.CIRCUIT_BREAKER_LOSSES
    consecutive_losses: int = 0
    is_open: bool = False

    def on_trade_closed(self, pnl: float) -> None:
        if pnl < 0:
            self.consecutive_losses += 1
            if self.consecutive_losses >= self.threshold:
                self.is_open = True
        else:
            self.consecutive_losses = 0

    def reset(self) -> None:
        self.consecutive_losses = 0
        self.is_open = False


@dataclass
class RiskManager:
    """Расчёт лотов с учётом риска и лимитов."""

    risk_percent: float = C.RISK_PERCENT
    max_positions: int = C.MAX_POSITIONS
    max_capital_per_trade: float = C.MAX_CAPITAL_PER_TRADE
    slippage_ticks: int = C.SLIPPAGE_TICKS
    circuit: CircuitBreaker = field(default_factory=CircuitBreaker)

    def calculate_position_lots(
        self,
        capital: float,
        entry_price: float,
        stop_loss: float,
        lot_size: int,
        min_price_step: float,
        open_positions: int = 0,
    ) -> int:
        """
        Количество лотов (не акций). Округление вниз.
        """
        if self.circuit.is_open:
            return 0
        if open_positions >= self.max_positions:
            return 0
        if capital <= 0 or lot_size <= 0:
            return 0

        risk_amount = capital * self.risk_percent
        price_risk_per_share = abs(entry_price - stop_loss)
        if price_risk_per_share <= 0:
            return 0

        slip = self.slippage_ticks * min_price_step
        total_risk_per_share = price_risk_per_share + slip
        risk_per_lot = total_risk_per_share * lot_size
        if risk_per_lot <= 0:
            return 0

        lots = int(risk_amount // risk_per_lot)
        max_by_capital = int((capital * self.max_capital_per_trade) // (entry_price * lot_size))
        lots = min(lots, max_by_capital)
        return max(0, lots)
