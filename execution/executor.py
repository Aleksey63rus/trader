"""
Исполнение заявок. Режим paper: без вызова брокера.
Реальная торговля — этап интеграции с broker/bcs_client.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from signals.generator import Signal

if TYPE_CHECKING:
    from broker.base import AbstractBrokerClient

log = logging.getLogger("investor.execution")


@dataclass
class PaperPosition:
    symbol: str
    side: str
    quantity_lots: int
    lot_size: int
    entry_price: float
    stop_loss: float
    take_profit: float


@dataclass
class PaperExecutor:
    """Виртуальное исполнение для тестов и демо."""

    positions: list[PaperPosition] = field(default_factory=list)

    async def execute_signal(
        self,
        signal: Signal,
        quantity_lots: int,
        lot_size: int,
        broker: AbstractBrokerClient | None = None,
    ) -> PaperPosition | None:
        if quantity_lots <= 0:
            return None
        if broker is not None:
            log.warning("Режим paper: брокер не вызывается")
        pos = PaperPosition(
            symbol=signal.symbol,
            side=signal.direction,
            quantity_lots=quantity_lots,
            lot_size=lot_size,
            entry_price=signal.entry_price,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
        )
        self.positions.append(pos)
        log.info("Paper: открыта %s %s лот=%s", pos.side, pos.symbol, quantity_lots)
        return pos

    def close_all_paper(self) -> None:
        self.positions.clear()
