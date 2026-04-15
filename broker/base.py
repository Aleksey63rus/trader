"""Абстрактный интерфейс брокера (реализация в bcs_client)."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Awaitable

import pandas as pd


class AbstractBrokerClient(ABC):
    """Единый контракт для REST/WebSocket брокера."""

    @abstractmethod
    async def get_candles(
        self,
        symbol: str,
        interval: str,
        start: str,
        end: str,
    ) -> pd.DataFrame:
        """Свечи: колонки datetime, open, high, low, close, volume."""

    @abstractmethod
    async def get_portfolio(self) -> dict[str, Any]:
        """cash, positions[{symbol, qty, avg_price}]."""

    @abstractmethod
    async def get_instrument_info(self, symbol: str) -> dict[str, Any]:
        """lot_size, min_price_step, name, ..."""

    @abstractmethod
    async def place_order(
        self,
        symbol: str,
        side: str,
        quantity: int,
        order_type: str,
        price: float | None = None,
    ) -> str:
        """Возвращает order_id."""

    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """Отмена заявки."""

    @abstractmethod
    async def subscribe_candles(
        self,
        symbol: str,
        interval: str,
        callback: Callable[[dict[str, Any]], Awaitable[None] | None],
    ) -> None:
        """WebSocket: новая свеча → callback."""

    @abstractmethod
    async def subscribe_ticks(
        self,
        symbol: str,
        callback: Callable[[dict[str, Any]], Awaitable[None] | None],
    ) -> None:
        """WebSocket: тик → callback."""
