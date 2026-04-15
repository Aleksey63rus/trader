"""
Реализация БКС Trade API (BCSPy или прямой REST — этап 3).
Пока заглушка: методы не вызываются до настройки .env.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Awaitable

import pandas as pd

from broker.base import AbstractBrokerClient

log = logging.getLogger("investor.broker")


class BcsBrokerClient(AbstractBrokerClient):
    """Клиент БКС; подключение реализуется на этапе 3."""

    def __init__(self, client_id: str | None, client_secret: str | None) -> None:
        self._client_id = client_id
        self._client_secret = client_secret

    async def get_candles(
        self, symbol: str, interval: str, start: str, end: str
    ) -> pd.DataFrame:
        raise NotImplementedError("Этап 3: подключить BCSPy / REST БКС")

    async def get_portfolio(self) -> dict[str, Any]:
        raise NotImplementedError("Этап 3: подключить BCSPy / REST БКС")

    async def get_instrument_info(self, symbol: str) -> dict[str, Any]:
        raise NotImplementedError("Этап 3: подключить BCSPy / REST БКС")

    async def place_order(
        self,
        symbol: str,
        side: str,
        quantity: int,
        order_type: str,
        price: float | None = None,
    ) -> str:
        raise NotImplementedError("Этап 3: подключить BCSPy / REST БКС")

    async def cancel_order(self, order_id: str) -> bool:
        raise NotImplementedError("Этап 3: подключить BCSPy / REST БКС")

    async def subscribe_candles(
        self,
        symbol: str,
        interval: str,
        callback: Callable[[dict[str, Any]], Awaitable[None] | None],
    ) -> None:
        raise NotImplementedError("Этап 3: WebSocket БКС")

    async def subscribe_ticks(
        self,
        symbol: str,
        callback: Callable[[dict[str, Any]], Awaitable[None] | None],
    ) -> None:
        raise NotImplementedError("Этап 3: WebSocket БКС")
