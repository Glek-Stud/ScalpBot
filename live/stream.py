"""WebSocket streaming of Binance 1m klines."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from binance import AsyncClient, BinanceSocketManager


def _make_client(api_key: str, api_secret: str) -> AsyncClient:
    return AsyncClient(api_key, api_secret)


@dataclass
class StreamConfig:
    symbol: str = "BTCUSDT"
    interval: str = "1m"


class KlineStream:
    def __init__(self, cfg: StreamConfig, client: AsyncClient) -> None:
        self.cfg = cfg
        self.client = client
        self.queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=100)
        self.bsm = BinanceSocketManager(client)
        self.ws = None

    async def start(self) -> None:
        self.ws = self.bsm.kline_socket(self.cfg.symbol.lower(), self.cfg.interval)
        async with self.ws as stream:
            async for msg in stream:
                await self.queue.put(msg)

    async def get(self) -> dict[str, Any]:
        return await self.queue.get()
