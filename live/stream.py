from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from binance import AsyncClient, BinanceSocketManager
from binance.exceptions import BinanceAPIException


async def _make_client(api_key: str, api_secret: str) -> AsyncClient:
    return await AsyncClient.create(api_key, api_secret)


@dataclass
class StreamConfig:
    symbol: str = "BTCUSDT"
    interval: str = "1m"
    resync_interval: int = 300


class KlineStream:
    def __init__(self, cfg: StreamConfig, client: AsyncClient) -> None:
        self.cfg = cfg
        self.client = client
        self.queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=100)
        self.bsm = BinanceSocketManager(client)
        self.ws = None

    async def start(self) -> None:
        while True:
            self.ws = self.bsm.kline_socket(
                self.cfg.symbol.lower(), self.cfg.interval
            )
            start = asyncio.get_running_loop().time()
            try:
                async with self.ws as stream:
                    async for msg in stream:
                        await self.queue.put(msg)
                        if (
                            self.cfg.resync_interval > 0
                            and asyncio.get_running_loop().time() - start
                            > self.cfg.resync_interval
                        ):
                            break
            except BinanceAPIException:
                try:
                    rest = await self.client.get_klines(
                        symbol=self.cfg.symbol,
                        interval=self.cfg.interval,
                        limit=1,
                    )
                    await self.queue.put({"k": rest[0]})
                except Exception:
                    pass
            await asyncio.sleep(1)

    async def get(self) -> dict[str, Any]:
        return await self.queue.get()
