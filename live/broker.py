"""Thin wrapper around Binance REST API."""

from __future__ import annotations

from dataclasses import dataclass

from binance.client import Client


def load_keys(path="~/.binance_keys.json"):
    import json
    from pathlib import Path
    p = Path(path).expanduser()
    with p.open() as f:
        data = json.load(f)
    return data["key"], data["secret"]


@dataclass
class BrokerConfig:
    symbol: str
    leverage: int
    dry_run: bool = True


class Broker:
    def __init__(self, cfg: BrokerConfig):
        key, sec = load_keys()
        self.client = Client(key, sec)
        self.cfg = cfg
        self.position = 0
        self.last_kline = None

    def update_kline(self, kline: dict):
        self.last_kline = {
            "close": float(kline["k"]["c"]),
            "open": float(kline["k"]["o"]),
            "high": float(kline["k"]["h"]),
            "low": float(kline["k"]["l"]),
            "volume": float(kline["k"]["v"]),
        }

    def execute(self, desired_pos: int):
        if self.cfg.dry_run or desired_pos == self.position:
            self.position = desired_pos
            return 0.0, self.position
        # For brevity we do not implement actual order placement here
        self.position = desired_pos
        return 0.0, self.position
