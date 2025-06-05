from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import numpy as np
import yaml

from .stream import _make_client, StreamConfig, KlineStream
from .features import FeatureExtractor, FeatureStats
from .broker import Broker, BrokerConfig
from .agent import DQNAgent
from .env_live import BTCRealTradingEnv


async def main(cfg_path: str, dry: bool) -> None:
    cfg = yaml.safe_load(Path(cfg_path).read_text())
    client = await _make_client(None, None)
    stream = KlineStream(StreamConfig(cfg["symbol"], cfg.get("interval", "1m")), client)
    broker = Broker(BrokerConfig(
        cfg["symbol"],
        cfg.get("leverage", 1),
        dry_run=dry or cfg.get("dry_run", True),
        starting_equity=cfg.get("starting_equity", 1000.0),
    ))
    stats_file = Path("collect/data_final/norm_stats.json")
    with stats_file.open() as f:
        ns = json.load(f)
    stats = FeatureStats(mean=np.array(ns["mean"]), std=np.array(ns["std"]), lowvol_p5=float(ns["p5_volume"]))
    extractor = FeatureExtractor(stats)
    env = BTCRealTradingEnv(extractor, broker)
    agent = DQNAgent(cfg["state_path"])

    async def producer():
        await stream.start()

    async def consumer():
        obs, _ = env.reset()
        while True:
            msg = await stream.get()
            broker.update_kline(msg)
            obs, _, term, _, _ = env.step(agent.act(obs))
            if term:
                obs, _ = env.reset()

    await asyncio.gather(producer(), consumer())


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="configs/live_cfg.yaml")
    ap.add_argument("--dry", action="store_true")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(args.cfg, args.dry))
