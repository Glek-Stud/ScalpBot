from __future__ import annotations

import argparse
import asyncio
import json
import signal
from pathlib import Path

import numpy as np
import yaml

from .stream import _make_client, StreamConfig, KlineStream
from .features import FeatureExtractor, FeatureStats
from .broker import Broker, BrokerConfig, load_keys
from .agent import DQNAgent
from .env_live import BTCRealTradingEnv
from .stats import StatsTracker


async def main(cfg_path: str, dry: bool) -> None:
    cfg = yaml.safe_load(Path(cfg_path).read_text())
    key, sec = load_keys()
    client = await _make_client(key, sec)
    stream = KlineStream(
        StreamConfig(
            cfg["symbol"],
            cfg.get("interval", "1m"),
            cfg.get("resync_interval", 300),
        ),
        client,
    )
    broker = Broker(BrokerConfig(
        cfg["symbol"],
        cfg.get("leverage", 1),
        dry_run=dry or cfg.get("dry_run", True),
        starting_equity=cfg.get("starting_equity", 1000.0),
        usd_per_trade=cfg.get("usd_per_trade", 10.0),
    ))
    stats_file = Path("collect/data_final/norm_stats.json")
    with stats_file.open() as f:
        ns = json.load(f)
    stats = FeatureStats(mean=np.array(ns["mean"]), std=np.array(ns["std"]), lowvol_p5=float(ns["p5_volume"]))
    extractor = FeatureExtractor(stats)
    env = BTCRealTradingEnv(extractor, broker)
    agent = DQNAgent(cfg["state_path"])
    logger = StatsTracker(Path(cfg.get("log_dir", "live_logs")),
                          start_equity=cfg.get("starting_equity", 1000.0))

    async def producer():
        await stream.start()

    async def consumer():
        obs, _ = env.reset()
        while True:
            msg = await stream.get()
            broker.update_kline(msg)
            action = agent.act(obs)
            obs, reward, term, _, _ = env.step(action)
            logger.update(reward, action != 0)
            if logger.step % cfg.get("log_interval", 100) == 0:
                logger.flush()
            if term:
                obs, _ = env.reset()

    prod_task = asyncio.create_task(producer())
    cons_task = asyncio.create_task(consumer())

    def _stop() -> None:
        prod_task.cancel()
        cons_task.cancel()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _stop)

    try:
        await asyncio.gather(prod_task, cons_task)
    except asyncio.CancelledError:
        pass
    finally:
        logger.flush()
        try:
            broker.close_position()
        except Exception:
            pass


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="configs/live_cfg.yaml")
    ap.add_argument("--dry", action="store_true")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(args.cfg, args.dry))
