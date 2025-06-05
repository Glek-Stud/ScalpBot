from __future__ import annotations

from pathlib import Path
import argparse
import yaml

from .broker import Broker, BrokerConfig


def main(cfg_path: str) -> None:
    cfg = yaml.safe_load(Path(cfg_path).read_text())
    broker = Broker(
        BrokerConfig(
            cfg["symbol"],
            cfg.get("leverage", 1),
            dry_run=cfg.get("dry_run", True),
            starting_equity=cfg.get("starting_equity", 1000.0),
        )
    )
    while True:
        act = input("Enter action (1=open long, 2=close, q=quit): ").strip()
        if act == "1":
            resp = broker.open_long()
            print("Opened:", resp)
        elif act == "2":
            resp = broker.close_position()
            print("Closed:", resp)
        elif act.lower() == "q":
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default="configs/live_cfg.yaml")
    args = parser.parse_args()
    main(args.cfg)
