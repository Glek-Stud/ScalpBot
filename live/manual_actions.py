"""Manual test CLI to open/close a long position using Broker."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - optional dependency for tests
    yaml = None

from .broker import Broker, BrokerConfig


def main(cfg_path: str) -> None:
    if yaml is None:
        raise RuntimeError("PyYAML is required for this script")
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
        act = input(
            "Enter action (1=open long, 2=open short, 3=close, q=quit): "
        ).strip()
        if act == "1":
            resp = broker.open_long()
            print("Opened:", resp)
            if isinstance(resp, dict) and "error" in resp:
                print("Error:", resp["error"])
        elif act == "2":
            resp = broker.open_short()
            print("Opened:", resp)
            if isinstance(resp, dict) and "error" in resp:
                print("Error:", resp["error"])
        elif act == "3":
            resp = broker.close_position()
            print("Closed:", resp)
            if isinstance(resp, dict) and "error" in resp:
                print("Error:", resp["error"])
        elif act.lower() == "q":
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default="configs/live_cfg.yaml")
    args = parser.parse_args()
    main(args.cfg)
