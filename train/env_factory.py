from __future__ import annotations
from typing import Literal

from envs.btc_dqn_env import BTCTradingEnv
from envs.utils.config import load_cfg


def make_env(split: Literal["train", "val", "test"],
             seed: int,
             cfg_name: str = "env_binance_tier0"
             ) -> BTCTradingEnv:

    cfg = load_cfg(cfg_name)
    env = BTCTradingEnv(mode=split, cfg=cfg)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    return env
