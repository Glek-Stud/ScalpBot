# train/env_factory.py
"""Helper to build Gym environments for each dataset slice."""
from __future__ import annotations
from typing import Literal

from envs.btc_dqn_env import BTCTradingEnv
from envs.utils.config import load_cfg  # your helper that reads YAML


def make_env(split: Literal["train", "val", "test"],
             seed: int,
             cfg_name: str = "env_binance_tier0"
             ) -> BTCTradingEnv:
    """
    Instantiate BTCTradingEnv with deterministic seeding.

    Parameters
    ----------
    split : "train" | "val" | "test"
        Which time window the env will operate on.
    seed : int
        RNG seed for action-space sampling etc.
    cfg_name : str
        YAML file inside /configs (without .yaml).

    Returns
    -------
    env : BTCTradingEnv
    """
    cfg = load_cfg(cfg_name)
    env = BTCTradingEnv(mode=split, cfg=cfg)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    return env
