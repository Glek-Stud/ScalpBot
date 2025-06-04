"""Live trading environment wrapping Binance broker and feature stream."""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .features import FeatureExtractor
from .broker import Broker


class BTCRealTradingEnv(gym.Env):
    """Gym-compatible environment for live BTC trading."""

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, extractor: FeatureExtractor, broker: Broker, max_steps: int = 10_000) -> None:
        super().__init__()
        self.extractor = extractor
        self.broker = broker
        self.max_steps = max_steps
        self.t = 0
        self.position = 0
        self.equity0 = self.equity = 1.0
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)

    def reset(self, *, seed: int | None = None, options=None):
        super().reset(seed=seed)
        self.t = 0
        self.position = 0
        self.equity0 = self.equity = 1.0
        obs = np.zeros(9, dtype=np.float32)
        return obs, {}

    def step(self, action: int):
        feature = self.extractor.update(self.broker.last_kline)
        if feature is None:
            return np.zeros(9, dtype=np.float32), 0.0, False, False, {}
        desired_pos = {0: self.position, 1: 1, 2: -1}[action]
        reward, filled = self.broker.execute(desired_pos)
        self.position = filled
        self.equity *= (1.0 + reward)
        obs = np.concatenate([feature, [self.position, self.equity / self.equity0 - 1.0]]).astype(np.float32)
        self.t += 1
        term = self.t >= self.max_steps
        return obs, reward, term, False, {"equity": self.equity}
