"""DQN agent wrapper for live trading."""

from __future__ import annotations

import numpy as np
import tensorflow as tf

from train.model import DuelingDQN


class DQNAgent:
    def __init__(self, model_path: str):
        """Load a saved Q-network with custom layers."""
        self.model = tf.keras.models.load_model(
            model_path,
            custom_objects={"DuelingDQN": DuelingDQN},
            compile=False,
        )

    def act(self, obs: np.ndarray) -> int:
        q = self.model(obs[None, :], training=False).numpy()[0]
        return int(q.argmax())
