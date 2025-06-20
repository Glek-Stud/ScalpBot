import numpy as np
import tensorflow as tf
from tempfile import NamedTemporaryFile

from live.agent import DQNAgent
from train.model import DuelingDQN


def test_agent_loads_dueling(tmp_path):
    model = DuelingDQN(num_actions=3, hidden_sizes=(4,))
    model.build((None, 2))
    f = tmp_path / "model.keras"
    model.save(f)

    agent = DQNAgent(str(f))
    obs = np.zeros(2, dtype=np.float32)
    action = agent.act(obs)
    assert action in {0, 1, 2}


def test_agent_loads_weights_only(tmp_path):
    """Loading from a weights file should also work."""
    model = DuelingDQN(num_actions=3, hidden_sizes=(4,))
    model.build((None, 2))
    f = tmp_path / "model.weights.h5"
    model.save_weights(f)

    agent = DQNAgent(str(f), obs_dim=2, n_actions=3)
    obs = np.zeros(2, dtype=np.float32)
    action = agent.act(obs)
    assert action in {0, 1, 2}

