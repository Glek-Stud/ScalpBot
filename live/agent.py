"""DQN agent wrapper for live trading."""

from __future__ import annotations

import numpy as np
import tensorflow as tf

from train.model import DuelingDQN

def _infer_arch(path: str) -> tuple[int, tuple[int, ...]]:
    """Return (obs_dim, hidden_sizes) from a ``save_weights`` HDF5 file."""
    import h5py

    obs_dim = 9
    hidden: list[int] = []
    try:
        with h5py.File(path, "r") as h:
            if "layers" in h:
                names = [n for n in h["layers"].keys() if n.startswith("dense")]
                names.sort(key=lambda x: (0 if x == "dense" else int(x.split("_")[-1])))
                weights = [h["layers"][n]["vars"]["0"] for n in names]
                obs_dim = weights[0].shape[0]
                # Exclude final two layers (value & advantage)
                hidden = [w.shape[1] for w in weights[:-2]]
    except Exception:
        pass
    return obs_dim, tuple(hidden) if hidden else (128, 128)


class DQNAgent:
    def __init__(self, model_path: str, obs_dim: int = 9, n_actions: int = 3):
        """Load a saved Q-network with custom layers or weights-only files."""
        try:
            self.model = tf.keras.models.load_model(
                model_path,
                custom_objects={"DuelingDQN": DuelingDQN},
                compile=False,
            )
        except (OSError, ValueError) as e:
            # Older checkpoints may contain weights only. Recreate the network
            # then load weights.
            msg = str(e)
            if "Layer count mismatch" not in msg and "No model config" not in msg:
                raise
            obs_dim, hidden = _infer_arch(model_path)
            hidden = hidden or (128, 128)
            model = DuelingDQN(num_actions=n_actions, hidden_sizes=hidden)
            model.build((None, obs_dim))
            model.load_weights(model_path)
            self.model = model

    def act(self, obs: np.ndarray) -> int:
        q = self.model(obs[None, :], training=False).numpy()[0]
        return int(q.argmax())
