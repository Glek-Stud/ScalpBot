# train/model.py
from __future__ import annotations
import tensorflow as tf
from tensorflow.keras import layers, optimizers


class DQN(tf.keras.Model):
    """Simple MLP for Deep-Q trading (Dense 64×3 + linear head)."""

    def __init__(self,
                 num_actions: int = 3,
                 hidden_sizes: tuple[int, ...] = (64, 64, 64),
                 **kwargs):
        super().__init__(**kwargs)
        self._hidden = [
            layers.Dense(units=h,
                         activation="relu",
                         kernel_initializer="he_normal",
                         name=f"fc{i}")
            for i, h in enumerate(hidden_sizes, start=1)
        ]
        self._out = layers.Dense(num_actions, activation=None, name="q_values")

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, x, training: bool = False):  # type: ignore[override]
        for layer in self._hidden:
            x = layer(x, training=training)
        return self._out(x)


# -------------------------------------------------------------------------- #
# factory helpers                                                            #
# -------------------------------------------------------------------------- #
def build_q_network(obs_dim: int,
                    num_actions: int = 3,
                    lr: float = 1e-3,
                    hidden_sizes: tuple[int, ...] = (64, 64, 64)
                    ) -> tuple[DQN, tf.keras.optimizers.Optimizer, tf.keras.losses.Loss]:
    """Instantiate & compile the online Q-network."""
    model = DQN(num_actions=num_actions, hidden_sizes=hidden_sizes)
    model.build((None, obs_dim))           # concrete input shape for TF
    optimiser = optimizers.Adam(learning_rate=lr, clipnorm=1.0)
    loss_fn = tf.keras.losses.Huber()      # smoother than MSE
    return model, optimiser, loss_fn


def hard_update(target_net: tf.keras.Model,
                online_net: tf.keras.Model) -> None:
    """Copy parameters from online to target net."""
    target_net.set_weights(online_net.get_weights())
