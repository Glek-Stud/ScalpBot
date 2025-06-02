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


# train/model.py  (additions only)

from tensorflow.keras import layers, optimizers
import tensorflow as tf

# ---------- NEW: DuelingDQN -------------------------------------------------
class DuelingDQN(tf.keras.Model):
    """Dueling architecture: separate V(s) and A(s,a) streams."""
    def __init__(self,
                 num_actions: int = 3,
                 hidden_sizes: tuple[int, ...] = (128, 128),
                 **kwargs):
        super().__init__(**kwargs)
        self._body = [
            layers.Dense(h, activation="relu", kernel_initializer="he_normal")
            for h in hidden_sizes
        ]
        self._V = layers.Dense(1,            name="V")           # scalar V(s)
        self._A = layers.Dense(num_actions,  name="A")           # advantage

    def call(self, x, training: bool = False):
        for l in self._body:
            x = l(x, training=training)
        v = self._V(x)
        a = self._A(x)
        return v + (a - tf.reduce_mean(a, axis=1, keepdims=True))
# --------------------------------------------------------------------------- #


def build_q_network(obs_dim: int,
                    num_actions: int = 3,
                    lr: float = 3e-4,
                    hidden_sizes: tuple[int, ...] = (128, 128),
                    dueling: bool = True,            # ← new switch
                    ) -> tuple[tf.keras.Model,
                               tf.keras.optimizers.Optimizer,
                               tf.keras.losses.Loss]:
    """Factory returning (model, optimiser, loss)."""
    if dueling:
        model = DuelingDQN(num_actions=num_actions,
                           hidden_sizes=hidden_sizes)
    else:
        model = DQN(num_actions=num_actions,
                    hidden_sizes=hidden_sizes)

    model.build((None, obs_dim))
    opt = optimizers.Adam(learning_rate=lr, clipnorm=1.0)
    loss_fn = tf.keras.losses.Huber()
    return model, opt, loss_fn


def hard_update(target_net: tf.keras.Model,
                online_net: tf.keras.Model) -> None:
    """Copy parameters from online to target net."""
    target_net.set_weights(online_net.get_weights())
