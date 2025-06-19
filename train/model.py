from __future__ import annotations
import tensorflow as tf
from tensorflow.keras import layers, optimizers


class DuelingDQN(tf.keras.Model):
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


class DQN(tf.keras.Model):

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



def build_q_network(obs_dim: int,
                    num_actions: int = 3,
                    lr: float = 3e-4,
                    hidden_sizes: tuple[int, ...] = (128, 128),
                    dueling: bool = False
                    ) -> tuple[tf.keras.Model,
                               tf.keras.optimizers.Optimizer,
                               tf.keras.losses.Loss]:
    net_cls = DuelingDQN if dueling else DQN
    model = net_cls(num_actions=num_actions,
                    hidden_sizes=hidden_sizes)
    model.build((None, obs_dim))

    optimiser = optimizers.Adam(learning_rate=lr, clipnorm=0.5)
    loss_fn   = tf.keras.losses.Huber()
    return model, optimiser, loss_fn


def hard_update(target_net: tf.keras.Model,
                online_net: tf.keras.Model) -> None:
    target_net.set_weights(online_net.get_weights())
