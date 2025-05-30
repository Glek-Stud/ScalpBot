# train/trainer.py
"""DQNTrainer ties everything together: env → replay → Q-learn → logs."""
from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Literal

import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorboard.plugins.hparams import api as hp

from env_factory      import make_env
from model            import build_q_network, hard_update
from replay_buffer    import ReplayBuffer, Transition
from epsilon          import LinearSchedule


# --------------------------------------------------------------------------- #
# utility: quick performance metrics                                          #
# --------------------------------------------------------------------------- #
def _sharpe(returns: np.ndarray, freq: int = 365 * 24 * 60) -> float:
    """Annualised Sharpe ratio (µ/σ * √freq)."""
    mu, sig = returns.mean(), returns.std(ddof=1)
    return float(mu / sig * np.sqrt(freq)) if sig > 0 else 0.0


def _drawdown(equity: np.ndarray) -> float:
    """Maximum drawdown in percentage."""
    cummax = np.maximum.accumulate(equity)
    dd     = 1.0 - equity / cummax
    return float(dd.max() * 100)


def _win_rate(trades: np.ndarray) -> float:
    return float((trades > 0).mean() * 100)


# --------------------------------------------------------------------------- #
# dataclass for trainer hyper-params (makes JSON serialisation easy)          #
# --------------------------------------------------------------------------- #
@dataclass
class TrainerParams:
    buffer_cap:    int   = 100_000
    batch_size:    int   = 64
    gamma:         float = 0.99
    lr:            float = 1e-3
    target_freq:   int   = 1_000      # env steps
    warmup_steps:  int   = 2_000
    eps_decay:     int   = 30_000
    val_freq:      int   = 20_000     # env steps
    patience:      int   = 5          # early-stop on val Sharpe
    prioritised:   bool  = False


class DQNTrainer:
    """Self-contained trainer with online + target nets, buffer & ε-schedule."""

    def __init__(self,
                 seed:      int,
                 logdir:    Path,
                 params:    TrainerParams = TrainerParams(),
                 cfg_name:  str = "env_binance_tier0") -> None:

        self.params   = params
        self.logdir   = logdir
        self.ckpt_dir = logdir / "checkpoints"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        # 1) envs --------------------------------------------------------------
        self.env_train = make_env("train", seed, cfg_name)
        self.env_val   = make_env("val",   seed + 1, cfg_name)

        obs_dim   = self.env_train.observation_space.shape[0]
        n_actions = self.env_train.action_space.n

        # 2) networks ----------------------------------------------------------
        self.online, self.opt, self.loss_fn = build_q_network(obs_dim, n_actions, params.lr)
        self.target, _, _ = build_q_network(obs_dim, n_actions, params.lr)

        dummy = tf.zeros((1, obs_dim), dtype=tf.float32)
        _ = self.online(dummy)  # create weights
        _ = self.target(dummy)

        hard_update(self.target, self.online)

        # 3) replay, ε-sched, misc --------------------------------------------
        self.buffer    = ReplayBuffer(params.buffer_cap, obs_dim,
                                      prioritised=params.prioritised)
        self.eps_sched = LinearSchedule(1.0, 0.1, params.eps_decay)
        self.step      = 0
        self.best_sharpe = -np.inf
        self.no_improve  = 0

        # 4) TensorBoard -------------------------------------------------------
        self.tb = tf.summary.create_file_writer(str(logdir))
        # log hyper-params once
        with self.tb.as_default():
            hp.hparams(asdict(params))

    # --------------------------------------------------------------------- #
    # public API                                                            #
    # --------------------------------------------------------------------- #
    def train(self, max_steps: int = 300_000) -> None:
        obs, _ = self.env_train.reset()
        episode_r = 0.0

        while self.step < max_steps:
            # ε-greedy action --------------------------------------------------
            eps = self.eps_sched.value(self.step)
            if np.random.rand() < eps:
                action = self.env_train.action_space.sample()
            else:
                q = self.online(obs[None])[0].numpy()
                action = int(q.argmax())

            # env step ---------------------------------------------------------
            nxt, rew, done, trunc, info = self.env_train.step(action)
            end = done or trunc
            self.buffer.add(Transition(obs, action, rew, nxt, end))
            obs = nxt
            episode_r += rew
            self.step += 1

            # learning after warm-up ------------------------------------------
            if self.buffer.size >= max(self.params.batch_size,
                                       self.params.warmup_steps):
                self._learn()

            # target net sync --------------------------------------------------
            if self.step % self.params.target_freq == 0:
                hard_update(self.target, self.online)

            # validation -------------------------------------------------------
            if self.step % self.params.val_freq == 0:
                v_sharpe = self._validate()
                self._log_scalar("val/sharpe", v_sharpe, self.step)

                # early stopping logic
                if v_sharpe > self.best_sharpe:
                    self.best_sharpe = v_sharpe
                    self.no_improve  = 0
                    self._save_ckpt("best")
                else:
                    self.no_improve += 1
                    if self.no_improve >= self.params.patience:
                        print(f"Early stop @ {self.step} steps")
                        break

            if end:
                self._log_scalar("train/episode_reward", episode_r, self.step)
                obs, _ = self.env_train.reset()
                episode_r = 0.0

        # always save final weights
        self._save_ckpt(f"final_{self.step}")

    def evaluate(self, split: Literal["val", "test"] = "test",
                 render: bool = False) -> dict[str, float]:
        """Run a full episode without exploration and compute metrics."""
        env = self.env_val if split == "val" else make_env(split, 999)
        obs, _ = env.reset()
        equity = []
        trades = []

        done = False
        while not done:
            q = self.online(obs[None])[0].numpy()
            action = int(q.argmax())
            obs, rew, done, trunc, info = env.step(action)
            equity.append(info.get("equity", 0.0))
            trades.append(rew)
            if render:
                env.render()

        equity     = np.array(equity)
        returns    = np.diff(equity) / equity[:-1]
        metrics = {
            "Sharpe":          _sharpe(returns),
            "WinRate[%]":      _win_rate(np.array(trades)),
            "MaxDD[%]":        _drawdown(equity),
            "ProfitFactor":    float(np.sum(trades[ trades > 0])
                                    / (1e-8 + np.abs(trades[ trades < 0 ]).sum()))
        }
        return metrics

    # --------------------------------------------------------------------- #
    # internals                                                             #
    # --------------------------------------------------------------------- #
    def _learn(self):
        s, a, r, ns, d, idx = self.buffer.sample(self.params.batch_size)

        # TD target
        max_next_q = tf.reduce_max(self.target(ns), axis=1)
        tgt = r + (1.0 - d.astype(np.float32)) * self.params.gamma * max_next_q

        with tf.GradientTape() as tape:
            q_pred = tf.gather_nd(self.online(s),
                                  np.stack([np.arange(self.params.batch_size), a], 1))
            loss = self.loss_fn(tgt, q_pred)

        grads = tape.gradient(loss, self.online.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.online.trainable_variables))

        # PER prio update
        td_err = (tgt - q_pred).numpy()
        self.buffer.update_priority(idx, td_err)

        self._log_scalar("train/loss", K.get_value(loss), self.step)
        self._log_scalar("train/epsilon", self.eps_sched.value(self.step), self.step)

    def _validate(self) -> float:
        metrics = self.evaluate("val")
        print(f"[val] step {self.step} — Sharpe {metrics['Sharpe']:.3f}  "
              f"WinRate {metrics['WinRate[%]']:.1f}%  MaxDD {metrics['MaxDD[%]']:.1f}%")
        return metrics["Sharpe"]

    # ---------------- tensorboard + checkpoint helpers ------------------- #
    def _log_scalar(self, tag: str, val: float, step: int):
        with self.tb.as_default():
            tf.summary.scalar(tag, val, step=step)

    def _save_ckpt(self, name: str):
        path = self.ckpt_dir / f"dqn_{name}.h5"
        self.online.save(path, include_optimizer=False)
        with open(self.ckpt_dir / f"{name}_params.json", "w") as f:
            json.dump(asdict(self.params), f, indent=2)
        print(f"✔ saved checkpoint → {path}")
