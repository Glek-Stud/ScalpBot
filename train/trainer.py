from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Literal

import json

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorboard.plugins.hparams import api as hp

from .env_factory import make_env
from .model import build_q_network, hard_update
from .replay_buffer import ReplayBuffer, Transition
from .epsilon import LinearSchedule


def _sharpe(returns: np.ndarray, freq: int = 365 * 24 * 60) -> float:
    mu, sig = returns.mean(), returns.std(ddof=1)
    return float(mu / sig * np.sqrt(freq)) if sig > 0 else 0.0


def _drawdown(equity: np.ndarray) -> float:
    cummax = np.maximum.accumulate(equity)
    dd     = 1.0 - equity / cummax
    return float(dd.max() * 100)


def _win_rate(trades: np.ndarray) -> float:
    return float((trades > 0).mean() * 100)


@dataclass
class TrainerParams:
    buffer_cap: int = 100_000
    batch_size: int = 256
    gamma: float = 0.99
    lr: float = 3e-4
    target_freq: int = 500
    warmup_steps: int = 2_000
    eps_decay: int = 120_000
    val_freq: int = 10_000
    patience: int = 20
    prioritised: bool  = True
    dueling: bool  = True
    hidden: tuple[int, ...] = (128,128)
    per_beta_start: float = 0.4


class DQNTrainer:
    def __init__(self,
                 seed:      int,
                 logdir:    Path,
                 params:    TrainerParams = TrainerParams(),
                 cfg_name:  str = "env_binance_tier0",
                 dueling:   bool = False
                 ) -> None:

        self.params   = params
        self.logdir   = logdir
        self.ckpt_dir = logdir / "checkpoints"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        self.env_train = make_env("train", seed, cfg_name)
        self.env_val   = make_env("val",   seed + 1, cfg_name)

        obs_dim   = self.env_train.observation_space.shape[0]
        n_actions = self.env_train.action_space.n

        self.online, self.opt, self.loss_fn = build_q_network(
            obs_dim,
            n_actions,
            lr=params.lr,
            hidden_sizes=(128, 128),
            dueling=dueling
        )

        self.target, _, _ = build_q_network(
            obs_dim,
            n_actions,
            lr=params.lr,
            hidden_sizes=(128, 128),
            dueling=dueling
        )

        dummy = tf.zeros((1, obs_dim), dtype=tf.float32)
        _ = self.online(dummy)
        _ = self.target(dummy)

        hard_update(self.target, self.online)

        self.buffer    = ReplayBuffer(params.buffer_cap, obs_dim,
                                      prioritised=True, alpha=0.7)
        self.eps_sched = LinearSchedule(1.0, 0.1, params.eps_decay)
        self.step      = 0
        self.best_sharpe = -np.inf
        self.no_improve  = 0

        self.tb = tf.summary.create_file_writer(str(logdir))

        safe_hparams = {
            k: (str(v) if isinstance(v, (tuple, list, dict)) else v)
            for k, v in asdict(params).items()
        }

        with self.tb.as_default():
            hp.hparams(safe_hparams)

    def train(self, max_steps: int = 300_000) -> None:
        obs, _ = self.env_train.reset()
        episode_r = 0.0

        while self.step < max_steps:
            eps = self.eps_sched.value(self.step)
            if np.random.rand() < eps:
                action = self.env_train.action_space.sample()
            else:
                q = self.online(obs[None])[0].numpy()
                action = int(q.argmax())

            nxt, rew, done, trunc, info = self.env_train.step(action)
            end = done or trunc
            self.buffer.add(Transition(obs, action, rew, nxt, end))
            obs = nxt
            episode_r += rew
            self.step += 1

            if self.buffer.size >= max(self.params.batch_size,
                                       self.params.warmup_steps):
                self._learn()

            if self.step % self.params.target_freq == 0:
                hard_update(self.target, self.online)

            if self.step % self.params.val_freq == 0:
                v_sharpe = self._validate()
                self._log_scalar("val/sharpe", v_sharpe, self.step)

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
                 render: bool = False,
                 save_dir: str | Path | None = None) -> dict[str, float]:
        env = self.env_val if split == "val" else make_env(split, 999)
        obs, _ = env.reset()
        equity, actions, trades = [], [], []

        done = False
        while not done:
            q = self.online(obs[None])[0].numpy()
            action = int(q.argmax())
            obs, rew, done, trunc, info = env.step(action)
            equity.append(info["equity"])
            actions.append(action)
            trades.append(rew)
            if render:
                env.render()

        if save_dir is not None:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            df = pd.DataFrame({"equity": equity, "action": actions, "reward": trades})
            df.to_csv(save_dir / f"equity_{split}.csv", index=False)

        equity = np.asarray(equity, dtype=np.float32)
        returns = np.diff(equity) / equity[:-1]
        trades = np.asarray(trades, dtype=np.float32)

        return {
            "Sharpe": _sharpe(returns),
            "WinRate[%]": _win_rate(trades),
            "MaxDD[%]": _drawdown(equity),
            "ProfitFactor": float(trades[trades > 0].sum() /
                                  (1e-8 + np.abs(trades[trades < 0]).sum())),
        }

    def _learn(self):
        beta0 = self.params.per_beta_start
        beta = beta0 + (1.0 - beta0) * min(1.0, self.step / 200_000)

        s, a, r, ns, d, idx, w = self.buffer.sample(self.params.batch_size, beta)
        w = tf.convert_to_tensor(w, dtype=tf.float32)

        bsz = self.params.batch_size
        gamma = self.params.gamma

        a_star = tf.argmax(self.online(ns), axis=1, output_type=tf.int32)  # select
        q_next = self.target(ns)  # evaluate
        idx2 = tf.stack([tf.range(bsz, dtype=tf.int32), a_star], axis=1)
        max_next_q = tf.gather_nd(q_next, idx2)  # Q_target(s', a*)

        tgt = r + (1.0 - d.astype(np.float32)) * gamma * max_next_q

        with tf.GradientTape() as tape:
            q_pred = tf.gather_nd(self.online(s),
                                  tf.stack([tf.range(bsz), a], axis=1))
            td_err = tgt - q_pred
            loss = self.loss_fn(tgt, q_pred) * w  # importance weighting
            loss = tf.reduce_mean(loss)

        grads = tape.gradient(loss, self.online.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.online.trainable_variables))

        self.buffer.update_priority(idx, tf.abs(td_err).numpy())
        self._log_scalar("train/loss", K.get_value(loss), self.step)
        self._log_scalar("train/epsilon", self.eps_sched.value(self.step), self.step)

        if self.step % 5_000 == 0:
            self._log_scalar("debug/avg_reward", float(np.mean(r)), self.step)
            max_q = tf.reduce_max(tf.abs(self.online(s))).numpy()
            self._log_scalar("debug/q_abs_max", float(max_q), self.step)


    def _validate(self) -> float:
        metrics = self.evaluate("val")
        print(f"[val] step {self.step} — Sharpe {metrics['Sharpe']:.3f}  "
              f"WinRate {metrics['WinRate[%]']:.1f}%  MaxDD {metrics['MaxDD[%]']:.1f}%")
        return metrics["Sharpe"]

    def _log_scalar(self, tag: str, val: float, step: int):
        with self.tb.as_default():
            tf.summary.scalar(tag, val, step=step)

    def _save_ckpt(self, name: str):
        path = self.ckpt_dir / f"dqn_{name}.h5"
        self.online.save(path, include_optimizer=False)
        with open(self.ckpt_dir / f"{name}_params.json", "w") as f:
            json.dump(asdict(self.params), f, indent=2)
        print(f"✔ saved checkpoint → {path}")

