from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass(slots=True)
class Transition:
    state:    np.ndarray
    action:   int
    reward:   float
    next_s:   np.ndarray
    done:     bool


class ReplayBuffer:
    def __init__(self,
                 capacity: int,
                 obs_dim: int,
                 prioritised: bool = False,
                 alpha: float = 0.7) -> None:

        self.capacity   = capacity
        self.obs_dim    = obs_dim
        self.prioritised = prioritised
        self.alpha      = alpha

        self._states  = np.empty((capacity, obs_dim), dtype=np.float32)
        self._actions = np.empty((capacity,),        dtype=np.int32)
        self._rewards = np.empty((capacity,),        dtype=np.float32)
        self._next_s  = np.empty((capacity, obs_dim), dtype=np.float32)
        self._dones   = np.empty((capacity,),        dtype=np.bool_)

        self._prio    = np.ones((capacity,), dtype=np.float32)

        self._idx  = 0
        self.size  = 0


    def add(self, tr: Transition, td_error: float | None = None) -> None:
        i = self._idx
        self._states[i]  = tr.state
        self._actions[i] = tr.action
        self._rewards[i] = tr.reward
        self._next_s[i]  = tr.next_s
        self._dones[i]   = tr.done
        if td_error is not None:
            self._prio[i] = self._td_to_priority(td_error)
        self._advance()

    def sample(self, batch: int, beta: float = 1.0) -> Tuple[np.ndarray, ...]:
        idx = self._sample_indices(batch)
        p = self._prio[idx]

        probs = p / p.sum()
        N = max(1, self.size)
        weights = (N * probs) ** (-beta)
        weights /= weights.max()

        batch = (self._states[idx],
                self._actions[idx],
                self._rewards[idx],
                self._next_s[idx],
                self._dones[idx],
                idx,
                weights.astype(np.float32))

        return batch

    def update_priority(self, idx: np.ndarray, td_errors: np.ndarray) -> None:
        if not self.prioritised:
            return
        self._prio[idx] = self._td_to_priority(td_errors)

    def _advance(self) -> None:
        self._idx = (self._idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def _sample_indices(self, batch: int) -> np.ndarray:
        if self.prioritised:
            ranks = np.argsort(-self._prio[:self.size])
            probs = 1.0 / (np.arange(self.size) + 1) ** self.alpha
            probs /= probs.sum()
            chosen = np.random.choice(self.size, size=batch, p=probs)
            return ranks[chosen]
        return np.random.randint(0, self.size, size=batch)

    @staticmethod
    def _td_to_priority(td: np.ndarray | float) -> np.ndarray:
        return np.abs(td) + 1e-6
