# train/replay_buffer.py
"""Ring-buffer replay memory with optional rank-based prioritisation."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass(slots=True)
class Transition:
    """Simple container to keep the typing clear."""
    state:    np.ndarray  # shape = (obs_dim,)
    action:   int
    reward:   float
    next_s:   np.ndarray  # shape = (obs_dim,)
    done:     bool


class ReplayBuffer:
    """Uniform or rank-based Prioritised Experience Replay (PER).

    Parameters
    ----------
    capacity : int
        Max transitions stored (oldest overwritten).
    obs_dim : int
        Length of observation vector.
    prioritised : bool
        If True, uses rank-based PER (Schaul et al. 2015) with exponent α.
    """
    def __init__(self,
                 capacity: int,
                 obs_dim: int,
                 prioritised: bool = False,
                 alpha: float = 0.7) -> None:

        self.capacity   = capacity
        self.obs_dim    = obs_dim
        self.prioritised = prioritised
        self.alpha      = alpha

        # Pre-allocate contiguous arrays
        self._states  = np.empty((capacity, obs_dim), dtype=np.float32)
        self._actions = np.empty((capacity,),        dtype=np.int32)
        self._rewards = np.empty((capacity,),        dtype=np.float32)
        self._next_s  = np.empty((capacity, obs_dim), dtype=np.float32)
        self._dones   = np.empty((capacity,),        dtype=np.bool_)

        # For PER: store priorities (1 = uninformative baseline)
        self._prio    = np.ones((capacity,), dtype=np.float32)

        self._idx  = 0      # next insert position
        self.size  = 0      # how many valid samples so far

    # ------------------------------------------------------------------ #
    # public API                                                          #
    # ------------------------------------------------------------------ #
    def add(self, tr: Transition, td_error: float | None = None) -> None:
        """Insert a transition (overwrite oldest)."""
        i = self._idx
        self._states[i]  = tr.state
        self._actions[i] = tr.action
        self._rewards[i] = tr.reward
        self._next_s[i]  = tr.next_s
        self._dones[i]   = tr.done
        if td_error is not None:
            self._prio[i] = self._td_to_priority(td_error)
        self._advance()

    def sample(self, batch: int) -> Tuple[np.ndarray, ...]:
        """Return a batch (states, actions, rewards, next_states, dones, idx)."""
        idx = self._sample_indices(batch)
        return (self._states[idx],
                self._actions[idx],
                self._rewards[idx],
                self._next_s[idx],
                self._dones[idx],
                idx)                       # needed to update prios

    def update_priority(self, idx: np.ndarray, td_errors: np.ndarray) -> None:
        """Post-learning priority update (only if prioritised)."""
        if not self.prioritised:
            return
        self._prio[idx] = self._td_to_priority(td_errors)

    # ------------------------------------------------------------------ #
    # internal helpers                                                    #
    # ------------------------------------------------------------------ #
    def _advance(self) -> None:
        self._idx = (self._idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def _sample_indices(self, batch: int) -> np.ndarray:
        if self.prioritised:
            # rank-based: sort prio, convert to ranks 1..N, prob ∝ 1/rank^α
            ranks = np.argsort(-self._prio[:self.size])  # highest first
            probs = 1.0 / (np.arange(self.size) + 1) ** self.alpha
            probs /= probs.sum()
            chosen = np.random.choice(self.size, size=batch, p=probs)
            return ranks[chosen]
        # uniform
        return np.random.randint(0, self.size, size=batch)

    @staticmethod
    def _td_to_priority(td: np.ndarray | float) -> np.ndarray:
        return np.abs(td) + 1e-6          # ensure non-zero
