# train/epsilon.py
"""Simple linear ε-decay schedule."""
from __future__ import annotations


class LinearSchedule:
    def __init__(self,
                 eps_start: float = 1.0,
                 eps_end: float = 0.1,
                 decay_steps: int = 30_000) -> None:
        self.start = eps_start
        self.end   = eps_end
        self.decay = decay_steps

    def value(self, step: int) -> float:
        """ε at a given global step (clipped)."""
        frac = min(step / self.decay, 1.0)
        return self.start + frac * (self.end - self.start)
