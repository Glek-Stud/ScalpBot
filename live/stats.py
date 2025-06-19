"""Performance statistics tracker for live trading."""

from __future__ import annotations

import json
from pathlib import Path
from dataclasses import dataclass, field
import numpy as np


@dataclass
class StatsTracker:
    log_dir: Path
    start_equity: float = 1.0
    step: int = 0
    equity: list[float] = field(default_factory=lambda: [1.0])
    returns: list[float] = field(default_factory=list)
    trades: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.log_dir = Path(self.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.equity[0] = self.start_equity

    # ------------------------------------------------------------------
    def update(self, reward: float, position_changed: bool) -> None:
        """Add step reward and optionally trade result."""
        self.step += 1
        new_eq = self.equity[-1] * (1.0 + reward)
        self.equity.append(new_eq)
        self.returns.append(reward)
        if position_changed:
            self.trades.append(reward)

    # ------------------------------------------------------------------
    def metrics(self) -> dict[str, float]:
        if not self.returns:
            return {"Sharpe": 0.0, "WinRate": 0.0, "MaxDD": 0.0}
        eq = np.asarray(self.equity)
        ret = np.asarray(self.returns)
        trade = np.asarray(self.trades) if self.trades else np.asarray([])
        sharpe = float(ret.mean() / ret.std(ddof=1) * np.sqrt(365 * 24 * 60)) if ret.std(ddof=1) > 0 else 0.0
        if eq.size > 1:
            cummax = np.maximum.accumulate(eq)
            dd = 1.0 - eq / cummax
            mdd = float(dd.max() * 100)
        else:
            mdd = 0.0
        win_rate = float((trade > 0).mean() * 100) if trade.size > 0 else 0.0
        return {"Sharpe": sharpe, "WinRate": win_rate, "MaxDD": mdd}

    # ------------------------------------------------------------------
    def flush(self) -> None:
        data = self.metrics()
        path = self.log_dir / f"stats_{self.step}.json"
        path.write_text(json.dumps(data, indent=2))

