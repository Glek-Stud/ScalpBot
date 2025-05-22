"""BTC‑USDT 1‑minute trading environment (Gymnasium).
Step4/9— trade‑execution engine (position flips, commission, slippage, leverage) ⚙️

➡️ **Patch1**: align Close‑price series to feature index to fix length mismatch.
Rewards still stubbed at 0.0; next step wires true reward calculus.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# Project‑local util
# -----------------------------------------------------------------------------
from utils.data_loader import load_features, DATA as DATA_PATH  # sibling package import

# Global paths (reuse data_loader's resolution logic)
HERE = Path(__file__).resolve()
PROJECT_ROOT = HERE.parents[1]
RAW_PARQUET = PROJECT_ROOT / "collect" / "data_final" / "btcusdt_1m_20240511-20250511.parquet"

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
_FEATURE_ORDER = [
    "ΔEMA",
    "RSI_14",
    "Stoch_%K",
    "Stoch_%D",
    "VWAP_Dev",
    "LogRet_1m",
    "LowVolFlag",
]

_COMMISSION_TABLE = {
    "spot_regular": dict(maker=0.00020, taker=0.00050),
    "spot_bnb10": dict(maker=0.00018, taker=0.00045),
    "usdtm_regular": dict(maker=0.00020, taker=0.00040),
}


class BTCTradingEnv(gym.Env):
    """Gymnasium environment for 1‑minute BTC‑USDT trading."""

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(
        self,
        mode: str = "train",  # {train, val, test}
        max_steps: int = 10_000,
        random_start: bool = False,
        commission_scheme: str = "usdtm_regular",
        leverage: float = 1.0,
        spread_pct: float = 1e-4,  # 1 bp slippage
        lambda_penalty: float | None = None,
    ) -> None:
        super().__init__()

        self._LVOL_COL: int = 6
        assert mode in {"train", "val", "test"}
        self.mode = mode
        self.max_steps = int(max_steps)
        self.random_start = bool(random_start)
        self.leverage = float(leverage)
        self.spread_pct = float(spread_pct)

        # Commission params ---------------------------------------------------
        if commission_scheme not in _COMMISSION_TABLE:
            raise ValueError(f"Unknown commission_scheme: {commission_scheme}")
        self.fee_taker = _COMMISSION_TABLE[commission_scheme]["taker"]

        self._commission_pct: float = _COMMISSION_TABLE[commission_scheme]["taker"]
        self._spread_pct: float = float(spread_pct)


        # Low‑vol penalty weight λ = 0.1 × commission by default
        self._lambda: float = (
            lambda_penalty if lambda_penalty is not None else 0.1 * self._commission_pct
        )

        # ------------------------------------------------------------------
        # Load features (7 cols) & align Close prices
        # ------------------------------------------------------------------
        self._df, self._splits = load_features(zscore=True, add_lowvol=True)
        self._features: np.ndarray = self._df.to_numpy(dtype=np.float32, copy=True)
        self._data_root: Path = DATA_PATH  # <<<<<<<<<<<<<<<<<<<<<<<< added

        # Align Close prices to feature index (drop warm‑up rows)
        raw_close = pd.read_parquet(
            self._data_root / "btcusdt_1m_20240511‑20250511.parquet",
            columns=["Close"],
        )
        raw_close = raw_close.reindex(self._df.index).dropna()
        self._close = raw_close["Close"].to_numpy(np.float32)
        assert len(self._close) == len(
            self._features
        ), "Price & features length mismatch after alignment!"

        # Spaces --------------------------------------------------------------
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)

        # Internal state vars -------------------------------------------------
        self._start_idx: int = 0
        self._idx: int = 0
        self._t: int = 0
        self._position: int = 0  # -1/0/+1
        self._equity0 = self._equity = self._cash = 1.0
        self._equity_floor: float = 0.5

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------
    def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None):
        super().reset(seed=seed)
        rng = np.random.default_rng(seed)
        start_low, start_high = self._slice_bounds(self.mode)
        self._start_idx = start_low if not self.random_start else rng.integers(
            start_low, start_high + 1 - self.max_steps, endpoint=True
        )
        self._idx = self._start_idx
        self._t = 0
        self._position = 0
        self._equity0 = self._equity = self._cash = 1.0
        obs = self._get_observation()
        return obs, {"t": self._t, "idx": self._idx}

    def step(self, action: int):
        # ---------- ACTION HANDLING ----------
        commission_cost = self._commission_pct
        slippage_cost   = self._spread_pct
        price_curr = self._close[self._idx]  # current close

        # execute trade if action changes position
        if action != self._position:
            # taker trade → commission applied once
            commission_cost = self._commission_pct
            # slippage (one-sided spread) applied once
            slippage_cost = self._spread_pct
            # flip position
            self._position = {0: 0, 1: 1, 2: -1}[action]

        # ---------- ADVANCE TIME ----------
        idx_prev = self._idx
        self._t += 1
        self._idx += 1

        terminated = self._t >= self.max_steps  # not _max_steps
        truncated = False  # no time truncation yet

        # ---------- P/L & EQUITY UPDATE ----------
        if self._idx < len(self._close):
            ret = self._bar_return(idx_prev, self._idx)
        else:
            # hit end of dataset (shouldn't in normal run)
            ret = 0.0
            terminated = True

        pnl = self.leverage * self._position * ret  # fraction of equity
        penalty_lv = self._penalty_lowvol(idx_prev, action)  # λ penalty

        # multiplicative equity update in *unit* space
        equity_change = pnl - commission_cost - slippage_cost - penalty_lv
        self._equity *= (1.0 + equity_change)

        # reward is exactly the fractional change
        reward = np.float32(equity_change)

        # ---------- TERMINATION BY DRAWDOWN ----------
        if self._equity < self._equity_floor:
            terminated = True

        # ---------- BUILD OBS & INFO ----------
        obs = self._get_observation()
        info = {
        "t": self._t,
        "idx": idx_prev,
        "price": np.float32(price_curr),
        "position": self._position,
        "equity": np.float32(self._equity),
        "commission": np.float32(commission_cost),
        "slippage": np.float32(slippage_cost),
        "reward_raw": reward,
        }

        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _slice_bounds(self, mode: str) -> Tuple[int, int]:
        s = self._splits
        if mode == "train":
            return 0, s["train_end"]
        if mode == "val":
            return s["train_end"], s["val_end"]
        return s["val_end"], len(self._features)

    def _get_observation(self) -> np.ndarray:
        market = self._features[self._idx]
        equity_ratio = self._equity / self._equity0 - 1.0
        obs = np.concatenate([market, [self._position, equity_ratio]]).astype(np.float32)
        assert obs.shape == (9,) and obs.dtype == np.float32
        return obs

    def _bar_return(self, idx_prev: int, idx_curr: int) -> float:
        p0 = self._close[idx_prev]
        p1 = self._close[idx_curr]
        return float((p1 - p0) / p0)

    def _penalty_lowvol(self, idx_curr: int, action: int) -> float:
        if action != 0 and self._features[idx_curr, self._LVOL_COL] == 1:
            return self._lambda
        return 0.0

    # ------------------------------------------------------------------
    def render(self, mode="human"):
        if mode != "human":
            raise NotImplementedError
        print(
            f"t={self._t} idx={self._idx} price={self._close[self._idx]:.2f} "
            f"pos={self._position} equity={self._equity:.4f}"
        )

    def close(self):
        pass


# ----------------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    env = BTCTradingEnv(mode="train", max_steps=5, random_start=False,
                        commission_scheme="usdtm_regular", leverage=2)
    obs, info = env.reset(seed=0)
    print("t pos reward equity")
    for a in [1, 0, 2, 0]:
        obs, r, term, trunc, info = env.step(a)
        print(info["t"], info["position"], f"{r:.6f}", f"{info['equity']:.4f}")
    env.close()
