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
from .utils.data_loader import load_features, DATA as DATA_PATH
from .utils.config import load_cfg

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
        cfg_name: str = "env_binance_tier0",
        noise_sigma: float = 0.0,
        **overrides,

    ) -> None:
        super().__init__()

        self._LVOL_COL: int = 6
        assert mode in {"train", "val", "test"}
        self.mode = mode
        self.max_steps = int(max_steps)
        self.random_start = bool(random_start)
        self.noise_sigma: float = float(noise_sigma)

        _cfg = load_cfg(cfg_name)
        _cfg.update(overrides)  # kwargs win over YAML

        # Short alias so later code is cleaner
        self.cfg = _cfg

        # ================================
        # --- economic parameters -------
        # ================================
        self.contract_value = _cfg["contract_value"]  # 1.0 USDT by default

        self._commission_maker = _cfg["maker_pct"]  # 0.00018
        self._commission_taker = _cfg["taker_pct"]  # 0.00036
        self._maker_prob = _cfg.get("use_maker_probability", 0.0)

        self._spread_pct = _cfg["spread_pct"]  # 1 bp
        self._lambda = _cfg["lambda_penalty_factor"] * self._commission_taker

        self.leverage = _cfg["leverage"]  # 2
        self._equity_floor = _cfg["equity_floor"]  # 0.5

        # funding support (optional)
        self.funding_enabled = _cfg.get("funding_enabled", False)
        if self.funding_enabled:
            fund_file = _cfg["funding_parquet"]
            fund_df = pd.read_parquet(DATA_PATH / fund_file, columns=["funding_rate"])
            self._fund_rates = fund_df["funding_rate"].to_numpy(np.float32)

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


        if mode == "train":
            left = 0
            right = self._splits["train_end"]
        elif mode == "val":
            left = self._splits["train_end"]
            right = self._splits["val_end"]
        else:  # test
            left = self._splits["val_end"]
            right = len(self._features)

        self._slice_left = left
        self._slice_right = right  # exclusive

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------
    def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None):
        super().reset(seed=seed)
        rng = np.random.default_rng(seed)

        # 1️⃣ pick episode start -----------------------------------------------
        start_low, start_high = self._slice_bounds(self.mode)
        self._start_idx = (
            start_low
            if not self.random_start
            else rng.integers(start_low, start_high + 1 - self.max_steps, endpoint=True)
        )
        self._idx = self._start_idx
        self._t = 0
        self._position = 0
        self._equity0 = self._equity = self._cash = 1.0

        # 2️⃣ set episode end ---------------------------------------------------
        # _slice_right was cached as an attribute in __init__
        self._end_idx = min(self._start_idx + self.max_steps, self._slice_right - 1)

        # 3️⃣ return first observation -----------------------------------------
        obs = self._get_observation()
        return obs, {"t": self._t, "idx": self._idx}

    def step(self, action: int):
        # ----- overflow guard -----
        if self._idx >= self._end_idx or self._t >= self.max_steps:
            raise RuntimeError("Step called after episode end. Call reset().")

        price_curr = self._close[self._idx]

        # ----- trading costs via helper -----
        commission_cost, slippage_cost = self._exec_trade(action)

        # ----- advance time -----
        idx_prev = self._idx
        self._t += 1
        self._idx += 1

        # time / slice limits
        terminated = truncated = False
        if self._idx >= self._end_idx:
            if self._idx >= self._slice_right - 1:
                truncated = True
            else:
                terminated = True

        # ----- bar return & P/L -----
        if self._idx < len(self._close):
            ret = self._bar_return(idx_prev, self._idx)
        else:
            ret = 0.0
            terminated = True

        pnl = self.leverage * self._position * ret  # fraction of equity
        penalty_lv = self._penalty_lowvol(idx_prev, action)  # λ penalty
        equity_change = pnl - commission_cost - slippage_cost - penalty_lv
        self._equity *= (1.0 + equity_change)
        if not np.isfinite(self._equity):
            raise FloatingPointError("Equity became NaN or inf")

        # ----- funding every 8 h (480 bars) -----
        if self.funding_enabled and (self._idx % 480 == 0):
            fr = float(self._fund_rates[self._idx])
            self._equity *= (1.0 + self.leverage * self._position * fr)

        reward = np.float32(equity_change)

        # draw-down kill-switch
        if self._equity < self._equity_floor:
            terminated = True

        # ----- observation & info -----
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
            "terminated": terminated,
            "truncated": truncated,
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

        if self.noise_sigma > 0.0 and self.mode == "train":
            noise = np.random.normal(0.0, self.noise_sigma, size=obs.shape).astype(np.float32)
            obs = obs + noise

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

    def _exec_trade(self, action: int) -> tuple[float, float]:
        desired_pos = (
            self._position  # Hold keeps current
            if action == 0 else
            1 if action == 1 else
            -1  # action == 2
        )

        if desired_pos == self._position:
            return 0.0, 0.0

        if self.np_random.random() < self._maker_prob:
            comm_pct = self._commission_maker
        else:
            comm_pct = self._commission_taker

        commission_cost = comm_pct
        slippage_cost = self._spread_pct

        self._position = desired_pos
        return commission_cost, slippage_cost

    # ------------------------------------------------------------------
    # ---------------------------------------------------------------------
    # RENDER: quick Matplotlib view (price + equity)
    # ---------------------------------------------------------------------
    def render(self, mode: str = "human"):
        if mode != "human":
            return  # only human mode supported

        import matplotlib.pyplot as plt

        # cache history in object attributes
        if not hasattr(self, "_render_cache"):
            self._render_cache = {"t": [], "price": [], "equity": []}

        self._render_cache["t"].append(self._t)
        self._render_cache["price"].append(float(self._close[self._idx]))
        self._render_cache["equity"].append(float(self._equity))

        # update plot once every 300 steps to avoid slowdown
        if self._t % 300 != 0 and self._t != self.max_steps:
            return

        plt.clf()
        ax1 = plt.gca()
        ax1.plot(self._render_cache["t"], self._render_cache["price"],
                 label="Close", alpha=0.6)
        ax1.set_xlabel("step")
        ax1.set_ylabel("Price (USDT)")

        ax2 = ax1.twinx()
        ax2.plot(self._render_cache["t"], self._render_cache["equity"],
                 label="Equity", color="tab:blue")
        ax2.set_ylabel("Equity (unit)")

        ax1.legend(loc="upper left")
        ax2.legend(loc="upper right")
        plt.pause(0.001)

    def close(self):
        if hasattr(self, "_render_cache"):
            self._render_cache.clear()



