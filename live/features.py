"""Incremental feature computation matching training pipeline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pandas_ta as ta


@dataclass
class FeatureStats:
    mean: np.ndarray
    std: np.ndarray
    lowvol_p5: float


class FeatureExtractor:
    def __init__(self, stats: FeatureStats):
        self.stats = stats
        self.buffer = []

    def update(self, kline: pd.Series) -> np.ndarray | None:
        self.buffer.append(kline)
        if len(self.buffer) < 21:
            return None
        df = pd.DataFrame(self.buffer)
        close = df["close"]
        high = df["high"]
        low = df["low"]
        vol = df["volume"]
        ema8 = ta.ema(close, length=8).iloc[-1]
        ema21 = ta.ema(close, length=21).iloc[-1]
        delta_ema = ema8 - ema21
        rsi = ta.rsi(close, length=14).iloc[-1]
        stoch = ta.stoch(high, low, close, k=14, d=3)
        k = stoch["STOCHk_14_3_3"].iloc[-1]
        d = stoch["STOCHd_14_3_3"].iloc[-1]
        vwap = ta.vwap(high, low, close, vol).iloc[-1]
        vwap_dev = close.iloc[-1] - vwap
        log_ret = np.log(close.iloc[-1] / close.iloc[-2])
        lowvol = 1.0 if vol.iloc[-1] < self.stats.lowvol_p5 else 0.0
        vec = np.array([delta_ema, rsi, k, d, vwap_dev, log_ret, lowvol], dtype=np.float32)
        return (vec - self.stats.mean) / self.stats.std
