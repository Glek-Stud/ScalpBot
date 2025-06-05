from __future__ import annotations
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
from envs.utils.data_loader import load_features


def load_dataset(zscore: bool = True,
                 add_lowvol: bool = True
                 ) -> Tuple[pd.DataFrame, Dict[str, slice]]:

    X, split_idx = load_features(zscore=zscore, add_lowvol=add_lowvol)

    splits = {
        "train": slice(None, split_idx["train_end"]),
        "val":   slice(split_idx["train_end"], split_idx["val_end"]),
        "test":  slice(split_idx["val_end"], None),
    }
    return X, splits
