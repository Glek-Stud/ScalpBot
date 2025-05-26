# utils/data_loader.py
import json
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve()
PROJECT_ROOT = HERE.parents[2]     # → project_root
DATA = PROJECT_ROOT / "collect" / "data_final"

def load_features(zscore=True, add_lowvol=True):
    # optional debug print:
    # print("Looking in", DATA)
    df = pd.read_parquet(
        DATA / ("btc1m_features_zscore.parquet"
                if zscore else "btc1m_features_matrix.parquet")
    )
    if add_lowvol:
        flag = pd.read_parquet(
            DATA / "btc1m_features_with_lowvol.parquet"
        )['LowVolFlag']
        df = df.join(flag)

    with open(DATA / "norm_stats.json") as f:
        splits = json.load(f)['split']
    return df, splits