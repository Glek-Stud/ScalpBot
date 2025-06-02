from envs.utils.data_loader import load_features
import numpy as np
from train.trainer import _sharpe
from envs.btc_dqn_env import BTCTradingEnv

# 1) prices from dataframe
X, split = load_features(zscore=False, add_lowvol=False)
price_ser = X['Close']          # or whichever column is raw close
val_prices = price_ser.iloc[split["train_end"]: split["val_end"]].to_numpy()

# 2) oracle actions
deltas = np.sign(np.diff(val_prices))
actions = np.where(deltas > 0, 1, 2)   # BUY if up, else SELL
actions = np.append(actions, 0)        # last step HOLD

# 3) replay them through the environment
env = BTCTradingEnv(mode="val")
env.reset(seed=123)
equity = []
for a in actions:
    _, _, done, trunc, info = env.step(int(a))
    equity.append(info["equity"])
returns = np.diff(equity) / equity[:-1]
print("Oracle Sharpe:", _sharpe(returns))
