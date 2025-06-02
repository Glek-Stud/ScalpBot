# train/scripts/oracle_eval.py
from envs.btc_dqn_env import BTCTradingEnv
import numpy as np
from train.trainer import _sharpe, _drawdown

env = BTCTradingEnv(mode="val")
obs, _ = env.reset(seed=1)

closes = env._df["Close"].to_numpy()             # 1-min close prices
start  = env._cursor
end    = len(closes) - 1                         # last usable index

# Build action list: 1 if next price↑, 2 if ↓, 0 final bar
actions = np.where(closes[start+1:end] > closes[start:end-1], 1, 2)
actions = np.append(actions, 0)

env.reset(seed=1)
equity = []
for a in actions:
    _, _, done, _, info = env.step(int(a))
    equity.append(info["equity"])

equity  = np.array(equity, dtype=np.float32)
returns = np.diff(equity) / equity[:-1]
print("Oracle Sharpe :", _sharpe(returns))
print("MaxDD [%]     :", _drawdown(equity))
