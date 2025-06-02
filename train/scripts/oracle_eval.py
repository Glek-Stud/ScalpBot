# train/scripts/oracle_eval.py
from envs.btc_dqn_env import BTCTradingEnv
import numpy as np, pandas as pd
from train.trainer import _sharpe, _drawdown

env = BTCTradingEnv(mode="val")
obs, _ = env.reset(seed=1)

equity = []
done = False
prices = []

while not done:
    prices.append(env._price)            # assume env exposes price
    done = env._cursor >= len(env._prices) - 2
# build oracle action list: +1 if next close up, -1 if down
deltas = np.sign(np.diff(prices))
actions = np.where(deltas > 0, 1, 2)     # BUY or SELL
actions = np.append(actions, 0)          # last step HOLD

env.reset(seed=1)
for a in actions:
    _, _, done, trunc, info = env.step(a)
    equity.append(info["equity"])

equity  = np.array(equity)
returns = np.diff(equity) / equity[:-1]
print("Oracle Sharpe:", _sharpe(returns))
