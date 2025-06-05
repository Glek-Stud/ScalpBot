import numpy as np
from envs.btc_dqn_env import BTCTradingEnv
from train.trainer import _sharpe, _drawdown

env = BTCTradingEnv(mode="val")
obs, _ = env.reset(seed=1)

prices = []
done = False
while not done:
    obs, r, done, trunc, info = env.step(0)     # HOLD
    prices.append(info["price"])

prices = np.array(prices, dtype=np.float32)

deltas  = np.sign(np.diff(prices))
actions = np.where(deltas > 0, 1, 2)
actions = np.append(actions, 0)


env.close()
env = BTCTradingEnv(mode="val")
env.reset(seed=1)
equity = []

for a in actions:
    _, _, done, trunc, info = env.step(int(a))
    equity.append(info["equity"])

equity  = np.array(equity, dtype=np.float32)
returns = np.diff(equity) / equity[:-1]

print("Oracle Sharpe :", _sharpe(returns))
print("Oracle MaxDD% :", _drawdown(equity))
