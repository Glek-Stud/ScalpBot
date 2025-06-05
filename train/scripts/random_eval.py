from envs.btc_dqn_env import BTCTradingEnv
import numpy as np
from train.trainer import _sharpe, _win_rate, _drawdown

env = BTCTradingEnv(mode="val")
obs, _ = env.reset(seed=123)

equity, trades = [], []
done = False
while not done:
    a = env.action_space.sample()
    obs, r, done, trunc, info = env.step(a)
    equity.append(info["equity"])
    trades.append(r)

equity  = np.array(equity, dtype=np.float32)
returns = np.diff(equity) / equity[:-1]
trades  = np.array(trades, dtype=np.float32)

print("Random policy:")
print("  Sharpe       :", _sharpe(returns))
print("  Win-Rate [%] :", _win_rate(trades))
print("  MaxDD  [%]   :", _drawdown(equity))
