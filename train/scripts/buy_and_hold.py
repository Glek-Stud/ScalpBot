from envs.btc_dqn_env import BTCTradingEnv
import numpy as np
from train.trainer import _sharpe, _drawdown

env = BTCTradingEnv(mode="val", random_start=True,
                    cfg_name="env_binance_tier0",
                    use_maker_probability=0.0,
                    funding_enabled=False)
obs, _ = env.reset(seed=123)

obs, r, done, trunc, info = env.step(1)

equity = []
while not done:
    obs, r, done, trunc, info = env.step(0)
    equity.append(info["equity"])

equity  = np.array(equity, dtype=np.float32)
returns = np.diff(equity) / equity[:-1]

print("Buy-and-Hold:")
print("  Sharpe     :", _sharpe(returns))
print("  MaxDD [%]  :", _drawdown(equity))
