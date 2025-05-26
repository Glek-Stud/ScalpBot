"""Unit‑tests for BTCTradingEnv (PhaseB).
Run with:  pytest -q
"""
import numpy as np
import pytest

from envs.btc_dqn_env import BTCTradingEnv

# ------------------------------------------------------------
# test helpers
# ------------------------------------------------------------

def make_env_train_fixed(max_steps=1_000, **kw):
    return BTCTradingEnv(
        mode="train",
        cfg_name="env_binance_tier0",
        random_start=False,
        noise_sigma=0.0,
        funding_enabled=False,
        use_maker_probability=0.0,   # <-- always taker
        max_steps=max_steps,
        **kw
    )



def manual_reward_trace(env: BTCTradingEnv, actions):
    """Re‑implement env reward logic to cross‑check per‑step deltas."""
    obs, _ = env.reset(seed=0)
    idx = env._idx
    pos = 0
    eq = 1.0
    outs = []
    close = env._close
    for a in actions:
        commission = slippage = 0.0
        if a != pos:
            commission = env._commission_taker  # assume taker for flips
            slippage = env._spread_pct
            pos = {0: 0, 1: 1, 2: -1}[a]
        ret = (close[idx + 1] - close[idx]) / close[idx]
        pnl = env.leverage * pos * ret
        delta = pnl - commission - slippage
        outs.append(np.float32(delta))
        eq *= 1 + delta
        idx += 1
    return np.asarray(outs, dtype=np.float32)

# ------------------------------------------------------------
# 1. P/L parity ------------------------------------------------
# ------------------------------------------------------------

def test_pnl_matches_env():
    actions = [1, 0, 2]
    env = make_env_train_fixed(max_steps=len(actions) + 1)
    env.reset(seed=0)
    rewards_env = []
    for a in actions:
        _, r, _, _, _ = env.step(a)
        rewards_env.append(r)
    rewards_env = np.asarray(rewards_env, dtype=np.float32)

    rewards_theo = manual_reward_trace(make_env_train_fixed(), actions)
    np.testing.assert_allclose(rewards_env, rewards_theo, rtol=1e-10, atol=1e-10)

# ------------------------------------------------------------
# 2. Reset determinism ---------------------------------------
# ------------------------------------------------------------

def test_reset_deterministic():
    env1 = make_env_train_fixed()
    env2 = make_env_train_fixed()
    obs1, info1 = env1.reset(seed=123)
    obs2, info2 = env2.reset(seed=123)
    np.testing.assert_array_equal(obs1, obs2)
    assert info1["idx"] == info2["idx"] == 0

# ------------------------------------------------------------
# 3. Low‑vol penalty check ------------------------------------
# ------------------------------------------------------------

def test_lowvol_penalty():
    """On a LowVolFlag bar, trading reduces reward by λ compared to Hold."""
    env = make_env_train_fixed(max_steps=500, leverage=0.0)
    obs, _ = env.reset(seed=0)

    # Advance to first LowVol bar
    while env._features[env._idx, 6] != 1:
        obs, _, term, trunc, _ = env.step(0)
        assert not term and not trunc, "No LowVolFlag within 500 steps"

    idx_flag = env._idx
    λ        = env._lambda

    # ----- Case A: Hold -----
    _, r_hold, _, _, _ = env.step(0)

    # ----- Case B: Buy (trade) -----
    env2 = make_env_train_fixed(max_steps=500, leverage=0.0)
    env2.reset(seed=0)
    while env2._idx < idx_flag:
        env2.step(0)
    _, r_trade, _, _, _ = env2.step(1)

    # Reward with trade should be exactly lower by λ
    fee = env._commission_taker
    slip = env._spread_pct
    expected_delta = fee + slip + λ
    np.testing.assert_allclose(r_hold - r_trade, expected_delta,
                               rtol=0, atol=1e-8)


# ------------------------------------------------------------
# 4. Overflow guard -------------------------------------------
# ------------------------------------------------------------

def test_overflow_raises():
    env = make_env_train_fixed(max_steps=3)
    env.reset(seed=0)
    for _ in range(3):
        env.step(0)
    with pytest.raises(RuntimeError):
        env.step(0)   # one step too many

# ------------------------------------------------------------
# 5. Equity floor termination ---------------------------------
# ------------------------------------------------------------

def test_equity_floor_terminates():
    env = make_env_train_fixed(max_steps=500, leverage=0.0)
    env.reset(seed=0)
    env._equity = 0.49          # already below 0.5 floor
    env._position = 0           # no trade needed
    _, _, term, _, _ = env.step(0)   # Any action; should terminate
    assert term, "Episode should terminate when equity is below drawdown floor"

