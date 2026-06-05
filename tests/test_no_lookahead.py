import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from statarb.kalman import KalmanHedgeRatio
from statarb.signals import pairs_strategy, compute_zscore_ou
from statarb.ou_process import OUParams
from statarb.backtest import run_backtest
import pandas as pd


def test_kalman_innovation_no_lookahead():
    rng = np.random.default_rng(0)
    n = 100
    x = np.cumsum(rng.normal(0, 0.01, n))
    y = 1.5 * x + rng.normal(0, 0.005, n)

    def run_and_get_innovations(y_arr, x_arr):
        kf = KalmanHedgeRatio(delta=1e-4, R=0.001)
        innovations = []
        for y_t, x_t in zip(y_arr, x_arr):
            _, _, innov = kf.update(float(y_t), float(x_t))
            innovations.append(innov)
        return np.array(innovations)

    innov_original = run_and_get_innovations(y, x)

    # Perturb future observations (indices 50 onward)
    y_modified = y.copy()
    y_modified[50:] = y_modified[50:] * 10.0  # large shock to future data

    innov_modified = run_and_get_innovations(y_modified, x)

    # Innovations at t < 50 must be identical (future data can't affect past)
    np.testing.assert_array_almost_equal(
        innov_original[:50], innov_modified[:50],
        decimal=10,
        err_msg="Kalman innovations before t=50 changed when future data was modified — look-ahead bias!"
    )


def test_pnl_uses_lagged_position():
    rng = np.random.default_rng(1)
    n = 200
    idx = pd.date_range("2015-01-01", periods=n, freq="B")
    x = pd.Series(np.cumsum(rng.normal(0, 0.01, n)), index=idx)
    y = pd.Series(1.5 * x.values + rng.normal(0, 0.005, n), index=idx)
    betas = np.full(n, 1.5)

    # Constant long position (all +1)
    positions = pd.Series(np.ones(n), index=idx)

    bt = run_backtest(y, x, betas, positions, costs_bps=0)

    # Day 0 PnL should be 0 (no lagged position yet)
    assert bt["pnl_gross"].iloc[0] == 0.0, "First-day PnL should be 0 (no prior position)"

    # PnL on day 1 = position[0] * return[1], not position[1] * return[1]
    ret_y1 = y.iloc[1] - y.iloc[0]
    ret_x1 = x.iloc[1] - x.iloc[0]
    expected_pnl_day1 = positions.iloc[0] * (ret_y1 - 1.5 * ret_x1)
    assert abs(bt["pnl_gross"].iloc[1] - expected_pnl_day1) < 1e-10, (
        f"Day 1 PnL mismatch: got {bt['pnl_gross'].iloc[1]:.6f}, expected {expected_pnl_day1:.6f}"
    )


def test_zscore_uses_formation_params_only():
    rng = np.random.default_rng(2)
    n = 300
    spread_values = rng.normal(0.5, 0.1, n)
    spread = pd.Series(spread_values)

    # Fit OU params on formation window only (first half)
    from statarb.ou_process import fit_ou_process
    ou_params = fit_ou_process(spread.iloc[:150])

    z = compute_zscore_ou(spread, ou_params)

    # Z-score should use formation-period mu, not full-sample mean
    # If it used full-sample mean, z.mean() ≈ 0; with formation params it can differ
    # The key test: changing future values should NOT change past z-scores
    spread_future_modified = spread.copy()
    spread_future_modified.iloc[200:] = 100.0  # extreme future shock

    z2 = compute_zscore_ou(spread_future_modified, ou_params)

    # Past z-scores (before index 200) must be identical
    pd.testing.assert_series_equal(
        z.iloc[:200], z2.iloc[:200],
        check_names=False,
    )


def test_pairs_strategy_no_internal_normalization():
    rng = np.random.default_rng(3)
    n = 100
    z_values = rng.normal(0, 2.5, n)  # std > 2.0 so we get entries
    z = pd.Series(z_values)

    pos = pairs_strategy(z, entry_z=2.0, exit_z=0.5)

    # If strategy internally re-normalized to N(0,1), far fewer entries would occur
    # With std=2.5, many values exceed 2.0 raw
    n_entries = (pos != 0).sum()
    assert n_entries > 5, f"Expected entries with high-std z-score, got {n_entries}"

    # Scale z-scores down by 10x — should get NO entries (all |z| < 2.0)
    z_small = z * 0.1  # max ~0.25 * 2.5 = 0.625 << 2.0
    pos_small = pairs_strategy(z_small, entry_z=2.0, exit_z=0.5)
    assert (pos_small == 0).all(), "Scaled-down z-scores should yield no trades"
