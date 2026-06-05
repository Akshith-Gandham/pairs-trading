import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from statarb.kalman import KalmanHedgeRatio, kalman_spread


def generate_synthetic_pair(n=500, true_beta=1.5, true_alpha=0.2, seed=42):
    rng = np.random.default_rng(seed)
    x = np.cumsum(rng.normal(0, 0.01, n))
    noise = rng.normal(0, 0.005, n)
    y = true_beta * x + true_alpha + noise
    return y, x

    # After warmup (first 300 obs), beta should be within 10% of true value
    late_beta = np.mean(betas[300:])
    assert abs(late_beta - true_beta) / true_beta < 0.10, (
        f"Kalman beta {late_beta:.4f} too far from true {true_beta}"
    )


def _run_kalman(y, x, delta=1e-4, R=0.001):
    kf = KalmanHedgeRatio(delta=delta, R=R)
    betas, alphas, innovations = [], [], []
    for y_t, x_t in zip(y, x):
        beta, alpha, innov = kf.update(float(y_t), float(x_t))
        betas.append(beta)
        alphas.append(alpha)
        innovations.append(innov)
    return np.array(betas), np.array(alphas), np.array(innovations)


def test_kalman_beta_convergence_fixed():
    true_beta = 1.5
    y, x = generate_synthetic_pair(n=500, true_beta=true_beta)
    betas, _, _ = _run_kalman(y, x)

    late_beta = np.mean(betas[300:])
    assert abs(late_beta - true_beta) / true_beta < 0.10, (
        f"Kalman beta {late_beta:.4f} too far from true {true_beta}"
    )


def test_kalman_innovations_near_zero_mean():
    y, x = generate_synthetic_pair(n=500)
    _, _, innovations = _run_kalman(y, x)

    # After warmup, innovations should have near-zero mean
    assert abs(np.mean(innovations[200:])) < 0.05, (
        f"Late innovations mean {np.mean(innovations[200:]):.4f} too large"
    )


def test_kalman_reset():
    kf = KalmanHedgeRatio()
    y, x = generate_synthetic_pair(n=100)
    for y_t, x_t in zip(y, x):
        kf.update(float(y_t), float(x_t))

    kf.reset()
    assert np.allclose(kf.x, [1.0, 0.0])
    assert np.allclose(kf.P, np.eye(2))


def test_kalman_returns_three_values():
    kf = KalmanHedgeRatio()
    result = kf.update(1.0, 1.0)
    assert len(result) == 3


def test_delta_affects_adaptation_speed():
    true_beta_pre = 1.0
    true_beta_post = 2.0
    n_pre, n_post = 200, 200
    rng = np.random.default_rng(0)

    x = np.cumsum(rng.normal(0, 0.01, n_pre + n_post))
    y = np.concatenate([
        true_beta_pre * x[:n_pre] + rng.normal(0, 0.005, n_pre),
        true_beta_post * x[n_pre:] + rng.normal(0, 0.005, n_post),
    ])

    betas_slow, _, _ = _run_kalman(y, x, delta=1e-5)
    betas_fast, _, _ = _run_kalman(y, x, delta=1e-3)

    # Fast delta should track the post-break beta more closely
    late_slow = np.mean(betas_slow[-50:])
    late_fast = np.mean(betas_fast[-50:])

    assert abs(late_fast - true_beta_post) < abs(late_slow - true_beta_post), (
        f"Fast delta {late_fast:.3f} should be closer to {true_beta_post} than slow {late_slow:.3f}"
    )
