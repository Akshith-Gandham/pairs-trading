from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize


@dataclass
class OUParams:
    kappa: float      # mean-reversion speed
    mu: float         # long-run mean
    sigma: float      # diffusion coefficient
    half_life: float  # log(2) / kappa, in same units as dt
    sigma_eq: float   # stationary std = sigma / sqrt(2*kappa)


def fit_ou_process(spread: pd.Series | np.ndarray) -> OUParams:
    # dX = kappa*(mu - X)*dt + sigma*dW, dt=1 (daily)
    if isinstance(spread, pd.Series):
        x = spread.dropna().values
    else:
        x = np.asarray(spread)
        x = x[~np.isnan(x)]

    dt = 1.0  # daily observations
    n = len(x)

    def neg_log_likelihood(params: np.ndarray) -> float:
        kappa, mu, sigma = params
        if sigma <= 0 or kappa <= 0:
            return 1e10
        e = np.exp(-kappa * dt)
        pred = mu + e * (x[:-1] - mu)
        # Conditional variance (one-step ahead)
        var = sigma**2 * (1 - e**2) / (2 * kappa)
        if var <= 0:
            return 1e10
        ll = -0.5 * (np.log(2 * np.pi * var) + (x[1:] - pred) ** 2 / var)
        return -ll.sum()

    x0 = np.array([0.1, float(np.mean(x)), float(np.std(x))])
    bounds = [(1e-6, None), (None, None), (1e-6, None)]
    res = minimize(neg_log_likelihood, x0, method="L-BFGS-B", bounds=bounds)

    kappa, mu, sigma = res.x
    half_life = np.log(2) / kappa
    sigma_eq = sigma / np.sqrt(2 * kappa)

    return OUParams(
        kappa=float(kappa),
        mu=float(mu),
        sigma=float(sigma),
        half_life=float(half_life),
        sigma_eq=float(sigma_eq),
    )
