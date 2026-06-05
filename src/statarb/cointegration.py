from itertools import combinations

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, coint


def find_cointegrated_pairs(
    log_prices: pd.DataFrame,
    significance: float = 0.05,
    buckets: dict[str, list[str]] | None = None,
) -> list[dict]:
    results = []
    tickers = list(log_prices.columns)

    if buckets is not None:
        pair_candidates = []
        for bucket_name, bucket_tickers in buckets.items():
            available = [t for t in bucket_tickers if t in tickers]
            for i, j in combinations(available, 2):
                pair_candidates.append((i, j, bucket_name))
    else:
        pair_candidates = [(tickers[i], tickers[j], "all")
                           for i, j in combinations(range(len(tickers)), 2)]

    for t1, t2, bucket in pair_candidates:
        s1 = log_prices[t1].dropna()
        s2 = log_prices[t2].dropna()
        idx = s1.index.intersection(s2.index)
        s1, s2 = s1[idx], s2[idx]
        if len(s1) < 100:
            continue
        try:
            t_stat, p_value, crit_vals = coint(s1, s2)
        except Exception:
            continue
        if p_value < significance:
            results.append({
                "pair": (t1, t2),
                "bucket": bucket,
                "p_value": p_value,
                "t_stat": t_stat,
                "crit_1pct": crit_vals[0],
                "crit_5pct": crit_vals[1],
                "crit_10pct": crit_vals[2],
                "n_obs": len(s1),
            })

    return sorted(results, key=lambda x: x["p_value"])


def adf_test(spread: pd.Series, name: str = "Spread") -> dict:
    clean = spread.dropna()
    result = adfuller(clean)
    return {
        "name": name,
        "adf_statistic": result[0],
        "p_value": result[1],
        "n_lags": result[2],
        "n_obs": result[3],
        "critical_values": result[4],
        "stationary_5pct": result[1] < 0.05,
    }


def compute_ols_spread(log_y: pd.Series, log_x: pd.Series) -> tuple[pd.Series, float, float]:
    from numpy.linalg import lstsq
    X = np.column_stack([log_x.values, np.ones(len(log_x))])
    coeffs, _, _, _ = lstsq(X, log_y.values, rcond=None)
    beta, alpha = coeffs
    spread = log_y - beta * log_x - alpha
    return spread, beta, alpha
