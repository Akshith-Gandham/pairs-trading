import numpy as np
import pandas as pd

from .ou_process import OUParams


def compute_zscore_ou(
    spread: pd.Series | np.ndarray,
    ou_params: OUParams,
    index: pd.Index | None = None,
) -> pd.Series:
    if isinstance(spread, pd.Series):
        values = spread.values
        idx = spread.index
    else:
        values = np.asarray(spread)
        idx = index if index is not None else pd.RangeIndex(len(values))

    z = (values - ou_params.mu) / ou_params.sigma_eq
    return pd.Series(z, index=idx, name="z_score")


def compute_zscore_kalman(
    innovations: pd.Series | np.ndarray,
    innov_stds: pd.Series | np.ndarray,
    index: pd.Index | None = None,
) -> pd.Series:
    if isinstance(innovations, pd.Series):
        vals = innovations.values
        idx = innovations.index
    else:
        vals = np.asarray(innovations)
        idx = index if index is not None else pd.RangeIndex(len(vals))

    stds = np.asarray(innov_stds) if not isinstance(innov_stds, pd.Series) else innov_stds.values
    z = vals / np.where(stds > 0, stds, np.nan)
    return pd.Series(z, index=idx, name="z_score")


def compute_zscore_rolling(
    spread: pd.Series,
    lookback: int = 60,
) -> pd.Series:
    mu = spread.rolling(lookback, min_periods=lookback // 2).mean()
    sigma = spread.rolling(lookback, min_periods=lookback // 2).std()
    return ((spread - mu) / sigma).rename("z_score_rolling")


def pairs_strategy(
    zscore: pd.Series,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
) -> pd.Series:
    z = zscore.values
    n = len(z)
    position = np.zeros(n)
    current_pos = 0

    for i in range(1, n):
        zi = z[i]
        if np.isnan(zi):
            position[i] = current_pos
            continue
        if current_pos == 0:
            if zi < -entry_z:
                current_pos = 1   # spread too low → long spread
            elif zi > entry_z:
                current_pos = -1  # spread too high → short spread
        elif current_pos == 1 and zi > -exit_z:
            current_pos = 0
        elif current_pos == -1 and zi < exit_z:
            current_pos = 0
        position[i] = current_pos

    return pd.Series(position, index=zscore.index, name="position")
