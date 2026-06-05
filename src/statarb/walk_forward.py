import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .cointegration import adf_test
from .kalman import KalmanHedgeRatio, calibrate_R
from .signals import compute_zscore_kalman, pairs_strategy
from .backtest import run_backtest
from .metrics import compute_metrics

logger = logging.getLogger(__name__)


@dataclass
class WalkForwardFold:
    fold_idx: int
    formation_start: pd.Timestamp
    formation_end: pd.Timestamp
    trading_start: pd.Timestamp
    trading_end: pd.Timestamp
    pair: tuple[str, str]
    cointegrated: bool
    half_life: float | None
    sharpe: float | None
    backtest_df: pd.DataFrame | None
    metrics: dict | None


def walk_forward_backtest(
    log_prices: pd.DataFrame,
    pairs: list[tuple[str, str]],
    formation_days: int = 252,
    trading_days: int = 63,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
    costs_bps: float = 5.0,
    min_coint_p: float = 0.05,
    benchmark_returns: pd.Series | None = None,
) -> tuple[list[WalkForwardFold], dict]:
    all_folds: list[WalkForwardFold] = []
    all_returns: list[pd.Series] = []

    dates = log_prices.index
    n_total = len(dates)
    window_size = formation_days + trading_days

    fold_starts = list(range(0, n_total - window_size + 1, trading_days))
    logger.info("Running walk-forward: %d folds x %d pairs", len(fold_starts), len(pairs))

    for pair in pairs:
        t1, t2 = pair
        if t1 not in log_prices.columns or t2 not in log_prices.columns:
            logger.warning("Pair %s/%s not in price data, skipping", t1, t2)
            continue

        log_y = log_prices[t1]
        log_x = log_prices[t2]

        for fold_idx, start_idx in enumerate(fold_starts):
            form_end_idx = start_idx + formation_days
            trade_end_idx = form_end_idx + trading_days

            if trade_end_idx > n_total:
                break

            form_slice = slice(start_idx, form_end_idx)
            trade_slice = slice(form_end_idx, trade_end_idx)

            ly_form = log_y.iloc[form_slice]
            lx_form = log_x.iloc[form_slice]
            ly_trade = log_y.iloc[trade_slice]
            lx_trade = log_x.iloc[trade_slice]

            fold = WalkForwardFold(
                fold_idx=fold_idx,
                formation_start=dates[start_idx],
                formation_end=dates[form_end_idx - 1],
                trading_start=dates[form_end_idx],
                trading_end=dates[trade_end_idx - 1],
                pair=pair,
                cointegrated=False,
                half_life=None,
                sharpe=None,
                backtest_df=None,
                metrics=None,
            )

            # Step 1: cointegration check on formation window
            from statsmodels.tsa.stattools import coint
            try:
                _, p_val, _ = coint(ly_form.values, lx_form.values)
            except Exception as e:
                logger.debug("Coint failed fold %d pair %s/%s: %s", fold_idx, t1, t2, e)
                all_folds.append(fold)
                continue

            if p_val >= min_coint_p:
                all_folds.append(fold)
                continue
            fold.cointegrated = True

            # Step 2: OLS spread for OU diagnostics (half-life reporting only)
            from numpy.linalg import lstsq
            from .ou_process import fit_ou_process
            X_form = np.column_stack([lx_form.values, np.ones(len(lx_form))])
            coeffs, _, _, _ = lstsq(X_form, ly_form.values, rcond=None)
            ols_beta, ols_alpha = coeffs
            ols_spread_form = ly_form.values - ols_beta * lx_form.values - ols_alpha

            try:
                ou_params = fit_ou_process(ols_spread_form)
                fold.half_life = ou_params.half_life
            except Exception:
                fold.half_life = None

            # Step 3: Kalman warm-up on formation window — collect for sigma_z calibration
            R_calib = calibrate_R(ly_form, lx_form)
            kf = KalmanHedgeRatio(delta=1e-4, R=R_calib)
            innov_form, istd_form = [], []
            for y_t, x_t in zip(ly_form.values, lx_form.values):
                _, _, inn = kf.update(float(y_t), float(x_t))
                innov_form.append(inn)
                istd_form.append(np.sqrt(kf.last_S))

            # sigma_z corrects for P_ss inflation in S_t (z-score calibration)
            warmup_n = max(50, len(innov_form) // 10)
            form_z = np.array(innov_form[warmup_n:]) / np.array(istd_form[warmup_n:])
            sigma_z = max(0.05, float(np.std(form_z)))

            # Step 4: Kalman on trading window — collect innovations + sqrt(S_t)
            betas_trade, innovations_trade, innov_stds_trade = [], [], []
            for y_t, x_t in zip(ly_trade.values, lx_trade.values):
                beta, _, innov = kf.update(float(y_t), float(x_t))
                betas_trade.append(beta)
                innovations_trade.append(innov)
                innov_stds_trade.append(np.sqrt(kf.last_S))

            betas_trade = np.array(betas_trade)
            innov_series = pd.Series(innovations_trade, index=ly_trade.index)
            innov_std_series = pd.Series(innov_stds_trade, index=ly_trade.index)

            # Step 5: z-score normalized by formation sigma_z → std~1 for entry_z=2 to work
            z_raw = compute_zscore_kalman(innov_series, innov_std_series)
            zscore = z_raw / sigma_z
            positions = pairs_strategy(zscore, entry_z=entry_z, exit_z=exit_z)

            # Step 6: backtest on trading window
            bt = run_backtest(ly_trade, lx_trade, betas_trade, positions, costs_bps=costs_bps)

            m = compute_metrics(bt, benchmark_returns)

            fold.sharpe = m["sharpe_ratio"]
            fold.backtest_df = bt
            fold.metrics = m

            all_returns.append(bt["pnl_net"])
            all_folds.append(fold)

    # Aggregate across all trading windows
    # Sum same-day returns from multiple pairs (portfolio-level aggregation)
    if all_returns:
        combined_returns = (
            pd.concat(all_returns)
            .groupby(level=0)
            .sum()
            .sort_index()
        )
        combined_bt = pd.DataFrame({
            "pnl_gross": combined_returns,
            "pnl_net": combined_returns,
            "cost": 0.0,
            "position": 0.0,
            "cumulative_pnl_net": combined_returns.cumsum(),
        }, index=combined_returns.index)
        agg_metrics = compute_metrics(combined_bt, benchmark_returns)
    else:
        agg_metrics = {}

    return all_folds, agg_metrics
