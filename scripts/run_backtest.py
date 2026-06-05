import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from statarb.data import download_prices, SECTOR_BUCKETS, EXTENDED_BUCKETS
from statarb.cointegration import find_cointegrated_pairs, adf_test, compute_ols_spread
from statarb.ou_process import fit_ou_process
from statarb.kalman import KalmanHedgeRatio, kalman_spread, calibrate_R
from statarb.signals import compute_zscore_kalman, pairs_strategy
from statarb.backtest import run_backtest
from statarb.metrics import compute_metrics, print_metrics
from statarb.walk_forward import walk_forward_backtest

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def run_single_pair(
    pair: tuple[str, str],
    start: str,
    end: str,
    entry_z: float,
    exit_z: float,
    costs_bps: float,
    formation_frac: float = 0.5,
):
    t1, t2 = pair
    logger.info("Running backtest: %s / %s  [%s -> %s]", t1, t2, start, end)

    # Download
    log_prices = download_prices([t1, t2, "SPY"], start=start, end=end)
    log_y = log_prices[t1]
    log_x = log_prices[t2]

    # Split formation / trading
    n = len(log_y)
    split = int(n * formation_frac)
    ly_form, lx_form = log_y.iloc[:split], log_x.iloc[:split]
    ly_trade, lx_trade = log_y.iloc[split:], log_x.iloc[split:]

    print(f"\nFormation: {ly_form.index[0].date()} -> {ly_form.index[-1].date()} ({len(ly_form)} days)")
    print(f"Trading:   {ly_trade.index[0].date()} -> {ly_trade.index[-1].date()} ({len(ly_trade)} days)")

    # Cointegration check
    from statsmodels.tsa.stattools import coint
    _, p_val, _ = coint(ly_form.values, lx_form.values)
    print(f"\nCointegration p-value (formation): {p_val:.4f}  {'PASS' if p_val < 0.05 else 'FAIL'}")

    # OLS spread + OU diagnostics on formation (for reporting; not used for z-score)
    from numpy.linalg import lstsq
    X = np.column_stack([lx_form.values, np.ones(len(lx_form))])
    coeffs, _, _, _ = lstsq(X, ly_form.values, rcond=None)
    ols_beta, ols_alpha = coeffs
    spread_form = ly_form.values - ols_beta * lx_form.values - ols_alpha

    ou = fit_ou_process(spread_form)
    print(f"\nOU Parameters (formation window — OLS spread):")
    print(f"  kappa={ou.kappa:.4f}  mu={ou.mu:.4f}  sigma={ou.sigma:.4f}")
    print(f"  sigma_eq={ou.sigma_eq:.4f}  half-life={ou.half_life:.1f} days")

    adf = adf_test(pd.Series(spread_form))
    print(f"\nADF Test (formation spread):")
    print(f"  statistic={adf['adf_statistic']:.4f}  p={adf['p_value']:.4f}  {'stationary' if adf['stationary_5pct'] else 'NOT stationary'}")

    # Kalman warm-up on formation — collect innovations + sqrt(S_t) for calibration
    R_calib = calibrate_R(ly_form, lx_form)
    kf = KalmanHedgeRatio(delta=1e-4, R=R_calib)
    innov_form, istd_form = [], []
    for y_t, x_t in zip(ly_form.values, lx_form.values):
        _, _, inn = kf.update(float(y_t), float(x_t))
        innov_form.append(inn)
        istd_form.append(np.sqrt(kf.last_S))
    print(f"\nKalman (after formation warm-up): beta={kf.x[0]:.4f}  alpha={kf.x[1]:.4f}")

    # sigma_z: formation-period z-score std (corrects for P_ss inflation in S_t)
    # Without this, e_t/sqrt(S_t) has std << 1 because P_ss*H^2 >> actual innovation variance.
    warmup_n = max(50, len(innov_form) // 10)
    form_z_arr = np.array(innov_form[warmup_n:]) / np.array(istd_form[warmup_n:])
    sigma_z = max(0.05, float(np.std(form_z_arr)))
    print(f"  Formation z-score std (sigma_z): {sigma_z:.4f}  "
          f"(effective entry at raw |z|={entry_z * sigma_z:.4f})")

    # Kalman on trading window — collect innovations AND their predicted std sqrt(S_t)
    betas_trade, innovations_trade, innov_stds_trade = [], [], []
    for y_t, x_t in zip(ly_trade.values, lx_trade.values):
        beta, _, innov = kf.update(float(y_t), float(x_t))
        betas_trade.append(beta)
        innovations_trade.append(innov)
        innov_stds_trade.append(np.sqrt(kf.last_S))

    betas_arr = np.array(betas_trade)
    innov_series = pd.Series(innovations_trade, index=ly_trade.index)
    innov_std_series = pd.Series(innov_stds_trade, index=ly_trade.index)

    print(f"\nKalman beta range (trading): [{betas_arr.min():.4f}, {betas_arr.max():.4f}]  "
          f"drift={betas_arr.max()-betas_arr.min():.4f}")

    # Z-score: e_t/sqrt(S_t) normalized by formation sigma_z -> std~1 in formation period
    z_raw = compute_zscore_kalman(innov_series, innov_std_series)
    zscore = z_raw / sigma_z
    print(f"Z-score stats: mean={zscore.mean():.3f}  std={zscore.std():.3f}  "
          f"max={zscore.max():.2f}  min={zscore.min():.2f}")
    positions = pairs_strategy(zscore, entry_z=entry_z, exit_z=exit_z)
    n_trades = (positions.diff().abs() > 0).sum() // 2
    print(f"\nSignals: {(positions != 0).sum()} days in position  ~{n_trades} round trips")

    # Backtest
    bt = run_backtest(ly_trade, lx_trade, betas_arr, positions, costs_bps=costs_bps)

    # SPY benchmark
    spy_returns = None
    if "SPY" in log_prices.columns:
        spy_log = log_prices["SPY"].iloc[split:]
        spy_returns = spy_log.diff()

    metrics = compute_metrics(bt, spy_returns)
    print_metrics(metrics)

    return bt, metrics


def run_sector_screen(start: str, end: str, significance: float = 0.05):
    all_tickers = list(set(
        EXTENDED_BUCKETS["financials"]
        + EXTENDED_BUCKETS["energy"]
        + EXTENDED_BUCKETS["commodity_etf"]
        + ["SPY"]
    ))
    logger.info("Downloading %d tickers...", len(all_tickers))
    log_prices = download_prices(all_tickers, start=start, end=end)

    print(f"\nLoaded {len(log_prices)} trading days, {len(log_prices.columns)-1} tickers")

    pairs = find_cointegrated_pairs(log_prices, significance=significance, buckets=EXTENDED_BUCKETS)

    if not pairs:
        print(f"No cointegrated pairs found at p < {significance}")
        return []

    print(f"\nCointegrated pairs (p < {significance}):")
    print(f"  {'Pair':<15} {'Bucket':<14} {'p-value':<10} {'t-stat':<10}")
    print(f"  {'-'*15} {'-'*14} {'-'*10} {'-'*10}")
    for p in pairs:
        t1, t2 = p["pair"]
        print(f"  {t1}/{t2:<12} {p['bucket']:<14} {p['p_value']:.4f}     {p['t_stat']:.4f}")

    return pairs


def main():
    parser = argparse.ArgumentParser(description="Statistical Arbitrage Backtest")
    parser.add_argument("--pair", nargs=2, metavar=("TICKER1", "TICKER2"),
                        help="Specific pair to trade (e.g. GS MS)")
    parser.add_argument("--start", default="2010-01-01")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--entry-z", type=float, default=2.0)
    parser.add_argument("--exit-z", type=float, default=0.5)
    parser.add_argument("--costs-bps", type=float, default=5.0)
    parser.add_argument("--walk-forward", action="store_true",
                        help="Run walk-forward validation")
    parser.add_argument("--screen", action="store_true",
                        help="Screen universe for cointegrated pairs")
    parser.add_argument("--significance", type=float, default=0.05,
                        help="Cointegration p-value threshold (default 0.05)")
    args = parser.parse_args()

    if args.screen or (not args.pair and not args.walk_forward):
        pairs_found = run_sector_screen(args.start, args.end, significance=args.significance)

    if args.walk_forward:
        all_tickers = list(set(
            EXTENDED_BUCKETS["financials"] + EXTENDED_BUCKETS["energy"]
            + EXTENDED_BUCKETS["commodity_etf"] + ["SPY"]
        ))
        log_prices = download_prices(all_tickers, start=args.start, end=args.end)
        pairs_found = find_cointegrated_pairs(
            log_prices, significance=args.significance, buckets=EXTENDED_BUCKETS
        )

        if not pairs_found:
            print("No cointegrated pairs for walk-forward")
            return

        top_pairs = [p["pair"] for p in pairs_found[:3]]
        spy_returns = log_prices["SPY"].diff() if "SPY" in log_prices.columns else None

        print(f"\nRunning walk-forward on {len(top_pairs)} pairs: {top_pairs}")
        folds, agg = walk_forward_backtest(
            log_prices,
            top_pairs,
            formation_days=252,
            trading_days=63,
            entry_z=args.entry_z,
            exit_z=args.exit_z,
            costs_bps=args.costs_bps,
            benchmark_returns=spy_returns,
        )

        traded = [f for f in folds if f.sharpe is not None]
        print(f"\nWalk-Forward Results: {len(traded)}/{len(folds)} folds traded")
        if traded:
            sharpes = [f.sharpe for f in traded]
            print(f"  Per-fold Sharpe: mean={np.mean(sharpes):.3f}  "
                  f"min={np.min(sharpes):.3f}  max={np.max(sharpes):.3f}")
        if agg:
            print(f"\nAggregate OOS Metrics:")
            print_metrics(agg)

    elif args.pair:
        run_single_pair(
            tuple(args.pair),
            start=args.start,
            end=args.end,
            entry_z=args.entry_z,
            exit_z=args.exit_z,
            costs_bps=args.costs_bps,
        )
    else:
        # Default: screen + run on best pair
        all_tickers = list(set(
            EXTENDED_BUCKETS["financials"] + EXTENDED_BUCKETS["energy"]
            + EXTENDED_BUCKETS["commodity_etf"] + ["SPY"]
        ))
        log_prices = download_prices(all_tickers, start=args.start, end=args.end)
        pairs_found = find_cointegrated_pairs(
            log_prices, significance=args.significance, buckets=EXTENDED_BUCKETS
        )
        if pairs_found:
            best_pair = pairs_found[0]["pair"]
            print(f"\nUsing best pair: {best_pair[0]}/{best_pair[1]}")
            run_single_pair(
                best_pair,
                start=args.start,
                end=args.end,
                entry_z=args.entry_z,
                exit_z=args.exit_z,
                costs_bps=args.costs_bps,
            )


if __name__ == "__main__":
    main()
