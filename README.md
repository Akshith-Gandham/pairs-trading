# Statistical Arbitrage — Pairs Trading with Kalman Filter

Pairs trading strategy with dynamic hedge ratio estimation via Kalman filter. Static OLS assumes a constant relationship between assets — the Kalman filter adapts continuously without look-ahead bias.

Five look-ahead bias sources addressed:
1. Z-score normalized by formation-window sigma_z (not full-sample mean/std)
2. Cointegration test on formation window only
3. OU params frozen during trading period (diagnostics only)
4. PnL: `positions.shift(1) * returns` — signal at t-1, return at t
5. Kalman state carries over from formation warm-up; fresh per walk-forward fold

## Quick Start

```bash
pip install -r requirements.txt

# Screen universe for cointegrated pairs
python scripts/run_backtest.py --screen

# Backtest canonical EWA/EWC pair (Ernie Chan benchmark)
python scripts/run_backtest.py --pair EWA EWC --start 2010-01-01 --end 2024-12-31

# Relax significance threshold to find more pairs
python scripts/run_backtest.py --screen --significance 0.10

# Walk-forward out-of-sample validation
python scripts/run_backtest.py --walk-forward --significance 0.10
```

## Project Structure

```
src/statarb/
├── data.py          # yfinance download + parquet cache; EXTENDED_BUCKETS incl. EWA/EWC
├── cointegration.py # Engle-Granger screening within sector buckets
├── ou_process.py    # MLE Ornstein-Uhlenbeck fitting (kappa, mu, sigma, half-life, sigma_eq)
├── kalman.py        # KalmanHedgeRatio class + kalman_spread() [returns 4 values]
├── signals.py       # compute_zscore_kalman, pairs_strategy (look-ahead-free)
├── backtest.py      # Dollar-neutral backtest + trade log
├── metrics.py       # Sharpe/Sortino/Calmar/drawdown + cost sensitivity table
└── walk_forward.py  # 252d formation / 63d trading rolling validation
notebooks/
├── 01_screening_diagnostics.ipynb   # pair screen, OU fit, Kalman demo + sigma_z
└── 02_backtest_walkforward.ipynb    # full backtest, metrics, OOS validation
```

## Universe

- **Financials (XLF):** JPM, BAC, WFC, GS, MS, C, BK, STT
- **Energy (XLE):** XOM, CVX, COP, EOG, SLB, PSX
- **Commodity ETFs:** EWA, EWC (Australia/Canada — Ernie Chan's canonical pair)
- Pairs tested within each bucket (avoids multiple-testing inflation from all-vs-all)

## Kalman Filter State Space

```
State:         x_t = [beta_t, alpha_t]^T        (random walk)
Observation:   y_t = [log_x_t, 1] @ x_t + v_t
Process noise: Q = delta/(1-delta) * I           (Chan's parameterization)
Innovation:    e_t = y_t - H_t @ x_{t|t-1}      (look-ahead-free spread)
Inn. variance: S_t = H_t @ P_pred @ H_t.T + R
Z-score:       z_t = e_t / sqrt(S_t) / sigma_z  (sigma_z from formation window)
```

`delta=1e-4` controls adaptation speed. The innovation `e_t` uses only data through `t-1` for the state estimate (look-ahead-free by construction).

### Why sigma_z calibration matters

Raw `e_t / sqrt(S_t)` has std << 1 during the formation period because the Kalman steady-state covariance `P_ss` inflates `S_t` well above the actual innovation variance. Without the `sigma_z` correction, `entry_z=2.0` is never triggered and the strategy generates zero trades.

Fix: compute `sigma_z = std(e_t / sqrt(S_t))` over the formation window (excluding early warmup), then divide: `z_t = e_t / sqrt(S_t) / sigma_z`. This rescales the z-score to std~1 in-sample, making the entry threshold meaningful.

### Why calibrate_R uses diff variance

`R` must match the daily-change variance of the spread (~0.0002), **not** the level variance (~0.016). Using level variance collapses all z-scores toward zero by making `sqrt(S_t)` ~8x too large.

```python
# Correct: daily-change variance
R = np.var(np.diff(OLS_spread))   # ~0.0002

# Wrong: level variance -- kills all trades
R = np.var(OLS_spread)            # ~0.016
```

## Key Metrics Reported

Sharpe, Sortino, Calmar, max drawdown + duration, hit rate, avg holding period, turnover, market beta (market-neutrality check), and a **cost sensitivity table** (Sharpe at 0–20 bps).

## Canonical Results (EWA/EWC, 2010–2024)

| Metric | Value |
|--------|-------|
| Sharpe (net, 5 bps) | ~1.01 |
| Trades | ~26 round trips |
| Hit rate | ~77% |
| Market beta | ~0.0 (market neutral) |
| Breakeven cost | ~17 bps |

## Walk-Forward Validation

252-day formation + 63-day trading windows, rolled by 63 days. Per fold:
1. Cointegration check (Engle-Granger) on formation window
2. OU fit for half-life reporting
3. Kalman warm-up on formation (collect sigma_z)
4. Kalman continues live on trading window with formation sigma_z
5. Strategy + backtest on trading window

Same-day returns across multiple pairs aggregated via `groupby(level=0).sum()` to avoid duplicate-index errors in portfolio combination.

## Tests

```bash
pytest tests/
```

- `test_kalman_convergence.py` — beta converges on synthetic data with known ground truth
- `test_no_lookahead.py` — modifying future data does not change past innovations/signals
