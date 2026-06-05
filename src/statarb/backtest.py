import numpy as np
import pandas as pd


def run_backtest(
    log_y: pd.Series,
    log_x: pd.Series,
    betas: np.ndarray | pd.Series,
    positions: pd.Series,
    costs_bps: float = 5.0,
) -> pd.DataFrame:
    idx = log_y.index
    beta_arr = np.asarray(betas) if not isinstance(betas, pd.Series) else betas.values
    pos = positions.reindex(idx).fillna(0).values

    ret_y = np.diff(log_y.values, prepend=np.nan)
    ret_x = np.diff(log_x.values, prepend=np.nan)

    # Shift positions by 1 — trade at close of signal day, return realized next day
    pos_lagged = np.roll(pos, 1)
    pos_lagged[0] = 0.0
    beta_lagged = np.roll(beta_arr, 1)
    beta_lagged[0] = beta_arr[0]

    pnl_gross = pos_lagged * (ret_y - beta_lagged * ret_x)
    pnl_gross[0] = 0.0

    # Transaction costs: applied at position changes (both legs)
    delta_pos = np.diff(pos, prepend=0.0)
    # Cost = |delta_pos| * (1 + |beta|) * costs_bps * 1e-4
    # Factor (1 + beta) accounts for trading both the y and x legs
    cost = np.abs(delta_pos) * (1.0 + np.abs(beta_arr)) * costs_bps * 1e-4
    cost[0] = 0.0

    pnl_net = pnl_gross - cost

    result = pd.DataFrame({
        "position": pos,
        "pnl_gross": pnl_gross,
        "cost": cost,
        "pnl_net": pnl_net,
        "cumulative_pnl_net": np.nancumsum(pnl_net),
    }, index=idx)

    return result


def compute_trade_log(backtest_df: pd.DataFrame) -> pd.DataFrame:
    pos = backtest_df["position"]
    trades = []
    entry_date = None
    entry_side = 0
    running_pnl = 0.0

    for date, row in backtest_df.iterrows():
        p = int(row["position"])
        if entry_side == 0 and p != 0:
            entry_date = date
            entry_side = p
            running_pnl = 0.0
        elif entry_side != 0:
            running_pnl += row["pnl_net"]
            if p == 0:
                trades.append({
                    "entry_date": entry_date,
                    "exit_date": date,
                    "side": entry_side,
                    "pnl_net": running_pnl,
                    "holding_days": (date - entry_date).days,
                })
                entry_side = 0
                entry_date = None
                running_pnl = 0.0

    return pd.DataFrame(trades)
