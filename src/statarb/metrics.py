import numpy as np
import pandas as pd


TRADING_DAYS = 252


def _max_drawdown(cum_returns: np.ndarray) -> tuple[float, int]:
    running_max = np.maximum.accumulate(cum_returns)
    drawdown = cum_returns - running_max
    max_dd = float(np.min(drawdown))

    # Duration of longest drawdown
    in_dd = drawdown < 0
    max_dur = 0
    cur_dur = 0
    for d in in_dd:
        if d:
            cur_dur += 1
            max_dur = max(max_dur, cur_dur)
        else:
            cur_dur = 0

    return max_dd, max_dur


def compute_metrics(
    backtest_df: pd.DataFrame,
    benchmark_returns: pd.Series | None = None,
) -> dict:
    from .backtest import compute_trade_log

    ret = backtest_df["pnl_net"].dropna()
    ret_gross = backtest_df["pnl_gross"].dropna()
    n_days = len(ret)
    n_years = n_days / TRADING_DAYS

    total_gross = float(ret_gross.sum())
    total_net = float(ret.sum())
    cagr = float((1 + total_net) ** (1 / n_years) - 1) if n_years > 0 else 0.0

    mu = float(ret.mean())
    sigma = float(ret.std())
    sharpe = float(mu / sigma * np.sqrt(TRADING_DAYS)) if sigma > 0 else 0.0

    downside = ret[ret < 0].std()
    sortino = float(mu / downside * np.sqrt(TRADING_DAYS)) if downside > 0 else 0.0

    cum = (1 + ret).cumprod().values
    max_dd, max_dd_dur = _max_drawdown(cum)
    calmar = float(cagr / abs(max_dd)) if max_dd < 0 else 0.0

    # Trade-level stats
    trade_log = compute_trade_log(backtest_df)
    num_trades = len(trade_log)
    hit_rate = float((trade_log["pnl_net"] > 0).mean()) if num_trades > 0 else 0.0
    avg_hold = float(trade_log["holding_days"].mean()) if num_trades > 0 else 0.0

    # Turnover: mean absolute daily position change, annualized
    pos_changes = backtest_df["position"].diff().abs()
    turnover = float(pos_changes.mean() * TRADING_DAYS)

    # Market neutrality
    market_beta = 0.0
    market_corr = 0.0
    if benchmark_returns is not None:
        common = ret.index.intersection(benchmark_returns.index)
        if len(common) > 30:
            r = ret[common].values
            b = benchmark_returns[common].values
            if np.std(r) > 0 and np.std(b) > 0:
                cov = np.cov(r, b)
                market_beta = float(cov[0, 1] / cov[1, 1]) if cov[1, 1] > 0 else 0.0
                market_corr = float(np.corrcoef(r, b)[0, 1])

    # Cost sensitivity: Sharpe at various cost levels
    cost_sensitivity = {}
    gross_ret = backtest_df["pnl_gross"].dropna()
    costs_raw = backtest_df["cost"].dropna()
    # costs_raw already at 5bps; scale to other levels
    base_bps = 5.0
    for test_bps in [0, 2, 5, 10, 15, 20]:
        if base_bps > 0:
            scale = test_bps / base_bps
        else:
            scale = 0.0
        adj_ret = gross_ret - costs_raw * scale
        mu_adj = float(adj_ret.mean())
        sigma_adj = float(adj_ret.std())
        cost_sensitivity[test_bps] = round(
            float(mu_adj / sigma_adj * np.sqrt(TRADING_DAYS)) if sigma_adj > 0 else 0.0, 3
        )

    return {
        "total_return_gross": round(total_gross, 4),
        "total_return_net": round(total_net, 4),
        "cagr": round(cagr, 4),
        "sharpe_ratio": round(sharpe, 3),
        "sortino_ratio": round(sortino, 3),
        "calmar_ratio": round(calmar, 3),
        "max_drawdown": round(max_dd, 4),
        "max_drawdown_duration_days": max_dd_dur,
        "num_trades": num_trades,
        "hit_rate": round(hit_rate, 3),
        "avg_holding_period_days": round(avg_hold, 1),
        "annualized_turnover": round(turnover, 3),
        "market_beta": round(market_beta, 4),
        "market_correlation": round(market_corr, 4),
        "sharpe_by_cost_bps": cost_sensitivity,
        "n_trading_days": n_days,
    }


def print_metrics(metrics: dict) -> None:
    print("\n" + "=" * 50)
    print("STRATEGY PERFORMANCE SUMMARY")
    print("=" * 50)
    fields = [
        ("Total Return (Gross)", f"{metrics['total_return_gross']:.2%}"),
        ("Total Return (Net)",   f"{metrics['total_return_net']:.2%}"),
        ("CAGR",                 f"{metrics['cagr']:.2%}"),
        ("Sharpe Ratio",         f"{metrics['sharpe_ratio']:.3f}"),
        ("Sortino Ratio",        f"{metrics['sortino_ratio']:.3f}"),
        ("Calmar Ratio",         f"{metrics['calmar_ratio']:.3f}"),
        ("Max Drawdown",         f"{metrics['max_drawdown']:.2%}"),
        ("Max DD Duration",      f"{metrics['max_drawdown_duration_days']} days"),
        ("Num Trades",           str(metrics['num_trades'])),
        ("Hit Rate",             f"{metrics['hit_rate']:.1%}"),
        ("Avg Hold (days)",      f"{metrics['avg_holding_period_days']:.1f}"),
        ("Annualized Turnover",  f"{metrics['annualized_turnover']:.2f}x"),
        ("Market Beta",          f"{metrics['market_beta']:.4f}"),
        ("Market Correlation",   f"{metrics['market_correlation']:.4f}"),
    ]
    for label, val in fields:
        print(f"  {label:<28} {val}")

    print("\n  Cost Sensitivity (Sharpe):")
    for bps, sharpe in metrics["sharpe_by_cost_bps"].items():
        bar = "#" * max(0, int(sharpe * 10))
        print(f"    {bps:>3} bps: {sharpe:>6.3f}  {bar}")
    print("=" * 50)
