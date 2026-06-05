import hashlib
import logging
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parents[2] / "data"

logger = logging.getLogger(__name__)


def _cache_path(tickers: list[str], start: str, end: str) -> Path:
    key = "_".join(sorted(tickers)) + f"_{start}_{end}"
    h = hashlib.md5(key.encode()).hexdigest()[:8]
    return DATA_DIR / f"prices_{h}.parquet"


def _fetch_yfinance(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    import yfinance as yf
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"]
    else:
        close = raw[["Close"]]
        close.columns = tickers
    close = close.dropna(how="all")
    return close


def _fetch_stooq(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    from pandas_datareader import data as pdr
    frames = {}
    for t in tickers:
        try:
            df = pdr.DataReader(t, "stooq", start=start, end=end)
            frames[t] = df["Close"].sort_index()
        except Exception as e:
            logger.warning("Stooq failed for %s: %s", t, e)
    if not frames:
        raise RuntimeError("All data sources failed")
    return pd.DataFrame(frames).dropna(how="all")


def download_prices(
    tickers: list[str],
    start: str = "2010-01-01",
    end: str = "2024-12-31",
    use_cache: bool = True,
) -> pd.DataFrame:
    DATA_DIR.mkdir(exist_ok=True)
    cache = _cache_path(tickers, start, end)

    if use_cache and cache.exists():
        logger.info("Loading from cache: %s", cache)
        return pd.read_parquet(cache)

    logger.info("Downloading %d tickers from %s to %s", len(tickers), start, end)
    try:
        close = _fetch_yfinance(tickers, start, end)
    except Exception as e:
        logger.warning("yfinance failed (%s), trying Stooq fallback", e)
        close = _fetch_stooq(tickers, start, end)

    # Drop tickers with excessive NaNs (>10% missing)
    threshold = 0.10
    valid = close.columns[close.isna().mean() < threshold]
    if len(valid) < len(tickers):
        dropped = set(tickers) - set(valid)
        logger.warning("Dropping tickers with >10%% missing data: %s", dropped)
    close = close[valid].dropna()

    log_prices = np.log(close)
    log_prices.to_parquet(cache)
    logger.info("Saved to cache: %s", cache)
    return log_prices


# Predefined sector universes
XLF_TICKERS = ["JPM", "BAC", "WFC", "GS", "MS", "C", "BK", "STT"]
XLE_TICKERS = ["XOM", "CVX", "COP", "EOG", "SLB", "PSX"]

# Chan's canonical commodity-currency ETF pair (Australia/Canada)
COMMODITY_ETF_TICKERS = ["EWA", "EWC"]

SECTOR_BUCKETS = {
    "financials": XLF_TICKERS,
    "energy": XLE_TICKERS,
}

# Extended buckets including commodity ETFs (all-in-one universe for screening)
EXTENDED_BUCKETS = {
    "financials": XLF_TICKERS,
    "energy": XLE_TICKERS,
    "commodity_etf": COMMODITY_ETF_TICKERS,
}
