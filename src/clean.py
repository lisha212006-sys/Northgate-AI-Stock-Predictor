import os
import yaml
import numpy as np
import pandas as pd
from loguru import logger


def load_config(config_path="config.yaml"):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def validate_ohlc(df: pd.DataFrame) -> pd.Series:
    """
    Validates High >= Low, High >= Open/Close, and Low <= Open/Close invariants.
    Handles both flat and multi-index DataFrames without deep nested checks.
    """
    try:
        # If columns are MultiIndex, extract metrics across level 0 or 1
        if isinstance(df.columns, pd.MultiIndex):
            o = df.xs("Open", axis=1, level=0 if "Open" in df.columns.levels[0] else 1)
            h = df.xs("High", axis=1, level=0 if "High" in df.columns.levels[0] else 1)
            l = df.xs("Low", axis=1, level=0 if "Low" in df.columns.levels[0] else 1)
            c = df.xs("Close", axis=1, level=0 if "Close" in df.columns.levels[0] else 1)
        else:
            o, h, l, c = df["Open"], df["High"], df["Low"], df["Close"]

        # Vectorized check for bad prices
        invalid = (l > o) | (l > c) | (l > h) | (h < o) | (h < c)
        return invalid.any(axis=1) if isinstance(invalid, pd.DataFrame) else invalid
    except KeyError as e:
        logger.warning(f"Could not perform complete OHLC validation: missing column {e}")
        return pd.Series(False, index=df.index)


def clean_prices(df: pd.DataFrame, calendar: pd.DatetimeIndex) -> tuple[pd.DataFrame, dict]:
    stats = {"rows_in": len(df)}

    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    # De-duplicate timestamp entries
    df = df[~df.index.duplicated(keep="last")].sort_index()

    # Align with target exchange trading calendar
    if calendar is not None and len(calendar) > 0:
        df = df.reindex(calendar)

    # Forward-fill gaps (max 3 consecutive days); NO backward fill to avoid future look-ahead bias
    df = df.ffill(limit=3)
    
    # Flag and quarantine OHLC invariant failures
    bad_rows = validate_ohlc(df)
    stats["ohlc_violations"] = int(bad_rows.sum())
    
    if stats["ohlc_violations"] > 0:
        logger.warning(f"Quarantining {stats['ohlc_violations']} bad OHLC rows.")
        df.loc[bad_rows, :] = np.nan
        df = df.ffill(limit=2)

    # Rolling Z-score on log returns to detect abnormal spikes
    try:
        price_col = "Adj Close" if "Adj Close" in df.columns else "Close"
        px = df[price_col] if not isinstance(df.columns, pd.MultiIndex) else df.xs(price_col, axis=1, level=0 if price_col in df.columns.levels[0] else 1)
        
        log_ret = np.log(px.astype(float)).diff()
        z_scores = (log_ret - log_ret.rolling(63, min_periods=10).mean()) / log_ret.rolling(63, min_periods=10).std()
        
        stats["outliers_flagged"] = int((z_scores.abs() > 5.0).sum().sum())
    except Exception as err:
        logger.debug(f"Skipped outlier computation: {err}")
        stats["outliers_flagged"] = 0

    df_cleaned = df.dropna(how="all")
    stats["rows_out"] = len(df_cleaned)

    return df_cleaned, stats


def clean_macro(macro_df: pd.DataFrame, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    macro_df = macro_df[~macro_df.index.duplicated(keep="last")].sort_index()
    if calendar is not None and len(calendar) > 0:
        macro_df = macro_df.reindex(calendar)
    # Forward fill macro releases (e.g. monthly CPI mapped onto daily grid)
    return macro_df.ffill()


def main():
    logger.info("Executing price and macro cleaning workflow")
    cfg = load_config()
    
    raw_dir = cfg["data"]["raw_dir"]
    proc_dir = cfg["data"]["processed_dir"]
    os.makedirs(proc_dir, exist_ok=True)

    prices_path = os.path.join(raw_dir, "prices.parquet")
    if not os.path.exists(prices_path):
        logger.error(f"Input file missing: {prices_path}")
        return

    prices_df = pd.read_parquet(prices_path)
    calendar = prices_df.dropna(how="all").index

    cleaned_prices, stats = clean_prices(prices_df, calendar)
    cleaned_prices.to_parquet(os.path.join(proc_dir, "clean_prices.parquet"))

    macro_path = os.path.join(raw_dir, "macro.parquet")
    if os.path.exists(macro_path):
        macro_df = pd.read_parquet(macro_path)
        clean_macro(macro_df, calendar).to_parquet(os.path.join(proc_dir, "clean_macro.parquet"))

    logger.info(f"Cleaned dataset: {stats['rows_in']} -> {stats['rows_out']} rows | Bad OHLC: {stats['ohlc_violations']} | Outliers: {stats['outliers_flagged']}")


if __name__ == "__main__":
    main()