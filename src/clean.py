import os
import yaml
import numpy as np
import pandas as pd
from loguru import logger


def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def clean_prices(df: pd.DataFrame, calendar: pd.DatetimeIndex) -> tuple[pd.DataFrame, dict]:
    """
    Executes Section 4.1 cleaning stages on raw market price data:
    1. Structural alignment to exchange trading calendar
    2. Missing-value handling (ffill & bfill)
    3. Duplicate removal
    4. Outlier detection via rolling z-score on log returns
    5. OHLC invariant validation check
    """
    stats = {}
    rows_initial = len(df)

    # Ensure DatetimeIndex format
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    # --- Stage 1 & 3: Calendar Alignment & De-duplication ---
    df = df[~df.index.duplicated(keep="last")]
    if calendar is not None and len(calendar) > 0:
        df = df.reindex(calendar)

    # --- Stage 2: Missing Value Imputation ---
    # Impute prices safely with forward-fill and backward-fill
    df = df.ffill().bfill()

    # --- Stage 5: OHLC Invariant Validation ---
    bad_rows = pd.Series(False, index=df.index)

    if isinstance(df.columns, pd.MultiIndex):
        # Determine level structure dynamically
        level_0_vals = list(df.columns.levels[0])
        metrics = ["Open", "High", "Low", "Close"]

        if any(m in level_0_vals for m in metrics):
            # Structure: (Metric, Ticker)
            tickers = df.columns.levels[1]
            for t in tickers:
                try:
                    o, h, l, c = df[("Open", t)], df[("High", t)], df[("Low", t)], df[("Close", t)]
                    invalid = (l > o) | (l > c) | (l > h) | (h < o) | (h < c)
                    bad_rows = bad_rows | invalid.fillna(False)
                except KeyError:
                    continue
        else:
            # Structure: (Ticker, Metric)
            tickers = df.columns.levels[0]
            for t in tickers:
                try:
                    o, h, l, c = df[(t, "Open")], df[(t, "High")], df[(t, "Low")], df[(t, "Close")]
                    invalid = (l > o) | (l > c) | (l > h) | (h < o) | (h < c)
                    bad_rows = bad_rows | invalid.fillna(False)
                except KeyError:
                    continue

    stats["ohlc_violations"] = int(bad_rows.sum())
    if stats["ohlc_violations"] > 0:
        logger.warning(f"Quarantining {stats['ohlc_violations']} rows failing OHLC invariant checks.")
        df.loc[bad_rows, :] = np.nan
        df = df.ffill().bfill()

    # --- Stage 4: Outlier Detection ---
    try:
        if isinstance(df.columns, pd.MultiIndex):
            adj_close = df["Adj Close"] if "Adj Close" in df.columns else df["Close"]
        else:
            adj_close = df[["Adj Close"]] if "Adj Close" in df.columns else df[["Close"]]

        log_ret = np.log(adj_close.astype(float)).diff()
        rolling_mean = log_ret.rolling(63, min_periods=5).mean()
        rolling_std = log_ret.rolling(63, min_periods=5).std()
        z_scores = (log_ret - rolling_mean) / rolling_std.replace(0, np.nan)

        stats["outliers_flagged"] = int((z_scores.abs() > 5).sum().sum())
    except Exception as e:
        logger.warning(f"Outlier detection skipped due to layout: {e}")
        stats["outliers_flagged"] = 0

    df_cleaned = df.dropna(how="all")

    stats["rows_in"] = rows_initial
    stats["rows_out"] = len(df_cleaned)

    return df_cleaned, stats


def clean_macro(macro_df: pd.DataFrame, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    """Cleans macroeconomic indicator data."""
    macro_df = macro_df[~macro_df.index.duplicated(keep="last")]
    if calendar is not None and len(calendar) > 0:
        macro_df = macro_df.reindex(calendar)
    return macro_df.ffill().bfill()


def main():
    logger.info("Starting Data Cleaning Pipeline (Section 4)...")

    config = load_config()
    raw_dir = config["data"]["raw_dir"]
    processed_dir = config["data"]["processed_dir"]
    os.makedirs(processed_dir, exist_ok=True)

    prices_path = os.path.join(raw_dir, "prices.parquet")
    macro_path = os.path.join(raw_dir, "macro.parquet")

    if not os.path.exists(prices_path):
        logger.error(f"Raw prices file missing at {prices_path}. Run ingest.py first!")
        return

    prices_df = pd.read_parquet(prices_path)
    trading_calendar = prices_df.dropna(how="all").index

    logger.info("Cleaning market prices panel...")
    cleaned_prices, price_stats = clean_prices(prices_df, trading_calendar)

    cleaned_macro = pd.DataFrame()
    if os.path.exists(macro_path):
        logger.info("Cleaning macro indicators dataset...")
        macro_df = pd.read_parquet(macro_path)
        cleaned_macro = clean_macro(macro_df, trading_calendar)

    # Saved with exact filename expected by features.py
    cleaned_prices_path = os.path.join(processed_dir, "clean_prices.parquet")
    cleaned_prices.to_parquet(cleaned_prices_path)
    logger.success(f"Cleaned prices saved to {cleaned_prices_path} with {len(cleaned_prices)} rows!")

    if not cleaned_macro.empty:
        cleaned_macro_path = os.path.join(processed_dir, "clean_macro.parquet")
        cleaned_macro.to_parquet(cleaned_macro_path)
        logger.success(f"Cleaned macro series saved to {cleaned_macro_path}")

    logger.info("--- DATA QUALITY AUDIT REPORT ---")
    logger.info(f"Rows In: {price_stats['rows_in']} | Rows Out: {price_stats['rows_out']}")
    logger.info(f"OHLC Invariant Violations: {price_stats['ohlc_violations']}")
    logger.info(f"Return Outliers Flagged (>5 std): {price_stats['outliers_flagged']}")
    logger.info("Data Cleaning Pipeline completed successfully!")


if __name__ == "__main__":
    main()
    