import os
import yaml
import numpy as np
import pandas as pd
from loguru import logger


def load_config(config_path="config.yaml"):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config path missing: {config_path}")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def compute_wilder_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Computes RSI using Wilder's Exponential Moving Average.
    Matches Bloomberg/TradingView calculations (standard SMA understates momentum).
    """
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # Wilder's smoothing uses alpha = 1 / period
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()

    rs = avg_gain / (avg_loss + 1e-9)
    return 100.0 - (100.0 / (1.0 + rs))


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates technical indicators on a sorted single-ticker DataFrame.
    """
    px_col = "Adj Close" if "Adj Close" in df.columns else "Close"
    px = df[px_col].astype(float)

    # Log returns over multiple lookback windows
    df["ret_1d"] = np.log(px / px.shift(1))
    df["ret_5d"] = np.log(px / px.shift(5))
    df["ret_21d"] = np.log(px / px.shift(21))

    # Volatility and Moving Averages
    df["sma_20"] = px.rolling(window=20, min_periods=15).mean()
    df["vol_20d"] = df["ret_1d"].rolling(window=20, min_periods=15).std() * np.sqrt(252)

    # Momentum
    df["rsi_14"] = compute_wilder_rsi(px, period=14)

    # Distance to moving average feature (z-score normalized ratio)
    df["dist_sma_20"] = (px - df["sma_20"]) / (px.rolling(20).std() + 1e-8)

    return df


def process_panel_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Processes both flat single-asset series and multi-ticker panel DataFrames.
    """
    # Normalize long-format structures
    if isinstance(df.columns, pd.MultiIndex):
        df = df.stack(level=-1).reset_index()

    if "Date" in df.columns:
        df = df.sort_values("Date")
    elif isinstance(df.index, pd.DatetimeIndex):
        df = df.sort_index()

    if "Ticker" in df.columns or "symbol" in df.columns:
        ticker_col = "Ticker" if "Ticker" in df.columns else "symbol"
        # Efficient vector grouping without breaking index tracking
        processed = (
            df.groupby(ticker_col, group_keys=False)
            .apply(extract_features)
            .reset_index(drop=True)
        )
    else:
        processed = extract_features(df)

    # Clean initial warm-up NaNs created by lagging windows
    initial_rows = len(processed)
    processed = processed.dropna(subset=["ret_21d", "rsi_14"]).reset_index(drop=True)
    logger.debug(f"Trimmed {initial_rows - len(processed)} warm-up rows post-feature generation.")

    return processed


def main():
    logger.info("Initializing feature transformation execution...")
    cfg = load_config()

    proc_dir = cfg["data"]["processed_dir"]
    in_file = os.path.join(proc_dir, "clean_prices.parquet")
    out_file = os.path.join(proc_dir, "features.parquet")

    if not os.path.exists(in_file):
        logger.error(f"Required cleaned price file not found: {in_file}")
        return

    raw_prices = pd.read_parquet(in_file)
    logger.info(f"Loaded input panel: shape={raw_prices.shape}")

    feature_matrix = process_panel_features(raw_prices)

    feature_matrix.to_parquet(out_file, index=False)
    logger.info(f"Successfully exported feature matrix ({len(feature_matrix)} rows) -> {out_file}")


if __name__ == "__main__":
    main()