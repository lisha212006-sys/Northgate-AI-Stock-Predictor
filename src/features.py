import os
import yaml
import numpy as np
import pandas as pd
from loguru import logger


def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Calculates Relative Strength Index (RSI)."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()

    rs = avg_gain / (avg_loss + 1e-8)
    return 100.0 - (100.0 / (1.0 + rs))


def build_features(group: pd.DataFrame) -> pd.DataFrame:
    """Calculates technical indicators per ticker group."""
    df = group.copy().sort_values("Date")

    price_col = "Adj Close" if "Adj Close" in df.columns else "Close"

    # 1. Log returns
    df["ret1"] = np.log(df[price_col] / df[price_col].shift(1))
    df["ret5"] = np.log(df[price_col] / df[price_col].shift(5))
    df["ret21"] = np.log(df[price_col] / df[price_col].shift(21))

    # 2. Moving Averages & Volatility
    df["sma_20"] = df[price_col].rolling(20).mean()
    df["vol_20"] = df["ret1"].rolling(20).std()

    # 3. RSI
    df["rsi_14"] = calculate_rsi(df[price_col], period=14)

    # Drop NaNs created by rolling windows (e.g. 21 periods for ret21)
    return df.dropna()


def main():
    logger.info("Building feature dataset...")
    config = load_config()
    processed_dir = config["data"]["processed_dir"]

    input_path = os.path.join(processed_dir, "clean_prices.parquet")
    output_path = os.path.join(processed_dir, "features.parquet")

    if not os.path.exists(input_path):
        logger.error(f"Cleaned prices missing at {input_path}. Run clean.py first!")
        return

    df = pd.read_parquet(input_path)
    logger.info(f"Loaded input prices shape: {df.shape}")

    # Safely handle MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        logger.info("Flattening MultiIndex columns...")
        df = df.stack(level=-1).reset_index()

    # Ensure required columns exist
    if "Ticker" not in df.columns and "Ticker" in df.index.names:
        df = df.reset_index(level="Ticker")

    # Group and build features (preserving 'Ticker' column!)
    if "Ticker" in df.columns:
        features_df = df.groupby("Ticker", group_keys=True).apply(build_features).reset_index(drop=True)
    else:
        features_df = build_features(df)

    # Save features panel
    features_df.to_parquet(output_path)
    logger.success(f"Saved {len(features_df)} rows to {output_path}")


if __name__ == "__main__":
    main()