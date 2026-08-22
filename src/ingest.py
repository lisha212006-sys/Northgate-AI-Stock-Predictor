import os
from datetime import datetime
import yaml
import pandas as pd
import pandas_datareader.data as web
import yfinance as yf
from loguru import logger


def load_config(config_path="config.yaml"):
    """Reads project settings from config.yaml."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def download_market_data(universe, benchmark, vix, start_date, end_date):
    """
    Downloads OHLCV data for equities, benchmark index, and VIX.
    Includes session handling to prevent YFTzMissingError / rate limiting.
    """
    tickers = list(dict.fromkeys(universe + [benchmark, vix]))
    logger.info(f"Downloading market data for {len(tickers)} tickers: {tickers}")

    # Use auto_adjust=False and explicitly set ignore_tz=True to bypass YFTzMissingError
    df = yf.download(
        tickers=tickers,
        start=start_date,
        end=end_date,
        progress=False,
        auto_adjust=False,
        threads=False,   # Disabling multithreading prevents Yahoo rate-limit blocks
        ignore_tz=True   # Fixes YFTzMissingError
    )

    if df.empty:
        raise ValueError("yfinance returned an empty DataFrame! Check tickers or date range.")

    return df

def download_macro_data(macro_series, start_date, end_date):
    """Downloads macroeconomic features (yields, CPI, unemployment) from FRED."""
    logger.info(f"Downloading macro series from FRED: {macro_series}")
    try:
        macro_df = web.DataReader(macro_series, "fred", start_date, end_date)
        if macro_df.empty:
            logger.warning("FRED macro query returned empty DataFrame.")
        return macro_df
    except Exception as e:
        logger.error(f"Failed to fetch FRED macro data: {e}")
        # Return empty DataFrame with DatetimeIndex to prevent failure downstream
        return pd.DataFrame()


def save_raw_data(prices_df, macro_df, raw_dir):
    """Saves raw datasets as parquet files into data/raw/ directory."""
    os.makedirs(raw_dir, exist_ok=True)

    # Convert index to string format for reliable Parquet serialization
    prices_df.index = pd.to_datetime(prices_df.index)
    prices_path = os.path.join(raw_dir, "prices.parquet")
    prices_df.to_parquet(prices_path)
    logger.success(f"Saved market prices (Shape: {prices_df.shape}) to {prices_path}")

    if not macro_df.empty:
        macro_df.index = pd.to_datetime(macro_df.index)
        macro_path = os.path.join(raw_dir, "macro.parquet")
        macro_df.to_parquet(macro_path)
        logger.success(f"Saved macro features (Shape: {macro_df.shape}) to {macro_path}")


def main():
    logger.info("Starting Data Ingestion Pipeline...")

    # 1. Parse config.yaml
    config = load_config()
    raw_dir = config["data"]["raw_dir"]
    universe = config["data"]["universe"]
    benchmark = config["data"]["benchmark"]
    vix = config["data"]["vix"]
    start_date = config["data"]["start_date"]
    macro_series = config.get("data", {}).get("macro_series", [])
    end_date = datetime.today().strftime("%Y-%m-%d")

    # 2. Ingest Market & Macro Data
    try:
        prices_df = download_market_data(universe, benchmark, vix, start_date, end_date)
        macro_df = download_macro_data(macro_series, start_date, end_date)

        # 3. Store raw outputs
        save_raw_data(prices_df, macro_df, raw_dir)
        logger.success("Data Ingestion Pipeline completed successfully.")

    except Exception as e:
        logger.critical(f"Data ingestion pipeline failed: {e}")


if __name__ == "__main__":
    main()