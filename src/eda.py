import os
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis, jarque_bera
from statsmodels.tsa.stattools import adfuller
from loguru import logger

# Set aesthetic styling for report figures
plt.style.use("seaborn-v0_8-darkgrid" if "seaborn-v0_8-darkgrid" in plt.style.available else "default")


def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def run_stationarity_tests(features_df: pd.DataFrame) -> dict:
    """
    Section 6.3: Runs ADF tests on raw Adj Close vs Log Returns.
    Rejects null hypothesis of non-stationarity if p-value < 0.05.
    """
    logger.info("--- 1. STATIONARITY ANALYSIS (ADF TEST) ---")
    
    # Pick representative ticker (e.g. first ticker in dataset)
    sample_ticker = features_df["Ticker"].iloc[0] if "Ticker" in features_df.columns else None
    df_sample = features_df[features_df["Ticker"] == sample_ticker] if sample_ticker else features_df

    price_series = df_sample["Adj Close"].dropna()
    return_series = df_sample["ret1"].dropna()

    price_adf_p = adfuller(price_series)[1]
    return_adf_p = adfuller(return_series)[1]

    results = {
        "ticker": sample_ticker,
        "price_adf_p": price_adf_p,
        "price_is_stationary": price_adf_p < 0.05,
        "return_adf_p": return_adf_p,
        "return_is_stationary": return_adf_p < 0.05
    }

    logger.info(f"Ticker: {sample_ticker}")
    logger.info(f"Price Series ADF p-value: {price_adf_p:.4f} -> Stationary: {results['price_is_stationary']}")
    logger.info(f"Log Return ADF p-value:  {return_adf_p:.4e} -> Stationary: {results['return_is_stationary']}")
    
    return results


def run_distribution_analysis(features_df: pd.DataFrame, figures_dir: str):
    """
    Section 6.1 & 6.2: Analyzes return skewness, excess kurtosis, and fat tails.
    Plots Return Distribution Histogram with Gaussian overlay.
    """
    logger.info("--- 2. DISTRIBUTION & FAT-TAILS ANALYSIS ---")
    returns = features_df["ret1"].dropna()

    s = skew(returns)
    k = kurtosis(returns)  # Excess kurtosis (Normal = 0)
    jb_stat, jb_p = jarque_bera(returns)

    logger.info(f"Return Skewness: {s:.2f}")
    logger.info(f"Excess Kurtosis: {k:.2f} (Kurtosis > 0 indicates fat tails)")
    logger.info(f"Jarque-Bera p-value: {jb_p:.4e}")

    # Plot Return Histogram vs Normal Distribution
    plt.figure(figsize=(9, 5))
    sns.histplot(returns, bins=100, kde=True, stat="density", color="royalblue", alpha=0.6, label="Empirical Log Returns")
    
    # Gaussian theoretical overlay
    mu, std = returns.mean(), returns.std()
    x = np.linspace(returns.min(), returns.max(), 200)
    p = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / std) ** 2)
    plt.plot(x, p, "r--", linewidth=2, label=f"Normal Fit (μ={mu:.4f}, σ={std:.4f})")

    plt.title(f"Daily Log Returns Distribution (Excess Kurtosis: {k:.2f})")
    plt.xlabel("1-Day Log Return")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    
    out_path = os.path.join(figures_dir, "01_returns_distribution.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    logger.success(f"Saved distribution plot to {out_path}")


def run_correlation_matrix(features_df: pd.DataFrame, figures_dir: str):
    """
    Section 6.2: Computes return correlation heatmap across the equity universe.
    """
    logger.info("--- 3. CORRELATION ANALYSIS ---")
    if "Ticker" in features_df.columns:
        pivoted_returns = features_df.pivot(columns="Ticker", values="ret1").dropna()
    else:
        logger.warning("Single-asset panel detected; skipping cross-asset correlation.")
        return

    corr = pivoted_returns.corr(method="pearson")

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1, fmt=".2f", linewidths=0.5)
    plt.title("Asset Returns Pearson Correlation Heatmap")
    plt.tight_layout()

    out_path = os.path.join(figures_dir, "02_correlation_heatmap.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    logger.success(f"Saved correlation heatmap to {out_path}")


def run_volatility_analysis(features_df: pd.DataFrame, raw_dir: str, figures_dir: str):
    """
    Section 6.2: Plots rolling volatility co-movement with VIX overlay.
    """
    logger.info("--- 4. VOLATILITY CLUSTERING & VIX OVERLAY ---")
    prices_path = os.path.join(raw_dir, "prices.parquet")
    
    if not os.path.exists(prices_path):
        return

    raw_prices = pd.read_parquet(prices_path)
    
    try:
        if isinstance(raw_prices.columns, pd.MultiIndex):
            vix_series = raw_prices["Close"]["^VIX"].dropna()
        else:
            vix_series = raw_prices["^VIX"].dropna()
            
        sample_ticker = features_df["Ticker"].iloc[0] if "Ticker" in features_df.columns else None
        df_sample = features_df[features_df["Ticker"] == sample_ticker].set_index(features_df.index) if sample_ticker else features_df

        fig, ax1 = plt.subplots(figsize=(12, 5))

        ax1.set_xlabel("Date")
        ax1.set_ylabel(f"{sample_ticker} 21-Day Volatility", color="tab:blue")
        ax1.plot(df_sample.index, df_sample["vol21"], color="tab:blue", label="21D Realized Volatility")
        ax1.tick_params(axis="y", labelcolor="tab:blue")

        ax2 = ax1.twinx()
        ax2.set_ylabel("CBOE VIX Index (^VIX)", color="tab:red")
        ax2.plot(vix_series.index, vix_series, color="tab:red", alpha=0.5, linestyle=":", label="VIX Index")
        ax2.tick_params(axis="y", labelcolor="tab:red")

        plt.title(f"Volatility Clustering & Market Risk Regime Overlay ({sample_ticker} vs ^VIX)")
        fig.tight_layout()

        out_path = os.path.join(figures_dir, "03_volatility_vix_overlay.png")
        plt.savefig(out_path, dpi=300)
        plt.close()
        logger.success(f"Saved volatility chart to {out_path}")

    except Exception as e:
        logger.warning(f"Could not construct VIX overlay plot: {e}")


def main():
    logger.info("Starting Exploratory Data Analysis Pipeline (Section 6)...")

    config = load_config()
    processed_dir = config["data"]["processed_dir"]
    raw_dir = config["data"]["raw_dir"]
    figures_dir = os.path.join("reports", "figures")
    os.makedirs(figures_dir, exist_ok=True)

    features_path = os.path.join(processed_dir, "features.parquet")
    if not os.path.exists(features_path):
        logger.error(f"Features file missing at {features_path}. Run features.py first!")
        return

    features_df = pd.read_parquet(features_path)

    # 1. ADF Stationarity Check
    stationarity_res = run_stationarity_tests(features_df)

    # 2. Distribution & Fat-Tails Check
    run_distribution_analysis(features_df, figures_dir)

    # 3. Cross-Asset Correlation Analysis
    run_correlation_matrix(features_df, figures_dir)

    # 4. Volatility Regime Analysis
    run_volatility_analysis(features_df, raw_dir, figures_dir)

    logger.success("EDA Pipeline complete. Statistical findings printed & figures exported to reports/figures/")


if __name__ == "__main__":
    main()