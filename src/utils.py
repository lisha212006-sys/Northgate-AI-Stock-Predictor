import os
import yaml
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from loguru import logger

# ==========================================
# 1. Directory & IO Helpers
# ==========================================

def load_config(config_path="config.yaml") -> dict:
    """Reads project settings from config.yaml."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def create_directories(paths: list[str]) -> None:
    """Creates directory paths if they do not exist."""
    for path in paths:
        os.makedirs(path, exist_ok=True)

def load_processed_features(processed_dir: str = "data/processed") -> pd.DataFrame:
    """Loads the main processed feature panel (Parquet format)."""
    path = os.path.join(processed_dir, "features.parquet")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Feature matrix missing at {path}. Run features.py first!")
    return pd.read_parquet(path)


# ==========================================
# 2. Section 7: Financial Math From Scratch
# ==========================================

def manual_log_returns(prices: np.ndarray) -> np.ndarray:
    """Calculates log returns: R_t = ln(P_t / P_{t-1})."""
    return np.log(prices[1:] / prices[:-1])

def manual_annualized_stats(returns: np.ndarray, periods: int = 252) -> tuple[float, float]:
    """
    Computes annualized mean return and volatility.
    mu_ann = mu_daily * 252 ; sigma_ann = sigma_daily * sqrt(252)
    """
    mean_daily = np.mean(returns)
    # ✅ Safe against single-element or empty return arrays
    n = len(returns)
    vol_daily = np.sqrt(np.sum((returns - mean_daily) ** 2) / (n - 1)) if n > 1 else 0.0
    
    mu_ann = float(mean_daily * periods)
    sigma_ann = float(vol_daily * np.sqrt(periods))
    return mu_ann, sigma_ann

def manual_covariance_matrix(returns_matrix: np.ndarray) -> np.ndarray:
    """Computes sample covariance matrix: Cov(X, Y) = sum((X - mu_x)(Y - mu_y)) / (N - 1)."""
    mean_vec = np.mean(returns_matrix, axis=0)
    centered = returns_matrix - mean_vec
    n_samples = returns_matrix.shape[0]
    return (centered.T @ centered) / (n_samples - 1)

def manual_beta(asset_returns: np.ndarray, market_returns: np.ndarray) -> float:
    """Calculates systematic risk Beta: Beta = Cov(r_i, r_m) / Var(r_m)."""
    asset_mean = np.mean(asset_returns)
    market_mean = np.mean(market_returns)
    
    cov_im = np.sum((asset_returns - asset_mean) * (market_returns - market_mean)) / (len(asset_returns) - 1)
    var_m = np.sum((market_returns - market_mean) ** 2) / (len(market_returns) - 1)
    
    return float(cov_im / var_m)

def manual_sharpe_ratio(returns: np.ndarray, rf_annual: float = 0.04, periods: int = 252) -> float:
    """Calculates Sharpe Ratio: (mu_p - r_f) / sigma_p."""
    mu_ann, sigma_ann = manual_annualized_stats(returns, periods)
    if sigma_ann == 0:
        return 0.0
    return float((mu_ann - rf_annual) / sigma_ann)

def manual_max_drawdown(price_series: np.ndarray) -> float:
    """Calculates Maximum Drawdown: min_t (V_t / max_{s<=t} V_s - 1)."""
    running_max = np.maximum.accumulate(price_series)
    drawdowns = (price_series - running_max) / running_max
    return float(np.min(drawdowns))

def manual_gradient_descent_ols(X: np.ndarray, y: np.ndarray, lr: float = 0.01, epochs: int = 5000) -> tuple[np.ndarray, float]:
    """
    Manual Gradient Descent OLS implementation (PRD Listing 7.1).
    Minimizes MSE loss via iterative gradient updates.
    """
    n, d = X.shape
    w = np.zeros(d)
    b = 0.0
    
    for _ in range(epochs):
        yhat = X @ w + b
        err = yhat - y
        w -= lr * (2 / n) * (X.T @ err)  # dL/dw
        b -= lr * (2 / n) * err.sum()    # dL/db
        
    return w, b


# ==========================================
# 3. Math Verification Suite (PRD Section 7.5)
# ==========================================

def verify_math_implementations():
    """Verifies manual formula outputs against NumPy, SciPy, and Scikit-Learn."""
    logger.info("Executing Financial Math Verification Suite (PRD Section 7)...")
    
    np.random.seed(42)
    p_asset = 100 * np.exp(np.cumsum(np.random.normal(0.0005, 0.01, 1000)))
    p_market = 100 * np.exp(np.cumsum(np.random.normal(0.0003, 0.008, 1000)))
    
    r_asset = manual_log_returns(p_asset)
    r_market = manual_log_returns(p_market)
    
    # 1. Verify Covariance
    matrix = np.column_stack([r_asset, r_market])
    cov_manual = manual_covariance_matrix(matrix)
    cov_lib = np.cov(matrix, rowvar=False)
    np.testing.assert_almost_equal(cov_manual, cov_lib, decimal=6)
    logger.success("✅ Covariance matches numpy.cov")
    
    # 2. Verify Beta
    beta_manual = manual_beta(r_asset, r_market)
    beta_lib = cov_lib[0, 1] / cov_lib[1, 1]
    np.testing.assert_almost_equal(beta_manual, beta_lib, decimal=6)
    logger.success(f"✅ Beta matches library benchmark ({beta_manual:.4f})")
    
    # 3. Verify Max Drawdown
    mdd_manual = manual_max_drawdown(p_asset)
    peaks = pd.Series(p_asset).cummax()
    mdd_lib = ((pd.Series(p_asset) - peaks) / peaks).min()
    np.testing.assert_almost_equal(mdd_manual, mdd_lib, decimal=6)
    logger.success(f"✅ Max Drawdown matches pandas logic ({mdd_manual:.2%})")
    
    # 4. Verify Gradient Descent Linear Regression vs sklearn OLS
    X = np.random.randn(200, 3)
    true_w = np.array([1.5, -2.0, 0.5])
    y = X @ true_w + 0.5 + np.random.normal(0, 0.1, 200)
    
    w_manual, b_manual = manual_gradient_descent_ols(X, y, lr=0.05, epochs=3000)
    sk_model = LinearRegression().fit(X, y)
    
    np.testing.assert_almost_equal(w_manual, sk_model.coef_, decimal=2)
    np.testing.assert_almost_equal(b_manual, sk_model.intercept_, decimal=2)
    logger.success("✅ Manual Gradient Descent OLS matches sklearn OLS")
    
    logger.success("All financial math verification checks passed!")

if __name__ == "__main__":
    verify_math_implementations()