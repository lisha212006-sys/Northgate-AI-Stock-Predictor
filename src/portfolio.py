import sys
import os
import yaml
import numpy as np
import pandas as pd
import scipy.optimize as sco
import matplotlib.pyplot as plt


def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# --- 1. Portfolio Performance Math ---
def portfolio_performance(weights, mean_returns, cov_matrix, risk_free_rate=0.02):
    returns = np.sum(mean_returns * weights) * 252
    std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
    sharpe_ratio = (returns - risk_free_rate) / std_dev
    return std_dev, returns, sharpe_ratio


def neg_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate=0.02):
    return -portfolio_performance(weights, mean_returns, cov_matrix, risk_free_rate)[2]


# --- 2. MPT Optimization & Max-Sharpe Weights ---
def optimize_max_sharpe(mean_returns, cov_matrix, risk_free_rate=0.02):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix, risk_free_rate)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(num_assets))
    init_guess = num_assets * [1.0 / num_assets]

    opts = sco.minimize(neg_sharpe_ratio, init_guess, args=args, method='SLSQP', bounds=bounds, constraints=constraints)
    return opts.x


# --- 3. Monte Carlo Simulation Cross-Check ---
def run_monte_carlo(mean_returns, cov_matrix, num_portfolios=5000, risk_free_rate=0.02):
    num_assets = len(mean_returns)
    results = np.zeros((3, num_portfolios))
    weights_record = []

    for i in range(num_portfolios):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        weights_record.append(weights)
        std_dev, returns, sharpe = portfolio_performance(weights, mean_returns, cov_matrix, risk_free_rate)
        results[0, i] = std_dev
        results[1, i] = returns
        results[2, i] = sharpe

    max_sharpe_idx = np.argmax(results[2])
    mc_best_weights = weights_record[max_sharpe_idx]
    return results, mc_best_weights, results[:, max_sharpe_idx]


# --- 4. Backtest vs. Baselines ---
def backtest_portfolio(returns_df, optimal_weights):
    """
    Backtests Max-Sharpe strategy against Equal-Weight (1/N) benchmark.
    """
    num_assets = len(returns_df.columns)
    equal_weights = np.array([1.0 / num_assets] * num_assets)

    # Calculate daily portfolio returns
    strat_daily_returns = returns_df.dot(optimal_weights)
    benchmark_daily_returns = returns_df.dot(equal_weights)

    # Cumulative growth
    strat_cum = (1 + strat_daily_returns).cumprod()
    benchmark_cum = (1 + benchmark_daily_returns).cumprod()

    # Metrics
    metrics = {
        "Max-Sharpe Cum Return": strat_cum.iloc[-1] - 1,
        "Max-Sharpe Sharpe": (strat_daily_returns.mean() * 252 - 0.02) / (strat_daily_returns.std() * np.sqrt(252)),
        "Equal-Weight Cum Return": benchmark_cum.iloc[-1] - 1,
        "Equal-Weight Sharpe": (benchmark_daily_returns.mean() * 252 - 0.02) / (benchmark_daily_returns.std() * np.sqrt(252))
    }

    return metrics, strat_cum, benchmark_cum


def main():
    config = load_config()
    processed_dir = config["data"]["processed_dir"]
    features_path = os.path.join(processed_dir, "features.parquet")
    reports_dir = "reports"
    figures_dir = os.path.join(reports_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    df = pd.read_parquet(features_path)
    price_col = "Close" if "Close" in df.columns else "Adj Close"

    # Pivot to get returns matrix per ticker
    price_pivot = df.pivot(index="Date", columns="Ticker", values=price_col).dropna()
    returns_df = price_pivot.pct_change().dropna()

    mean_returns = returns_df.mean()
    cov_matrix = returns_df.cov()

    # 1. SLSQP Optimization
    opt_weights = optimize_max_sharpe(mean_returns, cov_matrix)
    opt_vol, opt_ret, opt_sharpe = portfolio_performance(opt_weights, mean_returns, cov_matrix)

    # 2. Monte Carlo Cross-Check
    mc_results, mc_weights, mc_best = run_monte_carlo(mean_returns, cov_matrix)

    # 3. Backtest
    backtest_metrics, strat_cum, bench_cum = backtest_portfolio(returns_df, opt_weights)

    # 4. Save Weights Table
    weights_df = pd.DataFrame({
        "Ticker": returns_df.columns,
        "Max-Sharpe Weight (SLSQP)": opt_weights,
        "Monte-Carlo Weight": mc_weights
    })
    weights_df.to_csv(os.path.join(reports_dir, "portfolio_weights.csv"), index=False)

    # 5. Plot Efficient Frontier
    plt.figure(figsize=(9, 5))
    plt.scatter(mc_results[0, :], mc_results[1, :], c=mc_results[2, :], cmap='viridis', marker='o', s=10, alpha=0.3)
    plt.colorbar(label='Sharpe Ratio')
    plt.scatter(opt_vol, opt_ret, color='red', marker='*', s=200, label=f'Max Sharpe (SLSQP): {opt_sharpe:.2f}')
    plt.title('Efficient Frontier & Monte Carlo Simulation')
    plt.xlabel('Annualized Volatility')
    plt.ylabel('Annualized Return')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(figures_dir, "efficient_frontier.png"), bbox_inches="tight", dpi=300)
    plt.close()

    print("\n================ MPT Portfolio Optimization Results ================")
    print(weights_df.to_string(index=False))
    print("\n--- Backtest Performance vs Equal-Weight Baseline ---")
    for k, v in backtest_metrics.items():
        print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    main()