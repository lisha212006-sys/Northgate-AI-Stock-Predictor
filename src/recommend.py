import os
import yaml
import numpy as np
import pandas as pd


def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def generate_signal(expected_return, buy_threshold=0.015, sell_threshold=-0.005):
    """
    Transparent signal decision rule based on expected return thresholds.
    """
    if expected_return >= buy_threshold:
        return "BUY"
    elif expected_return <= sell_threshold:
        return "SELL"
    else:
        return "HOLD"


def calculate_rebalance_trades(current_portfolio, target_weights, total_portfolio_value=100000.0):
    """
    Calculates exact target values and required trade amounts to reach MPT optimal weights.
    """
    rebalance_orders = []

    for ticker, target_w in target_weights.items():
        current_val = current_portfolio.get(ticker, 0.0)
        current_w = current_val / total_portfolio_value if total_portfolio_value > 0 else 0.0
        
        target_val = total_portfolio_value * target_w
        trade_amount = target_val - current_val

        if trade_amount > 50:
            action = "BUY"
        elif trade_amount < -50:
            action = "SELL"
        else:
            action = "HOLD"

        rebalance_orders.append({
            "Ticker": ticker,
            "Current Weight (%)": round(current_w * 100, 2),
            "Target Weight (%)": round(target_w * 100, 2),
            "Current Value ($)": round(current_val, 2),
            "Target Value ($)": round(target_val, 2),
            "Rebalance Order ($)": round(trade_amount, 2),
            "Execution Action": action
        })

    return pd.DataFrame(rebalance_orders)


def main():
    config = load_config()
    reports_dir = "reports"
    weights_path = os.path.join(reports_dir, "portfolio_weights.csv")

    # Load MPT weights calculated in Layer 5 (portfolio.py)
    if os.path.exists(weights_path):
        weights_df = pd.read_csv(weights_path)
        target_weights = dict(zip(weights_df["Ticker"], weights_df["Max-Sharpe Weight (SLSQP)"]))
    else:
        # Fallback equal weights if portfolio_weights.csv doesn't exist
        target_weights = {"AAPL": 0.33, "MSFT": 0.33, "GOOGL": 0.34}

    # Simulate hypothetical portfolio input (e.g. $100,000 baseline)
    total_portfolio_value = 100000.0
    current_portfolio = {
        "AAPL": 40000.0,
        "MSFT": 20000.0,
        "GOOGL": 40000.0
    }

    # Forecast returns simulation for signals (Replace with predictions loaded from models/)
    expected_returns = {
        "AAPL": 0.025,   # +2.5% expected return -> BUY
        "MSFT": 0.005,   # +0.5% expected return -> HOLD
        "GOOGL": -0.010  # -1.0% expected return -> SELL
    }

    # 1. Generate Signal Fusion Table
    recommendations = []
    for ticker, target_w in target_weights.items():
        exp_ret = expected_returns.get(ticker, 0.0)
        signal = generate_signal(exp_ret)

        # Transparent reasoning text
        reasoning = (
            f"Forecasted Return: {exp_ret * 100:+.2f}%. "
            f"Signal [{signal}] driven by return relative to thresholds. "
            f"Target allocation adjusted to {target_w * 100:.1f}% via Max-Sharpe MPT."
        )

        recommendations.append({
            "Ticker": ticker,
            "Expected Return (%)": round(exp_ret * 100, 2),
            "Recommendation Signal": signal,
            "Target Weight (%)": round(target_w * 100, 2),
            "Transparent Rationale": reasoning
        })

    rec_df = pd.DataFrame(recommendations)
    rec_df.to_csv(os.path.join(reports_dir, "recommendations.csv"), index=False)

    # 2. Calculate Rebalancing Orders
    rebalance_df = calculate_rebalance_trades(current_portfolio, target_weights, total_portfolio_value)
    rebalance_df.to_csv(os.path.join(reports_dir, "rebalancing_orders.csv"), index=False)

    print("\n================ Layer 6: Signal Fusion & Recommendations ================")
    print(rec_df[["Ticker", "Expected Return (%)", "Recommendation Signal", "Target Weight (%)"]].to_string(index=False))

    print("\n================ Rebalancing Execution Orders ================")
    print(rebalance_df[["Ticker", "Current Weight (%)", "Target Weight (%)", "Rebalance Order ($)", "Execution Action"]].to_string(index=False))


if __name__ == "__main__":
    main()