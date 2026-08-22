import os
import yaml
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor


def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def evaluate_model_on_features(df, feature_cols, target_col="Target_Return"):
    """Splits data chronologically and evaluates XGBoost performance."""
    split_idx = int(len(df) * 0.8)

    X = df[feature_cols]
    y = df[target_col]

    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Use XGBoost as the primary benchmark model for the experiment
    model = XGBRegressor(n_estimators=100, learning_rate=0.05, random_state=42)
    model.fit(X_train_scaled, y_train)

    preds = model.predict(X_test_scaled)

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    return {"MAE": mae, "RMSE": rmse, "R2": r2}


def run_sentiment_experiment():
    config = load_config()
    processed_dir = config["data"]["processed_dir"]
    features_path = os.path.join(processed_dir, "features.parquet")
    reports_dir = "reports"
    os.makedirs(reports_dir, exist_ok=True)

    df = pd.read_parquet(features_path).dropna()
    df = df.sort_values("Date").reset_index(drop=True)

    target_col = "Target_Return" if "Target_Return" in df.columns else "Close"
    sentiment_col = "FinBERT_Sentiment_Lag1"

    if sentiment_col not in df.columns:
        raise KeyError(f"Missing {sentiment_col} in features matrix. Run src/sentiment.py first!")

    # Define Feature Sets
    ignore_cols = ["Date", "Ticker", target_col]
    features_with = [c for c in df.columns if c not in ignore_cols]
    features_without = [c for c in features_with if c != sentiment_col]

    # Evaluate both setups
    metrics_without = evaluate_model_on_features(df, features_without, target_col)
    metrics_with = evaluate_model_on_features(df, features_with, target_col)

    # Compute Deltas (Positive delta for R2 is good, negative delta for MAE/RMSE is good)
    delta_mae = metrics_with["MAE"] - metrics_without["MAE"]
    delta_rmse = metrics_with["RMSE"] - metrics_without["RMSE"]
    delta_r2 = metrics_with["R2"] - metrics_without["R2"]

    results = [
        {"Experiment": "Without FinBERT Sentiment", **metrics_without},
        {"Experiment": "With FinBERT Sentiment", **metrics_with},
        {
            "Experiment": "Impact / Delta (With vs Without)",
            "MAE": delta_mae,
            "RMSE": delta_rmse,
            "R2": delta_r2,
        },
    ]

    report_df = pd.DataFrame(results)
    output_path = os.path.join(reports_dir, "sentiment_experiment_results.csv")
    report_df.to_csv(output_path, index=False)

    print("\n================ FinBERT Sentiment Experiment Results ================")
    print(report_df.to_string(index=False))
    print(f"\nExperiment output saved to: {output_path}")

    # Written Diagnosis
    print("\n--- Measured Effect Summary ---")
    if delta_r2 > 0 and delta_rmse < 0:
        print("Verdict: FinBERT sentiment IMPROVED predictive performance (Higher R², Lower RMSE/MAE).")
    else:
        print("Verdict: FinBERT sentiment provided minimal or negative signal boost over technical features alone.")


if __name__ == "__main__":
    run_sentiment_experiment()