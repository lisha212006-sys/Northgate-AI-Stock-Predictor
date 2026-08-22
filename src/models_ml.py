import os
import yaml
import joblib
import numpy as np
import pandas as pd
from loguru import logger

from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR


def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def prepare_dataset(features_path):
    """Loads features and builds clean, leakage-free chronological train/test splits."""
    if not os.path.exists(features_path):
        raise FileNotFoundError(f"Feature matrix missing at {features_path}. Run features.py first!")

    df = pd.read_parquet(features_path).dropna()

    target_col = "Target_Return" if "Target_Return" in df.columns else "Close"

    # Compute Naive Baseline (Lag 1) grouped by Ticker to avoid crossing stock boundaries
    if "Ticker" in df.columns:
        df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)
        df["y_naive"] = df.groupby("Ticker")[target_col].shift(1)
    else:
        df = df.sort_values("Date").reset_index(drop=True)
        df["y_naive"] = df[target_col].shift(1)

    # Drop NaNs created by shifting
    df = df.dropna(subset=["y_naive"]).reset_index(drop=True)

    # Sort strictly by Date for global chronological split
    df = df.sort_values("Date").reset_index(drop=True)

    # Exclude non-feature columns
    ignore_cols = ["Date", "Ticker", target_col, "y_naive"]
    feature_cols = [c for c in df.columns if c not in ignore_cols]

    X = df[feature_cols]
    y = df[target_col]
    y_naive = df["y_naive"]

    # Chronological Out-of-Sample Split (80% Train, 20% Test)
    split_idx = int(len(df) * 0.8)

    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    y_test_naive = y_naive.iloc[split_idx:]

    return X_train, X_test, y_train, y_test, y_test_naive


def evaluate_predictions(y_true, y_pred):
    """Calculates quantitative evaluation metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {"MAE": float(mae), "RMSE": float(rmse), "R2": float(r2)}


def main():
    logger.info("Starting Machine Learning Training & Tuning Pipeline (Section 8)...")
    config = load_config()
    processed_dir = config["data"]["processed_dir"]
    models_dir = "models"
    reports_dir = "reports"
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    features_path = os.path.join(processed_dir, "features.parquet")
    X_train, X_test, y_train, y_test, y_test_naive = prepare_dataset(features_path)

    # Standardize features (Fit on train ONLY to prevent leakage)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save feature scaler
    joblib.dump(scaler, os.path.join(models_dir, "scaler.pkl"))

    # --- 1. Establish Naive Random-Walk Baseline ---
    baseline_metrics = evaluate_predictions(y_test, y_test_naive)
    results = [{
        "Model": "Naive Baseline (Random Walk)",
        **baseline_metrics
    }]
    logger.info(f"Naive Baseline | MAE: {baseline_metrics['MAE']:.4f} | RMSE: {baseline_metrics['RMSE']:.4f}")

    # --- 2. Configure TimeSeriesSplit Cross-Validation ---
    tscv = TimeSeriesSplit(n_splits=5)

    # --- 3. Define Models & Hyperparameter Grids ---
    model_grids = {
        "Ridge": (
            Ridge(),
            {"alpha": [0.1, 1.0, 10.0, 100.0]}
        ),
        "Random Forest": (
            RandomForestRegressor(random_state=42),
            {"n_estimators": [50, 100], "max_depth": [5, 10], "min_samples_split": [2, 5]}
        ),
        "XGBoost": (
            XGBRegressor(random_state=42, objective="reg:squarederror"),
            {"n_estimators": [50, 100], "learning_rate": [0.01, 0.1], "max_depth": [3, 6]}
        ),
        "SVR": (
            SVR(),
            {"C": [0.1, 1.0, 10.0], "epsilon": [0.01, 0.1], "kernel": ["rbf"]}
        )
    }

    # --- 4. Train & Tune Models ---
    for name, (model, grid) in model_grids.items():
        logger.info(f"Tuning {name} using TimeSeriesSplit...")

        grid_search = GridSearchCV(
            estimator=model,
            param_grid=grid,
            cv=tscv,
            scoring="neg_mean_squared_error",
            n_jobs=-1
        )

        grid_search.fit(X_train_scaled, y_train)
        best_model = grid_search.best_estimator_

        # Out-of-sample Test Predictions
        preds = best_model.predict(X_test_scaled)
        metrics = evaluate_predictions(y_test, preds)

        results.append({
            "Model": name,
            **metrics
        })

        logger.success(f"{name} Best Params: {grid_search.best_params_}")
        logger.success(f"{name} Test Set | MAE: {metrics['MAE']:.4f} | RMSE: {metrics['RMSE']:.4f} | R²: {metrics['R2']:.4f}")

        # Save model checkpoint
        model_filename = name.lower().replace(" ", "_") + "_model.pkl"
        joblib.dump(best_model, os.path.join(models_dir, model_filename))

    # --- 5. Compile and Save Metrics Summary ---
    summary_df = pd.DataFrame(results)
    logger.info("\n" + summary_df.to_string(index=False))

    summary_df.to_csv(os.path.join(reports_dir, "ml_model_comparison.csv"), index=False)
    joblib.dump(results, os.path.join(models_dir, "ml_metrics.pkl"))
    logger.success("All ML models trained, tuned, and evaluated successfully!")


if __name__ == "__main__":
    main()