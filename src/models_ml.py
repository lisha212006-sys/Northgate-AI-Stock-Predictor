import os
import yaml
import joblib
import numpy as np
import pandas as pd
from loguru import logger

from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR


def load_config(path="config.yaml"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing configuration file at {path}")
    with open(path, "r") as f:
        return yaml.safe_load(f)


def prepare_dataset(features_path: str):
    """
    Loads features and creates temporal train-test splits.
    Handles naive persistence baselines per asset group.
    """
    if not os.path.exists(features_path):
        raise FileNotFoundError(f"Feature dataset not found: {features_path}")

    df = pd.read_parquet(features_path).dropna().copy()
    target_col = "Target_Return" if "Target_Return" in df.columns else "Close"

    # Naive baseline calculation (lag-1 value per symbol)
    if "Ticker" in df.columns:
        df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)
        df["y_naive"] = df.groupby("Ticker")[target_col].shift(1)
    else:
        df = df.sort_values("Date").reset_index(drop=True)
        df["y_naive"] = df[target_col].shift(1)

    df = df.dropna(subset=["y_naive"]).sort_values("Date").reset_index(drop=True)

    ignore_cols = {"Date", "Ticker", "symbol", target_col, "y_naive"}
    feature_cols = [c for c in df.columns if c not in ignore_cols]

    X = df[feature_cols]
    y = df[target_col]
    y_naive = df["y_naive"]

    # 80/20 chronological split
    split_idx = int(len(df) * 0.8)

    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    y_test_naive = y_naive.iloc[split_idx:]

    return X_train, X_test, y_train, y_test, y_test_naive


def calculate_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict:
    """
    Computes statistical and directional evaluation metrics.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    # Calculate Directional Accuracy (sign agreement)
    dir_acc = np.mean(np.sign(np.array(y_true)) == np.sign(y_pred))

    return {
        "MAE": round(float(mae), 5),
        "RMSE": round(float(rmse), 5),
        "R2": round(float(r2), 4),
        "Dir_Acc": round(float(dir_acc), 4)
    }


def train_regressors(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series, models_dir: str):
    """
    Runs cross-validated hyperparameter optimization using Pipelines to avoid scaling leakage.
    """
    tscv = TimeSeriesSplit(n_splits=5)
    results = []

    # Pipeline setup guarantees StandardScaler is fit ONLY on internal CV training folds
    experiments = [
        (
            "Ridge",
            Pipeline([("scaler", StandardScaler()), ("model", Ridge())]),
            {"model__alpha": [0.1, 1.0, 10.0, 100.0]}
        ),
        (
            "RandomForest",
            Pipeline([("scaler", StandardScaler()), ("model", RandomForestRegressor(random_state=42))]),
            {"model__n_estimators": [50, 100], "model__max_depth": [4, 8]}
        ),
        (
            "XGBoost",
            Pipeline([("scaler", StandardScaler()), ("model", XGBRegressor(random_state=42, objective="reg:squarederror"))]),
            {"model__n_estimators": [50, 100], "model__learning_rate": [0.01, 0.1], "model__max_depth": [3, 5]}
        ),
        (
            "SVR",
            Pipeline([("scaler", StandardScaler()), ("model", SVR())]),
            {"model__C": [0.1, 1.0, 10.0], "model__epsilon": [0.01, 0.1]}
        )
    ]

    for name, pipe, grid in experiments:
        logger.info(f"Cross-validating {name} estimator...")

        gs = GridSearchCV(
            estimator=pipe,
            param_grid=grid,
            cv=tscv,
            scoring="neg_mean_squared_error",
            n_jobs=-1
        )

        gs.fit(X_train, y_train)
        best_pipe = gs.best_estimator_

        # Out-of-sample prediction
        preds = best_pipe.predict(X_test)
        metrics = calculate_metrics(y_test, preds)

        results.append({"Model": name, **metrics})

        logger.info(f"{name} | Best Params: {gs.best_params_}")
        logger.info(f"{name} Out-of-Sample | MAE: {metrics['MAE']} | RMSE: {metrics['RMSE']} | Dir Acc: {metrics['Dir_Acc']}")

        # Save pipeline (includes scaler + trained regressor)
        joblib.dump(best_pipe, os.path.join(models_dir, f"{name.lower()}_pipeline.pkl"))

    return results


def main():
    logger.info("Initializing classical ML model training workflow")
    cfg = load_config()

    proc_dir = cfg["data"]["processed_dir"]
    models_dir = "models"
    reports_dir = "reports"
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    features_path = os.path.join(proc_dir, "features.parquet")
    X_tr, X_te, y_tr, y_te, y_te_naive = prepare_dataset(features_path)

    # Persistence baseline metric evaluation
    baseline_metrics = calculate_metrics(y_te, y_te_naive.values)
    results = [{"Model": "Naive Baseline", **baseline_metrics}]

    # Run training execution loop
    model_results = train_regressors(X_tr, y_tr, X_te, y_te, models_dir)
    results.extend(model_results)

    summary_df = pd.DataFrame(results)
    summary_df.to_csv(os.path.join(reports_dir, "ml_model_comparison.csv"), index=False)
    
    logger.info(f"\n{summary_df.to_string(index=False)}")
    logger.info("ML evaluation pipeline complete.")


if __name__ == "__main__":
    main()