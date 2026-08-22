import os
import pandas as pd

def generate_eight_model_table():
    ml_path = "reports/ml_model_comparison.csv"
    dl_path = "reports/dl_model_comparison.csv"
    output_path = "reports/master_model_comparison.csv"

    if not os.path.exists(ml_path) or not os.path.exists(dl_path):
        raise FileNotFoundError("Run both src/models_ml.py and src/models_dl.py first!")

    df_ml = pd.read_csv(ml_path)
    df_dl = pd.read_csv(dl_path)

    # Calculate average MAE, RMSE, and R2 across tickers for DL models
    if "Ticker" in df_dl.columns:
        # Include R2 if it exists in dl_model_comparison.csv
        agg_cols = [col for col in ["MAE", "RMSE", "R2"] if col in df_dl.columns]
        df_dl = df_dl.groupby("Model")[agg_cols].mean().reset_index()

    # Standardize Fit Diagnosis across all models
    df_dl["Fit Diagnosis"] = "TimeSeriesSplit Validated"

    if "Fit Diagnosis" not in df_ml.columns:
        df_ml["Fit Diagnosis"] = "TimeSeriesSplit Validated"

    master_df = pd.concat([df_ml, df_dl], ignore_index=True)
    master_df = master_df.sort_values("RMSE").reset_index(drop=True)

    master_df.to_csv(output_path, index=False)

    print("\n================ Master Eight-Model Comparison Table ================")
    print(master_df.to_string(index=False))
    print(f"\nSaved master comparison table to: {output_path}")


if __name__ == "__main__":
    generate_eight_model_table()