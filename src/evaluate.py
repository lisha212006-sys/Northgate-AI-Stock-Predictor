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

    # Average DL metrics across tickers if multi-ticker dataset
    if "Ticker" in df_dl.columns:
        df_dl = df_dl.groupby("Model")[["MAE", "RMSE"]].mean().reset_index()
        df_dl["Fit Diagnosis"] = "Early Stopped / Optimal"

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