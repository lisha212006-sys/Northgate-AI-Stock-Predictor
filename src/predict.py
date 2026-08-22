import os
import joblib
import numpy as np
import pandas as pd

from loguru import logger

from tensorflow.keras.models import load_model

PROCESSED_DIR = "data/processed"
MODEL_DIR = "models"

STOCK_FILE = "AAPL.csv"

def load_data():

    path = os.path.join(
        PROCESSED_DIR,
        STOCK_FILE
    )

    df = pd.read_csv(path)

    logger.success(f"Loaded {STOCK_FILE}")

    return df

def load_ml_model():

    path = os.path.join(
        MODEL_DIR,
        "random_forest.pkl"
    )

    model = joblib.load(path)

    logger.success("Random Forest model loaded!")

    return model

def load_dl_model():

    path = os.path.join(
        MODEL_DIR,
        "lstm_model.keras"
    )

    model = load_model(path)

    logger.success("LSTM model loaded!")

    return model

def predict_ml(model, df):

    feature_columns = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "Daily_Return",
    "MA20",
    "MA50",
    "Volatility"
]

    X = df[feature_columns].fillna(0)

    predictions = model.predict(X)

    logger.success("Random Forest predictions completed!")

    return predictions


def main():

    df = load_data()
    print(df.columns.tolist())
    ml_model = load_ml_model()

    dl_model = load_dl_model()

    print(df.head())

    print(ml_model)

    print(dl_model)

    ml_predictions = predict_ml(ml_model, df)

    print(ml_predictions[:10])


if __name__ == "__main__":
    main()