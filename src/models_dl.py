import os
import copy
import yaml
import joblib
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from loguru import logger

torch.set_float32_matmul_precision("high")


def load_config(path="config.yaml"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config not found at {path}")
    with open(path, "r") as f:
        return yaml.safe_load(f)


class EarlyStopping:
    def __init__(self, patience=10, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.early_stop = False
        self.best_state = None

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.best_state = copy.deepcopy(model.state_dict())
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for time-series Transformer attention."""
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


# Modularized Network Implementations
class LSTMRegressor(nn.Module):
    def __init__(self, input_size=1, hidden_dim=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])


class GRURegressor(nn.Module):
    def __init__(self, input_size=1, hidden_dim=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.SiLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        out, _ = self.gru(x)
        return self.head(out[:, -1, :])


class BiLSTMRegressor(nn.Module):
    def __init__(self, input_size=1, hidden_dim=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.bilstm = nn.LSTM(input_size, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout, bidirectional=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, 32),
            nn.SiLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        out, _ = self.bilstm(x)
        return self.head(out[:, -1, :])


class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_size=1, d_model=32, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=64, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.proj(x)
        x = self.pos_encoder(x)
        out = self.transformer(x)
        return self.head(out[:, -1, :])


def create_sequences(data: np.ndarray, seq_len: int = 60):
    """Vectorized window extraction using sliding window strides."""
    windows = np.lib.stride_tricks.sliding_window_view(data[:, 0], seq_len + 1)
    X = windows[:, :-1, np.newaxis]
    y = windows[:, -1, np.newaxis]
    return X, y


def prepare_datasets(df: pd.DataFrame, symbol: str, seq_len: int = 60):
    sub = df[df["Ticker"] == symbol].sort_values("Date") if "Ticker" in df.columns else df.sort_values("Date")
    px_col = "Adj Close" if "Adj Close" in sub.columns else "Close"
    prices = sub[[px_col]].values

    if len(prices) < seq_len + 50:
        return None, None, None, None, None

    split = int(len(prices) * 0.8)
    train_raw, test_raw = prices[:split], prices[split:]

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_train = scaler.fit_transform(train_raw)
    scaled_test = scaler.transform(test_raw)

    X_train, y_train = create_sequences(scaled_train, seq_len)
    X_test, y_test = create_sequences(scaled_test, seq_len)

    return X_train, y_train, X_test, y_test, scaler


def run_training_loop(model, train_loader, val_loader, device, epochs=100, lr=1e-3):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.HuberLoss() # Robust to return outliers
    stopper = EarlyStopping(patience=10)

    train_losses, val_losses = [], []

    for epoch in range(epochs):
        model.train()
        t_loss = 0.0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            pred = model(bx)
            loss = criterion(pred, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            t_loss += loss.item() * bx.size(0)

        t_loss /= len(train_loader.dataset)
        train_losses.append(t_loss)

        model.eval()
        v_loss = 0.0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                v_loss += criterion(model(bx), by).item() * bx.size(0)

        v_loss /= len(val_loader.dataset)
        val_losses.append(v_loss)

        stopper(v_loss, model)
        if stopper.early_stop:
            break

    if stopper.best_state:
        model.load_state_dict(stopper.best_state)

    return model, train_losses, val_losses


def main():
    cfg = load_config()
    proc_dir = cfg["data"]["processed_dir"]
    models_dir = "models"
    reports_dir = "reports"
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    features_path = os.path.join(proc_dir, "features.parquet")
    df = pd.read_parquet(features_path)
    symbols = df["Ticker"].unique().tolist() if "Ticker" in df.columns else ["ASSET"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_registry = {
        "LSTM": lambda: LSTMRegressor(),
        "GRU": lambda: GRURegressor(),
        "BiLSTM": lambda: BiLSTMRegressor(),
        "Transformer": lambda: TimeSeriesTransformer()
    }

    metrics_log = []

    for sym in symbols:
        logger.info(f"Training deep learning architectures for symbol: {sym}")
        X_tr, y_tr, X_te, y_te, scaler = prepare_datasets(df, sym)

        if X_tr is None:
            logger.warning(f"Insufficient historical data points for {sym}, skipping.")
            continue

        train_ds = TensorDataset(torch.tensor(X_tr, dtype=torch.float32), torch.tensor(y_tr, dtype=torch.float32))
        test_ds = TensorDataset(torch.tensor(X_te, dtype=torch.float32), torch.tensor(y_te, dtype=torch.float32))

        train_ld = DataLoader(train_ds, batch_size=64, shuffle=False)
        test_ld = DataLoader(test_ds, batch_size=64, shuffle=False)

        for name, factory in model_registry.items():
            model = factory().to(device)
            fitted_model, tr_loss, va_loss = run_training_loop(model, train_ld, test_ld, device)

            # Evaluation
            fitted_model.eval()
            with torch.no_grad():
                preds_scaled = fitted_model(torch.tensor(X_te, dtype=torch.float32).to(device)).cpu().numpy()

            preds = scaler.inverse_transform(preds_scaled)
            actuals = scaler.inverse_transform(y_te)

            mae = float(np.mean(np.abs(actuals - preds)))
            rmse = float(np.sqrt(np.mean((actuals - preds) ** 2)))

            metrics_log.append({"Symbol": sym, "Model": name, "MAE": round(mae, 4), "RMSE": round(rmse, 4)})
            torch.save(fitted_model.state_dict(), os.path.join(models_dir, f"{sym.lower()}_{name.lower()}.pt"))

    pd.DataFrame(metrics_log).to_csv(os.path.join(reports_dir, "dl_model_comparison.csv"), index=False)
    logger.info("Deep learning model training and evaluation process complete.")


if __name__ == "__main__":
    main()