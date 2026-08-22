import os
import copy
import yaml
import joblib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Reproducibility
# Enable CUDA optimizations
torch.set_float32_matmul_precision('high')
torch.backends.cudnn.benchmark = True

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# --- 1. Custom Early Stopping with History Tracking ---
class EarlyStopping:
    def __init__(self, patience=7, delta=1e-4):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_loss = float("inf")
        self.early_stop = False
        self.best_model_weights = None

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.best_model_weights = copy.deepcopy(model.state_dict())
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


# --- 2. Loss Curve Plotting & Diagnostic Function ---
def save_loss_curve(train_losses, val_losses, ticker, model_name, save_dir="reports/figures"):
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(7, 4))
    plt.plot(train_losses, label="Train Loss", color="blue")
    plt.plot(val_losses, label="Validation Loss", color="orange", linestyle="--")
    plt.title(f"Loss Curve: {model_name} ({ticker})")
    plt.xlabel("Epochs")
    plt.ylabel("Loss (MSE)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    save_path = os.path.join(save_dir, f"{ticker.lower()}_{model_name.lower()}_loss_curve.png")
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()


def diagnose_fit(train_losses, val_losses):
    """Diagnoses model fit based on loss trajectory."""
    final_train = train_losses[-1]
    final_val = val_losses[-1]
    min_val = min(val_losses)
    
    # Val loss increasing while train drops -> Overfitting
    if final_val > min_val * 1.15:
        return "Overfitting"
    # Loss stays high/flat -> Underfitting
    elif final_train > 0.05 and final_val > 0.05:
        return "Underfitting"
    else:
        return "Optimal / Early Stopped"


# --- 3. Neural Architectures ---
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_dim=50, num_layers=2, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Sequential(nn.Linear(hidden_dim, 25), nn.ReLU(), nn.Linear(25, 1))

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class GRUModel(nn.Module):
    def __init__(self, input_size=1, hidden_dim=50, num_layers=2, dropout=0.2):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Sequential(nn.Linear(hidden_dim, 25), nn.ReLU(), nn.Linear(25, 1))

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])


class BiLSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_dim=50, num_layers=2, dropout=0.2):
        super(BiLSTMModel, self).__init__()
        self.bilstm = nn.LSTM(input_size, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout, bidirectional=True)
        self.fc = nn.Sequential(nn.Linear(hidden_dim * 2, 25), nn.ReLU(), nn.Linear(25, 1))

    def forward(self, x):
        out, _ = self.bilstm(x)
        return self.fc(out[:, -1, :])


class TransformerModel(nn.Module):
    def __init__(self, input_size=1, d_model=32, nhead=4, num_layers=2, dropout=0.2):
        super(TransformerModel, self).__init__()
        self.input_projection = nn.Linear(input_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=64, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Sequential(nn.Linear(d_model, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, x):
        x_proj = self.input_projection(x)
        out = self.transformer(x_proj)
        return self.fc(out[:, -1, :])


# --- 4. Data Preparation ---
def prepare_data_for_ticker(df, ticker, sequence_length=60, train_split=0.8):
    sub_df = df[df["Ticker"] == ticker].copy() if "Ticker" in df.columns else df.copy()
    sub_df = sub_df.sort_values("Date").reset_index(drop=True)
    price_col = "Adj Close" if "Adj Close" in sub_df.columns else "Close"
    data = sub_df[[price_col]].values

    if len(data) <= sequence_length + 20:
        return None, None, None, None, None

    split_idx = int(len(data) * train_split)
    train_data = data[:split_idx]
    test_data = data[split_idx:]

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_train = scaler.fit_transform(train_data)
    scaled_test = scaler.transform(test_data)

    def create_seqs(dataset, sequence_length):
    # Vectorized windowing without Python loops
        windows = np.lib.stride_tricks.sliding_window_view(dataset[:, 0], sequence_length + 1)
        X = windows[:, :-1, np.newaxis]
        y = windows[:, -1]
        return X, y

    X_train, y_train = create_seqs(scaled_train, sequence_length)
    X_test, y_test = create_seqs(scaled_test, sequence_length)

    return X_train, y_train, X_test, y_test, scaler


# --- 5. Main Execution ---
def main():
    config = load_config()
    processed_dir = config["data"]["processed_dir"]
    models_dir = "models"
    reports_dir = "reports"
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    features_path = os.path.join(processed_dir, "features.parquet")
    features_df = pd.read_parquet(features_path)
    tickers = features_df["Ticker"].unique().tolist() if "Ticker" in features_df.columns else ["ASSET"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_factories = {
        "LSTM": lambda: LSTMModel(),
        "GRU": lambda: GRUModel(),
        "BiLSTM": lambda: BiLSTMModel(),
        "Transformer": lambda: TransformerModel()
    }

    all_dl_metrics = []

    for ticker in tickers:
        print(f"\nProcessing ticker: {ticker}")
        X_train, y_train, X_test, y_test, scaler = prepare_data_for_ticker(features_df, ticker)
        if X_train is None:
            print(f"Skipping {ticker} (insufficient data)")
            continue

        train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32).unsqueeze(1))
        test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32).unsqueeze(1))

        train_loader = DataLoader(train_dataset, batch_size=128, 
            shuffle=False, pin_memory=True if device.type == "cuda" else False,num_workers=2 )
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        for model_name, model_fn in model_factories.items():
            model = model_fn().to(device)
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            early_stopping = EarlyStopping(patience=10)

            train_losses, val_losses = [], []

            for epoch in range(100):
                if (epoch + 1) % 10 == 0:
                    print(f"    Epoch {epoch+1}/100 | Train Loss: {t_loss:.6f} | Val Loss: {v_loss:.6f}")
                model.train()
                t_loss = 0.0
                for X_batch, y_batch in train_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    optimizer.zero_grad()
                    out = model(X_batch)
                    loss = criterion(out, y_batch)
                    loss.backward()
                    optimizer.step()
                    t_loss += loss.item() * X_batch.size(0)

                t_loss /= len(train_loader.dataset)
                train_losses.append(t_loss)

                # Validation phase
                model.eval()
                v_loss = 0.0
                with torch.no_grad():
                    for X_batch, y_batch in test_loader:
                        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                        out = model(X_batch)
                        v_loss += criterion(out, y_batch).item() * X_batch.size(0)

                v_loss /= len(test_loader.dataset)
                val_losses.append(v_loss)

                early_stopping(v_loss, model)
                if early_stopping.early_stop:
                    break

            # Plot Loss Curve & Diagnose Fit
            save_loss_curve(train_losses, val_losses, ticker, model_name)
            fit_diagnosis = diagnose_fit(train_losses, val_losses)

            # Load Best Model Weights
            if early_stopping.best_model_weights:
                model.load_state_dict(early_stopping.best_model_weights)

            # Metrics
            model.eval()
            X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
            with torch.no_grad():
                preds = model(X_test_tensor).cpu().numpy()

            preds_inv = scaler.inverse_transform(preds)
            actual_inv = scaler.inverse_transform(y_test.reshape(-1, 1))

            all_dl_metrics.append({
                "Ticker": ticker,
                "Model": model_name,
                "MAE": float(np.mean(np.abs(actual_inv - preds_inv))),
                "RMSE": float(np.sqrt(np.mean((actual_inv - preds_inv) ** 2))),
                "Fit Diagnosis": fit_diagnosis
            })

            torch.save(model.state_dict(), os.path.join(models_dir, f"{ticker.lower()}_{model_name.lower()}_model.pt"))

    # Save metrics summary
    dl_df = pd.DataFrame(all_dl_metrics)
    dl_df.to_csv(os.path.join(reports_dir, "dl_model_comparison.csv"), index=False)
    joblib.dump(all_dl_metrics, os.path.join(models_dir, "dl_metrics.pkl"))


if __name__ == "__main__":
    main()