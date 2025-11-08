"""
backend/vmd_lstm_model.py

CPU-friendly VMD + LSTM pipeline.

Requirements:
pip install vmdpy tensorflow scikit-learn matplotlib

Usage:
python vmd_lstm_model.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
import vmdpy
import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense
# from tensorflow.keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import EarlyStopping

# ---------- Config (CPU-friendly) ----------
K = 5                 # number of modes (IMFs)
ALPHA = 2000.0        # data-fidelity term (reduced from paper to speed up)
TAU = 0.0
DC = 0
INIT = 1
TOL = 1e-7

SEQ_LEN = 30          # input sequence length (days)
LSTM_UNITS = 32       # small LSTM for CPU
BATCH_SIZE = 32
EPOCHS = 50           # small number for fast CPU training
TEST_SPLIT = 0.2

DATA_DIR = "data"
MODELS_DIR = "models/vmd_lstm"
PLOTS_DIR = "plots"

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)


# ---------- Utility functions ----------
def create_supervised(series, seq_len=30):
    X, y = [], []
    for i in range(len(series) - seq_len):
        X.append(series[i:i + seq_len])
        y.append(series[i + seq_len])
    X = np.array(X)
    y = np.array(y)
    return X, y

def direction_accuracy(y_true, y_pred):
    # proportion where direction (up/down) is same
    return np.mean((np.sign(np.diff(y_true)) == np.sign(np.diff(y_pred))).astype(int))

def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    with np.errstate(divide='ignore', invalid='ignore'):
        res = np.abs((y_true - y_pred) / y_true)
        res = res[~np.isinf(res)]
        return np.nanmean(res) * 100.0


# ---------- LSTM model builder ----------
def build_lstm(seq_len, features=1, units=32):
    model = Sequential([
        LSTM(units, input_shape=(seq_len, features)),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model


# ---------- VMD + LSTM pipeline for one ticker ----------
def train_vmd_lstm_for_ticker(ticker_csv_path):
    print(f"\n=== Processing {ticker_csv_path} ===")
    df = pd.read_csv(ticker_csv_path)
    close = df["Close"].values.astype(float)

    # Normalize original close for plotting baseline later
    close_scaler = MinMaxScaler()
    close_scaled_all = close_scaler.fit_transform(close.reshape(-1, 1)).flatten()

    # 1) VMD decomposition
    print("Running VMD decomposition (this may take a few seconds)...")
    # vmdpy expects a 1D numpy array
    signal = close.copy().astype(float)
    # u, u_hat = vmdpy.VMD(signal, ALPHA, TAU, K, DC, INIT, TOL)
    # # u shape: (K, N)
    # u = vmdpy.VMD(signal, ALPHA, TAU, K, DC, INIT, TOL)
    u, u_hat, omega = vmdpy.VMD(signal, ALPHA, TAU, K, DC, INIT, TOL)
    imfs = np.array(u)
    # u is array of modes
    # imfs = np.array(u)  # shape (K, N)

    print(f"Obtained {imfs.shape[0]} IMFs, length {imfs.shape[1]}")

    # Prepare per-IMF LSTM
    imf_models = []
    imf_scalers = []

    # For evaluation aggregators
    reconstructed_pred = None
    reconstructed_true = None

    # For each IMF: train small LSTM to predict next value of that IMF
    for idx in range(imfs.shape[0]):
        mode = imfs[idx, :]
        mode = mode.reshape(-1, 1)

        scaler = MinMaxScaler()
        mode_scaled = scaler.fit_transform(mode).flatten()

        # supervised framing
        X, y = create_supervised(mode_scaled, seq_len=SEQ_LEN)
        # split
        split_idx = int(len(X) * (1 - TEST_SPLIT))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        # reshape for LSTM: (samples, timesteps, features)
        X_train = X_train.reshape((-1, SEQ_LEN, 1))
        X_test = X_test.reshape((-1, SEQ_LEN, 1))

        model = build_lstm(SEQ_LEN, features=1, units=LSTM_UNITS)

        # early stopping helps CPU-run time
        es = EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True, verbose=0)
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=[es],
            verbose=0
        )

        # Save model + scaler
        model_path = os.path.join(MODELS_DIR, f"lstm_mode_{idx}.h5")
        scaler_path = os.path.join(MODELS_DIR, f"scaler_mode_{idx}.pkl")
        model.save(model_path)
        joblib.dump(scaler, scaler_path)
        print(f"Saved model & scaler for IMF {idx}: {model_path}")

        # Evaluate on test set: reconstruct IMF predictions
        y_pred_test = model.predict(X_test).flatten()
        # invert scaling
        y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
        y_pred_inv = scaler.inverse_transform(y_pred_test.reshape(-1, 1)).flatten()

        # For reconstructing whole-signal predictions, we'll pad (since test set is shorter),
        # we'll predict the last part of the series by sliding window later.
        if reconstructed_pred is None:
            reconstructed_pred = np.zeros_like(close, dtype=float)
            reconstructed_true = np.zeros_like(close, dtype=float)

        # We will do iterative prediction for the last (len(y_test)+SEQ_LEN) positions:
        # Create sequences covering the region corresponding to the test set
        # index start for test set in full series:
        test_start = split_idx + SEQ_LEN
        # place inverse predictions into the reconstructed arrays at proper positions
        recon_indices = np.arange(test_start, test_start + len(y_pred_inv))
        reconstructed_pred[recon_indices] += y_pred_inv
        reconstructed_true[recon_indices] += y_test_inv

        imf_models.append(model)
        imf_scalers.append(scaler)

    # Now compare reconstructed_true vs reconstructed_pred where non-zero
    valid_mask = reconstructed_true != 0
    if valid_mask.sum() == 0:
        print("Warning: no test overlap to compute metrics. Skipping evaluation.")
    else:
        y_true = reconstructed_true[valid_mask]
        y_pred = reconstructed_pred[valid_mask]

        # Since these are IMF values, sum of IMFs equals original close.
        # For direct price-level metrics, compare reconstructed sums of IMFs to original close window.
        # Create price-level reconstructed predictions by mapping IMF predictions back:
        # For simplicity, compute predicted close by adding predicted IMF contributions where available,
        # and compare to actual close over that same range.
        actual_close_segment = close[valid_mask]
        # scale back to original units - we trained on IMF values already in price units
        # compute metrics
        mse = mean_squared_error(actual_close_segment, y_pred)
        mae = mean_absolute_error(actual_close_segment, y_pred)
        mape_val = mape(actual_close_segment, y_pred)
        dir_acc = direction_accuracy(actual_close_segment, y_pred)

        print("\nEvaluation (IMF-aggregated predictions) — note: rough estimate")
        print(f"MSE: {mse:.6f}, MAE: {mae:.6f}, MAPE: {mape_val:.3f}% , Direction Acc: {dir_acc:.3f}")

    # ---------- Generate final 1-day and iterative 5-day prediction ----------
    def predict_future_days(n_days=5):
        # For each IMF, perform iterative prediction starting from last SEQ_LEN values
        imf_future_preds = []
        for idx in range(imfs.shape[0]):
            model = imf_models[idx]
            scaler = imf_scalers[idx]
            mode = imfs[idx, :].reshape(-1, 1)
            mode_scaled = scaler.transform(mode).flatten()

            # seed sequence: last SEQ_LEN of scaled mode
            seq = mode_scaled[-SEQ_LEN:].tolist()
            preds = []
            for _ in range(n_days):
                x = np.array(seq[-SEQ_LEN:]).reshape((1, SEQ_LEN, 1))
                p_scaled = model.predict(x).flatten()[0]
                preds.append(p_scaled)
                seq.append(p_scaled)  # iterative
            # invert scale
            preds_inv = scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
            imf_future_preds.append(preds_inv)

        # sum across IMFs to get price predictions
        imf_future_preds = np.array(imf_future_preds)  # shape (K, n_days)
        price_preds = imf_future_preds.sum(axis=0)
        return price_preds

    pred_1 = predict_future_days(1)[0]
    pred_5 = predict_future_days(5)

    print(f"\nPredicted next-day close (estimate): {pred_1:.4f}")
    print(f"Predicted next-5 days (est): {np.array2string(pred_5, precision=4)}")

    # Save a small summary artifact
    summary = {
        "ticker": os.path.basename(ticker_csv_path),
        "pred_next_day": float(pred_1),
        "pred_next_5": [float(x) for x in pred_5]
    }
    joblib.dump(summary, os.path.join(MODELS_DIR, f"{os.path.basename(ticker_csv_path)}_vmd_lstm_summary.pkl"))
    print(f"Saved summary to models/vmd_lstm/{os.path.basename(ticker_csv_path)}_vmd_lstm_summary.pkl")

    # Plot last N days actual vs predicted (using IMF-aggregated predicted region)
    try:
        idxs = np.where(valid_mask)[0]
        if len(idxs) > 0:
            plt.figure(figsize=(10, 4))
            plt.plot(idxs, close[idxs], label="Actual Close")
            plt.plot(idxs, reconstructed_pred[idxs], label="IMF-agg Pred")
            plt.title(f"{os.path.basename(ticker_csv_path)} - IMF aggregated predictions (test region)")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(PLOTS_DIR, f"{os.path.basename(ticker_csv_path)}_vmd_lstm_plot.png"))
            plt.close()
            print(f"Saved plot to {PLOTS_DIR}")
    except Exception as e:
        print("Plotting failed:", e)

    return summary


# ---------- Main: process both tickers ----------
if __name__ == "__main__":
    tickers = ["data/TCS.NS_processed.csv", "data/TATAMOTORS.NS_processed.csv"]
    all_summaries = []
    for t in tickers:
        s = train_vmd_lstm_for_ticker(t)
        all_summaries.append(s)
    joblib.dump(all_summaries, os.path.join(MODELS_DIR, "all_vmd_lstm_summaries.pkl"))
    print("\nAll done.")
