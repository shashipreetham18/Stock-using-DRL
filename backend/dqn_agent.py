import os
import sys
import time
import random
from collections import deque

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import vmdpy

# tensorflow/keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import InputLayer, LSTM, Dense
from keras.optimizers import Adam

from sklearn.preprocessing import MinMaxScaler

# ---------------- CONFIG (edit if you changed training params) ----------------
K = 5                      # number of IMFs used in VMD step
SEQ_LEN = 30               # LSTM input window used in training
LSTM_UNITS = 32            # LSTM units used in training (must match training code)
MODELS_DIR = "models/vmd_lstm"
SCALER_FMT = "scaler_mode_{}.pkl"
LSTM_FMT = "lstm_mode_{}.h5"   # weights inside; load_weights() will read from .h5
DATA_DIR = "data"
PLOTS_DIR = "plots"
OUTPUT_POLICY = "models/dqn_policy.h5"

# DQN settings (CPU-friendly)
STATE_SIZE = 6             # RSI, SMA14, SMA50, Momentum, PredTrend, Sentiment
ACTION_SIZE = 3            # HOLD, LONG, SHORT
EPISODES = 150
GAMMA = 0.99
LR = 1e-3
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 0.995
MEMORY_CAPACITY = 5000
BATCH_SIZE = 32
TRAIN_START = 200
TRAIN_EVERY = 4

# processed ticker (change to whichever processed CSV you want)
TICKER_PROCESSED = "TCS.NS_processed.csv"

os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(os.path.dirname(OUTPUT_POLICY) or ".", exist_ok=True)


print("\n✅ DQN Agent (final) starting up...\n")

# ---------------- Helper: build LSTM architecture and load weights ----------------
def build_inference_lstm(seq_len=SEQ_LEN, units=LSTM_UNITS):
    """
    Build the same LSTM architecture used during training.
    """
    model = Sequential()
    model.add(InputLayer(input_shape=(seq_len, 1)))
    model.add(LSTM(units, return_sequences=False))
    model.add(Dense(1))
    # compile not required for inference, but compile to avoid warnings
    model.compile(optimizer=Adam(1e-3), loss="mse")
    return model

def load_lstm_weights(path, seq_len=SEQ_LEN, units=LSTM_UNITS):
    """
    Build the LSTM architecture and load weights from 'path'.
    Works when 'path' is a full model .h5 containing weights.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"LSTM weight/model file not found: {path}")
    model = build_inference_lstm(seq_len=seq_len, units=units)
    try:
        model.load_weights(path)
    except Exception as e:
        # fallback: try by_name
        try:
            model.load_weights(path, by_name=True)
        except Exception as e2:
            raise RuntimeError(f"Failed to load weights from {path}:\n{e}\n{e2}")
    return model

# ---------------- Compute predicted trend using VMD + IMF-LSTMs ----------------
def compute_predicted_trend_from_models(close_arr):
    """
    Run VMD on close_arr, then for each IMF load its scaler & LSTM weights
    and do stepwise predictions to produce predicted price series -> trend.
    Returns: trend array same length as close_arr (NaN for first SEQ_LEN positions).
    """
    print("🔄 Running VMD decomposition + per-IMF inference...")
    try:
        u, u_hat, omega = vmdpy.VMD(close_arr.astype(float), 2000.0, 0.0, K, 0, 1, 1e-7)
    except Exception as e:
        # Some vmdpy versions return just 'u' -- try single return
        try:
            u = vmdpy.VMD(close_arr.astype(float), 2000.0, 0.0, K, 0, 1, 1e-7)
            # ensure shape
            if isinstance(u, (list, tuple)):
                u = np.array(u)
            else:
                u = np.array(u)
            # fallback: continue
        except Exception as e2:
            raise RuntimeError(f"VMD decomposition failed: {e}\nFallback also failed: {e2}")

    imfs = np.array(u)   # shape (K, N) expected
    N = imfs.shape[1]
    preds = np.full(N, np.nan, dtype=float)

    # load scalers and LSTM models
    scalers = []
    lstm_models = []
    for i in range(K):
        scaler_path = os.path.join(MODELS_DIR, SCALER_FMT.format(i))
        lstm_path = os.path.join(MODELS_DIR, LSTM_FMT.format(i))
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Missing scaler: {scaler_path}")
        if not os.path.exists(lstm_path):
            raise FileNotFoundError(f"Missing LSTM file: {lstm_path}")
        scalers.append(joblib.load(scaler_path))
        lstm_models.append(load_lstm_weights(lstm_path, seq_len=SEQ_LEN, units=LSTM_UNITS))
        print(f"  ✓ Loaded IMF {i} scaler & LSTM")

    # Predict: for each timestep t >= SEQ_LEN, produce predicted price sum across IMFs
    for t in range(SEQ_LEN, N):
        total = 0.0
        for i in range(K):
            mode = imfs[i, :].reshape(-1, 1)
            scaler = scalers[i]
            # scaler expects 2D
            try:
                mode_scaled = scaler.transform(mode).flatten()
            except Exception:
                mode_scaled = scaler.transform(mode.reshape(-1, 1)).flatten()
            seq = mode_scaled[t-SEQ_LEN:t].reshape(1, SEQ_LEN, 1)
            p_scaled = lstm_models[i].predict(seq, verbose=0).flatten()[0]
            # inverse scale
            p_real = scaler.inverse_transform(np.array(p_scaled).reshape(-1, 1)).flatten()[0]
            total += p_real
        preds[t] = total

    # compute trend
    with np.errstate(divide='ignore', invalid='ignore'):
        trend = (preds - close_arr) / close_arr
    return trend

# ---------------- Build dataset (states + next_returns) ----------------
def build_dataset(csv_path):
    """
    Loads CSV (processed) and builds:
      - df
      - states_scaled (MinMax scaled)
      - next_returns (for reward shaping)
      - valid_idxs (mapping of state index -> df index)
    """
    print(f"\n📌 Loading processed CSV: {csv_path}")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Processed CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    # ensure numeric
    for c in ["Close", "RSI", "SMA_14", "SMA_50", "Momentum"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna().reset_index(drop=True)
    close = df["Close"].values

    # compute predicted trend
    pred_trend = compute_predicted_trend_from_models(close)

    # Sentiment column optional
    if "Sentiment" in df.columns:
        sentiment = df["Sentiment"].values
    else:
        sentiment = np.zeros(len(df))

    states = []
    next_returns = []
    valid_idxs = []
    for i in range(len(df)-1):
        if i >= len(pred_trend) or np.isnan(pred_trend[i]):
            continue
        s = [
            df.loc[i, "RSI"],
            df.loc[i, "SMA_14"],
            df.loc[i, "SMA_50"],
            df.loc[i, "Momentum"],
            pred_trend[i],
            float(sentiment[i])
        ]
        states.append(s)
        nr = (df.loc[i+1, "Close"] - df.loc[i, "Close"]) / df.loc[i, "Close"]
        next_returns.append(nr)
        valid_idxs.append(i)

    if len(states) == 0:
        raise RuntimeError("No valid states built — check pred_trend and data length")

    states = np.array(states, dtype=float)
    next_returns = np.array(next_returns, dtype=float)
    valid_idxs = np.array(valid_idxs, dtype=int)

    scaler = MinMaxScaler()
    states_scaled = scaler.fit_transform(states)

    print(f"Built dataset: {len(states_scaled)} rows.")
    return df, states_scaled, next_returns, valid_idxs, scaler

# ---------------- Simple Trading Environment ----------------
class TradingEnv:
    def __init__(self, df, valid_idxs, initial_cash=10000.0):
        self.df = df
        self.valid_idxs = valid_idxs
        self.initial_cash = initial_cash
        self.reset()

    def reset(self):
        self.cash = self.initial_cash
        self.holdings = 0.0
        self.step_idx = 0
        self.portfolio_value = self.initial_cash
        return self._get_obs()

    def _get_obs(self):
        idx = self.valid_idxs[self.step_idx]
        price = float(self.df.loc[idx, "Close"])
        return {"price": price, "cash": self.cash, "holdings": self.holdings, "pv": self.cash + self.holdings * price}

    def step(self, action):
        idx = self.valid_idxs[self.step_idx]
        price = float(self.df.loc[idx, "Close"])
        prev_pv = self.cash + self.holdings * price

        MAX_POSITION = self.initial_cash / price  # limit to full not infinite leverage

        if action == 1:  # BUY
            qty = int(self.cash // price)
            if qty > 0:
                self.holdings = min(self.holdings + qty, MAX_POSITION)
                self.cash -= qty * price

        elif action == 2:  # SHORT
            qty = int(self.cash // price)
            if qty > 0:
                self.holdings = max(self.holdings - qty, -MAX_POSITION)
                self.cash += qty * price

        new_pv = self.cash + self.holdings * price

    # reward = change in portfolio value
        reward = new_pv - prev_pv

        self.step_idx += 1
        done = (self.step_idx >= len(self.valid_idxs)-1)
        return reward, done


# ---------------- DQN Agent ----------------
class DQNAgent:
    def __init__(self, state_dim, action_dim=ACTION_SIZE):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.memory = deque(maxlen=MEMORY_CAPACITY)
        self.model = self._build_net()
        self.target = self._build_net()
        self.update_target()
        self.eps = EPS_START

    def _build_net(self):
        net = Sequential()
        net.add(InputLayer(input_shape=(self.state_dim,)))
        net.add(Dense(128, activation="relu"))
        net.add(Dense(64, activation="relu"))
        net.add(Dense(self.action_dim, activation="linear"))
        net.compile(optimizer=Adam(LR), loss="mse")
        return net

    def update_target(self):
        self.target.set_weights(self.model.get_weights())

    def act(self, state):
        if np.random.rand() < self.eps:
            return np.random.randint(self.action_dim)
        q = self.model.predict(state.reshape(1, -1), verbose=0)[0]
        return int(np.argmax(q))

    def remember(self, s, a, r, s2, done):
        self.memory.append((s, a, r, s2, done))

    def replay(self, batch_size=BATCH_SIZE):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        states = np.array([m[0] for m in minibatch])
        actions = np.array([m[1] for m in minibatch])
        rewards = np.array([m[2] for m in minibatch])
        next_states = np.array([m[3] for m in minibatch])
        dones = np.array([m[4] for m in minibatch])

        q_next = self.target.predict(next_states, verbose=0)
        q_target = self.model.predict(states, verbose=0)
        for i in range(len(minibatch)):
            if dones[i]:
                q_target[i, actions[i]] = rewards[i]
            else:
                q_target[i, actions[i]] = rewards[i] + GAMMA * np.max(q_next[i])
        self.model.train_on_batch(states, q_target)

# ---------------- Training ----------------
def train_dqn_for_ticker(processed_csv_path):
    df, states, next_returns, valid_idxs, scaler = build_dataset(processed_csv_path)
    env = TradingEnv(df, valid_idxs, initial_cash=10000.0)
    agent = DQNAgent(state_dim=states.shape[1], action_dim=ACTION_SIZE)

    print("\n🚀 Starting DQN training ...\n")
    for ep in range(EPISODES):
        env.reset()
        pv_trace = []
        for t in range(len(states)-1):
            s = states[t]
            a = agent.act(s)
            reward_env, done = env.step(a)

            # Use future-return based reward (next_returns) scaled; penalize negative more
            # r = next_returns[t] * 100.0
            # if r < 0:
            #     r *= 2.0
            trend_strength = abs(states[t][4])  # states[t][4] = predicted trend
            transaction_cost = 0.0008           # 0.08% per trade

            # reward weighted by trend confidence
            r = next_returns[t] * trend_strength * 200  

            # transaction penalty (reduces over-trading)
            if a != 0:
                r -= transaction_cost

            # discourage holding extreme long/short
            position_penalty = abs(env.holdings) * 0.0001
            r -= position_penalty


            s2 = states[t+1] if t+1 < len(states) else states[t]
            agent.remember(s, a, r, s2, done)

            if len(agent.memory) > TRAIN_START and (t % TRAIN_EVERY == 0):
                agent.replay(BATCH_SIZE)

            pv_trace.append(env._get_obs()["pv"])

            if done:
                break

        # agent.eps = max(EPS_END, agent.eps * EPS_DECAY)
        # agent.update_target()
        # last_pv = pv_trace[-1] if len(pv_trace) > 0 else env._get_obs()["pv"]
        # print(f"Episode {ep+1}/{EPISODES}  |  Final PV: {last_pv:.2f}  |  Eps: {agent.eps:.3f}")
        agent.eps = max(EPS_END, agent.eps * EPS_DECAY)

        # ✅ TARGET NETWORK UPDATE EVERY 10 EPISODES (stabilizes learning)
        if ep % 10 == 0:
            agent.update_target()

        last_pv = pv_trace[-1] if len(pv_trace) > 0 else env._get_obs()["pv"]
        print(f"Episode {ep+1}/{EPISODES}  |  Final PV: {last_pv:.2f}  |  Eps: {agent.eps:.3f}")


    print("\n✅ Training finished. Running deterministic evaluation ...")
    # deterministic evaluation (eps=0)
    agent.eps = 0.0
    env.reset()
    pv_trace = []
    for t in range(len(states)-1):
        s = states[t]
        q = agent.model.predict(s.reshape(1, -1), verbose=0)[0]
        a = int(np.argmax(q))
        reward_env, done = env.step(a)
        pv_trace.append(env._get_obs()["pv"])
        if done:
            break

    final_pv = pv_trace[-1] if len(pv_trace) > 0 else env._get_obs()["pv"]
    cum_return_pct = (final_pv - 10000.0) / 10000.0 * 100.0

    # Save policy
    try:
        agent.model.save(OUTPUT_POLICY)
        print(f"Saved trained DQN policy -> {OUTPUT_POLICY}")
    except Exception as e:
        print("⚠ Failed to save policy:", e)

    print(f"\nFinal deterministic evaluation: Final PV = {final_pv:.2f}, Cumulative Return = {cum_return_pct:.2f}%")

    # Save plot
    try:
        plt.figure(figsize=(10, 4))
        plt.plot(pv_trace, label="DQN PV")
        plt.hlines(10000, 0, len(pv_trace)-1, linestyles="dashed", label="Start PV")
        plt.title(f"DQN portfolio trace - {os.path.basename(processed_csv_path)}")
        plt.xlabel("Step")
        plt.ylabel("Portfolio value")
        plt.legend()
        plt.grid(True)
        plot_path = os.path.join(PLOTS_DIR, f"{os.path.basename(processed_csv_path)}_dqn_pv.png")
        plt.savefig(plot_path)
        print(f"Saved portfolio plot -> {plot_path}")
    except Exception as e:
        print("⚠ Failed to save plot:", e)

# ---------------- Main ----------------
if __name__ == "__main__":
    processed_path = os.path.join(DATA_DIR, TICKER_PROCESSED)
    try:
        train_dqn_for_ticker(processed_path)
    except Exception as e:
        print("\n❌ Training aborted due to error:")
        print(e)
        print("\nCheck that models are present under", MODELS_DIR, "and that the processed CSV exists:", processed_path)
        sys.exit(1)
