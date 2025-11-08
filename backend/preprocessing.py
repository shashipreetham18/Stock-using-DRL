import pandas as pd
import numpy as np

# ---------- Technical Indicators ----------

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def compute_sma(series, period=14):
    return series.rolling(window=period).mean()

def compute_momentum(series, period=10):
    return series.diff(period)

# ---------- Feature Engineering Function ----------

def preprocess_stock_data(csv_path):
    df = pd.read_csv(csv_path)

    # Convert numeric columns safely (Adj Close removed)
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows where market data is missing
    df = df.dropna().reset_index(drop=True)

    # Basic features
    df["Return"] = df["Close"].pct_change()

    # Technical Indicators
    df["RSI"] = compute_rsi(df["Close"])
    df["SMA_14"] = compute_sma(df["Close"], 14)
    df["SMA_50"] = compute_sma(df["Close"], 50)
    df["Momentum"] = compute_momentum(df["Close"], 10)

    # Final cleanup
    df = df.dropna().reset_index(drop=True)

    return df


# ---------- Save Preprocessed Files for Both Stocks ----------

if __name__ == "__main__":
    tickers = ["TCS.NS", "TATAMOTORS.NS"]
    for t in tickers:
        df = preprocess_stock_data(f"data/{t}.csv")
        df.to_csv(f"data/{t}_processed.csv", index=False)
        print(f"Saved data/{t}_processed.csv")
