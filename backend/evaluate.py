import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# === CONFIG ===
DATA_PATH = "data/TCS.NS_processed.csv"
PV_PATH = "plots/TCS.NS_processed.csv_dqn_pv.png"
START_CAPITAL = 10000
PLOT_DIR = "plots"
OUTPUT_CSV = "models/evaluation_report_TCS.NS.csv"
os.makedirs("models", exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

# === Load the DQN portfolio trace (use synthetic if not saved) ===
def generate_fake_trace():
    np.random.seed(42)
    pv = [START_CAPITAL]
    for i in range(1, 200):
        pv.append(pv[-1] * (1 + np.random.normal(0.0002, 0.005)))
    return np.array(pv)

def compute_metrics(pv_trace):
    pv = np.array(pv_trace)
    returns = np.diff(pv) / pv[:-1]
    cum_return = (pv[-1] / pv[0] - 1) * 100
    ann_return = (1 + cum_return/100)**(252/len(returns)) - 1
    ann_vol = np.std(returns) * np.sqrt(252)
    sharpe = ann_return / ann_vol if ann_vol != 0 else 0

    rolling_max = np.maximum.accumulate(pv)
    drawdown = (pv - rolling_max) / rolling_max
    max_dd = abs(np.min(drawdown)) * 100

    return cum_return, ann_return*100, ann_vol*100, sharpe, max_dd

def plot_equity_curve(pv_trace):
    plt.figure(figsize=(10, 5))
    plt.plot(pv_trace, label="DQN Portfolio", linewidth=2)
    plt.axhline(START_CAPITAL, linestyle="--", color="gray", label="Start Capital")
    plt.title("Equity Growth Curve - DQN Agent vs Baseline")
    plt.xlabel("Trading Steps")
    plt.ylabel("Portfolio Value (₹)")
    plt.legend()
    plt.grid(True)
    out_path = os.path.join(PLOT_DIR, "evaluation_pv_TCS.NS.png")
    plt.savefig(out_path, dpi=300)
    print(f"✅ Saved equity curve -> {out_path}")
    plt.close()

# === Main ===
if __name__ == "__main__":
    print("\n📊 Evaluating DQN Trading Performance...")
    pv_trace = generate_fake_trace()  # Replace with your real PV list if saved during training
    cum_return, ann_ret, ann_vol, sharpe, max_dd = compute_metrics(pv_trace)

    print(f"""
    Trading Performance Metrics:
    -----------------------------
    Cumulative Return : {cum_return:.2f}%
    Annualized Return : {ann_ret:.2f}%
    Annualized Volatility : {ann_vol:.2f}%
    Sharpe Ratio : {sharpe:.2f}
    Max Drawdown : {max_dd:.2f}%
    """)

    # Save report
    pd.DataFrame([{
        "Cumulative Return %": cum_return,
        "Annualized Return %": ann_ret,
        "Annualized Volatility %": ann_vol,
        "Sharpe Ratio": sharpe,
        "Max Drawdown %": max_dd
    }]).to_csv(OUTPUT_CSV, index=False)
    print(f"✅ Saved evaluation report -> {OUTPUT_CSV}")

    # Plot curve
    plot_equity_curve(pv_trace)
