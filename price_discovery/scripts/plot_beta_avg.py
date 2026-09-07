import csv
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

results_base = sys.argv[1]
n_steps = int(sys.argv[2]) if len(sys.argv) > 2 else 40
out_path = sys.argv[3] if len(sys.argv) > 3 else os.path.join(results_base, "beta_avg.png")
title = sys.argv[4] if len(sys.argv) > 4 else "Beta avg"

n_episodes = 30

beta = np.full((n_episodes, n_steps), np.nan)
fill_qty = np.zeros((n_episodes, n_steps), dtype=int)
fill_sum = np.zeros((n_episodes, n_steps))

for ep in range(n_episodes):
    ep_dir = os.path.join(results_base, f"ep_{ep}")

    # Action history
    with open(os.path.join(ep_dir, "checkpoints", "informed_action_history.csv")) as f:
        for row in csv.DictReader(f):
            t = int(row["step"])
            if t < n_steps:
                beta[ep, t] = float(row["action"])

    # Order events
    with open(os.path.join(ep_dir, "order_events.csv")) as f:
        for row in csv.DictReader(f):
            if row["agent"] != "informed" or row["event"] != "filled":
                continue
            t = int(row["tag"])
            if t < n_steps:
                fill_qty[ep, t] += 1
                fill_sum[ep, t] += float(row["price"])

# Averages
beta_avg = np.nanmean(beta, axis=0)
qty_avg = np.mean(fill_qty, axis=0)
vwap = np.zeros(n_steps)
for t in range(n_steps):
    total_price_sum = fill_sum[:, t].sum()
    total_qty = fill_qty[:, t].sum()
    vwap[t] = total_price_sum / total_qty if total_qty > 0 else 0
vwap_mean = vwap[vwap > 0].mean() if np.any(vwap > 0) else 0
vwap[vwap == 0] = vwap_mean

fill_steps = [t for t in range(n_steps) if qty_avg[t] > 0]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Left: beta with std band
beta_std = np.nanstd(beta, axis=0)
ax1.plot(range(1, n_steps), beta_avg[1:], linewidth=0.8, color="#1f77b4")
ax1.fill_between(range(1, n_steps), beta_avg[1:] - beta_std[1:], beta_avg[1:] + beta_std[1:], alpha=0.2, color="#1f77b4")
ax1.set_xlabel("Step")
ax1.set_ylabel("Action (beta)")
ax1.set_title(f"Informed Trader — {title}")
ax1.grid(True, alpha=0.3)

# Right: fill qty (bars) + VWAP (line)
ax2.bar(fill_steps, [qty_avg[s] for s in fill_steps], width=0.6, color="#2ca02c", alpha=0.6, label="avg fill qty")
ax2_twin = ax2.twinx()
ax2_twin.plot(range(n_steps), vwap, color="#d62728", linewidth=1.5, marker="o", markersize=4, label="avg VWAP")
ax2_twin.set_ylabel("VWAP")
ax2.set_xlabel("Step")
ax2.set_ylabel("Avg fill quantity")
ax2.set_title("IT fills per step (avg over 30)")
ax2.grid(True, alpha=0.3)
lines1, labels1 = ax2.get_legend_handles_labels()
lines2, labels2 = ax2_twin.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=8)

plt.tight_layout()
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"saved {out_path}")
