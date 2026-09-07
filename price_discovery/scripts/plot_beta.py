import csv
import sys

import matplotlib.pyplot as plt
import numpy as np

csv_path = sys.argv[1]
out_path = sys.argv[2] if len(sys.argv) > 2 else csv_path.replace(".csv", ".png")
title = sys.argv[3] if len(sys.argv) > 3 else "Beta"
orders_path = sys.argv[4] if len(sys.argv) > 4 else None

steps = []
actions = []
with open(csv_path) as f:
    for row in csv.DictReader(f):
        steps.append(int(row["step"]))
        actions.append(float(row["action"]))

if orders_path:
    # Per-step fill data
    fill_count = {}
    fill_sum = {}
    with open(orders_path) as f:
        for row in csv.DictReader(f):
            if row["agent"] != "informed" or row["event"] != "filled":
                continue
            t = int(row["tag"])
            p = float(row["price"])
            fill_count[t] = fill_count.get(t, 0) + 1
            fill_sum[t] = fill_sum.get(t, 0) + p

    max_step = max(steps)
    qty = [fill_count.get(s, 0) for s in range(max_step + 1)]
    all_vwap = [fill_sum.get(s, 0) / fill_count.get(s, 1) if fill_count.get(s, 0) > 0 else 0 for s in range(max_step + 1)]
    nonzero = [v for v in all_vwap if v > 0]
    fill_gap = np.mean(nonzero) if nonzero else 0
    vwap = [v if v > 0 else fill_gap for v in all_vwap]
    fill_steps = [s for s in range(max_step + 1) if fill_count.get(s, 0) > 0]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Left: beta
    ax1.plot(steps, actions, linewidth=0.4, alpha=0.7, color="#1f77b4")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Action (beta)")
    ax1.set_title(f"Informed Trader — {title}")
    ax1.grid(True, alpha=0.3)

    # Right: fill qty (bars) + VWAP (line)
    ax2.bar(fill_steps, [fill_count[s] for s in fill_steps], width=0.6, color="#2ca02c", alpha=0.6, label="fill qty")
    ax2_twin = ax2.twinx()
    ax2_twin.plot(range(max_step + 1), vwap, color="#d62728", linewidth=1.5, marker="o", markersize=4, label="VWAP")
    ax2_twin.set_ylabel("VWAP")
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Fill quantity")
    ax2.set_title("IT fills per step")
    ax2.grid(True, alpha=0.3)
    # Legends merged
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=8)

else:
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(steps, actions, linewidth=0.4, alpha=0.7, color="#1f77b4")
    ax.set_xlabel("Step")
    ax.set_ylabel("Action (beta)")
    ax.set_title(f"Informed Trader — {title}")
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"saved {out_path}")
