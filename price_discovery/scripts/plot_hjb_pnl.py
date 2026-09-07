import csv
import sys

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

results_dir = sys.argv[1]

pnl_rows = []
with open(f"{results_dir}/pnl_history.csv") as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row["agent"] == "hjb_mm":
            pnl_rows.append(row)

assert pnl_rows, f"no hjb_mm rows in {results_dir}/pnl_history.csv"
sim_t = [float(r["sim_t"]) for r in pnl_rows]
inventory = [int(r["inventory"]) for r in pnl_rows]
total_pnl = [float(r["total_pnl"]) for r in pnl_rows]
mid = [float(r["mid"]) for r in pnl_rows]

order_rows = []
with open(f"{results_dir}/order_events.csv") as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row["agent"] == "hjb_mm":
            order_rows.append(row)

fig = plt.figure(figsize=(14, 10))
gs = fig.add_gridspec(3, 1, height_ratios=[2, 1, 1], hspace=0.15)

ax_quotes = fig.add_subplot(gs[0])
ax_inv = fig.add_subplot(gs[1], sharex=ax_quotes)
ax_pnl = fig.add_subplot(gs[2], sharex=ax_quotes)

for r in order_rows:
    t = float(r["sim_t"])
    px = float(r["price"])
    ev = r["event"]
    side = int(r["side"])
    if ev == "sent":
        color = "#1f77b4" if side == 0 else "#d62728"
        ax_quotes.scatter(t, px, s=6, color=color, alpha=0.3, linewidth=0)
    elif ev == "filled":
        color = "#2ca02c" if side == 0 else "#ff7f0e"
        ax_quotes.scatter(t, px, s=60, color=color, marker="D", edgecolors="black", linewidth=0.5, zorder=5)

legend_elements = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor="#1f77b4", alpha=0.4, markersize=6, label="Bid sent"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor="#d62728", alpha=0.4, markersize=6, label="Ask sent"),
    Line2D([0], [0], marker="D", color="w", markerfacecolor="#2ca02c", markeredgecolor="black", markersize=8, label="Bid filled"),
    Line2D([0], [0], marker="D", color="w", markerfacecolor="#ff7f0e", markeredgecolor="black", markersize=8, label="Ask filled"),
    Line2D([0], [0], color="gray", linewidth=0.8, alpha=0.7, label="Mid price"),
]

ax_quotes.plot(sim_t, mid, color="gray", linewidth=0.6, alpha=0.7)
ax_quotes.set_ylabel("Price (ticks)")
ax_quotes.set_title("HJB Market Maker Quote Activity (200 ticks)")
ax_quotes.grid(True, alpha=0.3)
ax_quotes.legend(handles=legend_elements, loc="upper right", fontsize=8)

ax_inv.plot(sim_t, inventory, linewidth=0.5, color="#1f77b4")
ax_inv.set_ylabel("Inventory (lots)")
ax_inv.set_title("HJB Market Maker — Inventory over Time")
ax_inv.grid(True, alpha=0.3)
ax_inv.axhline(0, color="gray", linewidth=0.4, linestyle="--")

ax_pnl.plot(sim_t, total_pnl, linewidth=0.5, color="#2ca02c")
ax_pnl.set_xlabel("Simulation Time (ticks)")
ax_pnl.set_ylabel("Total PnL (ticks)")
ax_pnl.set_title("HJB Market Maker — Cumulative PnL")
ax_pnl.grid(True, alpha=0.3)

plt.savefig(f"{results_dir}/hjb_pnl.png", dpi=150)
print(f"saved {results_dir}/hjb_pnl.png")
