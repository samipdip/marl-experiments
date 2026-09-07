import csv
import sys

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

results_base = sys.argv[1]
seeds = [int(s) for s in sys.argv[2:]]

fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(3, 1, height_ratios=[2, 1, 1], hspace=0.2)

ax_quotes = fig.add_subplot(gs[0])
ax_inv = fig.add_subplot(gs[1], sharex=ax_quotes)
ax_pnl = fig.add_subplot(gs[2], sharex=ax_quotes)

colors = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd"]

for i, s in enumerate(seeds):
    dir_name = f"{results_base}/1000steps_{s}"
    pnl_rows = []
    with open(f"{dir_name}/pnl_history.csv") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["agent"] == "hjb_mm":
                pnl_rows.append(row)
    sim_t = [float(r["sim_t"]) for r in pnl_rows]
    inventory = [int(r["inventory"]) for r in pnl_rows]
    total_pnl = [float(r["total_pnl"]) for r in pnl_rows]
    mid = [float(r["mid"]) for r in pnl_rows]

    order_rows = []
    with open(f"{dir_name}/order_events.csv") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["agent"] == "hjb_mm":
                order_rows.append(row)

    c = colors[i % len(colors)]
    mid_label = f"seed={s} mid" if i == 0 else f"seed={s} mid"
    ax_quotes.plot(sim_t, mid, color=c, linewidth=0.6, alpha=0.7, label=f"mid (seed {s})")

    for r in order_rows:
        t = float(r["sim_t"])
        px = float(r["price"])
        ev = r["event"]
        side = int(r["side"])
        if ev == "filled":
            fc = "#2ca02c" if side == 0 else "#ff7f0e"
            ax_quotes.scatter(t, px, s=40, color=fc, marker="D", edgecolors=c, linewidth=0.4, alpha=0.7, zorder=5)

    ax_inv.plot(sim_t, inventory, color=c, linewidth=0.6, alpha=0.7, label=f"seed {s}")
    ax_pnl.plot(sim_t, total_pnl, color=c, linewidth=0.6, alpha=0.7, label=f"seed {s}")

legend_elements = [
    Line2D([0], [0], marker="D", color="w", markerfacecolor="#2ca02c", markeredgecolor="gray", markersize=8, label="Bid filled"),
    Line2D([0], [0], marker="D", color="w", markerfacecolor="#ff7f0e", markeredgecolor="gray", markersize=8, label="Ask filled"),
]

ax_quotes.set_ylabel("Price (ticks)")
ax_quotes.set_title("HJB Market Maker — Mid Price (all seeds)")
ax_quotes.grid(True, alpha=0.3)
ax_quotes.legend(fontsize=7, ncol=2)

ax_inv.set_ylabel("Inventory (lots)")
ax_inv.set_title("HJB Market Maker — Inventory")
ax_inv.grid(True, alpha=0.3)
ax_inv.axhline(0, color="gray", linewidth=0.4, linestyle="--")
ax_inv.legend(fontsize=7)

ax_pnl.set_xlabel("Simulation Time (ticks)")
ax_pnl.set_ylabel("Total PnL (ticks)")
ax_pnl.set_title("HJB Market Maker — Cumulative PnL")
ax_pnl.grid(True, alpha=0.3)
ax_pnl.legend(fontsize=7)

plt.savefig(f"{results_base}/multi_seed_overlay.png", dpi=150)
print(f"saved {results_base}/multi_seed_overlay.png")

for s in seeds:
    dir_name = f"{results_base}/1000steps_{s}"
    pnl_rows = []
    with open(f"{dir_name}/pnl_history.csv") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["agent"] == "hjb_mm":
                pnl_rows.append(row)
    sim_t = [float(r["sim_t"]) for r in pnl_rows]
    inventory = [int(r["inventory"]) for r in pnl_rows]
    total_pnl = [float(r["total_pnl"]) for r in pnl_rows]
    mid = [float(r["mid"]) for r in pnl_rows]

    order_rows = []
    with open(f"{dir_name}/order_events.csv") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["agent"] == "hjb_mm":
                order_rows.append(row)

    fig2 = plt.figure(figsize=(14, 10))
    gs2 = fig2.add_gridspec(3, 1, height_ratios=[2, 1, 1], hspace=0.15)
    ax_q = fig2.add_subplot(gs2[0])
    ax_i = fig2.add_subplot(gs2[1], sharex=ax_q)
    ax_p = fig2.add_subplot(gs2[2], sharex=ax_q)

    for r in order_rows:
        t = float(r["sim_t"])
        px = float(r["price"])
        ev = r["event"]
        side = int(r["side"])
        if ev == "sent":
            color = "#1f77b4" if side == 0 else "#d62728"
            ax_q.scatter(t, px, s=5, color=color, alpha=0.25, linewidth=0)
        elif ev == "filled":
            color = "#2ca02c" if side == 0 else "#ff7f0e"
            ax_q.scatter(t, px, s=50, color=color, marker="D", edgecolors="black", linewidth=0.4, zorder=5)

    ax_q.plot(sim_t, mid, color="gray", linewidth=0.6, alpha=0.7)
    ax_q.set_ylabel("Price (ticks)")
    ax_q.set_title(f"HJB Market Maker — seed {s}")
    ax_q.grid(True, alpha=0.3)
    leg = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#1f77b4", alpha=0.4, markersize=6, label="Bid sent"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#d62728", alpha=0.4, markersize=6, label="Ask sent"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor="#2ca02c", markeredgecolor="black", markersize=8, label="Bid filled"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor="#ff7f0e", markeredgecolor="black", markersize=8, label="Ask filled"),
        Line2D([0], [0], color="gray", linewidth=0.8, alpha=0.7, label="Mid"),
    ]
    ax_q.legend(handles=leg, loc="upper right", fontsize=7)

    ax_i.plot(sim_t, inventory, linewidth=0.5, color="#1f77b4")
    ax_i.set_ylabel("Inventory (lots)")
    ax_i.grid(True, alpha=0.3)
    ax_i.axhline(0, color="gray", linewidth=0.4, linestyle="--")

    ax_p.plot(sim_t, total_pnl, linewidth=0.5, color="#2ca02c")
    ax_p.set_xlabel("Simulation Time (ticks)")
    ax_p.set_ylabel("Total PnL (ticks)")
    ax_p.grid(True, alpha=0.3)

    plt.savefig(f"{dir_name}/hjb_pnl.png", dpi=150)
    print(f"saved {dir_name}/hjb_pnl.png")
