import csv
import os
import re
import sys

import matplotlib.pyplot as plt
import numpy as np

results_base = sys.argv[1]
n_steps = int(sys.argv[2]) if len(sys.argv) > 2 else 40
ep_idx = int(sys.argv[3]) if len(sys.argv) > 3 else None
n_episodes = 30

# ── Read inventory/pnl time series per agent ──
agent_data = {}
agent_pnl = {}
for ep in range(n_episodes):
    pnl_path = os.path.join(results_base, f"ep_{ep}", "pnl_history.csv")
    if not os.path.exists(pnl_path):
        continue
    with open(pnl_path) as f:
        for row in csv.DictReader(f):
            name = row["agent"]
            if name not in agent_data:
                agent_data[name] = np.full((n_episodes, n_steps), np.nan)
                agent_pnl[name] = np.full((n_episodes, n_steps), np.nan)
            t = int(float(row["sim_t"])) - 1
            if 0 <= t < n_steps:
                agent_data[name][ep, t] = float(row["inventory"])
                agent_pnl[name][ep, t] = float(row["total_pnl"])

if not agent_data:
    print("No pnl_history data")
    sys.exit(1)


def agent_sort_key(k):
    if k == "informed":
        return (0, 0)
    if k == "liquidity":
        return (1, 0)
    m = re.match(r"ppo_hjb_mm_(\d+)$", k)
    if m:
        return (2, int(m.group(1)))
    m = re.match(r"hjb_mm_(\d+)$", k)
    if m:
        return (3, int(m.group(1)))
    return (4, 0)


keys = sorted(agent_data.keys(), key=agent_sort_key)

if ep_idx is not None:
    keys = [k for k in keys if not k.startswith("uniform") and not k.startswith("noise")]
    n_agents = len(keys)
    x = np.arange(n_steps)

    actions = {}
    for k in keys:
        act_path = os.path.join(results_base, f"ep_{ep_idx}", "checkpoints", f"{k}_action_history.csv")
        if os.path.exists(act_path):
            cols = {}
            with open(act_path) as f:
                for row in csv.DictReader(f):
                    t = int(float(row["step"]))
                    if t >= n_steps:
                        continue
                    for col, val in row.items():
                        if col == "step":
                            continue
                        if col not in cols:
                            cols[col] = np.full(n_steps, np.nan)
                        cols[col][t] = float(val)
            actions[k] = cols

    fig, axes = plt.subplots(4, n_agents, figsize=(4.5 * n_agents, 12), sharex=True, squeeze=False)

    for i, k in enumerate(keys):
        inv = agent_data[k][ep_idx]
        pnl = agent_pnl[k][ep_idx]
        signed = np.diff(inv, prepend=0)
        buy = np.where(signed > 0, signed, 0)
        sell = np.where(signed < 0, signed, 0)

        ax = axes[0, i]
        ax.plot(x, inv, linewidth=1.2, color="#1f77b4")
        ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
        ax.set_title(k, fontsize=9)
        ax.set_ylabel("Inventory", fontsize=8)
        ax.grid(True, alpha=0.2)

        ax = axes[1, i]
        ax.plot(x, pnl, linewidth=1.2, color="#2ca02c")
        ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
        ax.set_ylabel("Total PnL", fontsize=8)
        ax.grid(True, alpha=0.2)

        ax = axes[2, i]
        ax.bar(x, buy, color="green", alpha=0.6, label="buy vol")
        ax.bar(x, sell, color="red", alpha=0.6, label="sell vol")
        ax.axhline(0, color="gray", linewidth=0.5)
        ax.set_ylabel("signed fill vol", fontsize=8)
        ax.grid(True, alpha=0.2)
        if i == 0:
            ax.legend(fontsize=7)

        ax = axes[3, i]
        if k in actions:
            for col, vals in actions[k].items():
                ax.plot(x, vals, linewidth=1.0, label=col)
            ax.legend(fontsize=7)
        else:
            ax.text(0.5, 0.5, "no action data", transform=ax.transAxes, ha="center", fontsize=8)
        ax.set_ylabel("action params", fontsize=8)
        ax.set_xlabel("Step")
        ax.grid(True, alpha=0.2)

    plt.suptitle(f"{os.path.basename(results_base)} ep_{ep_idx}", fontsize=11)
    plt.tight_layout()
    out = os.path.join(results_base, f"agents_ep{ep_idx}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved {out}")
    sys.exit(0)

n_agents = len(keys)
n_cols = min(4, n_agents)
n_rows = int(np.ceil(n_agents / n_cols))

fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.5 * n_cols, 4 * n_rows), squeeze=False)
axes_flat = axes.flatten()

x = np.arange(n_steps)
for i, k in enumerate(keys):
    ax = axes_flat[i]
    data = agent_data[k]
    inv_avg = np.nanmean(data, axis=0)
    inv_std = np.nanstd(data, axis=0)

    # Signed fills: per-episode inventory diff, split buy/sell, average
    signed = np.diff(data, axis=1, prepend=0)
    buy_avg = np.nanmean(np.where(signed > 0, signed, 0), axis=0)
    sell_avg = np.nanmean(np.where(signed < 0, signed, 0), axis=0)

    ax.bar(x, buy_avg, color="green", alpha=0.6, label="buy vol")
    ax.bar(x, sell_avg, color="red", alpha=0.6, label="sell vol")
    ax.axhline(0, color="gray", linewidth=0.5)

    ax2 = ax.twinx()
    ax2.plot(x, inv_avg, linewidth=1.5, color="#1f77b4", label="inventory")
    ax2.fill_between(x, inv_avg - inv_std, inv_avg + inv_std, alpha=0.15, color="#1f77b4")
    ax2.axhline(0, color="gray", linewidth=0.5, linestyle="--")

    ax.set_title(k, fontsize=9)
    ax.set_xlabel("Step")
    ax.set_ylabel("signed fill vol", fontsize=8)
    ax2.set_ylabel("Inventory", fontsize=8)
    ax.grid(True, alpha=0.2)

for j in range(len(keys), len(axes_flat)):
    axes_flat[j].set_visible(False)

plt.suptitle(os.path.basename(results_base), fontsize=11, y=1.01)
plt.tight_layout()
out = os.path.join(results_base, "fills_avg.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"saved {out}")
