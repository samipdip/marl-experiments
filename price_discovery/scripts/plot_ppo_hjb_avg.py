import csv
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

results_base = sys.argv[1]
n_steps = int(sys.argv[2]) if len(sys.argv) > 2 else 40
n_episodes = 30

# ── Collect PPO MM action histories ──
ppo_files = []
for ep in range(n_episodes):
    ckpt_dir = os.path.join(results_base, f"ep_{ep}", "checkpoints")
    if not os.path.isdir(ckpt_dir):
        continue
    for fname in sorted(os.listdir(ckpt_dir)):
        if fname.startswith("ppo_hjb_mm") and fname.endswith("_action_history.csv"):
            if fname not in ppo_files:
                ppo_files.append(fname)

n_dims = 1
depth_units = False
kappa_bid = {}
kappa_ask = {}
for fname in ppo_files:
    key = fname.replace("_action_history.csv", "")
    kappa_bid[key] = np.full((n_episodes, n_steps), np.nan)
    kappa_ask[key] = np.full((n_episodes, n_steps), np.nan)

beta = np.full((n_episodes, n_steps), np.nan)
theta = np.full((n_episodes, n_steps), np.nan)

for ep in range(n_episodes):
    ckpt_dir = os.path.join(results_base, f"ep_{ep}", "checkpoints")
    for afname in ppo_files:
        key = afname.replace("_action_history.csv", "")
        path = os.path.join(ckpt_dir, afname)
        if os.path.exists(path):
            with open(path) as f:
                header = f.readline().strip()
                f.seek(0)
                is_2d = "kappa_bid" in header or "bid_depth" in header
                if is_2d:
                    n_dims = 2
                for row in csv.DictReader(f):
                    t = int(row["step"])
                    if t < n_steps:
                        if "bid_depth" in header:
                            depth_units = True
                            kappa_bid[key][ep, t] = float(row.get("bid_depth", 0))
                            kappa_ask[key][ep, t] = float(row.get("ask_depth", 0))
                        elif is_2d:
                            kappa_bid[key][ep, t] = float(row.get("kappa_bid", 0))
                            kappa_ask[key][ep, t] = float(row.get("kappa_ask", 0))
                        else:
                            a = float(row["action"])
                            kappa_bid[key][ep, t] = a
                            kappa_ask[key][ep, t] = a

    informed_path = os.path.join(ckpt_dir, "informed_action_history.csv")
    if os.path.exists(informed_path):
        with open(informed_path) as f:
            for row in csv.DictReader(f):
                t = int(row["step"])
                if t < n_steps:
                    beta[ep, t] = float(row["action"])

    liq_path = os.path.join(ckpt_dir, "liquidity_action_history.csv")
    if os.path.exists(liq_path):
        with open(liq_path) as f:
            for row in csv.DictReader(f):
                t = int(row["step"])
                if t < n_steps:
                    theta[ep, t] = float(row["action"])

# ── Compute per-agent mean and std across episodes ──
mm_keys = sorted(kappa_bid.keys())
mm_means_bid = []
mm_stds_bid = []
mm_means_ask = []
mm_stds_ask = []
for k in mm_keys:
    mm_means_bid.append(np.nanmean(kappa_bid[k], axis=0))
    mm_stds_bid.append(np.nanstd(kappa_bid[k], axis=0))
    mm_means_ask.append(np.nanmean(kappa_ask[k], axis=0))
    mm_stds_ask.append(np.nanstd(kappa_ask[k], axis=0))

has_beta = not np.isnan(beta).all()
has_theta = not np.isnan(theta).all()
beta_avg = np.nanmean(beta, axis=0) if has_beta else None
beta_std = np.nanstd(beta, axis=0) if has_beta else None
theta_avg = np.nanmean(theta, axis=0) if has_theta else None
theta_std = np.nanstd(theta, axis=0) if has_theta else None

# ── Build grid plot ──
n_mms = len(mm_keys)
n_total = n_mms + (1 if has_beta else 0) + (1 if has_theta else 0)
if n_total == 0:
    print("No data")
    sys.exit(0)

n_cols = min(3, n_total)
n_rows = int(np.ceil(n_total / n_cols))

fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.5 * n_cols, 4 * n_rows), squeeze=False)
axes_flat = axes.flatten()

x = range(n_steps)
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

idx = 0
for i, k in enumerate(mm_keys):
    ax = axes_flat[idx]
    idx += 1
    c = colors[i % len(colors)]
    # Bid
    mb, sb = mm_means_bid[i], mm_stds_bid[i]
    ax.plot(x, mb, linewidth=1.5, color=c, linestyle="-",
            label="bid_depth" if depth_units else ("kappa_bid" if n_dims == 2 else "kappa"))
    ax.fill_between(x, mb - sb, mb + sb, alpha=0.2, color=c)
    # Ask (only for 2D)
    if n_dims == 2:
        ma, sa = mm_means_ask[i], mm_stds_ask[i]
        ax.plot(x, ma, linewidth=1.5, color=c, linestyle="--",
                label="ask_depth" if depth_units else "kappa_ask")
        ax.fill_between(x, ma - sa, ma + sa, alpha=0.15, color=c)
    ax.set_title(k, fontsize=9)
    ax.set_xlabel("Step")
    ax.set_ylabel("Depth" if depth_units else "Kappa")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)

if has_beta:
    ax = axes_flat[idx]
    idx += 1
    ax.plot(x, beta_avg, linewidth=1.5, color="#2ca02c", label="action")
    ax.fill_between(x, beta_avg - beta_std, beta_avg + beta_std, alpha=0.2, color="#2ca02c")
    ax.set_title("Informed Trader — Action", fontsize=9)
    ax.set_xlabel("Step")
    ax.set_ylabel("Action")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)

if has_theta:
    ax = axes_flat[idx]
    idx += 1
    ax.plot(x, theta_avg, linewidth=1.5, color="#ff7f0e", label="theta")
    ax.fill_between(x, theta_avg - theta_std, theta_avg + theta_std, alpha=0.2, color="#ff7f0e")
    ax.set_title("Liquidity Trader — Theta", fontsize=9)
    ax.set_xlabel("Step")
    ax.set_ylabel("Theta")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)

# Hide unused subplots
for j in range(idx, len(axes_flat)):
    axes_flat[j].set_visible(False)

plt.suptitle(os.path.basename(results_base), fontsize=11, y=1.01)
plt.tight_layout()
out = os.path.join(results_base, "ppo_hjb_avg.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"saved {out}")
