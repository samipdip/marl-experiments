"""Belief-trader ACTION dynamics: beta (conviction) + inventory over the episode.

Reads {ep}/checkpoints/belief_*_action_history.csv (step, beta) and
{ep}/pnl_history.csv (inventory per agent), averaged over episodes.

Usage: python plot_belief_actions.py <results_dir> [n_steps]
"""
import csv
import glob
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

results_base = sys.argv[1]
n_steps = int(sys.argv[2]) if len(sys.argv) > 2 else 240

eps = sorted(d for d in glob.glob(os.path.join(results_base, "ep_*")) if os.path.isdir(d))
if not eps:
    raise SystemExit(f"no ep_* dirs in {results_base}")
n_episodes = len(eps)

traders = sorted(
    os.path.basename(f)[:-len("_action_history.csv")]
    for f in glob.glob(os.path.join(eps[0], "checkpoints", "belief_*_action_history.csv"))
)
if not traders:
    raise SystemExit(f"no belief_*_action_history.csv under {eps[0]}/checkpoints")
print(f"{len(traders)} traders: {traders}  | {n_episodes} episodes")

beta = {t: np.full((n_episodes, n_steps), np.nan) for t in traders}
inv = {t: np.full((n_episodes, n_steps), np.nan) for t in traders}
first_active = {t: [] for t in traders}

for ei, ep in enumerate(eps):
    inv_now = {t: {} for t in traders}
    pnl = os.path.join(ep, "pnl_history.csv")
    if os.path.exists(pnl):
        with open(pnl) as f:
            for r in csv.DictReader(f):
                a = r.get("agent", "")
                if a not in traders:
                    continue
                t = int(float(r["sim_t"])) - 1
                if t < 0 or t >= n_steps:
                    continue
                q = int(r["inventory"])
                inv[a][ei, t] = q
                inv_now[a][t] = q
    for t in traders:
        ap = os.path.join(ep, "checkpoints", t + "_action_history.csv")
        if not os.path.exists(ap):
            continue
        with open(ap) as f:
            for r in csv.DictReader(f):
                s = int(r["step"])
                if s < 0 or s >= n_steps:
                    continue
                beta[t][ei, s] = float(r["action"])
        fa = next((s for s in range(n_steps) if inv_now[t].get(s, 0) != 0), None)
        if fa is not None:
            first_active[t].append(fa)

x = np.arange(n_steps)
colors = plt.cm.tab10(np.linspace(0, 1, max(len(traders), 1)))

fig, (ax_b, ax_i) = plt.subplots(
    2, 1, figsize=(14, 9), sharex=True, gridspec_kw={"hspace": 0.15}
)

all_fa = [v for t in traders for v in first_active[t]]
if all_fa:
    frac = np.array([np.mean([1.0 if fa <= s else 0.0 for fa in all_fa]) for s in x])
    ax_b.fill_between(x, 0, 10 * frac, color="gray", alpha=0.15, zorder=0)
    ax_b.text(2, 9.4, f"frac active (max={np.max(frac):.2f})", color="dimgray", fontsize=8)

for c, t in zip(colors, traders):
    bm = np.nanmean(beta[t], axis=0)
    bs = np.nanstd(beta[t], axis=0)
    ax_b.plot(x, bm, color=c, linewidth=1.6, label=f"{t} β")
    ax_b.fill_between(x, bm - bs, bm + bs, color=c, alpha=0.15)
    ax_i.plot(x, np.nanmean(inv[t], axis=0), color=c, linewidth=1.6, label=f"{t} inv")

ax_b.set_ylim(0, 10)
ax_b.set_ylabel("β  (action ∈ [0,10])")
ax_b.set_title(
    f"Belief-trader actions — avg over {n_episodes} eps  ({os.path.basename(results_base)})"
)
ax_b.legend(fontsize=8, ncol=2, loc="upper right")
ax_b.grid(alpha=0.3)
ax_i.set_ylabel("inventory")
ax_i.set_xlabel("step")
ax_i.axhline(0, color="gray", linewidth=0.5, linestyle="--")
ax_i.legend(fontsize=8, ncol=2, loc="upper right")
ax_i.grid(alpha=0.3)

out = os.path.join(results_base, "belief_actions.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"saved {out}")
