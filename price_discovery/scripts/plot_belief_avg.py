"""Belief-trader eval plot: LOB heatmap + mid + beliefs, per-trader inventory & PnL.

Generalised over however many belief_* traders exist (globbed from ep_0).

Usage: python plot_belief_avg.py <results_dir> [n_steps] [clip_ticks] [ep_idx]
"""
import csv
import glob
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

results_base = sys.argv[1]
n_episodes = 30
n_steps = int(sys.argv[2]) if len(sys.argv) > 2 else 40
clip_ticks = int(sys.argv[3]) if len(sys.argv) > 3 else None
ep_idx = int(sys.argv[4]) if len(sys.argv) > 4 else None

# discover belief traders + mm agents from ep_0
ep0 = os.path.join(results_base, "ep_0")
traders = sorted(
    os.path.basename(f)[:-len("_lob_data.csv")]
    for f in glob.glob(os.path.join(ep0, "checkpoints", "belief_*_lob_data.csv"))
)
if not traders:
    traders = ["belief_0", "belief_1"]
print(f"traders: {traders}")

mids = np.full((n_episodes, n_steps), np.nan)
mid_flush = np.full((n_episodes, n_steps), np.nan)
bids = np.full((n_episodes, n_steps), np.nan)
asks = np.full((n_episodes, n_steps), np.nan)

ppo_keys = set()
ppo_inv = {}
ppo_pnl = {}
hjb_keys = set()
hjb_inv = {}
hjb_pnl = {}

tr_inv = {t: np.full((n_episodes, n_steps), np.nan) for t in traders}
tr_pnl = {t: np.full((n_episodes, n_steps), np.nan) for t in traders}
tr_base = {t: {} for t in traders}
belief_series = {t: {} for t in traders}

lob_vol = {}
lob_sources = traders[:2] + ["ppo_hjb_mm_0"]

fv = 1000.0

for ep in range(n_episodes):
    ep_dir = os.path.join(results_base, f"ep_{ep}")
    pnl_path = os.path.join(ep_dir, "pnl_history.csv")
    if not os.path.exists(pnl_path):
        continue
    with open(pnl_path) as f:
        for row in csv.DictReader(f):
            t = int(float(row["sim_t"])) - 1
            if t < 0 or t >= n_steps:
                continue
            agent = row["agent"]

            def _grab_market():
                if float(row["mid"]) > 0:
                    mids[ep, t] = float(row["mid"])
                mf = row.get("mid_flush")
                if mf and float(mf) > 0:
                    mid_flush[ep, t] = float(mf)
                bf = row.get("bid_flush")
                af = row.get("ask_flush")
                if bf and float(bf) > 0:
                    bids[ep, t] = float(bf)
                if af and float(af) > 0:
                    asks[ep, t] = float(af)

            if agent in tr_inv:
                tr_inv[agent][ep, t] = int(row["inventory"])
                if ep not in tr_base[agent]:
                    tr_base[agent][ep] = float(row["total_pnl"])
                tr_pnl[agent][ep, t] = float(row["total_pnl"]) - tr_base[agent][ep]
                if np.isnan(mids[ep, t]):
                    _grab_market()
            elif agent.startswith("ppo_hjb_mm"):
                ppo_keys.add(agent)
                if agent not in ppo_inv:
                    ppo_inv[agent] = np.full((n_episodes, n_steps), np.nan)
                    ppo_pnl[agent] = np.full((n_episodes, n_steps), np.nan)
                v = int(row["inventory"])
                ppo_inv[agent][ep, t] = (ppo_inv[agent][ep, t] + v
                                         if not np.isnan(ppo_inv[agent][ep, t]) else v)
                p = float(row["total_pnl"])
                ppo_pnl[agent][ep, t] = (ppo_pnl[agent][ep, t] + p
                                         if not np.isnan(ppo_pnl[agent][ep, t]) else p)
                if np.isnan(mids[ep, t]):
                    _grab_market()
            elif agent.startswith("hjb_mm"):
                hjb_keys.add(agent)
                if agent not in hjb_inv:
                    hjb_inv[agent] = np.full((n_episodes, n_steps), np.nan)
                    hjb_pnl[agent] = np.full((n_episodes, n_steps), np.nan)
                v = int(row["inventory"])
                hjb_inv[agent][ep, t] = (hjb_inv[agent][ep, t] + v
                                         if not np.isnan(hjb_inv[agent][ep, t]) else v)
                p = float(row["total_pnl"])
                hjb_pnl[agent][ep, t] = (hjb_pnl[agent][ep, t] + p
                                         if not np.isnan(hjb_pnl[agent][ep, t]) else p)
                if np.isnan(mids[ep, t]):
                    _grab_market()

    # belief values from lob_data.csv
    for src in traders:
        cand = os.path.join(ep_dir, "checkpoints", f"{src}_lob_data.csv")
        if not os.path.exists(cand):
            continue
        with open(cand) as f:
            for row in csv.DictReader(f):
                t = int(row["step"])
                if t >= n_steps:
                    continue
                if row.get("v"):
                    belief_series[src][(ep, t)] = float(row["v"])

    # LOB volume (use first available lob source)
    lob_path = None
    for src in lob_sources:
        cand = os.path.join(ep_dir, "checkpoints", f"{src}_lob_data.csv")
        if os.path.exists(cand):
            lob_path = cand
            break
    if lob_path:
        with open(lob_path) as f:
            for row in csv.DictReader(f):
                t = int(row["step"])
                if t >= n_steps:
                    continue
                lob = json.loads(row["lob_json"])
                for side in ("bids", "asks"):
                    for p, q in lob.get(side, []):
                        pi = int(p)
                        if pi not in lob_vol:
                            lob_vol[pi] = np.zeros((n_episodes, n_steps))
                        lob_vol[pi][ep, t] += int(q)

x = np.arange(n_steps)
mid_avg = np.nanmean(mids, axis=0)
mid_flush_avg = np.nanmean(mid_flush, axis=0)
bid_avg = np.nanmean(bids, axis=0)
ask_avg = np.nanmean(asks, axis=0)


def _stack_total(keyset, invd, pnld):
    inv_tot = np.full((n_episodes, n_steps), np.nan)
    pnl_tot = np.full((n_episodes, n_steps), np.nan)
    for k in keyset:
        for ep in range(n_episodes):
            for t in range(n_steps):
                if not np.isnan(invd[k][ep, t]):
                    if np.isnan(inv_tot[ep, t]):
                        inv_tot[ep, t] = 0
                    inv_tot[ep, t] += invd[k][ep, t]
                if not np.isnan(pnld[k][ep, t]):
                    if np.isnan(pnl_tot[ep, t]):
                        pnl_tot[ep, t] = 0
                    pnl_tot[ep, t] += pnld[k][ep, t]
    return inv_tot, pnl_tot


ppo_inv_total, ppo_pnl_total = _stack_total(ppo_keys, ppo_inv, ppo_pnl)
hjb_inv_total, hjb_pnl_total = _stack_total(hjb_keys, hjb_inv, hjb_pnl)
ppo_inv_avg = np.nanmean(ppo_inv_total, axis=0)
ppo_pnl_avg = np.nanmean(ppo_pnl_total, axis=0)

tr_inv_avg = {t: np.nanmean(tr_inv[t], axis=0) for t in traders}
tr_pnl_avg = {t: np.nanmean(tr_pnl[t], axis=0) for t in traders}

price_range = sorted(lob_vol.keys())
price_to_idx = {p: i for i, p in enumerate(price_range)}
n_prices = len(price_range)
vol_avg = np.zeros((n_prices, n_steps))
for p, mat in lob_vol.items():
    vol_avg[price_to_idx[p]] = np.mean(mat, axis=0)

if ep_idx is not None:
    mid_avg = mids[ep_idx]
    mid_flush_avg = mid_flush[ep_idx]
    bid_avg = bids[ep_idx]
    ask_avg = asks[ep_idx]
    ppo_inv_avg = ppo_inv_total[ep_idx]
    ppo_pnl_avg = ppo_pnl_total[ep_idx]
    for t in traders:
        tr_inv_avg[t] = tr_inv[t][ep_idx]
        tr_pnl_avg[t] = tr_pnl[t][ep_idx]
    for p, mat in lob_vol.items():
        vol_avg[price_to_idx[p]] = mat[ep_idx]

if clip_ticks is not None:
    mid_lo = np.nanmin(mid_avg) - clip_ticks
    mid_hi = np.nanmax(mid_avg) + clip_ticks
    keep = [i for i, p in enumerate(price_range) if mid_lo <= p <= mid_hi]
    price_range = [price_range[i] for i in keep]
    vol_avg = vol_avg[keep]
    n_prices = len(price_range)
    print(f"clipped LOB to mid ± {clip_ticks}: {n_prices} levels")

fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(3, 1, height_ratios=[2, 1, 1], hspace=0.2)
ax_lob = fig.add_subplot(gs[0])
ax_inv = fig.add_subplot(gs[1], sharex=ax_lob)
ax_pnl = fig.add_subplot(gs[2], sharex=ax_lob)

if n_prices > 0:
    _viridis = plt.get_cmap("viridis").copy()
    _viridis.set_under("white")
    cmap = ax_lob.pcolormesh(x, price_range, vol_avg, shading="auto",
                             cmap=_viridis, vmin=0.001)
    fig.colorbar(cmap, ax=ax_lob, label="Avg volume")

ax_lob.plot(x, mid_avg, color="#0A3D5C", linewidth=1.8, linestyle="--",
            label="mid", zorder=5)
if not np.isnan(mid_flush_avg).all():
    ax_lob.plot(x, mid_flush_avg, color="#5B2A86", linewidth=1.8, linestyle=":",
                label="mid (post-flush)", zorder=5)
tr_colors = plt.cm.tab10(np.linspace(0, 1, max(len(traders), 1)))
for c, src in zip(tr_colors, traders):
    series = belief_series[src]
    if series:
        bs = np.full(n_steps, np.nan)
        for t in range(n_steps):
            vals = [series[(ep, t)] for ep in range(n_episodes) if (ep, t) in series]
            if vals:
                bs[t] = np.mean(vals)
        ax_lob.plot(x, bs, color=c, linewidth=1.4, linestyle="--",
                    label=f"{src} belief", zorder=4)
ax_lob.plot(x, bid_avg, color="#5CD1FF", linewidth=1.3, label="best bid", zorder=5)
ax_lob.plot(x, ask_avg, color="#FFB07A", linewidth=1.3, label="best ask", zorder=5)
ax_lob.set_ylabel("Price")
ax_lob.set_title(f"Episode {ep_idx}" if ep_idx is not None
                 else f"Avg over {n_episodes} episodes")
ax_lob.legend(fontsize=7, ncol=2)

if ep_idx is None:
    for ep in range(n_episodes):
        if not np.isnan(ppo_inv_total[ep]).all():
            ax_inv.plot(x, ppo_inv_total[ep], color="#2ca02c", linewidth=0.2, alpha=0.2)
ax_inv.plot(x, ppo_inv_avg, color="#2ca02c", linewidth=1.5, label="PPO HJB MM Σ")
for c, t in zip(tr_colors, traders):
    ax_inv.plot(x, tr_inv_avg[t], color=c, linewidth=1.4, label=f"{t} inv")
ax_inv.set_ylabel("Inventory")
ax_inv.grid(True, alpha=0.3)
ax_inv.axhline(0, color="gray", linewidth=0.4, linestyle="--")
ax_inv.legend(fontsize=7, ncol=3)

if ep_idx is None:
    for ep in range(n_episodes):
        if not np.isnan(ppo_pnl_total[ep]).all():
            ax_pnl.plot(x, ppo_pnl_total[ep], color="#2ca02c", linewidth=0.2, alpha=0.2)
ax_pnl.plot(x, ppo_pnl_avg, color="#2ca02c", linewidth=1.5, label="PPO HJB MM Σ")
for c, t in zip(tr_colors, traders):
    ax_pnl.plot(x, tr_pnl_avg[t], color=c, linewidth=1.4, label=f"{t} PnL")
ax_pnl.set_xlabel("Step")
ax_pnl.set_ylabel("Total PnL (baseline-sub)")
ax_pnl.grid(True, alpha=0.3)
ax_pnl.legend(fontsize=7, ncol=3)

out_png = os.path.join(results_base,
                       f"eval_ep{ep_idx}.png" if ep_idx is not None else "eval_avg.png")
plt.savefig(out_png, dpi=150, bbox_inches="tight")
print(f"saved {out_png}")
