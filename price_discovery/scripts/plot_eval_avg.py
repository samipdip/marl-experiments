import csv
import itertools
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

mids = np.full((n_episodes, n_steps), np.nan)
mid_flush = np.full((n_episodes, n_steps), np.nan)
bids = np.full((n_episodes, n_steps), np.nan)
asks = np.full((n_episodes, n_steps), np.nan)
info_inv = np.full((n_episodes, n_steps), np.nan)
info_pnl = np.full((n_episodes, n_steps), np.nan)

hjb_keys = set()
ppo_keys = set()
hjb_inv = {}
hjb_pnl = {}
ppo_inv = {}
ppo_pnl = {}
liq_inv = np.full((n_episodes, n_steps), np.nan)
liq_pnl = np.full((n_episodes, n_steps), np.nan)

lob_vol = {}

lob_sources = ["informed", "liquidity", "ppo_hjb_mm_0"]

import re
fv = 1000.0
m = re.search(r"v(\d+)", results_base)
if m:
    fv = float(m.group(1))

for ep in range(n_episodes):
    ep_dir = os.path.join(results_base, f"ep_{ep}")
    pnl_path = os.path.join(ep_dir, "pnl_history.csv")
    info_pnl_cum = 0.0
    info_prev_inv = 0
    with open(pnl_path) as f:
        for row in csv.DictReader(f):
            t = int(float(row["sim_t"])) - 1
            if t < 0 or t >= n_steps:
                continue
            agent = row["agent"]
            if agent == "informed":
                mids[ep, t] = float(row["mid"]) if float(row["mid"]) > 0 else np.nan
                mf = row.get("mid_flush")
                if mf and float(mf) > 0:
                    mid_flush[ep, t] = float(mf)
                bf = row.get("bid_flush")
                af = row.get("ask_flush")
                if bf and float(bf) > 0:
                    bids[ep, t] = float(bf)
                if af and float(af) > 0:
                    asks[ep, t] = float(af)
                info_inv[ep, t] = int(row["inventory"])
                inv = int(row["inventory"])
                d_inv = inv - info_prev_inv
                mid_p = float(row["mid"]) if float(row["mid"]) > 0 else fv
                ask_p = float(row["ask"]) if row.get("ask") and float(row["ask"]) > 0 else mid_p
                bid_p = float(row["bid"]) if row.get("bid") and float(row["bid"]) > 0 else mid_p
                if d_inv > 0:
                    info_pnl_cum += d_inv * (fv - ask_p)
                elif d_inv < 0:
                    info_pnl_cum += -d_inv * (bid_p - fv)
                info_pnl[ep, t] = info_pnl_cum
                info_prev_inv = inv
            elif agent.startswith("hjb_mm"):
                hjb_keys.add(agent)
                if agent not in hjb_inv:
                    hjb_inv[agent] = np.full((n_episodes, n_steps), np.nan)
                    hjb_pnl[agent] = np.full((n_episodes, n_steps), np.nan)
                v = int(row["inventory"])
                if not np.isnan(hjb_inv[agent][ep, t]):
                    hjb_inv[agent][ep, t] += v
                else:
                    hjb_inv[agent][ep, t] = v
                p = float(row["total_pnl"])
                if not np.isnan(hjb_pnl[agent][ep, t]):
                    hjb_pnl[agent][ep, t] += p
                else:
                    hjb_pnl[agent][ep, t] = p
                if np.isnan(mids[ep, t]) and float(row["mid"]) > 0:
                    mids[ep, t] = float(row["mid"])
                    bf = row.get("bid_flush")
                    af = row.get("ask_flush")
                    if bf and float(bf) > 0:
                        bids[ep, t] = float(bf)
                    if af and float(af) > 0:
                        asks[ep, t] = float(af)
            elif agent.startswith("ppo_hjb_mm"):
                ppo_keys.add(agent)
                if agent not in ppo_inv:
                    ppo_inv[agent] = np.full((n_episodes, n_steps), np.nan)
                    ppo_pnl[agent] = np.full((n_episodes, n_steps), np.nan)
                v = int(row["inventory"])
                if not np.isnan(ppo_inv[agent][ep, t]):
                    ppo_inv[agent][ep, t] += v
                else:
                    ppo_inv[agent][ep, t] = v
                p = float(row["total_pnl"])
                if not np.isnan(ppo_pnl[agent][ep, t]):
                    ppo_pnl[agent][ep, t] += p
                else:
                    ppo_pnl[agent][ep, t] = p
                if np.isnan(mids[ep, t]) and float(row["mid"]) > 0:
                    mids[ep, t] = float(row["mid"])
                    bf = row.get("bid_flush")
                    af = row.get("ask_flush")
                    if bf and float(bf) > 0:
                        bids[ep, t] = float(bf)
                    if af and float(af) > 0:
                        asks[ep, t] = float(af)
            elif agent == "liquidity":
                liq_inv[ep, t] = int(row["inventory"])
                liq_pnl[ep, t] = float(row["total_pnl"])
                if np.isnan(mids[ep, t]) and float(row["mid"]) > 0:
                    mids[ep, t] = float(row["mid"])
                    bf = row.get("bid_flush")
                    af = row.get("ask_flush")
                    if bf and float(bf) > 0:
                        bids[ep, t] = float(bf)
                    if af and float(af) > 0:
                        asks[ep, t] = float(af)

    lob_path = None
    for src in lob_sources:
        candidate = os.path.join(ep_dir, "checkpoints", f"{src}_lob_data.csv")
        if os.path.exists(candidate):
            lob_path = candidate
            break
    if lob_path and os.path.exists(lob_path):
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
info_inv_avg = np.nanmean(info_inv, axis=0) if not np.isnan(info_inv).all() else np.full(n_steps, np.nan)
info_pnl_avg = np.nanmean(info_pnl, axis=0) if not np.isnan(info_pnl).all() else np.full(n_steps, np.nan)

hjb_inv_total = np.full((n_episodes, n_steps), np.nan)
hjb_pnl_total = np.full((n_episodes, n_steps), np.nan)
for k in hjb_keys:
    for ep in range(n_episodes):
        for t in range(n_steps):
            if not np.isnan(hjb_inv[k][ep, t]):
                if np.isnan(hjb_inv_total[ep, t]):
                    hjb_inv_total[ep, t] = 0
                hjb_inv_total[ep, t] += hjb_inv[k][ep, t]
            if not np.isnan(hjb_pnl[k][ep, t]):
                if np.isnan(hjb_pnl_total[ep, t]):
                    hjb_pnl_total[ep, t] = 0
                hjb_pnl_total[ep, t] += hjb_pnl[k][ep, t]

ppo_inv_total = np.full((n_episodes, n_steps), np.nan)
ppo_pnl_total = np.full((n_episodes, n_steps), np.nan)
for k in ppo_keys:
    for ep in range(n_episodes):
        for t in range(n_steps):
            if not np.isnan(ppo_inv[k][ep, t]):
                if np.isnan(ppo_inv_total[ep, t]):
                    ppo_inv_total[ep, t] = 0
                ppo_inv_total[ep, t] += ppo_inv[k][ep, t]
            if not np.isnan(ppo_pnl[k][ep, t]):
                if np.isnan(ppo_pnl_total[ep, t]):
                    ppo_pnl_total[ep, t] = 0
                ppo_pnl_total[ep, t] += ppo_pnl[k][ep, t]

hjb_inv_avg = np.nanmean(hjb_inv_total, axis=0)
hjb_pnl_avg = np.nanmean(hjb_pnl_total, axis=0)
ppo_inv_avg = np.nanmean(ppo_inv_total, axis=0)
ppo_pnl_avg = np.nanmean(ppo_pnl_total, axis=0)

liq_inv_avg = np.nanmean(liq_inv, axis=0)
liq_pnl_avg = np.nanmean(liq_pnl, axis=0)

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
    info_inv_avg = info_inv[ep_idx]
    info_pnl_avg = info_pnl[ep_idx]
    hjb_inv_avg = hjb_inv_total[ep_idx]
    hjb_pnl_avg = hjb_pnl_total[ep_idx]
    ppo_inv_avg = ppo_inv_total[ep_idx]
    ppo_pnl_avg = ppo_pnl_total[ep_idx]
    liq_inv_avg = liq_inv[ep_idx]
    liq_pnl_avg = liq_pnl[ep_idx]
    for p, mat in lob_vol.items():
        vol_avg[price_to_idx[p]] = mat[ep_idx]

if clip_ticks is not None:
    mid_lo = np.nanmin(mid_avg) - clip_ticks
    mid_hi = np.nanmax(mid_avg) + clip_ticks
    keep = [i for i, p in enumerate(price_range) if mid_lo <= p <= mid_hi]
    price_range = [price_range[i] for i in keep]
    vol_avg = vol_avg[keep]
    n_prices = len(price_range)
    print(f"clipped LOB to mid +/- {clip_ticks} ticks (mid range {np.nanmin(mid_avg):.1f}..{np.nanmax(mid_avg):.1f}): {n_prices} price levels kept")

fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(3, 1, height_ratios=[2, 1, 1], hspace=0.2)
ax_lob = fig.add_subplot(gs[0])
ax_inv = fig.add_subplot(gs[1], sharex=ax_lob)
ax_pnl = fig.add_subplot(gs[2], sharex=ax_lob)

if n_prices > 0:
    _viridis = plt.get_cmap("viridis").copy()
    _viridis.set_under("white")
    cmap = ax_lob.pcolormesh(x, price_range, vol_avg, shading="auto", cmap=_viridis, vmin=0.001)
    fig.colorbar(cmap, ax=ax_lob, label="Avg volume")

ax_lob.plot(x, mid_avg, color="#0A3D5C", linewidth=1.8, linestyle="--", label="mid", zorder=5)
ax_lob.plot(x, mid_flush_avg, color="#5B2A86", linewidth=1.8, linestyle=":", label="mid (post-flush)", zorder=5)
ax_lob.axhline(fv, color="#FF5555", linewidth=1.5, linestyle="--", label=f"v={fv:.0f}")
ax_lob.plot(x, bid_avg, color="#5CD1FF", linewidth=1.5, label="best bid", zorder=5)
ax_lob.plot(x, ask_avg, color="#FF9D8A", linewidth=1.5, label="best ask", zorder=5)
ax_lob.set_ylabel("Price")
ax_lob.set_title(f"Episode {ep_idx}" if ep_idx is not None else f"Avg over {n_episodes} episodes")
ax_lob.legend(fontsize=8)

if ep_idx is None:
    for ep in range(n_episodes):
        ax_inv.plot(x, info_inv[ep], color="#1f77b4", linewidth=0.2, alpha=0.2)
        if not np.isnan(hjb_inv_total[ep]).all():
            ax_inv.plot(x, hjb_inv_total[ep], color="#d62728", linewidth=0.2, alpha=0.2)
        if not np.isnan(ppo_inv_total[ep]).all():
            ax_inv.plot(x, ppo_inv_total[ep], color="#2ca02c", linewidth=0.2, alpha=0.2)
        if not np.isnan(liq_inv[ep]).all():
            ax_inv.plot(x, liq_inv[ep], color="#ff7f0e", linewidth=0.2, alpha=0.2)
ax_inv.plot(x, info_inv_avg, color="#1f77b4", linewidth=1.5, label="informed avg")
if not np.isnan(hjb_inv_avg).all():
    ax_inv.plot(x, hjb_inv_avg, color="#d62728", linewidth=1.5, label="HJB MM avg")
if not np.isnan(ppo_inv_avg).all():
    ax_inv.plot(x, ppo_inv_avg, color="#2ca02c", linewidth=1.5, label="PPO HJB MM avg")
if not np.isnan(liq_inv_avg).all():
    ax_inv.plot(x, liq_inv_avg, color="#ff7f0e", linewidth=1.5, label="liquidity avg")
ax_inv.set_ylabel("Inventory")
ax_inv.grid(True, alpha=0.3)
ax_inv.axhline(0, color="gray", linewidth=0.4, linestyle="--")
ax_inv.legend(fontsize=8)

if ep_idx is None:
    for ep in range(n_episodes):
        ax_pnl.plot(x, info_pnl[ep], color="#1f77b4", linewidth=0.2, alpha=0.2)
        if not np.isnan(hjb_pnl_total[ep]).all():
            ax_pnl.plot(x, hjb_pnl_total[ep], color="#d62728", linewidth=0.2, alpha=0.2)
        if not np.isnan(ppo_pnl_total[ep]).all():
            ax_pnl.plot(x, ppo_pnl_total[ep], color="#2ca02c", linewidth=0.2, alpha=0.2)
        if not np.isnan(liq_pnl[ep]).all():
            ax_pnl.plot(x, liq_pnl[ep], color="#ff7f0e", linewidth=0.2, alpha=0.2)
ax_pnl.plot(x, info_pnl_avg, color="#1f77b4", linewidth=1.5, label="informed avg")
if not np.isnan(hjb_pnl_avg).all():
    ax_pnl.plot(x, hjb_pnl_avg, color="#d62728", linewidth=1.5, label="HJB MM avg")
if not np.isnan(ppo_pnl_avg).all():
    ax_pnl.plot(x, ppo_pnl_avg, color="#2ca02c", linewidth=1.5, label="PPO HJB MM avg")
if not np.isnan(liq_pnl_avg).all():
    ax_pnl.plot(x, liq_pnl_avg, color="#ff7f0e", linewidth=1.5, label="liquidity avg")
ax_pnl.set_xlabel("Step")
ax_pnl.set_ylabel("Total PnL")
ax_pnl.grid(True, alpha=0.3)
ax_pnl.legend(fontsize=8)

out_png = os.path.join(results_base, f"eval_ep{ep_idx}.png" if ep_idx is not None else "eval_avg.png")
plt.savefig(out_png, dpi=150, bbox_inches="tight")
print(f"saved {out_png}")
