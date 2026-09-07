import csv
import json
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np


def load_pnl(path, agent):
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            if row["agent"] == agent:
                rows.append(row)
    sim_t = [float(r["sim_t"]) for r in rows]
    inventory = [int(r["inventory"]) for r in rows]
    total_pnl = [float(r["total_pnl"]) for r in rows]
    return sim_t, inventory, total_pnl


def load_lob(path):
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def reconstruct_lob(lob_rows, order_events, n_steps, ref_price=1000):
    """Reconstruct post-flush LOB for each step.
    
    Tracks HJB MM limit orders through sent/cancelled/filled events,
    combines with pre-flush lob_data snapshots and uniform MM replenishment.
    Returns (best_bids, best_asks) arrays of length n_steps.
    """
    tag_state = {}
    best_bids = []
    best_asks = []

    for t in range(n_steps):
        for e in order_events:
            step = int(float(e["sim_t"]))
            if step != t or e["agent"] != "hjb_mm":
                continue
            tag = int(e["tag"])
            price = int(float(e["price"]))
            side = int(e["side"])
            if e["event"] == "sent":
                tag_state[tag] = {"price": price, "side": side, "active": True}
            elif e["event"] in ("cancelled", "filled"):
                if tag in tag_state:
                    tag_state[tag]["active"] = False

        hjb_bids = {v["price"] for v in tag_state.values() if v["active"] and v["side"] == 0}
        hjb_asks = {v["price"] for v in tag_state.values() if v["active"] and v["side"] == 1}

        lob = json.loads(lob_rows[t]["lob_json"])
        pre_bids = {int(p) for p, _ in lob.get("bids", []) if int(p) > 0}
        pre_asks = {int(p) for p, _ in lob.get("asks", []) if int(p) > 0}

        uni_bids = {ref_price - d for d in range(1, 6)}
        uni_asks = {ref_price + d for d in range(1, 6)}

        all_bids = pre_bids | hjb_bids | uni_bids
        all_asks = pre_asks | hjb_asks | uni_asks

        best_bid = max(all_bids) if all_bids else None
        best_ask = min(all_asks) if all_asks else None
        best_bids.append(best_bid)
        best_asks.append(best_ask)

    return best_bids, best_asks


res_dir = sys.argv[1]

pnl_path = f"{res_dir}/pnl_history.csv"
lob_path = f"{res_dir}/checkpoints/informed_lob_data.csv"
orders_path = f"{res_dir}/order_events.csv"

info_t, info_inv, info_pnl = load_pnl(pnl_path, "informed")
hjb_t, hjb_inv, hjb_pnl = load_pnl(pnl_path, "hjb_mm")
lob_rows = load_lob(lob_path)

n_steps = len(lob_rows)
mids_raw = [float(r["mid"]) for r in lob_rows]
v = float(lob_rows[0]["v"]) if lob_rows else 0.0
first_valid = next((m for m in mids_raw if m > 0), v - 6.0)
last = first_valid
mids = []
for m in mids_raw:
    if m > 0:
        last = m
    mids.append(last)

mid_flush_raw = [float(r.get("mid_flush") or 0) for r in lob_rows]
last = first_valid
mid_flush = []
for m in mid_flush_raw:
    if m > 0:
        last = m
    mid_flush.append(last)

order_events = list(csv.DictReader(open(orders_path)))
bb, ba = reconstruct_lob(lob_rows, order_events, n_steps)

all_prices = set()
for t in range(n_steps):
    lob = json.loads(lob_rows[t]["lob_json"])
    for p, _ in lob.get("bids", []):
        all_prices.add(int(p))
    for p, _ in lob.get("asks", []):
        all_prices.add(int(p))
with open(pnl_path) as f:
    for row in csv.DictReader(f):
        if row["agent"] == "hjb_mm":
            if row.get("quote_bid") and float(row["quote_bid"]) > 0:
                all_prices.add(int(float(row["quote_bid"])))
            if row.get("quote_ask") and float(row["quote_ask"]) > 0:
                all_prices.add(int(float(row["quote_ask"])))
if not all_prices:
    all_prices = {int(v) - 5, int(v) + 5}
price_range = list(range(min(all_prices), max(all_prices) + 1))
price_to_idx = {p: i for i, p in enumerate(price_range)}
n_prices = len(price_range)

vol_sum = np.zeros((n_prices, n_steps), dtype=np.float64)
vol_count = np.zeros((n_prices, n_steps), dtype=np.float64)

for t in range(n_steps):
    lob = json.loads(lob_rows[t]["lob_json"])
    for p, q in lob.get("bids", []):
        pi = int(p)
        if pi in price_to_idx:
            vol_sum[price_to_idx[pi], t] += int(q)
            vol_count[price_to_idx[pi], t] += 1
    for p, q in lob.get("asks", []):
        pi = int(p)
        if pi in price_to_idx:
            vol_sum[price_to_idx[pi], t] += int(q)
            vol_count[price_to_idx[pi], t] += 1

with np.errstate(invalid="ignore"):
    vol_mean = np.where(vol_count > 0, vol_sum / vol_count, np.nan)

fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(3, 1, height_ratios=[2, 1, 1], hspace=0.2)
ax_lob = fig.add_subplot(gs[0])
ax_inv = fig.add_subplot(gs[1], sharex=ax_lob)
ax_pnl = fig.add_subplot(gs[2], sharex=ax_lob)

cmap = ax_lob.pcolormesh(
    np.arange(n_steps),
    price_range,
    vol_mean,
    shading="auto",
    cmap="viridis",
    vmin=0,
    vmax=np.nanmax(vol_mean) if not np.all(np.isnan(vol_mean)) else 1,
)

ax_lob.plot(np.arange(n_steps), mids, color="cyan", linewidth=1.5, linestyle="--", label="mid (pre-flush)", zorder=5)
ax_lob.plot(np.arange(n_steps), mid_flush, color="magenta", linewidth=1.5, linestyle=":", label="mid (post-flush)", zorder=5)
ax_lob.axhline(v, color="red", linewidth=1.5, label=f"v={v:.0f}")

bb_plot = [b if b is not None else np.nan for b in bb]
ba_plot = [a if a is not None else np.nan for a in ba]
ax_lob.fill_between(np.arange(n_steps), bb_plot, ba_plot, color="gray", alpha=0.15, zorder=2, label="bid-ask spread (reconstructed)")

# Best bid/ask from exchange (post-flush via informed trader)
best_bid_vals = {}
best_ask_vals = {}
with open(pnl_path) as f:
    for row in csv.DictReader(f):
        if row["agent"] != "informed":
            continue
        t = int(float(row["sim_t"]))
        bf = row.get("bid_flush")
        af = row.get("ask_flush")
        if bf and float(bf) > 0:
            best_bid_vals[t] = float(bf)
        if af and float(af) > 0:
            best_ask_vals[t] = float(af)
bb_exchange = [best_bid_vals.get(i + 1, np.nan) for i in range(n_steps)]
ba_exchange = [best_ask_vals.get(i + 1, np.nan) for i in range(n_steps)]
ax_lob.plot(np.arange(n_steps), bb_exchange, color="#003057", linewidth=1.5, label="best bid", zorder=5)
ax_lob.plot(np.arange(n_steps), ba_exchange, color="#8B0000", linewidth=1.5, label="best ask", zorder=5)

hjb_bid_x, hjb_bid_y, hjb_ask_x, hjb_ask_y = [], [], [], []
with open(pnl_path) as f:
    for row in csv.DictReader(f):
        if row["agent"] != "hjb_mm":
            continue
        x = float(row["sim_t"])
        bid = row.get("quote_bid")
        ask = row.get("quote_ask")
        if bid and float(bid) > 0:
            hjb_bid_x.append(x); hjb_bid_y.append(float(bid))
        if ask and float(ask) > 0:
            hjb_ask_x.append(x); hjb_ask_y.append(float(ask))
ax_lob.scatter(hjb_bid_x, hjb_bid_y, marker="o", color="#003057", s=15, alpha=0.5, zorder=7, label="HJB bid")
ax_lob.scatter(hjb_ask_x, hjb_ask_y, marker="o", color="#8B0000", s=15, alpha=0.5, zorder=7, label="HJB ask")

ax_lob.set_ylabel("Price")
ax_lob.set_title("LOB Volume Heatmap")
ax_lob.legend(fontsize=8)
fig.colorbar(cmap, ax=ax_lob, label="Avg volume")

ax_inv.plot(info_t, info_inv, color="#1f77b4", linewidth=0.8, alpha=0.8, label="informed")
ax_inv.plot(hjb_t, hjb_inv, color="#d62728", linewidth=0.8, alpha=0.8, label="HJB MM")
ax_inv.set_ylabel("Inventory")
ax_inv.grid(True, alpha=0.3)
ax_inv.axhline(0, color="gray", linewidth=0.4, linestyle="--")
ax_inv.legend(fontsize=8)

ax_pnl.plot(info_t, info_pnl, color="#1f77b4", linewidth=0.8, alpha=0.8, label="informed")
ax_pnl.plot(hjb_t, hjb_pnl, color="#d62728", linewidth=0.8, alpha=0.8, label="HJB MM")
ax_pnl.set_xlabel("Simulation Time (ticks)")
ax_pnl.set_ylabel("Total PnL")
ax_pnl.grid(True, alpha=0.3)
ax_pnl.legend(fontsize=8)

plt.savefig(f"{res_dir}/price_discovery_eval.png", dpi=150, bbox_inches="tight")
print(f"saved {res_dir}/price_discovery_eval.png")
