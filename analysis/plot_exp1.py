"""plot_exp1.py -- Price discovery trajectory plots (paper Figures 2-3).

Usage:
    python -m analysis.plot_exp1 \
        --eval-up-dir results/exp1_exec_cost_aware/eval_up \
        --eval-down-dir results/exp1_exec_cost_aware/eval_down \
        --train-dir results/exp1_exec_cost_aware/train \
        --experiment-dir results/exp1_exec_cost_aware
"""

import argparse
import csv
import os
from collections import defaultdict

import json as json_mod

import matplotlib.pyplot as plt
import numpy as np


def load_per_step(path):
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            parsed = {}
            for k, v in row.items():
                if k == "lob_json":
                    parsed[k] = v
                else:
                    parsed[k] = float(v)
            rows.append(parsed)
    return rows


def plot_combined_grid(up_rows, down_rows, *, title, out_path, n_steps=60):

    def _to_array(rows, col="mid"):
        """Build (n_episodes, n_steps) array with forward-fill for 0/missing."""
        eps = {}
        for r in rows:
            e = int(r["episode"])
            if e not in eps:
                eps[e] = []
            eps[e].append(r)
        arr = []
        for e in sorted(eps):
            traj = []
            last = 0.0
            for s in range(min(len(eps[e]), n_steps)):
                val = float(eps[e][s].get(col, 0))
                if val > 0:
                    last = val
                traj.append(last)
            while len(traj) < n_steps:
                traj.append(last)
            arr.append(traj[:n_steps])
        return np.array(arr)

    up_mid = _to_array(up_rows, "mid")
    down_mid = _to_array(down_rows, "mid")
    up_vwap = _to_array(up_rows, "step_vwap")
    down_vwap = _to_array(down_rows, "step_vwap")
    v = float(up_rows[0]["v"])

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    steps = np.arange(n_steps)
    colors = {"Up": "C1", "Down": "C0"}

    for ax, mdata, vdata, label in [
        (axes[0, 0], down_mid, down_vwap, "Down"),
        (axes[0, 1], up_mid, up_vwap, "Up"),
    ]:
        mmean = np.nanmean(mdata, axis=0)
        mstd = np.nanstd(mdata, axis=0)
        ax.plot(steps, mmean, color=colors[label], label="Mid", linewidth=1.5)
        ax.fill_between(steps, mmean - mstd, mmean + mstd, alpha=0.2, color=colors[label])

        vmean = np.nanmean(np.where(vdata > 0, vdata, np.nan), axis=0)
        ax.scatter(steps, vmean, color=colors[label], marker=".", s=12, alpha=0.5, label="Step VWAP")

        ax.axhline(v, color="red", ls="--", label=f"v={v:.0f}")
        ax.set_title(f"{label} eval")
        ax.set_xlabel("Step")
        ax.set_ylabel("Price")
        ax.legend(fontsize=8)

    for ax, data in [(axes[1, 0], down_mid), (axes[1, 1], up_mid)]:
        errors = v - data
        mean = np.nanmean(errors, axis=0)
        std = np.nanstd(errors, axis=0)
        ax.plot(steps, mean, color="black")
        ax.fill_between(steps, mean - std, mean + std, alpha=0.2, color="gray")
        ax.axhline(0, color="red", ls="--")
        ax.set_xlabel("Step")
        ax.set_ylabel("Pricing error")

    axes[1, 0].set_title("Pricing error (Down)")
    axes[1, 1].set_title("Pricing error (Up)")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path)
    fig.savefig(out_path.replace(".png", ".pdf"))
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_training_curves_split(log_path, output_dir, prefix="Exp1"):
    import json

    import pandas as pd

    df = pd.read_csv(log_path)
    df.columns = df.columns.str.strip()

    # Parse stats_json into separate columns
    stats_df = df["stats_json"].apply(lambda s: pd.Series(json.loads(s)))
    df = pd.concat([df, stats_df], axis=1)
    metrics = ["policy_loss", "value_loss", "entropy"]

    df["is_mm"] = df["agent"].str.startswith("mm_")

    for m in metrics:
        if m not in df.columns:
            print(f"  Column {m} not found; skipping")
            continue
        fig, ax = plt.subplots(figsize=(10, 4))
        # Averaged MMs
        mm = df[df["is_mm"]].groupby("global_step")[m].agg(["mean", "std"]).reset_index()
        if not mm.empty:
            ax.plot(mm["global_step"], mm["mean"], linewidth=0.8, color="C0", label="MM (avg ± std)")
            ax.fill_between(mm["global_step"], mm["mean"] - mm["std"], mm["mean"] + mm["std"], alpha=0.2, color="C0")
        # Informed trader
        it = df[df["agent"] == "informed"].dropna(subset=[m])
        if not it.empty:
            ax.plot(it["global_step"], it[m], linewidth=0.8, color="C1", label="Informed")
        ax.set_xlabel("Global step")
        if m == "entropy":
            ax.set_ylabel("Entropy")
            ax.set_title(f"{prefix}: Policy Entropy")
        else:
            ax.set_ylabel(m.replace("_", " ").title())
            ax.set_title(f"{prefix}: {m.replace('_', ' ').title()}")
            ax.set_yscale("log")
        ax.legend(fontsize=8)
        out = os.path.join(output_dir, f"{prefix}_{m}.png")
        fig.savefig(out)
        fig.savefig(out.replace(".png", ".pdf"))
        plt.close(fig)
        print(f"Saved: {out}")


def _lob_mid(lob):
    """Compute mid from lob_json dict, handling empty side."""
    bids = lob.get("bids", [])
    asks = lob.get("asks", [])
    if bids and asks:
        return (bids[0][0] + asks[0][0]) / 2.0
    if bids:
        return float(bids[0][0])
    if asks:
        return float(asks[0][0])
    return None


def plot_lob_heatmap(eval_dir, label, output_dir):
    """Matplotlib heatmap of LOB volume across steps, averaged over episodes."""
    path = os.path.join(eval_dir, "per_step.csv")
    if not os.path.exists(path):
        print(f"  Warning: {path} not found; skipping LOB heatmap")
        return

    rows = load_per_step(path)
    episodes = {}
    for r in rows:
        ep = int(r["episode"])
        if ep not in episodes:
            episodes[ep] = []
        episodes[ep].append(r)

    n_steps = max(len(v) for v in episodes.values())
    ep_mids_list = []
    ep_lob_list = []
    all_ticks = set()

    for ep_id in sorted(episodes.keys()):
        ep_rows = episodes[ep_id]
        mids = []
        lobs = []
        last_mid = 0.0
        for r in ep_rows[:n_steps]:
            try:
                lob = json_mod.loads(r.get("lob_json", "{}"))
            except Exception:
                lob = {"bids": [], "asks": []}
            bids = lob.get("bids", [])
            asks = lob.get("asks", [])
            if bids and asks:
                mid = (bids[0][0] + asks[0][0]) / 2.0
                last_mid = mid
            else:
                mid = last_mid
            mids.append(mid)
            lobs.append(lob)
            if bids and asks:
                for price, _ in bids:
                    d = int(round((price - mid)))
                    all_ticks.add(d)
                for price, _ in asks:
                    d = int(round((price - mid)))
                    all_ticks.add(d)
        ep_mids_list.append(mids)
        ep_lob_list.append(lobs)

    if not all_ticks:
        print("  No LOB data for heatmap")
        return

    lo, hi = min(all_ticks), max(all_ticks)
    tick_range = list(range(lo, hi + 1))
    tick_to_idx = {t: i for i, t in enumerate(tick_range)}
    n_ticks = len(tick_range)

    vol_sum = np.zeros((n_ticks, n_steps), dtype=np.float64)
    vol_count = np.zeros((n_ticks, n_steps), dtype=np.float64)

    for ep_idx in range(len(ep_lob_list)):
        for t_idx in range(min(len(ep_lob_list[ep_idx]), n_steps)):
            mid = ep_mids_list[ep_idx][t_idx]
            lob = ep_lob_list[ep_idx][t_idx]
            bids = lob.get("bids", [])
            asks = lob.get("asks", [])
            if not (bids and asks):
                continue
            for price, qty in bids:
                d = int(round((price - mid)))
                if d in tick_to_idx:
                    vol_sum[tick_to_idx[d], t_idx] += qty
                    vol_count[tick_to_idx[d], t_idx] += 1
            for price, qty in asks:
                d = int(round((price - mid)))
                if d in tick_to_idx:
                    vol_sum[tick_to_idx[d], t_idx] += qty
                    vol_count[tick_to_idx[d], t_idx] += 1

    with np.errstate(invalid="ignore"):
        vol_mean = np.where(vol_count > 0, vol_sum / vol_count, np.nan)

    fig, ax = plt.subplots(figsize=(10, 8))
    cmap = ax.pcolormesh(
        np.arange(n_steps),
        tick_range,
        vol_mean,
        shading="auto",
        cmap="viridis",
        vmin=0,
        vmax=np.nanmax(vol_mean) if not np.all(np.isnan(vol_mean)) else 1,
    )
    ax.axhline(0, color="white", linestyle="--", linewidth=1, alpha=0.7, label="mid")
    ax.set_xlabel("Step")
    ax.set_ylabel("Ticks from mid")
    ax.set_title(f"LOB Volume Heatmap ({label})")
    fig.colorbar(cmap, ax=ax, label="Avg volume")
    fig.tight_layout()
    out = os.path.join(output_dir, f"exp1_lob_heatmap_{label.lower().replace(' ', '_')}.png")
    fig.savefig(out)
    fig.savefig(out.replace(".png", ".pdf"))
    plt.close(fig)
    print(f"Saved: {out}")


def plot_agent_metrics(down_rows, up_rows, output_dir):
    """Plot IT beta, IT fill qty, and MM lambda per step, averaged over episodes."""
    import pandas as pd

    def _avg(df, col):
        return df.groupby("step")[col].agg(["mean", "std"]).reset_index()

    dfd = pd.DataFrame(down_rows)
    dfu = pd.DataFrame(up_rows)
    beta_d = _avg(dfd, "action_raw")
    beta_u = _avg(dfu, "action_raw")
    fq_d = _avg(dfd, "step_fill_qty")
    fq_u = _avg(dfu, "step_fill_qty")
    lam_d = _avg(dfd, "mm_lambda_mean")
    lam_u = _avg(dfu, "mm_lambda_mean")

    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(16, 4))

    ax0.plot(beta_d["step"], beta_d["mean"], label="Down", color="C0")
    ax0.fill_between(beta_d["step"],
                     beta_d["mean"] - beta_d["std"],
                     beta_d["mean"] + beta_d["std"], alpha=0.2, color="C0")
    ax0.plot(beta_u["step"], beta_u["mean"], label="Up", color="C1")
    ax0.fill_between(beta_u["step"],
                     beta_u["mean"] - beta_u["std"],
                     beta_u["mean"] + beta_u["std"], alpha=0.2, color="C1")
    ax0.set_xlabel("Step")
    ax0.set_ylabel("Beta")
    ax0.set_title("IT Beta (action_raw)")
    ax0.legend(fontsize=8)

    ax1.plot(fq_d["step"], fq_d["mean"], label="Down", color="C0")
    ax1.fill_between(fq_d["step"],
                     fq_d["mean"] - fq_d["std"],
                     fq_d["mean"] + fq_d["std"], alpha=0.2, color="C0")
    ax1.plot(fq_u["step"], fq_u["mean"], label="Up", color="C1")
    ax1.fill_between(fq_u["step"],
                     fq_u["mean"] - fq_u["std"],
                     fq_u["mean"] + fq_u["std"], alpha=0.2, color="C1")
    ax1.axhline(500, color="gray", ls=":", label="max_order")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Fill qty")
    ax1.set_title("IT Fill Qty per Step")
    ax1.legend(fontsize=8)

    ax2.plot(lam_d["step"], lam_d["mean"], label="Down", color="C0")
    ax2.fill_between(lam_d["step"],
                     lam_d["mean"] - lam_d["std"],
                     lam_d["mean"] + lam_d["std"], alpha=0.2, color="C0")
    ax2.plot(lam_u["step"], lam_u["mean"], label="Up", color="C1")
    ax2.fill_between(lam_u["step"],
                     lam_u["mean"] - lam_u["std"],
                     lam_u["mean"] + lam_u["std"], alpha=0.2, color="C1")
    ax2.axhline(0.05, color="gray", ls=":", label="λ_max")
    ax2.axhline(0.03, color="gray", ls=":", label="λ_min")
    ax2.set_xlabel("Step")
    ax2.set_ylabel("λ mean")
    ax2.set_title("MM Lambda")
    ax2.legend(fontsize=8)

    fig.tight_layout()
    out = os.path.join(output_dir, "exp1_agent_metrics.png")
    fig.savefig(out)
    fig.savefig(out.replace(".png", ".pdf"))
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-up-dir", default=None,
                        help="Dir containing per_step.csv for up (open=1.3u_v) eval")
    parser.add_argument("--eval-down-dir", default=None,
                        help="Dir containing per_step.csv for down (open=0.7u_v) eval")
    parser.add_argument("--train-dir", default=None,
                        help="Train results dir containing training_log.csv")
    parser.add_argument("--config-name", default="Exp1", help="Title prefix")
    parser.add_argument("--experiment-dir", required=True,
                        help="Experiment folder, e.g. results/exp1_exec_cost_aware")
    parser.add_argument("--lob-down-dir", default=None,
                        help="Dir with per_step.csv to produce LOB heatmap for down eval")
    parser.add_argument("--lob-up-dir", default=None,
                        help="Dir with per_step.csv to produce LOB heatmap for up eval")
    args = parser.parse_args()

    experiment_dir = os.path.abspath(args.experiment_dir)
    output_dir = os.path.join(experiment_dir, "figures")
    os.makedirs(output_dir, exist_ok=True)

    up_rows = []
    down_rows = []
    if args.eval_up_dir:
        up_path = os.path.join(args.eval_up_dir, "per_step.csv")
        if os.path.exists(up_path):
            up_rows = load_per_step(up_path)
        else:
            print(f"  Warning: {up_path} not found")
    if args.eval_down_dir:
        down_path = os.path.join(args.eval_down_dir, "per_step.csv")
        if os.path.exists(down_path):
            down_rows = load_per_step(down_path)
        else:
            print(f"  Warning: {down_path} not found")

    if up_rows and down_rows:
        plot_combined_grid(
            up_rows, down_rows,
            title=f"{args.config_name}: Price Discovery (Up vs Down)",
            out_path=os.path.join(output_dir, "exp1_price_discovery.png"),
        )
        plot_agent_metrics(down_rows, up_rows, output_dir)
    elif up_rows or down_rows:
        print("  Warning: only one of up/down eval dirs has data; skipping combined grid")
    else:
        print("  No eval data; skipping price-discovery plots")

    if args.train_dir:
        log_path = os.path.join(args.train_dir, "training_log.csv")
        if os.path.exists(log_path):
            plot_training_curves_split(log_path, output_dir, prefix="exp1")
        else:
            print(f"  Warning: {log_path} not found; skipping training plots")

    if args.lob_down_dir:
        plot_lob_heatmap(args.lob_down_dir, "Down", output_dir)
    if args.lob_up_dir:
        plot_lob_heatmap(args.lob_up_dir, "Up", output_dir)
