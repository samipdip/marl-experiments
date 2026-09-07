"""plot_exp3.py -- Execution strategy comparison (paper Figures 5-6 + Tables 2-3).

Usage:
    python -m analysis.plot_exp3 \
        --results-dir results/exp3_eval_... \
        --output-dir results/figures
"""

import argparse
import csv
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PLOT_STYLE = {
    "font.family": "serif",
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
}

STRATEGIES = ["ppo", "vwap", "twap", "analytical", "ppo_single"]
STRATEGY_LABELS = {
    "ppo": "PPO (multi-agent)",
    "vwap": "VWAP",
    "twap": "TWAP",
    "analytical": "Analytical (Kyle-λ)",
    "ppo_single": "PPO-Single",
}


def load_csv(path):
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            rows.append({k: float(v) for k, v in row.items()})
    return rows


def plot_strategy_panel(lt_rows, strategy_name, ax_price, ax_inventory, v_line=None):
    """Plot one strategy's price + inventory on given axes."""
    by_step = defaultdict(list)
    by_step_inv = defaultdict(list)
    for r in lt_rows:
        s = int(r["step"])
        by_step[s].append(r["mid"])
        by_step_inv[s].append(r.get("inventory", 0))

    steps = sorted(by_step.keys())
    mean_prices = [np.mean(by_step[s]) for s in steps]
    std_prices = [np.std(by_step[s]) for s in steps]
    mean_inv = [np.mean(by_step_inv[s]) for s in steps]
    std_inv = [np.std(by_step_inv[s]) for s in steps]

    mean_v = np.mean([r.get("v", 0) for r in lt_rows if r.get("v", 0) > 0]) or v_line

    ax_price.plot(steps, mean_prices, "b-", linewidth=1.5)
    ax_price.fill_between(
        steps,
        np.array(mean_prices) - np.array(std_prices),
        np.array(mean_prices) + np.array(std_prices),
        alpha=0.2,
        color="blue",
    )
    if mean_v:
        ax_price.axhline(y=mean_v, color="red", linestyle="--", linewidth=1)
    ax_price.set_title(strategy_name, fontsize=11)
    ax_price.set_xlabel("Step")
    ax_price.set_ylabel("Price")

    ax_inventory.plot(steps, mean_inv, "g-", linewidth=1.5)
    ax_inventory.fill_between(
        steps,
        np.array(mean_inv) - np.array(std_inv),
        np.array(mean_inv) + np.array(std_inv),
        alpha=0.2,
        color="green",
    )
    ax_inventory.set_xlabel("Step")
    ax_inventory.set_ylabel("Unfilled Inventory")
    ax_inventory.set_title(strategy_name, fontsize=11)


def plot_strategy_comparison(strategy_data, title, out_path, v_line=None):
    """5-panel grid: one per strategy, each with price + inventory."""
    plt.rcParams.update(PLOT_STYLE)

    fig, axes = plt.subplots(3, 2, figsize=(14, 14))
    axes_flat = axes.flatten()

    found = 0
    for strat in STRATEGIES:
        if strat not in strategy_data:
            continue
        label = STRATEGY_LABELS.get(strat, strat)
        row, col = divmod(found, 2)
        ax_p = axes[row][col] if col == 0 else axes[row][col]
        ax_i = axes[row][col] if col == 1 else axes[row][col]

        # Use twin approach: left col = price, right col = inventory
        # Actually let's do price and inventory on same subplot with twin y
        by_step = defaultdict(list)
        by_step_inv = defaultdict(list)
        for r in strategy_data[strat]:
            s = int(r["step"])
            by_step[s].append(r["mid"])
            by_step_inv[s].append(r.get("inventory", 0))

        steps = sorted(by_step.keys())
        mean_p = [np.mean(by_step[s]) for s in steps]
        std_p = [np.std(by_step[s]) for s in steps]
        mean_inv = [np.mean(by_step_inv[s]) for s in steps]
        mean_v = np.mean([r.get("v", 0) for r in strategy_data[strat] if r.get("v", 0) > 0]) or v_line

        ax = axes_flat[found]
        ax2 = ax.twinx()

        ax.plot(steps, mean_p, "b-", linewidth=1.5, label="Price")
        ax.fill_between(
            steps,
            np.array(mean_p) - np.array(std_p),
            np.array(mean_p) + np.array(std_p),
            alpha=0.15,
            color="blue",
        )
        if mean_v:
            ax.axhline(y=mean_v, color="red", linestyle="--", linewidth=1)

        ax2.plot(steps, mean_inv, "g--", linewidth=1.2, label="Inventory")
        ax.set_title(label, fontsize=11)
        ax.set_xlabel("Step")
        ax.set_ylabel("Price", color="blue")
        ax2.set_ylabel("Inventory", color="green")

        found += 1

    # Hide unused subplots
    for i in range(found, len(axes_flat)):
        axes_flat[i].set_visible(False)

    fig.suptitle(title, fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path)
    fig.savefig(out_path.replace(".png", ".pdf"))
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_is_bar_chart(is_data, title, out_path):
    """Grouped bar chart of IS values across strategies."""
    plt.rcParams.update(PLOT_STYLE)

    configs = sorted(is_data.keys())
    strategies = STRATEGIES

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(configs))
    width = 0.15
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    for i, strat in enumerate(strategies):
        means = []
        stds = []
        for cfg in configs:
            val = is_data[cfg].get(strat, {})
            means.append(val.get("mean", 0))
            stds.append(val.get("std", 0))
        ax.bar(x + i * width, means, width, yerr=stds, label=STRATEGY_LABELS[strat],
               color=colors[i], alpha=0.8, capsize=3)

    ax.set_xlabel("Configuration")
    ax.set_ylabel("Implementation Shortfall")
    ax.set_title(title)
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(configs, rotation=45, ha="right", fontsize=9)
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path)
    fig.savefig(out_path.replace(".png", ".pdf"))
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True,
                        help="Dir with strategy subdirs (ppo/, vwap/, twap/, etc.)")
    parser.add_argument("--config-name", default="Exp3")
    parser.add_argument("--output-dir", default="results/figures")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    strategy_data = {}
    for d in sorted(os.listdir(args.results_dir)):
        full = os.path.join(args.results_dir, d)
        csv_path = os.path.join(full, "per_step.csv")
        if os.path.exists(csv_path):
            strategy_data[d] = load_csv(csv_path)

    if strategy_data:
        plot_strategy_comparison(
            strategy_data, f"{args.config_name}: Strategy Comparison",
            os.path.join(args.output_dir, "exp3_strategy_comparison.png"),
        )
    else:
        print("No strategy data found")
