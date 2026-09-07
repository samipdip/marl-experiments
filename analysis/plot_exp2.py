"""plot_exp2.py -- LT vs MM price + inventory dynamics (paper Figure 4).

Usage:
    python -m analysis.plot_exp2 \
        --results-dir results/train_kyle_exp2_.../eval/step_20000 \
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
    "font.size": 12,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
}


def load_csv(path):
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            rows.append({k: float(v) for k, v in row.items()})
    return rows


def plot_price_inventory(lt_rows, mm_rows_list, title, out_path, v_line=None):
    """Two-panel: price trajectory (left), LT inventory (right)."""
    plt.rcParams.update(PLOT_STYLE)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # --- Price trajectory ---
    by_step = defaultdict(list)
    for r in lt_rows:
        by_step[int(r["step"])].append(r["mid"])

    steps = sorted(by_step.keys())
    mean_prices = [np.mean(by_step[s]) for s in steps]
    std_prices = [np.std(by_step[s]) for s in steps]

    ax1.plot(steps, mean_prices, "b-", linewidth=1.5, label="Step mid price")
    ax1.fill_between(
        steps,
        np.array(mean_prices) - np.array(std_prices),
        np.array(mean_prices) + np.array(std_prices),
        alpha=0.2,
        color="blue",
    )

    mean_v = np.mean([r.get("v", 0) for r in lt_rows if r.get("v", 0) > 0]) or v_line
    if mean_v:
        ax1.axhline(y=mean_v, color="red", linestyle="--", linewidth=1.2, label="Fundamental value v")

    ax1.set_xlabel("Step")
    ax1.set_ylabel("Price")
    ax1.set_title(f"{title} - Price")
    ax1.legend(fontsize=10)

    # --- LT inventory ---
    by_step_inv = defaultdict(list)
    for r in lt_rows:
        by_step_inv[int(r["step"])].append(r.get("inventory", 0))

    mean_inv = [np.mean(by_step_inv[s]) for s in steps]
    std_inv = [np.std(by_step_inv[s]) for s in steps]

    ax2.plot(steps, mean_inv, "g-", linewidth=1.5, label="LT acquired inventory")
    ax2.fill_between(
        steps,
        np.array(mean_inv) - np.array(std_inv),
        np.array(mean_inv) + np.array(std_inv),
        alpha=0.2,
        color="green",
    )
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Inventory")
    ax2.set_title(f"{title} - LT Acquisition")
    ax2.legend(fontsize=10)

    fig.tight_layout()
    fig.savefig(out_path)
    fig.savefig(out_path.replace(".png", ".pdf"))
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_mm_inventory(mm_rows_list, title, out_path):
    """Plot aggregate MM inventory over time."""
    plt.rcParams.update(PLOT_STYLE)

    fig, ax = plt.subplots(figsize=(7, 4.5))

    for mm_rows in mm_rows_list:
        agent_name = mm_rows[0].get("agent", "mm") if mm_rows else "mm"
        by_step = defaultdict(list)
        for r in mm_rows:
            by_step[int(r["step"])].append(r.get("inventory", 0))
        steps = sorted(by_step.keys())
        mean_inv = [np.mean(by_step[s]) for s in steps]
        ax.plot(steps, mean_inv, linewidth=1, alpha=0.7)

    ax.set_xlabel("Step")
    ax.set_ylabel("Inventory")
    ax.set_title(f"{title} - MM Inventories")
    fig.savefig(out_path)
    fig.savefig(out_path.replace(".png", ".pdf"))
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True, help="Eval step dir with agent subdirs")
    parser.add_argument("--config-name", default="Exp2: LT vs 20 MM")
    parser.add_argument("--output-dir", default="results/figures")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Find LT and MM agent dirs
    lt_dir = None
    mm_dirs = []
    for d in sorted(os.listdir(args.results_dir)):
        full = os.path.join(args.results_dir, d)
        if os.path.isdir(full):
            csv_path = os.path.join(full, "per_step.csv")
            if os.path.exists(csv_path):
                if "lt" in d.lower() or "liquid" in d.lower():
                    lt_dir = full
                elif "mm" in d.lower():
                    mm_dirs.append(full)

    if lt_dir:
        lt_rows = load_csv(os.path.join(lt_dir, "per_step.csv"))
        mm_rows_list = [load_csv(os.path.join(d, "per_step.csv")) for d in mm_dirs]
        plot_price_inventory(
            lt_rows, mm_rows_list, args.config_name,
            os.path.join(args.output_dir, "exp2_price_inventory.png"),
        )
        if mm_rows_list:
            plot_mm_inventory(
                mm_rows_list, args.config_name,
                os.path.join(args.output_dir, "exp2_mm_inventory.png"),
            )
    else:
        print("ERROR: No LT agent dir found")
