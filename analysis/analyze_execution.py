"""analyze_execution.py -- compute Implementation Shortfall (IS) for Experiment 3.

IS = (sum_n step_vwap(n) * q(n) / Q - p(0)) / p(0)

Usage:
    python -m phase3_multi_agent.analysis.analyze_execution \
        --results-dir phase3_multi_agent/results/eval_.../lt
"""

import argparse
import csv
import json
import os
import sys
from collections import defaultdict

import numpy as np


def load_per_step(path):
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            rows.append({k: float(v) for k, v in row.items()})
    return rows


def compute_is(rows, q_target):
    """Compute implementation shortfall for one episode."""
    if not rows:
        return float("nan")
    p0 = rows[0]["mid"]
    if p0 == 0:
        return float("nan")
    total_cost = sum(r.get("mid", 0) * r.get("inventory_delta", 0) for r in rows)
    acquired = rows[-1].get("inventory", 0)
    if acquired == 0:
        return float("nan")
    avg_price = total_cost / acquired
    is_value = (avg_price - p0) / p0
    return float(is_value)


def analyze(results_dir, q_target=1000):
    per_step_path = os.path.join(results_dir, "per_step.csv")
    eval_path = os.path.join(results_dir, "eval_results.csv")

    if not os.path.exists(per_step_path):
        print(f"ERROR: {per_step_path} not found")
        sys.exit(1)

    step_rows = load_per_step(per_step_path)

    episodes = defaultdict(list)
    for r in step_rows:
        episodes[int(r["episode"])].append(r)

    is_values = []
    for ep_id in sorted(episodes.keys()):
        ep_rows = episodes[ep_id]
        p0 = ep_rows[0]["mid"]
        final_inv = ep_rows[-1].get("inventory", 0)
        total_reward = ep_rows[-1].get("ep_reward", 0)
        is_values.append({
            "episode": ep_id,
            "p0": p0,
            "final_inventory": final_inv,
            "completion": final_inv / q_target if q_target > 0 else 0,
            "total_reward": total_reward,
        })

    rewards = [e["total_reward"] for e in is_values]
    completions = [e["completion"] for e in is_values]

    summary = {
        "n_episodes": len(is_values),
        "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
        "std_reward": float(np.std(rewards)) if rewards else 0.0,
        "mean_completion": float(np.mean(completions)) if completions else 0.0,
        "q_target": q_target,
    }

    print(json.dumps(summary, indent=2))

    out_path = os.path.join(results_dir, "execution_metrics.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to {out_path}")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--q-target", type=int, default=1000)
    args = parser.parse_args()
    analyze(args.results_dir, args.q_target)
