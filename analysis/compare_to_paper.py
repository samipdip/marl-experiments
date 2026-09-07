"""compare_to_paper.py -- Compare our results to the ABIDES-MARL paper.

Usage:
    python -m analysis.compare_to_paper --results-dir results/
"""

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


PAPER_VALUES = {
    "exp1": {
        "down": {
            "phi_ar1": 0.745, "phi_pval": 3.18e-88,
            "half_life": 2.36, "kyle_lambda": 2.30, "kurt": 0.02,
            "ad_pval": 4.2e-07, "arch_pval": 0.275,
        },
        "up": {
            "phi_ar1": 0.804, "phi_pval": 0.0,
            "half_life": 3.18, "kyle_lambda": 2.78, "kurt": 0.62,
            "ad_pval": 1.2e-06, "arch_pval": 0.028,
        },
    },
    "exp3_is": {
        "down": {"ppo": 0.876, "vwap": 0.834, "twap": 0.750, "analytical": 0.775},
        "up": {"ppo": 0.149, "vwap": 0.082, "twap": -0.028, "analytical": 0.117},
    },
}


def find_latest_train_dir(results_root, pattern):
    subdirs = {"kyle_exp1": "exp1", "kyle_exp2": "exp2", "kyle_exp3": "exp3"}
    if pattern in subdirs:
        subdir = os.path.join(results_root, subdirs[pattern])
        if os.path.isdir(subdir) and os.path.exists(os.path.join(subdir, "training_log.csv")):
            return subdir
    matches = sorted(Path(results_root).glob(f"train_{pattern}*"))
    if matches:
        return str(matches[-1])
    return None


def load_training_log(train_dir):
    path = os.path.join(train_dir, "training_log.csv")
    if not os.path.exists(path):
        return {}
    agents = defaultdict(lambda: {"steps": [], "policy_loss": [], "value_loss": [], "entropy": []})
    with open(path) as f:
        for row in csv.DictReader(f):
            name = row["agent"]
            stats = json.loads(row["stats_json"])
            agents[name]["steps"].append(int(row["global_step"]))
            agents[name]["policy_loss"].append(stats.get("policy_loss", 0))
            agents[name]["value_loss"].append(stats.get("value_loss", 0))
            agents[name]["entropy"].append(stats.get("entropy", 0))
    return dict(agents)


def load_eval_data(train_dir, agent_name=None):
    eval_root = os.path.join(train_dir, "eval")
    if not os.path.isdir(eval_root):
        return None, None
    eval_steps = sorted(Path(eval_root).glob("step_*"), key=lambda p: int(p.name.split("_")[1]))
    if not eval_steps:
        return None, None
    final_step = str(eval_steps[-1])
    if agent_name:
        agents_to_check = [agent_name]
    else:
        agents_to_check = [d.name for d in Path(final_step).iterdir() if d.is_dir()]
    all_data = {}
    for agent in agents_to_check:
        per_step_path = os.path.join(final_step, agent, "per_step.csv")
        if os.path.exists(per_step_path):
            rows = []
            with open(per_step_path) as f:
                for row in csv.DictReader(f):
                    rows.append({k: (float(v) if v != "" else 0.0) for k, v in row.items()})
            all_data[agent] = rows
        summary_path = os.path.join(final_step, agent, "summary.json")
        if os.path.exists(summary_path):
            with open(summary_path) as f:
                all_data[f"{agent}_summary"] = json.load(f)
    return all_data, final_step


def compute_ar1(errors):
    from scipy import stats
    e = np.array(errors)
    if len(e) < 3 or len(set(e)) < 2:
        return 0.0, 1.0
    phi, _, _, pval, _ = stats.linregress(e[:-1], e[1:])
    return float(phi), float(pval)


def half_life(phi):
    if abs(phi) < 1e-10 or abs(phi) >= 1.0:
        return float("inf")
    return float(-np.log(2) / np.log(abs(phi)))


def excess_kurtosis(returns):
    from scipy import stats
    if len(returns) < 4:
        return 0.0
    return float(stats.kurtosis(returns, fisher=True))


def arch_lm_test(returns, lags=5):
    try:
        from statsmodels.stats.diagnostic import het_arch
        if len(returns) < lags + 5:
            return float("nan")
        return float(het_arch(np.array(returns, dtype=float), nlags=lags)[1])
    except Exception:
        return float("nan")


def ad_test_pvalue(returns):
    from scipy import stats
    if len(returns) < 8:
        return 1.0
    try:
        result = stats.anderson(returns, dist="norm")
        if result.statistic < result.critical_values[0]:
            return 0.5
        for i in range(len(result.critical_values) - 1):
            if result.critical_values[i] <= result.statistic < result.critical_values[i + 1]:
                p_vals = [10, 5, 2.5, 1]
                frac = (result.statistic - result.critical_values[i]) / (result.critical_values[i + 1] - result.critical_values[i] + 1e-10)
                return float((p_vals[i] + frac * (p_vals[i + 1] - p_vals[i])) / 100)
        return 0.001
    except Exception:
        return 1.0


def kyle_lambda_regression(dprices, order_flows):
    from scipy import stats
    if len(dprices) < 3:
        return 0.0, 1.0
    if len(set(order_flows)) < 2:
        return 0.0, 1.0
    lam, _, _, pval, _ = stats.linregress(order_flows, dprices)
    return float(lam), float(pval)


def get_mid_or_vwap(row):
    mid = row.get("mid", 0)
    if mid and mid > 0:
        return mid
    vwap = row.get("step_vwap", 0)
    if vwap and vwap > 0:
        return vwap
    return 0.0


def split_by_direction(rows):
    """Split eval rows by episode parity: even=down (v=0.7*mu_v), odd=up (v=1.3*mu_v)."""
    episodes = defaultdict(list)
    for r in rows:
        episodes[int(r["episode"])].append(r)
    down, up = {}, {}
    for ep_id, ep_rows in episodes.items():
        if ep_id % 2 == 0:
            down[ep_id] = ep_rows
        else:
            up[ep_id] = ep_rows
    return down, up


def compute_price_metrics(episodes_dict):
    all_errors = []
    all_returns = []
    all_dprices = []
    all_order_flows = []
    for ep_id, ep_rows in sorted(episodes_dict.items()):
        v = ep_rows[0].get("v", 0)
        prices = [get_mid_or_vwap(r) for r in ep_rows]
        prices = [p for p in prices if p > 0]
        if not prices:
            continue
        if v and v > 0:
            errors = [v - p for p in prices]
            all_errors.extend(errors)
        rets = [prices[i] - prices[i - 1] for i in range(1, len(prices))]
        all_returns.extend(rets)
        for i in range(1, len(ep_rows)):
            dp = prices[i] - prices[i - 1] if prices[i - 1] > 0 else 0
            q = ep_rows[i].get("step_fill_qty", 0)
            if dp != 0 and q > 0:
                all_dprices.append(dp)
                all_order_flows.append(q)
    phi, phi_pval = compute_ar1(all_errors) if all_errors else (0.0, 1.0)
    hl = half_life(phi)
    kurt = excess_kurtosis(all_returns)
    ad_pval = ad_test_pvalue(all_returns)
    arch_pval = arch_lm_test(all_returns)
    lam, lam_pval = kyle_lambda_regression(all_dprices, all_order_flows) if all_dprices else (0.0, 1.0)
    return {
        "phi_ar1": round(phi, 4),
        "phi_pval": f"{phi_pval:.2e}",
        "half_life": round(hl, 2),
        "kyle_lambda": round(lam, 4),
        "kyle_lambda_pval": f"{lam_pval:.2e}",
        "excess_kurtosis": round(kurt, 4),
        "ad_pvalue": f"{ad_pval:.2e}",
        "arch_lm_pvalue": round(arch_pval, 4),
        "n_episodes": len(episodes_dict),
    }


def print_comparison_table(label, our_metrics, paper_metrics):
    print(f"\n  {label} — Comparison to paper (Table 1: linear, 20 MM, exchange):")
    print(f"  {'Metric':25s} {'Our Result':>15s} {'Paper':>15s} {'Match?':>10s}")
    print(f"  {'-'*65}")
    for metric, paper_val in paper_metrics.items():
        ours = our_metrics.get(metric, "N/A")
        if isinstance(ours, str):
            match = "N/A"
        elif isinstance(paper_val, (int, float)):
            ratio = abs(ours - paper_val) / max(abs(paper_val), 1e-6)
            match = "OK" if ratio < 0.5 else "DIFF"
        else:
            match = "N/A"
        print(f"  {metric:25s} {str(ours):>15s} {str(paper_val):>15s} {match:>10s}")


def analyze_exp1(train_dir):
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Price Discovery (20 MM, exchange, linear)")
    print("=" * 70)

    training = load_training_log(train_dir)
    if training:
        mm_count = sum(1 for k in training if k.startswith("mm_"))
        print(f"\nTraining data: {len(training)} agents ({mm_count} MMs)")
        for name in ["informed"] + [f"mm_{i}" for i in range(min(3, mm_count))]:
            if name in training:
                d = training[name]
                if d["policy_loss"]:
                    print(f"  {name:12s}: pol={d['policy_loss'][-1]:.4f}, val={d['value_loss'][-1]:.2f}, ent={d['entropy'][-1]:.4f}")

    eval_data, eval_step = load_eval_data(train_dir)
    if not eval_data:
        print("  No eval data found")
        return None

    informed_key = None
    for k in eval_data:
        if "inform" in k.lower() and not k.endswith("_summary"):
            informed_key = k
            break
    if not informed_key:
        informed_key = list(eval_data.keys())[0]

    rows = eval_data[informed_key]
    print(f"\nEval data ({informed_key}): {len(rows)} rows from {eval_step}")

    down_eps, up_eps = split_by_direction(rows)
    print(f"  Down episodes: {len(down_eps)}, Up episodes: {len(up_eps)}")

    down_metrics = compute_price_metrics(down_eps)
    up_metrics = compute_price_metrics(up_eps)

    print(f"\n  DOWN (v=0.7*mu_v={0.7*1000:.0f}):")
    for k, v in down_metrics.items():
        print(f"    {k:25s}: {v}")
    print_comparison_table("DOWN", down_metrics, PAPER_VALUES["exp1"]["down"])

    print(f"\n  UP (v=1.3*mu_v={1.3*1000:.0f}):")
    for k, v in up_metrics.items():
        print(f"    {k:25s}: {v}")
    print_comparison_table("UP", up_metrics, PAPER_VALUES["exp1"]["up"])

    summary_key = f"{informed_key}_summary"
    if summary_key in eval_data:
        s = eval_data[summary_key]
        print(f"\n  IT reward: mean={s.get('mean_reward', 0):.2f}, std={s.get('std_reward', 0):.2f}")

    return {"down": down_metrics, "up": up_metrics}


def analyze_exp2(train_dir):
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: LT vs Market Makers (20 MM)")
    print("=" * 70)

    training = load_training_log(train_dir)
    if training:
        mm_count = sum(1 for k in training if k.startswith("mm_"))
        print(f"\nTraining data: {len(training)} agents ({mm_count} MMs)")
        for name in ["lt"] + [f"mm_{i}" for i in range(min(3, mm_count))]:
            if name in training:
                d = training[name]
                if d["policy_loss"]:
                    print(f"  {name:12s}: pol={d['policy_loss'][-1]:.4f}, val={d['value_loss'][-1]:.2f}, ent={d['entropy'][-1]:.4f}")

    eval_data, eval_step = load_eval_data(train_dir)
    if not eval_data:
        print("  No eval data found")
        return None

    lt_key = "lt"
    if lt_key not in eval_data:
        for k in eval_data:
            if "lt" in k.lower() and not k.endswith("_summary"):
                lt_key = k
                break

    rows = eval_data[lt_key]
    print(f"\nEval data ({lt_key}): {len(rows)} rows from {eval_step}")

    episodes = defaultdict(list)
    for r in rows:
        episodes[int(r["episode"])].append(r)

    completions = []
    rewards = []
    for ep_id in sorted(episodes.keys()):
        ep_rows = episodes[ep_id]
        final_inv = ep_rows[-1].get("inventory", 0)
        q_target = 1000
        completions.append(min(final_inv / q_target, 1.0) if final_inv > 0 else 0.0)
        rewards.append(ep_rows[-1].get("ep_reward", 0))

    summary_key = f"{lt_key}_summary"
    is_bps = None
    if summary_key in eval_data:
        s = eval_data[summary_key]
        is_bps = s.get("mean_is_bps")
        print(f"\n  LT summary: reward={s.get('mean_reward',0):.2f} ± {s.get('std_reward',0):.2f}")
        if is_bps is not None:
            print(f"  IS: {is_bps:.2f} bps")
        comp = s.get("mean_completion")
        if comp is not None:
            print(f"  Completion: {comp:.1%}")

    results = {
        "mean_completion": round(float(np.mean(completions)), 4),
        "std_completion": round(float(np.std(completions)), 4),
        "mean_reward": round(float(np.mean(rewards)), 2),
        "std_reward": round(float(np.std(rewards)), 2),
        "mean_is_bps": round(is_bps, 2) if is_bps else None,
        "n_episodes": len(episodes),
    }

    print(f"\nResults:")
    for k, v in results.items():
        print(f"  {k:25s}: {v}")

    return results


def analyze_exp3(train_dir):
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Full Game (IT + LT + 20 MMs)")
    print("=" * 70)

    training = load_training_log(train_dir)
    if training:
        mm_count = sum(1 for k in training if k.startswith("mm_"))
        print(f"\nTraining data: {len(training)} agents ({mm_count} MMs)")
        for name in ["informed", "lt"] + [f"mm_{i}" for i in range(min(3, mm_count))]:
            if name in training:
                d = training[name]
                if d["policy_loss"]:
                    print(f"  {name:12s}: pol={d['policy_loss'][-1]:.4f}, val={d['value_loss'][-1]:.2f}, ent={d['entropy'][-1]:.4f}")

    eval_data, eval_step = load_eval_data(train_dir)
    if not eval_data:
        print("  No eval data found")
        return None

    results = {}

    # LT execution metrics
    lt_key = "lt"
    if lt_key in eval_data:
        rows = eval_data[lt_key]
        down_eps, up_eps = split_by_direction(rows)

        for label, eps_dict in [("DOWN", down_eps), ("UP", up_eps)]:
            completions = []
            rewards = []
            is_vals = []
            for ep_id, ep_rows in sorted(eps_dict.items()):
                final_inv = ep_rows[-1].get("inventory", 0)
                completions.append(min(final_inv / 1000, 1.0) if final_inv > 0 else 0.0)
                rewards.append(ep_rows[-1].get("ep_reward", 0))
                arrival = ep_rows[0].get("mid", 0) or ep_rows[0].get("step_vwap", 0) or 1000
                cash = ep_rows[-1].get("cash", 0)
                if final_inv > 0 and arrival > 0:
                    avg_fill = abs(cash / final_inv) if final_inv > 0 else 0
                    is_vals.append((avg_fill - arrival) / arrival)
            results[f"lt_{label.lower()}"] = {
                "mean_completion": round(float(np.mean(completions)), 4),
                "mean_reward": round(float(np.mean(rewards)), 2),
                "mean_is": round(float(np.mean(is_vals)), 4) if is_vals else None,
                "n_episodes": len(eps_dict),
            }
            print(f"\n  LT ({label}): completion={results[f'lt_{label.lower()}']['mean_completion']:.1%}, "
                  f"reward={results[f'lt_{label.lower()}']['mean_reward']:.2f}, "
                  f"IS={results[f'lt_{label.lower()}']['mean_is']}")

    # IT profit metrics
    it_key = "informed"
    if it_key in eval_data:
        rows = eval_data[it_key]
        down_eps, up_eps = split_by_direction(rows)

        for label, eps_dict in [("DOWN", down_eps), ("UP", up_eps)]:
            rewards = [ep_rows[-1].get("ep_reward", 0) for ep_rows in eps_dict.values()]
            results[f"it_{label.lower()}"] = {
                "mean_reward": round(float(np.mean(rewards)), 2),
                "n_episodes": len(eps_dict),
            }
            print(f"  IT ({label}): reward={results[f'it_{label.lower()}']['mean_reward']:.2f}")

        # Price discovery
        down_metrics = compute_price_metrics(down_eps)
        up_metrics = compute_price_metrics(up_eps)
        results["price_discovery_down"] = down_metrics
        results["price_discovery_up"] = up_metrics
        print(f"\n  Price Discovery (DOWN): phi={down_metrics['phi_ar1']}, hl={down_metrics['half_life']}, kurt={down_metrics['excess_kurtosis']}")
        print(f"  Price Discovery (UP):   phi={up_metrics['phi_ar1']}, hl={up_metrics['half_life']}, kurt={up_metrics['excess_kurtosis']}")

    # Compare IS to paper
    print(f"\n  IS Comparison (Table 2: linear, phi=0.01, LOB=0):")
    print(f"  {'Direction':12s} {'Our PPO':>10s} {'Paper PPO':>10s} {'Paper TWAP':>10s} {'Paper VWAP':>10s}")
    print(f"  {'-'*55}")
    for d in ["down", "up"]:
        our = results.get(f"lt_{d}", {}).get("mean_is", "N/A")
        paper = PAPER_VALUES["exp3_is"][d]
        our_str = f"{our:.3f}" if isinstance(our, float) else str(our)
        print(f"  {d.upper():12s} {our_str:>10s} {paper['ppo']:>10.3f} {paper['twap']:>10.3f} {paper['vwap']:>10.3f}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()

    results_root = args.results_dir
    all_results = {}

    for pattern, analyzer, label in [
        ("kyle_exp1", analyze_exp1, "Exp1"),
        ("kyle_exp2", analyze_exp2, "Exp2"),
        ("kyle_exp3", analyze_exp3, "Exp3"),
    ]:
        d = find_latest_train_dir(results_root, pattern)
        if d:
            print(f"\nFound {label}: {d}")
            all_results[label.lower()] = analyzer(d)
        else:
            print(f"No {label} results found")

    out_path = os.path.join(results_root, "comparison_report.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n{'='*70}")
    print(f"Report saved to: {out_path}")


if __name__ == "__main__":
    main()
