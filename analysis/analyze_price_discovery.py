"""analyze_price_discovery.py -- compute Table 1 metrics from eval logs.

Usage:
    python -m phase3_multi_agent.analysis.analyze_price_discovery \
        --results-dir phase3_multi_agent/results/train_.../eval/step_20000/informed

Reads per_step.csv and eval_results.csv, computes:
    - AR(1) coefficient + p-value on pricing error e(n) = v - mid(n)
    - Half-life of decay
    - Kyle lambda regression: delta_mid ~ q(n)
    - Excess kurtosis of returns
    - Anderson-Darling normality test
    - Engle ARCH-LM test
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
            parsed = {}
            for k, v in row.items():
                if k == "lob_json":
                    continue
                parsed[k] = float(v)
            rows.append(parsed)
    return rows


def compute_ar1(errors):
    """Fit AR(1): e(n) = phi * e(n-1) + eps. Return phi, p-value."""
    from scipy import stats
    e = np.array(errors)
    if len(e) < 3:
        return 0.0, 1.0
    e_prev = e[:-1]
    e_curr = e[1:]
    phi, intercept, r_value, p_value, std_err = stats.linregress(e_prev, e_curr)
    return float(phi), float(p_value)


def half_life(phi):
    if abs(phi) < 1e-10 or abs(phi) >= 1.0:
        return float("inf")
    return float(-np.log(2) / np.log(abs(phi)))


def compute_kyle_lambda(mids, order_flows):
    """Kyle regression: delta_mid(n) = lambda * q(n) + xi."""
    from scipy import stats
    mids = np.array(mids, dtype=float)
    q = np.array(order_flows, dtype=float)
    if len(mids) < 3:
        return 0.0, 1.0
    delta_mid = np.diff(mids)
    q_aligned = q[1:]
    if np.std(q_aligned) < 1e-10:
        return 0.0, 1.0
    lam, intercept, r_value, p_value, std_err = stats.linregress(q_aligned, delta_mid)
    return float(lam), float(p_value)


def excess_kurtosis(returns):
    from scipy import stats
    if len(returns) < 4:
        return 0.0
    return float(stats.kurtosis(returns, fisher=True))


def ad_test_pvalue(returns):
    from scipy import stats
    if len(returns) < 8:
        return 1.0
    try:
        result = stats.anderson(returns, dist="norm")
        # Convert to approximate p-value using the statistic
        stat = result.statistic
        sig_levels = result.significance_level
        crit_vals = result.critical_values
        # Interpolate p-value at the 5% level
        p_vals = [10, 5, 2.5, 1]
        if stat < crit_vals[0]:
            return 0.5
        for i in range(len(crit_vals) - 1):
            if crit_vals[i] <= stat < crit_vals[i + 1]:
                frac = (stat - crit_vals[i]) / (crit_vals[i + 1] - crit_vals[i] + 1e-10)
                return float((p_vals[i] + frac * (p_vals[i + 1] - p_vals[i])) / 100)
        return 0.001
    except Exception:
        return 1.0


def arch_lm_test(returns, lags=5):
    """Engle's ARCH-LM test. Returns p-value."""
    try:
        from statsmodels.stats.diagnostic import het_arch
        r = np.array(returns, dtype=float)
        if len(r) < lags + 5:
            return 1.0
        result = het_arch(r, nlags=lags)
        return float(result[1])
    except ImportError:
        return float("nan")
    except Exception:
        return 1.0


def analyze(results_dir):
    per_step_path = os.path.join(results_dir, "per_step.csv")
    if not os.path.exists(per_step_path):
        print(f"ERROR: {per_step_path} not found")
        sys.exit(1)

    rows = load_per_step(per_step_path)

    episodes = defaultdict(list)
    for r in rows:
        episodes[int(r["episode"])].append(r)

    all_errors = []
    all_mids = []
    all_returns = []
    all_order_flows = []

    for ep_id in sorted(episodes.keys()):
        ep_rows = episodes[ep_id]
        v = ep_rows[0]["v"] if ep_rows[0]["v"] != 0 else None

        mids = [r.get("step_vwap", r["mid"]) for r in ep_rows]
        # Skip zero-VWAP entries (no trades that step)
        valid = [(s, m) for s, m in enumerate(mids) if m > 0]
        if len(valid) < 2:
            continue
        vsteps, vwap = zip(*valid)
        all_mids.extend(vwap)

        if v and v > 0:
            errors = [v - m for m in vwap]
            all_errors.extend(errors)

        rets = [vwap[i] - vwap[i - 1] for i in range(1, len(vwap))]
        all_returns.extend(rets)

    phi, phi_pval = compute_ar1(all_errors) if all_errors else (0.0, 1.0)
    hl = half_life(phi)
    lam, lam_pval = compute_kyle_lambda(all_mids, [0] * len(all_mids))
    kurt = excess_kurtosis(all_returns)
    ad_pval = ad_test_pvalue(all_returns)
    arch_pval = arch_lm_test(all_returns)

    result = {
        "n_episodes": len(episodes),
        "n_steps_total": len(rows),
        "phi_ar1": round(phi, 4),
        "p_phi": f"{phi_pval:.2e}",
        "half_life": round(hl, 2),
        "kyle_lambda": round(lam, 4),
        "p_lambda": f"{lam_pval:.2e}",
        "excess_kurtosis": round(kurt, 4),
        "ad_pvalue": f"{ad_pval:.2e}",
        "arch_lm_pvalue": round(arch_pval, 4),
    }

    print(json.dumps(result, indent=2))

    out_path = os.path.join(results_dir, "price_discovery_metrics.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {out_path}")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True)
    args = parser.parse_args()
    analyze(args.results_dir)
