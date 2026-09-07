"""Post-session analysis: mid vs F, captured %, agent states, arrival fairness.

Usage: python analyze_live.py runs/<session>
Reads <session>/.arena_state.json (or session_meta fallback), mm_*/belief_*
cycle logs, noise_log.csv. Writes analysis.png + summary.csv in the session dir.
"""

from __future__ import annotations

import csv
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_cycle_logs(session_dir):
    series = {}
    for entry in sorted(os.listdir(session_dir)):
        sub = os.path.join(session_dir, entry)
        if not os.path.isdir(sub):
            continue
        for f in os.listdir(sub):
            if f.endswith("_cycle_log.csv") and not f.endswith("_ack_cycle_log.csv"):
                rows = list(csv.DictReader(open(os.path.join(sub, f))))
                if rows:
                    series[entry] = rows
    return series


def main():
    session_dir = sys.argv[1]
    meta = {}
    for name in (".arena_state.json", "session_meta.json"):
        p = os.path.join(session_dir, name)
        if os.path.exists(p):
            meta = json.load(open(p))
            break
    scenario = meta.get("scenario", os.path.basename(session_dir).rsplit("_", 1)[0])
    F = float(meta.get("f", 0.0))
    if not F:
        cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "config", "scenarios.yaml")
        try:
            import yaml
            F = float(yaml.safe_load(open(cfg_path))["scenarios"][scenario]["f"])
        except Exception:
            F = 1000.0

    series = read_cycle_logs(session_dir)
    if not series:
        print(f"no cycle logs in {session_dir}")
        return

    mms = {k: v for k, v in series.items() if k.startswith("mm_")}
    beliefs = {k: v for k, v in series.items() if k.startswith("belief_")}

    n_spawns = len([k for k in series if k.startswith("belief")])

    # Cycle-indexed aggregation (churn sessions have agents with different
    # lifetimes; rectangular min-truncation would clip everything to the
    # shortest spawn). mid<=0 (no book yet) treated as missing. Belief agents
    # align on their logged global cycle (gcycle) when present — churn spawns
    # run local clocks shifted by their spawn cycle.
    n = max(len(v) for v in series.values())
    cycles = np.arange(n)

    def _row_cycle(rows, j):
        try:
            return int(float(rows[j].get("gcycle")))
        except (TypeError, ValueError):
            return j

    def _col(source, field, use_gcycle=False):
        mat = np.full((max(len(source), 1), n), np.nan)
        for i, rows in enumerate(source.values()):
            for j, r in enumerate(rows):
                try:
                    mat[i, _row_cycle(rows, j) if use_gcycle else j] = float(r.get(field))
                except (TypeError, ValueError):
                    pass
        return mat

    mid_mat = _col(series, "mid", use_gcycle=True)
    mid_mat[mid_mat <= 0] = np.nan
    with np.errstate(invalid="ignore"):
        mid_mean = np.nanmean(mid_mat, axis=0)
        mid_std = np.nanstd(mid_mat, axis=0)
    valid_mid = ~np.isnan(mid_mean)
    valid_cycles = cycles[valid_mid]
    vm = mid_mean[valid_mid]

    tail = vm[max(0, len(vm) - 50):]
    head = vm[: min(20, len(vm))]
    pre, post = float(head.mean()), float(tail.mean())
    delta = F - pre
    captured = (post - pre) / delta * 100.0 if abs(delta) > 1e-9 else float("nan")

    # Cross-sectional mean belief per cycle (alive belief agents only).
    if beliefs:
        v_mat = _col(beliefs, "v", use_gcycle=True)
        v_mat[v_mat <= 0] = np.nan
        with np.errstate(invalid="ignore"):
            mean_v = np.nanmean(v_mat, axis=0)
        mean_v_end = float(np.nanmean(mean_v[valid_mid][-50:])) \
            if valid_mid.any() else float("nan")
    else:
        mean_v = np.full(n, np.nan)
        mean_v_end = float("nan")
    delta_v = mean_v_end - pre
    captured_v = ((post - pre) / delta_v * 100.0
                  if beliefs and abs(delta_v) > 1e-9 else float("nan"))

    # Net informed demand: signed sent qty (taker flow intent) from beliefs,
    # aligned on global cycle for churn spawns.
    net_demand = np.zeros(n)
    for rows in beliefs.values():
        for j, r in enumerate(rows):
            side, qty = r.get("side"), float(r.get("qty") or 0)
            d = qty if side == "0" else (-qty if side == "1" else 0.0)
            net_demand[_row_cycle(rows, j)] += d
    cum_demand = np.cumsum(net_demand)
    fit_mask = valid_mid & (np.abs(cum_demand) > 0)
    if fit_mask.sum() > 10 and np.ptp(cum_demand[fit_mask]) > 1e-9:
        impact_coef = float(np.polyfit(cum_demand[fit_mask],
                                       mid_mean[fit_mask], 1)[0])
    else:
        impact_coef = float("nan")

    def _reject_totals(source):
        tot = cash = 0
        for rows in source.values():
            for r in rows:
                tot += int(r.get("n_rejects") or 0)
                cash += int(r.get("n_cash_rejects") or 0)
        return tot, cash

    mm_rejects, _ = _reject_totals(mms)
    belief_rejects, belief_cash_rejects = _reject_totals(beliefs)

    fig, axes = plt.subplots(3, 2, figsize=(13, 13))
    ax = axes[0][0]
    ax.plot(cycles, mid_mean, color="#1f77b4", label="mid (mean over agents)")
    win = 20
    if len(vm) > win:
        roll = np.convolve(vm, np.ones(win) / win, mode="valid")
        ax.plot(valid_cycles[win - 1:], roll, color="#1f77b4", linewidth=2.5,
                alpha=0.8, label=f"mid rolling mean ({win})")
    ax.fill_between(cycles, mid_mean - mid_std, mid_mean + mid_std, color="#1f77b4", alpha=0.2)
    if beliefs:
        ax.plot(cycles, mean_v, color="#ff7f0e", linewidth=1.5,
                label="mean(v_i) beliefs")
    ax.axhline(F, color="#d62728", linestyle="--", linewidth=2, label=f"F={F:.0f}")
    first_valid_mid = float(vm[0]) if len(vm) else float("nan")
    ax.axhline(first_valid_mid, color="#7f7f7f", linestyle=":",
               linewidth=1.5, label=f"first mid={first_valid_mid:.1f}")
    ax.set_title(f"{scenario}: mid vs F  (captured={captured:.1f}%, "
                 f"vs mean(v)={captured_v:.1f}%)")
    ax.set_xlabel("cycle"); ax.set_ylabel("price"); ax.legend(); ax.grid(alpha=0.3)

    ax = axes[0][1]
    names, pnls, invs = [], [], []
    for name, rows in sorted(series.items()):
        last = rows[min(len(rows), n) - 1]
        mid_end = float(last["mid"]) or F
        pnl = float(last["cash"]) + float(last["inventory"]) * mid_end
        names.append(name); pnls.append(pnl); invs.append(float(last["inventory"]))
    y = np.arange(len(names))
    colors = ["#2ca02c" if nm.startswith("mm_") else "#1f77b4" for nm in names]
    ax.barh(y - 0.2, pnls, height=0.4, color=colors, label="PnL (mtm)")
    ax.barh(y + 0.2, invs, height=0.4, color="#7f7f7f", label="inventory")
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=7)
    ax.set_title("final PnL / inventory per agent"); ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1][0]
    if mms:
        kb = [float(v[min(len(v), n) - 1]["kappa_bid"]) for v in mms.values()]
        ka = [float(v[min(len(v), n) - 1]["kappa_ask"]) for v in mms.values()]
        ax.scatter(kb, ka, c=range(len(kb)), cmap="tab20", s=60)
        ax.plot([0, 2], [0, 2], color="gray", linestyle="--", alpha=0.5)
        ax.set_xlabel("kappa_bid"); ax.set_ylabel("kappa_ask")
        ax.set_title("MM kappas (last cycle)"); ax.grid(alpha=0.3)

    ax = axes[1][1]
    ts_by_cycle = {}
    for name, rows in series.items():
        for i, r in enumerate(rows[:n]):
            ts_by_cycle.setdefault(i, []).append((float(r["ts"]), name))
    ranks = {name: [] for name in series}
    for i, lst in sorted(ts_by_cycle.items()):
        if len(lst) < 2:
            continue
        for rank, (_ts, name) in enumerate(sorted(lst)):
            ranks[name].append(rank)
    names_r = sorted(ranks)
    mean_ranks = [np.mean(ranks[nm]) if ranks[nm] else np.nan for nm in names_r]
    expected = (len(names_r) - 1) / 2.0
    ax.bar(range(len(names_r)), mean_ranks, color="#9467bd")
    ax.axhline(expected, color="#d62728", linestyle="--", label=f"uniform expectation {expected:.1f}")
    ax.set_xticks(range(len(names_r)))
    ax.set_xticklabels(names_r, rotation=90, fontsize=7)
    ax.set_ylabel("mean arrival rank per cycle")
    ax.set_title("fairness check (lower = consistently earlier)")
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[2][0]
    ax.plot(cum_demand, mid_mean, color="#2ca02c", linewidth=1.2)
    if impact_coef == impact_coef:
        ax.set_title(f"impact: mid vs cumulative net informed demand "
                     f"(slope={impact_coef:.3f} price/unit)")
    else:
        ax.set_title("impact: mid vs cumulative net informed demand (degenerate)")
    ax.set_xlabel("cumulative net informed demand (units, + = buyer pressure)")
    ax.set_ylabel("mid"); ax.grid(alpha=0.3)

    ax = axes[2][1]
    spr_mat = _col(series, "ask", use_gcycle=True) - _col(series, "bid", use_gcycle=True)
    spr_mat[(spr_mat < 0) | ~np.isfinite(spr_mat)] = np.nan
    with np.errstate(invalid="ignore"):
        spread = np.nanmean(spr_mat, axis=0)
    ax.plot(cycles, spread, color="#9467bd", linewidth=1.0)
    ax.set_title(f"spread (first agent's book); rejects: mm={mm_rejects} "
                 f"belief={belief_rejects} (cash={belief_cash_rejects})")
    ax.set_xlabel("cycle"); ax.set_ylabel("ask - bid"); ax.grid(alpha=0.3)

    plt.tight_layout()
    out_png = os.path.join(session_dir, "analysis.png")
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"saved {out_png}")

    summary = {
        "scenario": scenario, "F": F, "n_cycles": n, "n_agents": len(series),
        "n_trader_spawns": n_spawns,
        "pre_mid": round(pre, 3), "post_mid": round(post, 3),
        "first_valid_mid": round(first_valid_mid, 3)
        if first_valid_mid == first_valid_mid else "nan",
        "max_mid": round(float(np.nanmax(vm)), 3) if len(vm) else "nan",
        "min_mid": round(float(np.nanmin(vm)), 3) if len(vm) else "nan",
        "captured_pct": round(captured, 1) if captured == captured else "nan",
        "mean_v_end": round(mean_v_end, 3) if mean_v_end == mean_v_end else "nan",
        "captured_vs_meanv_pct": round(captured_v, 1) if captured_v == captured_v else "nan",
        "impact_coef_per_unit": round(impact_coef, 4) if impact_coef == impact_coef else "nan",
        "cum_net_demand_units": round(float(cum_demand[-1]), 1),
        "mm_rejects": mm_rejects, "belief_rejects": belief_rejects,
        "belief_cash_rejects": belief_cash_rejects,
        "mid_std_end": round(float(np.nanmean(mid_std[valid_mid][-50:]))
                             if valid_mid.any() else float("nan"), 3),
        "fairness_min_rank": round(float(np.nanmin(mean_ranks)), 2),
        "fairness_max_rank": round(float(np.nanmax(mean_ranks)), 2),
        "fairness_expected": round(expected, 2),
    }
    out_csv = os.path.join(session_dir, "summary.csv")
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary))
        w.writeheader(); w.writerow(summary)
    print("summary:", summary)


if __name__ == "__main__":
    main()
