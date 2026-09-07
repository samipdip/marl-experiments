"""Statistical validation: local (sandbox) vs arena-live inference transfer.

Question: are live arena sessions statistically consistent with sandbox eval
episodes, given the same policies (ckpt125k) and matched generative settings?

Matched settings (verified): 7 PPO-HJB MM + 3 BeliefTrader, noise Poisson(0.47)
per side, 240 cycles, q_max 20, belief biases iid U{-3..3} (sandbox:
BeliefCoordinator._init_episode; arena: val2_* scenarios, bias_mode random,
fresh --seed per session), arrivals U{0..60}. Remaining structural difference
(what is being tested): sandbox ticks are atomic with shuffled order flush;
arena cycles are 10 async processes on jittered 0.5s clocks.

Comparison: 30 sandbox episodes (ep_0..ep_29, seeds 10000..10029) vs 30 arena
sessions (runs/val2_<scen>_s<K>, seeds 101..130) per scenario.

Metric definitions (computed IDENTICALLY on both sides):
  mid            (best_bid+best_ask)/2 from the book snapshot; forward-filled
                 on cycles where the book is not two-sided.
  end_mid        mean(mid, last 50 ticks)          mid_move  mean(last 50) - mean(first 20)
  mid_std        std(mid, all 240 ticks)           min/max_mid  extremes of mid
  spread_mean    mean(ask - bid) over two-sided snapshots only (never ff-filled)
  fill_events    sum of fill events logged by OUR 10 agents. Every trade has
                 exactly one MM maker side; belief takers log their own fill
                 too, so MM-belief trades count twice - identically on both
                 sides (sandbox order_events.csv rows; arena n_fills columns).
  flow_units     sum over our 10 agents of |inventory delta| between
                 consecutive ticks/cycles, in units. Noise-MM trades contribute
                 their qty once (MM side), belief-MM trades twice - identical
                 convention both sides.
  best_qty       mean over two-sided snapshots of (best bid qty + best ask qty)/2
  depth_within_k mean cumulative qty within k display ticks of mid, both sides
  n_levels       mean number of visible levels (max 5 per side)
  imbalance      mean (sum bid qty - sum ask qty)/(sum bid qty + sum ask qty)
Sandbox snapshots: checkpoints/ppo_hjb_mm_0_lob_data.csv lob_json (top 5
levels used). Arena snapshots: md_log.csv (orchestrator cycle boundary, top 5
levels per side). Both are one book snapshot per tick/cycle, display units.

Statistics per metric: percentile ranks of each arena value inside the
sandbox sample (midrank, tie-safe), Mann-Whitney U (normal approx with tie +
continuity correction), Cliff's delta.

Usage (QFRCAI): .venv/bin/python validate_transfer.py
Writes validation2/validation_report.csv + validation2/validation_<scen>.png
"""

from __future__ import annotations

import csv
import glob
import json
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
# Defaults = the 30v30 val2 study (HANDOFF 5g). CLI (optional, in order):
#   python validate_transfer.py <arena_glob_prefix> <sb_root_fmt> <out_dir>
# e.g. for val4 vs the MarlV2 ft@15000 in-env reference:
#   python validate_transfer.py val4 \
#     '~/marl-experiments/async_marlv2/results_ft15000_ref_{scen}' \
#     validation4
_args = sys.argv[1:]
ARENA_PREFIX = _args[0] if len(_args) > 0 else "val2"
ARENA_RUNS = os.path.join(HERE, "runs")
SB_ROOT = (_args[1] if len(_args) > 1 else os.path.expanduser(
    "~/marl-experiments/price_discovery/"
    "results_matched_7x3_g99_ckpt125000_{scen}"))
if not SB_ROOT.startswith("/"):
    SB_ROOT = os.path.expanduser(SB_ROOT)
OUT_DIR = os.path.join(HERE, _args[2] if len(_args) > 2 else "validation2")
SCENARIOS = {"calm": 1000.0, "bull": 1003.0, "bear": 997.0}
N_TICKS, HEAD, TAIL, N_EPS = 240, 20, 50, 30
DEPTH_TICKS = (1, 2, 5)
BLUE, RED, GRAY = "#1f77b4", "#d62728", "#7f7f7f"


# ── statistics (numpy-only; remote venvs have no scipy) ───────────────────

def _norm_sf(z):
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def mann_whitney(x, y):
    """U for x vs y; normal approx, tie + continuity correction."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    n1, n2 = len(x), len(y)
    allv = np.concatenate([x, y])
    order = allv.argsort(kind="mergesort")
    ranks = np.empty(len(allv))
    ranks[order] = np.arange(1, len(allv) + 1)
    # average ranks within tie groups; tie correction for sigma
    tie_sum = 0.0
    for v in np.unique(allv):
        c = int((allv == v).sum())
        if c > 1:
            ranks[allv == v] = ranks[allv == v].mean()
            tie_sum += c ** 3 - c
    r1 = ranks[:n1].sum()
    u1 = r1 - n1 * (n1 + 1) / 2.0
    mu = n1 * n2 / 2.0
    n = n1 + n2
    sigma = math.sqrt(n1 * n2 / 12.0 * ((n + 1) - tie_sum / (n * (n - 1))))
    if sigma <= 0:
        return float(u1), 1.0
    z = (u1 - mu) / sigma
    z = math.copysign(max(abs(z) - 0.5 / sigma, 0.0), z)  # continuity corr.
    return float(u1), min(max(2.0 * _norm_sf(abs(z)), 0.0), 1.0)


def cliffs_delta(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    gt = sum(float((xi > y).sum()) for xi in x)
    lt = sum(float((xi < y).sum()) for xi in x)
    return (gt - lt) / (len(x) * len(y))


def pct_ranks(x, ref):
    """Midrank percentile of each x inside ref (tie-safe)."""
    ref = np.asarray(ref, float)
    out = []
    for xi in x:
        lt = float((ref < xi).sum())
        eq = float((ref == xi).sum())
        out.append((lt + 0.5 * eq) / len(ref) * 100.0)
    return np.array(out)


# ── loaders ───────────────────────────────────────────────────────────────

def _depth_stats(bids, asks):
    """Depth metrics for one two-sided snapshot (display units, qty units)."""
    mid = (bids[0][0] + asks[0][0]) / 2.0
    out = {"best_qty": (bids[0][1] + asks[0][1]) / 2.0,
           "n_levels": len(bids) + len(asks),
           "imbalance": (sum(q for _, q in bids) - sum(q for _, q in asks))
                    / max(sum(q for _, q in bids) + sum(q for _, q in asks), 1e-9)}
    for k in DEPTH_TICKS:
        out[f"depth_within_{k}"] = (
            sum(q for p, q in bids if mid - p <= k) +
            sum(q for p, q in asks if p - mid <= k))
    return out


def _ffill_mid(mid_raw):
    """Forward-fill interior gaps, back-fill leading NaNs (book not yet
    two-sided at t=0 on BOTH sides: sandbox lob step 0 and arena cycle 0)."""
    mid = mid_raw.copy()
    for i in range(1, len(mid)):
        if np.isnan(mid[i]):
            mid[i] = mid[i - 1]
    if np.isnan(mid[0]):
        first = np.argmax(~np.isnan(mid))
        if first == 0 and np.isnan(mid[0]):
            return None
        mid[:first] = mid[first]
    return mid


def _series_metrics(mid, spreads, depths, fill_events, flow_units):
    m = mid[~np.isnan(mid)]
    met = {
        "end_mid": float(np.mean(m[-TAIL:])),
        "mid_move": float(np.mean(m[-TAIL:]) - np.mean(m[:HEAD])),
        "mid_std": float(np.std(m)),
        "min_mid": float(np.min(m)), "max_mid": float(np.max(m)),
        "spread_mean": float(np.mean(spreads)) if len(spreads) else float("nan"),
        "fill_events": float(fill_events),
        "flow_units": float(flow_units),
    }
    if depths:
        for k in list(depths[0]):
            met[f"mean_{k}"] = float(np.mean([d[k] for d in depths]))
    else:
        for k in ("best_qty", "n_levels", "imbalance",
                  *(f"depth_within_{j}" for j in DEPTH_TICKS)):
            met[f"mean_{k}"] = float("nan")
    return met


def load_sandbox_ep(ep_dir):
    lob_path = os.path.join(ep_dir, "checkpoints", "ppo_hjb_mm_0_lob_data.csv")
    if not os.path.exists(lob_path):
        return None
    mid_raw = np.full(N_TICKS, np.nan)
    spreads, depths = [], []
    with open(lob_path) as fh:
        for r in csv.DictReader(fh):
            t = int(float(r["step"]))
            if not (0 <= t < N_TICKS):
                continue
            lob = json.loads(r["lob_json"])
            bids = [(float(p), float(q)) for p, q in lob.get("bids", [])][:5]
            asks = [(float(p), float(q)) for p, q in lob.get("asks", [])][:5]
            if bids and asks:
                mid_raw[t] = (bids[0][0] + asks[0][0]) / 2.0
                spreads.append(asks[0][0] - bids[0][0])
                depths.append(_depth_stats(bids, asks))
    # inventory series per agent -> flow_units; order_events -> fill_events
    inv = {}
    with open(os.path.join(ep_dir, "pnl_history.csv")) as fh:
        for r in csv.DictReader(fh):
            a = r["agent"]
            if a.startswith(("ppo_hjb_mm_", "belief_")):
                inv.setdefault(a, []).append((float(r["sim_t"]),
                                              float(r["inventory"])))
    flow = sum(float(np.abs(np.diff([q for _, q in sorted(v)])).sum())
               for v in inv.values())
    fills = sum(1 for _ in open(os.path.join(ep_dir, "order_events.csv"))) - 1
    mid = _ffill_mid(mid_raw)
    if mid is None:
        return None
    return _series_metrics(mid, spreads, depths, fills, flow)


def load_arena_session(sdir):
    md_path = os.path.join(sdir, "md_log.csv")
    if not os.path.exists(md_path):
        return None
    rows = list(csv.DictReader(open(md_path)))
    if len(rows) < N_TICKS:
        return None  # in-progress session
    mid_raw = np.full(N_TICKS, np.nan)
    spreads, depths = [], []
    for r in rows:
        t = int(float(r["cycle"]))
        if not (0 <= t < N_TICKS) or int(r["md_ok"]) != 1:
            continue
        bid, ask = float(r["bid"]), float(r["ask"])
        if bid > 0 and ask > 0:
            mid_raw[t] = (bid + ask) / 2.0
            spreads.append(ask - bid)
            lob = json.loads(r["lob_json"])
            bids = [(float(p), float(q)) for p, q in lob.get("bids", [])][:5]
            asks = [(float(p), float(q)) for p, q in lob.get("asks", [])][:5]
            if bids and asks:
                depths.append(_depth_stats(bids, asks))
    fills, inv = 0.0, {}
    for sub in os.listdir(sdir):
        sdir2 = os.path.join(sdir, sub)
        if not os.path.isdir(sdir2) or not (sub.startswith(("mm_", "belief_"))):
            continue
        logs = [f for f in os.listdir(sdir2)
                if f.endswith("_cycle_log.csv")
                and not f.endswith("_ack_cycle_log.csv")]
        if not logs:
            continue
        arows = list(csv.DictReader(open(os.path.join(sdir2, logs[0]))))
        if not arows:
            continue
        fills += sum(int(float(r.get("n_fills") or 0)) for r in arows)
        if len(arows) >= N_TICKS:
            qs = [float(r["inventory"]) for r in arows[:N_TICKS]]
            inv[sub] = abs(np.diff(qs)).sum()
    mid = _ffill_mid(mid_raw)
    if mid is None:
        return None
    return _series_metrics(mid, spreads, depths, fills, sum(inv.values()))


# ── reporting ─────────────────────────────────────────────────────────────

HIST_METRICS = ["end_mid", "mid_move", "mid_std", "spread_mean",
                "fill_events", "flow_units", "mean_best_qty",
                "mean_depth_within_2", "mean_imbalance"]


def _hist_panel(ax, sb, ar, k):
    ax.hist(sb, bins=12, color=BLUE, alpha=0.6, density=True,
            label=f"sandbox n={len(sb)}")
    for xi in ar:
        ax.axvline(xi, color=RED, alpha=0.6, linewidth=1.0)
    ax.axvline(float(np.mean(ar)), color=RED, linewidth=2.0,
               label=f"arena n={len(ar)} (mean)")
    _, p = mann_whitney(ar, sb)
    cd = cliffs_delta(ar, sb)
    ax.set_title(f"{k}   MW p={p:.3f}  Cliff's delta={cd:+.2f}", fontsize=9)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    report = []
    for scen, F in SCENARIOS.items():
        sb_dirs = [os.path.join(SB_ROOT.format(scen=scen), f"ep_{i}")
                   for i in range(N_EPS)]
        sb = [m for d in sb_dirs if (m := load_sandbox_ep(d))]
        ar_dirs = sorted(glob.glob(os.path.join(
            ARENA_RUNS, f"{ARENA_PREFIX}_{scen}_s*")),
                         key=lambda p: int(p.rsplit("_s", 1)[1]))
        ar = [m for d in ar_dirs if (m := load_arena_session(d))]
        print(f"[{scen}] sandbox eps={len(sb)}  arena sessions={len(ar)}")
        if len(sb) < N_EPS or not ar:
            print(f"[{scen}] incomplete - skipping report for now")
            continue

        keys = [k for k in sb[0] if k in ar[0]]
        sbv = {k: np.array([m[k] for m in sb]) for k in keys}
        arv = {k: np.array([m[k] for m in ar]) for k in keys}
        for k in keys:
            x, y = arv[k][~np.isnan(arv[k])], sbv[k][~np.isnan(sbv[k])]
            u, p = mann_whitney(x, y)
            cd = cliffs_delta(x, y)
            pr = pct_ranks(x, y)
            report.append({
                "scenario": scen, "metric": k,
                "n_arena": len(x), "n_sandbox": len(y),
                "arena_mean": round(float(x.mean()), 4),
                "arena_std": round(float(x.std()), 4),
                "sandbox_mean": round(float(y.mean()), 4),
                "sandbox_std": round(float(y.std()), 4),
                "mw_p": round(p, 4), "cliffs_delta": round(cd, 4),
                "arena_pct_min": round(float(pr.min()), 1),
                "arena_pct_median": round(float(np.median(pr)), 1),
                "arena_pct_max": round(float(pr.max()), 1),
            })

        # mid-path band from the same snapshot source used for the metrics
        def _mid_series_sandbox(d):
            mid = np.full(N_TICKS, np.nan)
            for r in csv.DictReader(open(os.path.join(
                    d, "checkpoints", "ppo_hjb_mm_0_lob_data.csv"))):
                lob = json.loads(r["lob_json"])
                b, a = lob.get("bids", []), lob.get("asks", [])
                if b and a:
                    t = int(float(r["step"]))
                    if 0 <= t < N_TICKS:
                        mid[t] = (b[0][0] + a[0][0]) / 2.0
            return _ffill_mid(mid)

        def _mid_series_arena(d):
            mid = np.full(N_TICKS, np.nan)
            for r in csv.DictReader(open(os.path.join(d, "md_log.csv"))):
                if int(r["md_ok"]) == 1 and float(r["bid"]) > 0 \
                        and float(r["ask"]) > 0:
                    t = int(float(r["cycle"]))
                    if 0 <= t < N_TICKS:
                        mid[t] = (float(r["bid"]) + float(r["ask"])) / 2.0
            return _ffill_mid(mid)

        sb_paths = np.array([_mid_series_sandbox(d) for d in sb_dirs
                             if _mid_series_sandbox(d) is not None])
        ar_paths = [(m, os.path.basename(d)) for d in ar_dirs
                    if (m := _mid_series_arena(d)) is not None]
        t = np.arange(N_TICKS)

        fig, axes = plt.subplots(3, 3, figsize=(14, 12))
        ax = axes[0][0]
        ax.fill_between(t, np.percentile(sb_paths, 5, axis=0),
                        np.percentile(sb_paths, 95, axis=0),
                        color=BLUE, alpha=0.25, label="sandbox p5-p95")
        ax.plot(t, np.median(sb_paths, axis=0), color=BLUE,
                linewidth=1.8, label="sandbox median")
        for m, name in ar_paths:
            ax.plot(t, m, color=RED, linewidth=0.6, alpha=0.55)
        ax.axhline(F, color=GRAY, linestyle="--", linewidth=1.5, label=f"F={F:.0f}")
        ax.set_title(f"{scen}: mid paths - arena (red) vs sandbox band", fontsize=10)
        ax.set_xlabel("tick"); ax.set_ylabel("mid")
        ax.grid(alpha=0.3); ax.legend(fontsize=7)

        for ax, k in zip(axes.flat[1:], HIST_METRICS):
            _hist_panel(ax, sbv[k], arv[k], k)
        fig.suptitle(f"{scen}: arena vs sandbox - 30 v 30 "
                     f"(ckpt125k, iid U{{-3..3}} biases both sides)", fontsize=12)
        plt.tight_layout()
        out_png = os.path.join(OUT_DIR, f"validation_{scen}.png")
        plt.savefig(out_png, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {out_png}")

    if report:
        cols = ["scenario", "metric", "n_arena", "n_sandbox", "arena_mean",
                "arena_std", "sandbox_mean", "sandbox_std", "mw_p",
                "cliffs_delta", "arena_pct_min", "arena_pct_median",
                "arena_pct_max"]
        out_csv = os.path.join(OUT_DIR, "validation_report.csv")
        with open(out_csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            for row in report:
                w.writerow(row)
        print(f"saved {out_csv}")


if __name__ == "__main__":
    main()
