"""analyze_gate.py -- gate verdict metrics for MarlV2Env eval runs.

Computes, per episode, from ppo_hjb_mm_0's pnl_history rows (its own 1 s
firing grid, matching the sandbox reference layout):
  end_mid, mid_move = end_mid - 1000, mid_std,
  jump_frac  = fraction of consecutive mid changes with |delta| >= 5,
  mean spread, fill count (order_events rows).

Usage:
    python -m async_marlv2.analyze_gate <ep_root> [<ep_root> ...]

Each <ep_root> holds ep_*/ subdirs with pnl_history.csv (+order_events.csv).
"""

from __future__ import annotations

import csv
import glob
import os
import sys

import numpy as np


def load_eps(root: str):
    eps = sorted(glob.glob(os.path.join(root, "ep_*")),
                 key=lambda d: int(d.rsplit("_", 1)[-1]))
    if not eps:
        raise SystemExit(f"no ep_* dirs under {root}")
    rows_out = []
    for ep in eps:
        rows = [r for r in csv.DictReader(open(os.path.join(ep, "pnl_history.csv")))
                if r["agent"] == "ppo_hjb_mm_0"]
        mids = [float(r["mid"]) for r in rows if r["mid"] not in ("", "0.0")]
        if not mids:  # empty-book fallback: keep the 0.0 series verbatim
            mids = [float(r["mid"]) for r in rows if r["mid"]]
        spreads = []
        for r in rows:
            b, a = r["bid"], r["ask"]
            if b not in ("", "0.0", None) and a not in ("", "0.0", None):
                spreads.append(float(a) - float(b))
        n_fills = 0
        ev_path = os.path.join(ep, "order_events.csv")
        if os.path.exists(ev_path):
            n_fills = sum(1 for e in csv.DictReader(open(ev_path))
                          if e["event"] == "filled")
        mids_a = np.asarray(mids)
        jumps = np.abs(np.diff(mids_a)) >= 5.0 if len(mids_a) > 1 else np.array([])
        rows_out.append({
            "ep": os.path.basename(ep),
            "n_obs": len(mids),
            "end_mid": mids_a[-1] if len(mids_a) else float("nan"),
            "mid_move": (mids_a[-1] - 1000.0) if len(mids_a) else float("nan"),
            "mid_std": float(np.std(mids_a)) if len(mids_a) else float("nan"),
            "jump_frac": float(jumps.mean()) if len(jumps) else float("nan"),
            "mean_spread": float(np.mean(spreads)) if spreads else float("nan"),
            "fills": n_fills,
        })
    return rows_out


def summarize(root: str) -> dict:
    eps = load_eps(root)
    moves = np.array([e["mid_move"] for e in eps])
    ends = np.array([e["end_mid"] for e in eps])
    stds = np.array([e["mid_std"] for e in eps])
    jumps = np.array([e["jump_frac"] for e in eps])
    fills = np.array([e["fills"] for e in eps], dtype=float)
    return {
        "root": root, "n_eps": len(eps),
        "frac_down": float((moves < 0).mean()),
        "end_mid_mean": float(ends.mean()), "end_mid_min": float(ends.min()),
        "end_mid_max": float(ends.max()),
        "move_mean": float(moves.mean()), "move_median": float(np.median(moves)),
        "move_min": float(moves.min()),
        "mid_std_mean": float(stds.mean()),
        "jump_frac_mean": float(jumps.mean()),
        "fills_mean": float(fills.mean()),
        "eps": eps,
    }


def main() -> None:
    print(f"{'ep':>6} {'n':>4} {'end_mid':>8} {'move':>7} {'std':>6} "
          f"{'jump%':>6} {'sprd':>5} {'fills':>6}")
    for root in sys.argv[1:]:
        s = summarize(root)
        print(f"\n== {s['root']}  ({s['n_eps']} eps)")
        for e in s["eps"]:
            print(f"{e['ep']:>6} {e['n_obs']:>4} {e['end_mid']:>8.1f} "
                  f"{e['mid_move']:>7.1f} {e['mid_std']:>6.2f} "
                  f"{100*e['jump_frac']:>6.2f} {e['mean_spread']:>5.2f} "
                  f"{e['fills']:>6}")
        print(f"SUMMARY frac_down={s['frac_down']:.2f} "
              f"end_mid[{s['end_mid_min']:.1f},{s['end_mid_max']:.1f}] "
              f"mean={s['end_mid_mean']:.1f} move_mean={s['move_mean']:.1f} "
              f"move_med={s['move_median']:.1f} move_min={s['move_min']:.1f} "
              f"mid_std={s['mid_std_mean']:.2f} jump%={100*s['jump_frac_mean']:.2f} "
              f"fills={s['fills_mean']:.0f}")


if __name__ == "__main__":
    main()
