"""score_ckpts.py -- sweep a training run's checkpoints through eval configs.

For each requested ckpt step, rewrites every agent's `checkpoint` path in an
eval template to <ckpt_root>/<agent_dir>/<agent_name>_step_<step>.pt, runs
n_episodes eval episodes, and prints analyze_gate summary metrics (§5e-safe
recipe: score every 5k ckpt, attractors form late -- never trust finals).

Usage:
    python -m async_marlv2.score_ckpts <template.yaml> <ckpt_root> \
        <steps: 5000,10000,...> <n_eps> [seed0]

Skips steps where any of the 10 trainable agents lacks a checkpoint file.
"""

from __future__ import annotations

import glob
import os
import re
import shutil
import sys

import yaml

from .analyze_gate import summarize
from .runner import run_eval


def rewrite_ckpt(path: str, ckpt_root: str, step: int) -> str | None:
    agent_dir = path.rstrip("/").split("/")[-2]          # ppo_0 / belief_1
    base = os.path.basename(path).split("_step_")[0]      # ppo_hjb_mm_0
    cand = os.path.join(ckpt_root, agent_dir, f"{base}_step_{step}.pt")
    if os.path.exists(cand):
        return cand
    hits = sorted(glob.glob(os.path.join(
        ckpt_root, agent_dir, f"{base}_step_{step}*.pt")))
    return hits[0] if hits else None


def main() -> None:
    template, ckpt_root = sys.argv[1], sys.argv[2]
    steps = [int(s) for s in sys.argv[3].split(",")]
    n_eps = int(sys.argv[4]) if len(sys.argv) > 4 else 10
    seed0 = int(sys.argv[5]) if len(sys.argv) > 5 else 10000
    scratch = "/tmp/marlv2_score_scratch"

    print(f"template={template} root={ckpt_root} eps={n_eps} seed0={seed0}")
    print(f"{'step':>7} {'move_mean':>9} {'move_min':>8} {'frac_down':>9} "
          f"{'end_mean':>8} {'mid_std':>7} {'jump%':>6} {'fills':>6} {'ok':>3}")
    for step in steps:
        cfg = yaml.safe_load(open(template))
        missing = []
        for a in cfg["agents"]:
            if "checkpoint" not in a:
                continue
            new = rewrite_ckpt(a["checkpoint"], ckpt_root, step)
            if new is None:
                missing.append(a["id"])
            else:
                a["checkpoint"] = new
        if missing:
            print(f"{step:>7}  SKIP missing ckpts: {','.join(missing)}")
            continue
        if os.path.isdir(scratch):
            shutil.rmtree(scratch)
        run_eval(cfg, n_episodes=n_eps, out_root=scratch, seed0=seed0)
        s = summarize(scratch)
        print(f"{step:>7} {s['move_mean']:>9.2f} {s['move_min']:>8.1f} "
              f"{s['frac_down']:>9.2f} {s['end_mid_mean']:>8.1f} "
              f"{s['mid_std_mean']:>7.2f} {100*s['jump_frac_mean']:>6.2f} "
              f"{s['fills_mean']:>6.0f} {'yes':>3}", flush=True)


if __name__ == "__main__":
    main()
