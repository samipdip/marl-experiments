"""Attribution gate: run ckpt125k in the MarlV2 stop-the-clock env.

If the corrected mechanism (deep-ladder exposure under async actuation) is
right, ckpt125k must show the arena failure signature HERE for the first
time: downward mid drift + elevated jump rate, unlike the phase3 sync
sandbox. Usage:

    python -m async_marlv2.gate <config.yaml> <out_root> [n_episodes] [seed0]
"""

from __future__ import annotations

import sys

import yaml

from .runner import run_eval


def main() -> None:
    cfg_path = sys.argv[1]
    out_root = sys.argv[2]
    n_eps = int(sys.argv[3]) if len(sys.argv) > 3 else 30
    seed0 = int(sys.argv[4]) if len(sys.argv) > 4 else 10000
    with open(cfg_path) as f:
        config = yaml.safe_load(f)
    run_eval(config, n_episodes=n_eps, out_root=out_root, seed0=seed0)
    print(f"gate eval done -> {out_root}")


if __name__ == "__main__":
    main()
