"""train.py -- launch continuous training in the MarlV2Env.

Usage:
    python -m async_marlv2.train <config.yaml> <out_root>

Seed comes from the config's top-level `seed` (phase3 parity: 42/43/44).
"""

from __future__ import annotations

import os
import sys

import yaml

from .runner import run_train


def main() -> None:
    cfg_path, out_root = sys.argv[1], sys.argv[2]
    with open(cfg_path) as f:
        config = yaml.safe_load(f)
    seed = int(config.get("seed", 42))
    os.makedirs(out_root, exist_ok=True)
    run_train(config, results_dir=out_root, seed=seed)
    print(f"train done -> {out_root}")


if __name__ == "__main__":
    main()
