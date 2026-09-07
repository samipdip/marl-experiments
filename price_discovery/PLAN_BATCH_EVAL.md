# Plan: Speed Up Eval via Batch Processing

## Current State

The pipeline ran in ~48 min total (05:18–06:06 on 2026-07-28). Per-experiment breakdown:

| Step | Time | Notes |
|------|------|-------|
| Training | 2–5 min | 13 experiments, sequential |
| Eval | ~1 min | 30 episodes × ~2s each (30 Python process spawns) |
| Plots | ~30s | 3 plot scripts |

Eval is ~33 min of the 48 min total. Each of the 990 episode runs (33 experiments × 30 episodes) spawns a fresh Python process that:

1. Imports torch/numpy/SimExchange/Trading_agents (~0.7s)
2. Parses YAML config (~0.1s)
3. Constructs 5 MMs + 1 IT + noise trader (~0.2s — includes solving HJB PDE 5 times)
4. Loads 6 checkpoint files from disk (~0.2s)
5. Runs 80 ticks of simulation (~0.6s — the only useful work)
6. Dumps CSVs and exits

Steps 1–4 and 6 are pure overhead, repeated 990 times. That's ~990 × 1.1s = ~18 min of wasted compute.

## Proposed Fix: `batch_eval.py`

Write a single script that processes all episodes for one experiment **in one Python process**.

### What it does

```
For each experiment:
  1. Import all modules (once)
  2. Parse config (once)
  3. Build market + agents (once) → loads checkpoints, solves HJB PDE
  4. For ep in 0..N-1:
     a. _seed_everything(10000 + ep)
     b. market.reset(); agent.reset() for all agents
     c. Run 80 ticks: market.tick() → agent.on_tick() → shuffler.flush() → agent.on_shuffle_done()
     d. Dump pnl_history.csv + order_events.csv to ep_N/
  5. Exit
```

### Expected speedup

| Metric | Current | Batched |
|--------|---------|---------|
| Process spawns | 990 | 33 |
| Import overhead | 990 × 0.7s = 11.5 min | 33 × 0.7s = 0.4 min |
| Checkpoint loads | 990 × 6 files | 33 × 6 files |
| HJB PDE solves | 990 × 5 = 4950 | 33 × 5 = 165 |
| **Eval total (30 ep)** | **~33 min** | **~12 min** |
| **Eval total (100 ep)** | **~110 min** | **~30 min** |

## Files to Create/Modify

### 1. New: `price_discovery/scripts/batch_eval.py`

```python
#!/usr/bin/env python3
"""Batch eval: all episodes for one experiment in a single process.

Usage: batch_eval.py <config.yaml> <results_dir> [n_episodes] [n_steps]
"""
import sys, os, json
import numpy as np, torch

sys.path.insert(0, os.path.expanduser("~/SimExchangeSandbox"))
sys.path.insert(0, os.path.expanduser("~/Trading_agents"))

from phase3_multi_agent.src.factory import load_config, build_env
from phase3_multi_agent.src.runner import _seed_everything, dump_order_events, dump_pnl_history


def batch_eval(config_path, results_dir, n_episodes=30, n_steps=80, base_seed=10000):
    config = load_config(config_path)
    config["mode"] = "eval"
    os.makedirs(results_dir, exist_ok=True)

    # Build everything ONCE: market, agents, checkpoints, HJB PDE
    env_build = build_env(config, device="cpu", results_dir=results_dir)
    market = env_build.env.market
    agents = env_build.all_agents
    shuffler = env_build.shuffler

    for ep in range(n_episodes):
        seed = base_seed + ep
        _seed_everything(seed)
        market.reset()
        for a in agents:
            a.reset()
            if hasattr(a, "on_episode_start"):
                a.on_episode_start(market.get_md(1))

        # Run n_steps ticks (identical to SimulationRunner.run loop)
        for _ in range(n_steps):
            market.tick()
            sim_t = market.sim_t
            for a in agents:
                a.on_tick(sim_t)
            shuffler.flush()
            for a in agents:
                if hasattr(a, "on_shuffle_done"):
                    a.on_shuffle_done()

        # Dump per-episode results (same format as current)
        ep_dir = os.path.join(results_dir, f"ep_{ep}")
        os.makedirs(ep_dir, exist_ok=True)
        dump_pnl_history(agents, ep_dir)
        dump_order_events(agents, ep_dir)

    print(f"  {n_episodes} episodes done -> {results_dir}")


if __name__ == "__main__":
    config_path = sys.argv[1]
    results_dir = sys.argv[2]
    n_episodes = int(sys.argv[3]) if len(sys.argv) > 3 else 30
    n_steps = int(sys.argv[4]) if len(sys.argv) > 4 else 80
    batch_eval(config_path, results_dir, n_episodes, n_steps)
```

### 2. Modify: `price_discovery/scripts/run_all_sequential.sh`

Replace the `run_eval()` function — remove the 30-iteration `for ep` loop, call `batch_eval.py` once:

```bash
# OLD (30 Python process spawns per experiment):
run_eval() {
  local CFG_NAME=$1
  local OUTDIR=$2
  echo "=== EVAL ${CFG_NAME} ==="
  rm -rf "$OUTDIR"; mkdir -p "$OUTDIR"
  for ep in $(seq 0 29); do
    seed=$((10000 + ep))
    epdir="${OUTDIR}/ep_${ep}"
    mkdir -p "$epdir"
    PYTHONUNBUFFERED=1 $VENV -c "
import sys, os
sys.path.insert(0, os.path.expanduser('~/SimExchangeSandbox'))
sys.path.insert(0, os.path.expanduser('~/Trading_agents'))
from phase3_multi_agent.src.runner import run_simulation
run_simulation('${CFG}/${CFG_NAME}', device='cpu', seed=${seed},
               results_dir='${epdir}', episode_ticks=0)
" 2>&1 | tail -1
  done
  echo "  DONE: ${CFG_NAME}"
}

# NEW (1 Python process per experiment):
run_eval() {
  local CFG_NAME=$1
  local OUTDIR=$2
  echo "=== EVAL ${CFG_NAME} ==="
  rm -rf "$OUTDIR"; mkdir -p "$OUTDIR"
  PYTHONUNBUFFERED=1 $VENV scripts/batch_eval.py \
    "${CFG}/${CFG_NAME}" "$OUTDIR" ${N_EPISODES:-30} 80
  echo "  DONE: ${CFG_NAME}"
}
```

No other changes to `run_all_sequential.sh` — the main loop calling `run_eval` stays the same.

## Validation

Before adopting, verify batch eval produces identical results to the current per-episode approach:

```bash
# Run batch eval on one experiment with same seeds
$VENV scripts/batch_eval.py configs/eval_v1003_ppo.yaml /tmp/batch_test 30 80

# Compare episode 0 and episode 15 results
diff /tmp/batch_test/ep_0/pnl_history.csv  results_v1003_ppo_reset/ep_0/pnl_history.csv
diff /tmp/batch_test/ep_15/pnl_history.csv results_v1003_ppo_reset/ep_15/pnl_history.csv
```

If diffs exist, investigate `agent.reset()` — ensure `_inventory`, `_cash`, `_step_fills`, `_pnl_history`, `_order_events` are fully cleared between episodes. The training loop already calls `market.reset()` + `agent.reset()` every 80 ticks, so this mechanism is proven.

## Key Design Decisions

- **Same seed scheme**: `seed = 10000 + ep` for all experiments. Paired comparison preserved.
- **Configurable episode count**: `N_EPISODES` shell variable (default 30). Easy to bump to 50–100.
- **Same output format**: identical directory structure (`ep_0/pnl_history.csv`, `ep_0/order_events.csv`) — no plot script changes needed.
- **No parallelism**: experiments still run sequentially. Can be added later if needed (eval is lightweight — no gradients — so 2–4 concurrent experiments would be safe).

## Future Considerations (Not In Scope)

- **Parallel eval across experiments**: eval uses minimal CPU (no gradient computation). Could run 2–4 experiments concurrently with `&` + `wait`. Would cut eval from ~12 min to ~3–6 min.
- **Training speedup**: training is already fast (~2–5 min per experiment). Not worth optimizing.
- **Increasing episode count**: with batched eval, 100 episodes takes ~30 min instead of ~110 min. Worth doing for tighter confidence intervals on overshoot/kappa estimates.
