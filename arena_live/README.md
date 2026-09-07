# arena_live — deploy trained g99 agents on the live FastBoxEx Arena

Trained in the local phase3 sandbox (`price_discovery/`, checkpoints
`beliefs_pop_240_g99`), deployed here **inference-only** (deterministic policy
head, no learning) against the real FastBoxEx exchange arena.

## What runs

- **15 × PPO-Hjb-MM joiners** (slots 0–14): obs → deterministic κ_bid/κ_ask →
  HJB quote depths → cancel-by-tag + repost each cycle.
- **6 × BeliefTrader joiners** (slots 15–20): obs (incl. belief `v_i = F + b_i`)
  → deterministic β → capped market order.
- **1 × noise bot** (generator slot): Poisson market orders, λ=1/side — the
  background flow, driven from the orchestrator process.
- One arena per scenario (calm / bull / bear = F 1000 / 1003 / 997).

### `biased` scenario — churn

No aggregate news (F = 1000 flat); the only signal is each trader's own bias
`b_i ~ U{-3..3}`. Belief traders **enter and exit like real participants**:
random exponential lifetimes (~60 cycles, 20–150 clamped), a fixed slot pool,
respawn after a short gap with a FRESH bias, all managed by the orchestrator's
master loop (`churn_log.csv` records every spawn/exit with bias and lifetime).
Run: `python orchestrator.py --scenario biased` (or via
`run_scenarios_live.sh 45 biased`).

## Fairness (the one hard requirement)

Each joiner is an independent process looping `sleep(1s + U(0, 0.5s))` → read
book → act. The jitter is re-drawn independently every cycle, so arrival order
at the exchange is a fresh random permutation each round — no agent is
systematically first. Send timestamps are logged per cycle; `analyze_live.py`
checks the arrival-rank distribution per agent.

## Layout

```
Config.toml                  fastboxex profile (host 172.16.22.83)
.env                         FASTBOXEX_USER / FASTBOXEX_PASS (never committed)
config/scenarios.yaml        scenario params (F, seeds, rounds, jitter)
joiners/                     adapters; one process per agent at runtime
  common.py                  config/state/creds, order submission, cycle logger
  obs_mapper.py              live market-data dict → sandbox agent md dict
  mm_joiner.py               g99 PPO-Hjb-MM joiner (slot 0-14)
  belief_joiner.py           g99 BeliefTrader joiner (slot 15-20)
  noise_generator.py         Poisson noise bot (generator slot)
orchestrator.py              arena lifecycle + spawns joiners + noise loop
kill_arenas.py               list/destroy stray arenas (--dry-run default)
mock_test.py                 offline adapter tests vs mock connection
analyze_live.py              mid-vs-F plots, captured%, fairness check
runs/<session>/              per-session outputs (gitignored)
```

## Usage (on QFRCAI, from this directory)

```bash
~/marl-experiments/.venv/bin/python mock_test.py                 # rung 0
~/marl-experiments/.venv/bin/python orchestrator.py --scenario calm --rung 1
~/marl-experiments/.venv/bin/python orchestrator.py --scenario calm
```

## Running + watching live (TUI dashboard)

The exchange ships a terminal UI (`bin/fastboxex-tui`). As the arena owner you
see every participant's portfolio, orders and fills — the professor view.

```bash
# on QFRCAI (typically tmux window 1):
bash run_scenarios_live.sh calm bull bear     # 10s hold (default) before each session
bash run_scenarios_live.sh 45 bull            # explicit 45s hold

# watch live — second terminal:
bash watch_tui.sh                               # starts TUI in tmux session "tui"
tmux attach -t tui                              # detach: Ctrl-b d (TUI keeps running)

# from your LOCAL machine, one command:
bash watch_tui_local.sh
```

`--hold-secs N` keeps the arena in BookOnly for N seconds and prints the
invite code, giving you time to attach the TUI before trading starts. Sessions
run ~240 cycles ≈ 6 min each. After each scenario, `analyze_live.py` writes
`analysis.png` + `summary.csv` into `runs/<scenario>_live/`.

The real agent classes (`PPOHjbMarketMaker`, `BeliefTrader`) are imported from
`~/Trading_agents` and hand-driven: the joiner feeds a live-derived `md` dict,
sets `_last_best_bid/ask`, calls `build_observation` → `sample_action` →
`action_to_orders`, then translates the sandbox `Order` objects into exchange
calls. No obs/depth/cap formulas are duplicated.

Prices: the arena is created with `tick_size=100` raw per display unit,
`reference_price=from_float(1000)`, `lot_size=1`; `obs_mapper` divides raw
prices by 100 so agents see the training scale (mid ≈ 1000). Fills are fed back
to the agents at display scale so their internal cash/inventory accounting
matches training units; the exchange's own endowments are oversized so raw
cash/position limits never bind.
