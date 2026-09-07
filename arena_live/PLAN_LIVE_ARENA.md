# PLAN_LIVE_ARENA — deploy g99 agents on the live FastBoxEx Arena

## Status
- [x] Exchange located: `172.16.22.83` (auth 11001, arena 11002/11003, obs 11005, relay 12000 — probed OPEN from QFRCAI). Stale hosts: 202.120.13.117, 172.16.210.57 (= QFRCAI itself).
- [x] Deployment host: QFRCAI (`ssh -J ZhongMa QFRCAI`), torch venv `~/marl-experiments/.venv/bin/python`, g99 finals `~/checkpoints/beliefs_pop_240_g99/*_step_180029_final.pt` (15 MM + 6 belief).
- [ ] Rung 0: mock_test + SDK install + login verify
- [ ] Rung 1: 1 MM + noise, 60 cycles
- [ ] Rung 2: full calm session
- [ ] Rung 3: bull + bear

## Design (approved)
Inference-only deployment of the trained g99 population: 15 MM joiners + 6
BeliefTrader joiners + noise bot (generator slot). One arena per scenario
(calm/bull/bear, F = 1000/1003/997), beliefs v_i = F + b_i. Deterministic
policy heads (`mu_z`), no learning.

**Fairness** (the one hard requirement): each joiner loops
`sleep(1s + U(0, 0.5s))` → read book → act. Independent re-drawn jitter ⇒
fresh random arrival permutation each cycle; no agent systematically first.
Verified post-hoc via logged send timestamps (arrival-rank check in
analyze_live.py). No snapshot alignment, no heartbeat — this is a live test,
not a sandbox reproduction.

## Adapter approach (no formula duplication)
Joiners instantiate the REAL sandbox agents (`PPOHjbMarketMaker`,
`BeliefTrader` imported from `~/Trading_agents`), load the g99 checkpoint
(`{"model_state_dict", ...}` via `agent.load`), and hand-drive them:

1. drain fills → `agent.on_fill` (dict with display-scale price; PositionAgent
   updates inventory/cash exactly as in training)
2. `md = obs_mapper.map_md(market.get_latest(asset_id), 100)` — raw→display
   prices, `volume_buy/sell` → `taker_volume_*` (feeds the MM λ-EMA)
3. set `agent._last_best_bid/ask` (mirrors kyle_base.on_tick)
4. `obs = agent.build_observation(sim_t=cycle+1, md)` → `sample_action(det=True)`
   → `action_to_orders(action, sim_t, md)` (uses `_decision_step` for HJB step)
5. translate sandbox `Order`s → `cancel_by_tag` / `submit_limit` / `submit_market`
   (raw prices = display × 100)

Params mirror the g99 eval configs exactly (MM: obs 9 / action 2, κ∈[0.01,2],
q±20, φ=0.001, α=0.01, hjb_dt 0.5, N=240; trader: linear β∈[0,10], q_max 20,
arrival U{0..60}, sim_duration 60000 ⇒ t_norm≈0).

Arena market structure: `reference_price=from_float(1000)`, `tick_size=100`
raw per display unit, `lot_size=1`, fees 0, breaker 20% — agents see the
training price scale (mid ≈ 1000). Exchange endowments oversized so raw
cash/position limits never bind.

## Gotchas
- `cancel_by_tag(tag, asset_id, side)` needs the side.
- Rust connection binding is `!Send`: one connection per process (joiners are
  subprocesses; noise bot runs on the orchestrator's generator connection).
- Joiners run with CWD = arena_live/ (fastboxex `load_config()` reads
  Config.toml from CWD; `.env` likewise).
- MM joiners must NOT wait for a book (they ARE the book — cold start with
  empty book matches sandbox tick 1). Belief joiners wait (2-min timeout).
- `sim_t = cycle + 1` (nominal, not wall-clock) so t_norm and the HJB
  step index advance exactly as in training.
- Sandbox session = 240 ticks; live session = 240 cycles ≈ 6 min.
- Trainer `_step_fills`/`_action_history` are cleared/skipped to bound memory
  (reward is never computed live).

## Run ladder
| Rung | Command | Pass criteria |
|---|---|---|
| 0 | `python mock_test.py` | all adapter tests pass offline |
| 0b | install SDK wheel into marl venv; `python kill_arenas.py` (dry) | login works, quota visible |
| 1 | `python orchestrator.py --scenario calm --rung 1 --rounds 60` | quotes post/cancel/fill, no orphans, PnL finite |
| 2 | `python orchestrator.py --scenario calm` | mid ≈ 1000, all 22 alive, ranks uniform |
| 3 | `--scenario bull`, `--scenario bear` | mid moves toward F; captured% vs sandbox g99 (bull +0.69 / F 1003) |
| live | `bash run_scenarios_live.sh 45 calm bull bear` + `bash watch_tui.sh` | same ladder, watched live in the TUI |

## Live viewing (TUI)
- `bin/fastboxex-tui -c Config.toml` — owner view: all participants' portfolios/orders/fills.
- `watch_tui.sh` runs it in tmux session `tui` (attach/detach freely);
  `watch_tui_local.sh` attaches from the laptop via `ssh -J ZhongMa QFRCAI`.
- `orchestrator.py --hold-secs N` holds BookOnly N s (prints invite code) so
  the TUI can be attached before trading; `t_start` shifts by the hold.

Cleanup is mandatory: joiners `cancel_all` in `finally`; orchestrator destroys
the arena in `finally`; `kill_arenas.py --yes` sweeps strays.

## Later rungs (deferred)
- Dynamic F (step/poisson): belief joiners recompute F_t from the session
  clock + shared params — no coordinator process needed.
- Free-running (non-aligned) cadence as a realism comparison.
- Live retraining is out of scope (wall-clock pacing vs PPO sample efficiency).
