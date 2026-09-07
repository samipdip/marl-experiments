# Belief-Trader Experiment (Set 3) — Recovery & Reproduction Guide

> Single-file context dump. An LLM resuming this work should be able to reconstruct
> the full experiment state from this document alone. Read top-to-bottom before acting.

---

## 0. TL;DR — what this is

A new agent type **`BeliefTrader`** (a prop-desk-style informed trader with its own
*estimate* of fair value, not the truth) was built and run against 15 PPO HJB market
makers, 240-tick reset episodes, on the **ZhongMa** server. The experiment mirrors the
existing `set2R_2way_ppo_240_uni` family but replaces the single Kyle IT with **2
BeliefTraders**. Beliefs are two-sided (buy or sell), self-sampled per episode from
`U[994, 1006]` (±6 ticks around ref 1000) in training; evals fix beliefs to probe
directional aggregation.

**Current state:** train + 4 evals done (random / fixed 1003·997 / bull 1003·1005 /
flat 1000·1006). Plots pulled locally. Reward = **mark-to-belief** (inherited from
`KyleInformedTrader`) — an earlier mark-to-mid attempt was abandoned (caused abstention).

---

## 1. Repos, paths, servers

| Thing | Local | Remote (ZhongMa) |
|---|---|---|
| Configs/scripts/plots | `~/Documents/marl-experiments/price_discovery/` | `~/marl-experiments/price_discovery/` |
| Agent code | `~/Documents/Trading_agents/Agents_phase2/RL_Agents/ABIDES_MARL/` | `~/Trading_agents/Agents_phase2/RL_Agents/ABIDES_MARL/` |
| Phase3 framework (**DO NOT MODIFY**) | `~/Documents/SimExchangeSandbox/phase3_multi_agent/` | `~/SimExchangeSandbox/phase3_multi_agent/` |
| Checkpoints | — | `~/checkpoints/` |
| Python venv | `~/Documents/marl-experiments/.venv/bin/python` | `~/marl-experiments/.venv/bin/python` |

**SSH hosts** (`~/.ssh/config`):
- `ZhongMa` — tailscale `100.117.249.23`, User `samip`, 32 cores, 61 GB RAM. **Use this one.**
- `ZhongMa_Local` — `172.16.210.71` (LAN; currently unreachable/times out).
- (Other servers exist — `QFRCAI`, `QFRCAI_Local` — not used for this experiment.)

**Python sys.path on remote** (required for any run):
```python
import sys, os
sys.path.insert(0, os.path.expanduser('~/SimExchangeSandbox'))
sys.path.insert(0, os.path.expanduser('~/Trading_agents'))
```

---

## 2. Framework rules (HARD constraints — from `EXPERIMENT_GUIDE.md`)

1. **NEVER modify** anything under `phase3_multi_agent/` (`factory.py`, `runner.py`,
   `env.py`, `market.py`) nor `kyle_base.py`. All experiment work = config YAMLs +
   agent code under `ABIDES_MARL/`.
2. Always use the venv `~/marl-experiments/.venv/bin/python`.
3. Eval = one fresh `run_simulation` call per episode, seeds `10000+ep`, `episode_ticks=0`.

---

## 3. The agent code (what was built/changed)

All in `Trading_agents/Agents_phase2/RL_Agents/ABIDES_MARL/`:

### `belief_trader.py` — `BeliefTrader(KyleInformedTrader)`  [NEW]
- **Belief per episode:** self-sampled `v = randint(belief_min, belief_max)`; OR
  externally assigned by a `BeliefCoordinator` via `set_belief(v)` (sets a sticky
  `_externally_driven` flag so `on_episode_start` never re-samples mid-episode).
- **Obs:** `obs_extra=2` → `obs_dim = 24` = `[t_norm, edge, q_norm, cash_norm] + 20 LOB`.
  `q_norm = clip(q/q_max, -1, 1)`, `cash_norm = cash/initial_cash`.
- **Constraints (hard, in `action_to_orders`):** buys capped by remaining cash AND
  `q ≤ +q_max`; sells capped by `q ≥ -q_max`. `initial_cash` endowment reset each episode.
- **Two-sided:** no `buy_only`/`sell_only`; uses inherited `_signed_edge`
  (`v-ask` buys / `v-bid` sells / `0` inside spread → structural dead zone).
- **Reward: INHERITS `KyleInformedTrader.compute_reward`** = mark-to-belief,
  `(buy_vol*(v-ask) + sell_vol*(bid-v))/100`. **Do NOT override with mark-to-mid**
  (see §7 — it caused learned abstention).
- `_registry` class list for coordinator discovery (pattern from `KyleMarketMaker`).

### `belief_coordinator.py` — `BeliefCoordinator(Agent)`  [NEW, OPTIONAL]
- `dispatch_priority = -1` (runs before traders each tick).
- Params `f_min/f_max/bias_min/bias_max` (**scalars, not lists** — see §5). At
  `market.tick_count == 0` (or first tick) draws `F ~ U{f_min..f_max}` and per-trader
  `b_i ~ U{bias_min..bias_max}`, calls `trader.set_belief(F + b_i)`.
- Logs `F` series to `{results_dir}/{name}_f_series.csv` when `market.done`.
- **Not used in Set 3** (beliefs self-sampled); kept for a future correlated-F run.

### `kyle_informed_trader.py`  [MODIFIED — 2 changes]
1. Added `obs_extra: int = 0` param → `obs_dim = 2 + obs_extra + 4*n_lob_levels`
   (default 0 = no behavior change; lets `BeliefTrader` widen obs).
2. `_signed_edge()` helper + dead zone (predates this experiment; already on remote).

### `Agents_phase2/__init__.py`  [MODIFIED]
- Registered `BeliefTrader` and `BeliefCoordinator` (so factory's
  `from Agents_phase2 import *` exposes them). Remote already lacked the legacy
  `order_coordinator` import, so this file is safe to rsync whole.

---

## 4. Gotchas (read before touching configs/code)

1. **Factory `_resolve_param` list trap** (`factory.py:40`): a 2-element `[lo, hi]`
   list is resolved to ONE random scalar **at build time** (once per run); a 7-element
   list passes through but `on_episode_start` reads only `[0]` and `[1]`. ⇒ Belief
   ranges MUST be separate scalars (`belief_min`/`belief_max`), never lists. (This also
   means the old vrand configs' `fair_value: [1002,1006]` drew v once per run, not per
   episode — a pre-existing reproducibility wart, not ours to fix.)
2. **PPO-episode vs market-episode drift:** runner resets the market every
   `episode_ticks` (= min `n_steps`) AND dispatches `on_tick` at tick 0 after reset,
   adding one extra `decision_step` per episode. PPO buffer boundaries (every `n_steps`
   stores) drift ~1 step from market boundaries. `BeliefTrader` handles this via the
   sticky `_externally_driven` flag — never weaken it, or beliefs re-sample mid-episode.
3. **Belief lob CSV location in eval:** agents' `results_dir` is force-set by the factory
   to the runner `results_dir` (the ep dir); `checkpoint_dir` is NOT. Eval configs must
   OMIT `results_dir`/`checkpoint_dir` from agent `params` so lob CSVs land in
   `{epdir}/checkpoints/{name}_lob_data.csv` (where `plot_belief_avg.py` reads them).
4. **Final checkpoint step** = `total_timesteps + 29` (e.g. 180000 → `step_180029_final`).
   The run scripts `sed`-patch `eval_*.yaml` from a placeholder to the real step.

---

## 5. Configs — `marl-experiments/price_discovery/configs/market_reset/set3_beliefs_240/`

Common: `ref_price 1000, tick 1, lot 1, env_dt 1.0`. Roster: `PoissonNoiseTrader`
(λ±=1) + **15× `PPOHjbMarketMaker`** (2D: `obs_dim`/`action_dim` unpinned → 2D, obs 9;
`N 240, sim_duration 240, n_steps 240, kappa_ref 0.8, q ±20, total_timesteps 180000`,
checkpoint dirs `/home/samip/checkpoints/beliefs_240/ppo_N`) + **2× `BeliefTrader`**
(`policy linear, max_order 500, initial_cash 20000.0, q_max 20, n_steps 240,
sim_duration 60000, total_timesteps 180000, entropy_coef 0.01`, checkpoint dirs
`.../belief_N`). **No coordinator. No buy_only/sell_only (two-sided).**

| File | mode | env sim_duration | belief_0 | belief_1 | notes |
|---|---|---|---|---|---|
| `train_beliefs.yaml` | train | 60000 | `belief_min/max 994/1006` | same | self-sample per episode |
| `eval_beliefs.yaml` | eval | 240 | 994/1006 | 994/1006 | random per episode (paired seeds) |
| `eval_beliefs_fixed.yaml` | eval | 240 | 1003 | 997 | fixed (min=max) |
| `eval_beliefs_bull.yaml` | eval | 240 | 1003 | 1005 | both above ref |
| `eval_beliefs_flat.yaml` | eval | 240 | 1000 | 1006 | one in dead zone, one extreme |
| `eval_beliefs_bear.yaml` | — | — | — | — | proposed (997/995), NOT run (mirror of bull) |

Eval belief fixing = set `belief_min == belief_max` (no code change). Eval agents carry
`checkpoint: /home/samip/checkpoints/beliefs_240/{ppo_N|belief_N}/{name}_step_180029_final.pt`.

---

## 6. Scripts — `marl-experiments/price_discovery/scripts/`

All set `OMP_NUM_THREADS=MKL=OPENBLAS=NUMEXPR=4` (don't overcook ZhongMa; load was
~5→8 on 32 cores — safe).

- **`run_set3_beliefs_240.sh`** — TRAIN → `rm -rf ~/checkpoints/beliefs_240` → run
  train (seed 42) → find `*_final.pt`, extract step → `sed` patch all `eval_*.yaml` →
  EVAL `eval_beliefs.yaml` (30 eps, seeds 10000+ep, `episode_ticks=0`) → plots.
- **`run_set3_beliefs_240_eval_fixed.sh`** — eval `eval_beliefs_fixed.yaml` only → plots.
- **`run_set3_beliefs_240_eval_asym.sh`** — eval `eval_beliefs_bull.yaml` then
  `eval_beliefs_flat.yaml` → plots each.
- **`plot_belief_avg.py`** — adapted from `plot_eval_avg.py`: `belief_0`→"informed"
  slot (blue), `belief_1`→"liquidity" slot (orange); `lob_sources=["belief_0",
  "belief_1","ppo_hjb_mm_0"]`; belief lines drawn from the `v` column of the lob CSVs;
  PnL baseline-subtracted per episode (handles the `initial_cash` endowment).
  Args: `results_dir n_steps [clip_ticks]`. **Use `clip_ticks=15`** (clips LOB heatmap
  to mid±15 ≈ ref±15).

**Run everything (tmux on ZhongMa):**
```bash
ssh ZhongMa "tmux new-session -d -s beliefsmtm 'bash ~/marl-experiments/price_discovery/scripts/run_set3_beliefs_240.sh && bash ~/marl-experiments/price_discovery/scripts/run_set3_beliefs_240_eval_fixed.sh && bash ~/marl-experiments/price_discovery/scripts/run_set3_beliefs_240_eval_asym.sh 2>&1 | tee ~/marl-experiments/price_discovery/logs/set3_full.log'"
```
Monitor: `ssh ZhongMa "tail -f ~/marl-experiments/price_discovery/logs/beliefs_240.log"`
(train log) / `.../set3_full.log` (chain). Done when `TRAIN + EVAL + PLOTS COMPLETE`.
Training ≈ 16 min; each eval ≈ 2–4 min. **Evals are quick — do NOT sleep >120s between checks.**

---

## 7. Rsync conventions (USER HARD RULES)

- **To remote: code ONLY** (no logs, no checkpoints). After editing agent code:
  ```bash
  cd ~/Documents
  rsync -avz Trading_agents/Agents_phase2/__init__.py ZhongMa:~/Trading_agents/Agents_phase2/__init__.py
  rsync -avz Trading_agents/Agents_phase2/RL_Agents/ABIDES_MARL/{belief_trader.py,belief_coordinator.py,kyle_informed_trader.py,ppo_hjb_market_maker.py} ZhongMa:~/Trading_agents/Agents_phase2/RL_Agents/ABIDES_MARL/
  rsync -avz marl-experiments/price_discovery/configs/market_reset/set3_beliefs_240/ ZhongMa:~/marl-experiments/price_discovery/configs/market_reset/set3_beliefs_240/
  rsync -avz marl-experiments/price_discovery/scripts/<script>.sh ZhongMa:~/marl-experiments/price_discovery/scripts/
  ```
  ZhongMa's `factory.py` is already the fixed version (no `_AgentOrderShuffler`
  reference) and its `__init__.py` already lacks the `order_coordinator` import —
  safe to overwrite `__init__.py` whole.
- **To local: PLOTS ONLY** (no logs/CSVs/checkpoints):
  ```bash
  for EXP in "" "_fixed" "_bull" "_flat"; do
    mkdir -p ~/Documents/marl-experiments/price_discovery/plots_beliefs_240${EXP}
    rsync -avz --include="*.png" --exclude="*" \
      ZhongMa:~/marl-experiments/price_discovery/results_beliefs_240${EXP}/ \
      ~/Documents/marl-experiments/price_discovery/plots_beliefs_240${EXP}/
  done
  ```
- Local plot dirs use the **`plots_`** prefix (`plots_beliefs_240*`) — matches
  `plots_2way_it_240`, `plots_reset_*`. Remote result dirs keep `results_` prefix.

---

## 8. Results so far (mark-to-belief reward — the kept version)

Checkpoints: `~/checkpoints/beliefs_240/{ppo_0..14,belief_0,belief_1}/*_step_180029_final.pt`.

| Eval | belief_0 | belief_1 | mid behavior | trader behavior |
|---|---|---|---|---|
| fixed | 1003 | 997 | sits at midpoint ~1000 | both max conviction: +20 / −20 |
| bull | 1003 | 1005 | rises ~1.5 ticks to ~1001.5 (toward aggregate 1004, but stops short) | both max +20 within ~10 ticks |
| flat | 1000 | 1006 | — | 1000 in dead zone (abstains), 1006 maxes position |
| random | U[994,1006] | U[994,1006] | pinned ~1000 (mean belief) | sparse, near break-even |

**Findings:**
- Mark-to-belief reward ⇒ traders fully express beliefs (no abstention). MMs specialize
  (κ_bid/κ_ask disperse, no collapse), earn ~+600/ep intermediating.
- **Mid moves directionally with net belief imbalance but stops far short of the
  aggregate** (bull: +1.5 of a 4-tick gap). Cause: both traders exhaust `q_max=20` in
  ~10 ticks (40 lots total, one-shot) while 15 MMs re-quote 15 lots at the touch every
  tick (±300 aggregate absorption). After caps bind, informed flow dies, MMs re-pin.

**ABANDONED — mark-to-mid reward** (`reward = (Δcash + q_prev·Δmid)/ref_price`):
caused learned abstention — the weaker believer's β collapsed to ~0 because buying only
pays if the mid rises, and the mid didn't rise (absorption) → self-fulfilling pinning.
**Do not reintroduce it.** The kept reward is the inherited `KyleInformedTrader`
mark-to-belief.

---

## 9. Open / proposed next steps (NOT yet done)

- **Counterfactual A — loosen caps (NEEDS RETRAIN):** `q_max 200, initial_cash 200000`.
  Tests whether persistent informed flow walks the mid to the aggregate. Retrain
  required because `q_norm`/`cash_norm` obs normalization changes (policy would be OOD
  if eval-only).
- **Counterfactual B — thin the market (EVAL-ONLY):** 3 MMs instead of 15. Same
  informed flow vs 60-lot absorption; isolates MM depth as the pinning force.
- **Latent-F coordinator run:** wire `BeliefCoordinator` (`f_min/max 998/1002`,
  `bias_min/max -2/2`) for correlated beliefs around a measurable F ≠ mid — gives a
  real aggregate to discover. Code is built (`belief_coordinator.py`), just add the
  coordinator agent block to a train config (list it FIRST so its tick-0 dispatch
  precedes trader `on_episode_start` in the runner's reset block, which uses raw list
  order not priority sort).
- Worth-flagging: the legacy vrand `fair_value: [lo,hi]` configs drew v once per run,
  not per episode (§4.1) — affects comparability of older Set 1R/2R results.

---

## 10. Reference docs to read for deeper context

- `price_discovery/EXPERIMENT_GUIDE.md` — framework rules, remote paths, full workflow.
- `price_discovery/ANALYSIS_RESET_VS_NORESET.md` — reset vs no-reset mechanics, bugs A–F.
- `price_discovery/configs/market_reset/CONFIGURATIONS.md` — shared RL hyperparams.
- `price_discovery/scripts/run_2way_ppo_240_uni.sh` + `run_2way_eval_240_uni.sh` —
  the base family scripts Set 3 was modeled on.

## 11. Reproduce from scratch (cheat sheet)

```bash
# 1. push code+configs+scripts (code only) to ZhongMa — see §7
# 2. launch full chain in tmux — see §6
# 3. when done, pull plots only — see §7
# 4. inspect beliefs/quantities on remote (read-only), e.g.:
ssh ZhongMa "~/marl-experiments/.venv/bin/python -c \"
import csv, os
base=os.path.expanduser('~/marl-experiments/price_discovery/results_beliefs_240_fixed')
for ep in range(30):
  p=os.path.join(base,f'ep_{ep}','checkpoints','belief_0_lob_data.csv')
  if os.path.exists(p):
    print(ep, sorted({r['v'] for r in csv.DictReader(open(p))}))
\""
```
