# Set 4 — Belief-Trader Population Experiment

**Goal:** Test whether 15 PPO HJB market makers *aggregate* a population of staggered,
noisy belief traders into the mid price — and finally **see the traders' action dynamics**
(beta / conviction over the episode), which the current plots lack.

---

## The decision on `initial_cash` (asked directly)

**Do not increase it.** Grounded in the result numbers:

- Every active trader stops at **+19 lots, not +20** (bull/fixed/flat evals). That is the
  **cash bind**: `20000 // 1003 = 19`. `initial_cash=20000` is calibrated to `q_max=20`
  (`20000 / 1000 = 20`), so the two constraints are co-binding at ~19–20.
- Raising cash **alone** gains ~+1 lot/trader (q_max still binds at 20). Raising to a huge
  number gains **nothing**. To get more flow *per trader* you must raise **both**
  (`q_max 50, cash 50000` → +50).
- But scaling via **trader count** (many small desks) is more realistic and isolates the
  *aggregation* question from brute-force whale pressure. So: **keep q_max=20, cash=20000
  per trader; scale the population.**

---

## What the current results prove (the motivation)

| eval | aggregate belief | terminal mid | reading |
|---|---|---|---|
| flat (1000 / 1006) | 1003 | **999.98** | one maxed +19 trader → **zero** mid move (fully absorbed) |
| bull (1003 / 1005) | 1004 | **1000.85** | two maxed traders close only ~21% of the gap |
| random (iid) | ~1000 | **999.2** | iid beliefs net to ~0, no aggregate signal to discover |

Informed:MM capacity ratio ≈ 40:300 = **0.13**. Two levers, both addressed here:
1. **Population** 2 → 6 (ratio 0.13 → ~0.38).
2. **Correlated signal**: replace iid self-sampling with the **BeliefCoordinator** (latent `F`
   + per-trader bias) → a measurable aggregate `F` for the mid to discover. The iid setup
   has *no* aggregate signal (beliefs net to 1000), which is why "random" is uninformative.

---

## Composition (decided)

| agents | count | role | key params |
|---|---|---|---|
| PoissonNoiseTrader | 1 | background noise | λ±=1 (unchanged) |
| **BeliefCoordinator** | 1 (NEW in config) | draws `F`, assigns `v_i = F + b_i` at tick 0 | train: F∼U{998,1002}, b∼U{−3,3}; eval: F fixed, b∼U{−3,3} |
| **BeliefTrader** | **6** (was 2) | informed, staggered arrival | q_max 20, cash 20000, **arrival random U{0..60} per episode**, obs_extra=3 |
| PPOHjbMarketMaker | 15 | absorption side | 2D, κ_ref 0.8, q ±20, **hjb_dt 0.5, hjb_update_freq 50** (adaptive whole-grid solve) |

- **Staggered, randomized arrival:** each trader draws `active_from ~ U{arrival_lo..arrival_hi}`
  (= [0,60], first quarter of the episode) **per episode** in `on_episode_start`; before its tick it
  abstains (returns `[]`). Window [0,60] → full population active by tick 60, leaving 180 ticks
  (75%) of sustained full-pressure trading for the MMs to aggregate. Randomizing (not fixed per
  trader) makes all 6 learn the *same* general "arrive → assess → trade" policy instead of each
  over-fitting to one slot, and eval samples varied arrival patterns (averaged over 30 eps) —
  more honest than a contrived schedule.
- **`active` flag added to obs** (obs_extra 2 → 3, obs_dim 24 → 25): `1.0` once `sim_t >=
  active_from`, else `0.0`. Essential now that arrival is randomized — it's the only clean
  signal separating "dormant" from "trading" (t_norm is ~0 throughout because belief
  `sim_duration=60000`). Also prevents the learned-abstention failure mode.

---

## Code changes (all in allowed agent code / scripts / configs — framework untouched)

### `belief_trader.py`
1. Add **scalar** params `arrival_lo: int = 0`, `arrival_hi: int = 0` (scalars — NOT a list,
   per the factory `_resolve_param` build-time trap). In `on_episode_start`, draw
   `self._active_from = random.randint(arrival_lo, arrival_hi)` (when `lo==hi==0` → always
   active, current behavior). Drawn fresh each episode → randomized arrival.
2. Add `active` flag to obs: `obs_extra` 2 → 3, build_observation appends
   `1.0 if int(sim_t) >= self._active_from else 0.0` (place it with q_norm/cash_norm in the
   front block so the LOB tail stays aligned).
3. Gate `action_to_orders`: `if int(sim_t) < self._active_from: return []`.
4. `on_episode_start`: reset inventory/cash (already done); belief still comes from the
   coordinator's `set_belief` (sticky `_externally_driven`) — unchanged.

*(No logging code needed — `kyle_base.py` already dumps `{name}_action_history.csv` with
`step,action` (= beta) at simulation end, in eval too. Verified present in current results.)*

### `belief_coordinator.py`
No code change. First real wiring. **List it FIRST** in the config agents list (runner's
reset dispatch uses raw list order, not priority sort — so its tick-0 belief assignment
must precede traders' `on_episode_start`).

### `__init__.py`
No change (both classes already registered).

---

## HJB grid coverage — "make sure the HJB PPO is solving for the whole grid"

How the PPO-HJB MM works (`ppo_hjb_market_maker.py`): the Guéant HJB value function
`h(t,q)` is solved once by backward induction over a **time × inventory** grid; the PPO then
learns `[κ_bid, κ_ask]`, which enter only the closed-form depth `1/κ + h(q) − h(q±1)`.
Two problems in the set3 config that I am fixing:

1. **Static solve.** `_maybe_resolve_hjb` re-solves `h` with *estimated* arrival λ's — but
   only if `hjb_update_freq > 0`, and set3 leaves it at **0**. So `h` is frozen at the init
   values (κ_ref=0.8, λ=1.0) for the whole run. With a **staggered population**, arrival
   intensity varies strongly over the episode → a frozen `h` quotes wrong depths.
   **Fix (config only):** `hjb_update_freq: 50` → re-solve over the full grid every 50
   decision steps with the MM's running λ estimates. No change to the MM code; the
   `_decision_step % N` re-solve guard is already on remote.

2. **Coarse time grid.** set3 used `hjb_dt: 1.0` → only 240 sub-steps over the episode;
   the backward Euler step is large and the `exp` terms get clamped at `max_exp=5`.
   **Fix (config):** `hjb_dt: 0.5` → `_n_hjb = 480` sub-steps, a genuinely solved PDE.
   Cost: each re-solve is 480×41 ≈ 20k ops (<50 ms); at freq 50 over 180k steps that's
   ~3600 re-solves/MM — adds only ~1–2 min total across 15 MMs.

**Deliberately NOT widening the inventory grid.** The grid is `q_grid = arange(q_min, q_max+1)`
and MMs **self-limit** at the edges (`q=q_max` → bid depth `None`, stops buying). So [-20,+20]
is the whole *reachable* inventory space by construction. Widening to ±40 would raise
aggregate absorption (15×40=600 vs 114 informed → ratio 0.19) and **suppress** price discovery
— the opposite of the goal. Keeping ±20 holds the ratio at ~0.38.

**Verification step (Phase 1 smoke test):** dump `self._h.shape` and confirm `(481, 41)` and
that `h` actually changes between re-solves (log `h[0, 20]` before/after). Also confirm
`_maybe_resolve_hjb` fires (it's gated on `_decision_step % 50 == 0`).

**Known limitation (flagged, not fixing):** `h` is re-solved for `κ_ref`, not the learned κ —
the 2D-κ / 1D-`h` factorisation is the existing design. Making the re-solve use a representative
`κ_eff = (κ_bid+κ_ask)/2` is a one-line change in `_maybe_resolve_hjb` if you want it; say so.

---

## Configs — new `configs/market_reset/set4_beliefs_240/`

- `train_beliefs.yaml` — roster above; coordinator first; `seed: 42`; env `sim_duration 60000`;
  all RL agents `total_timesteps 180000, n_steps 240, eval_freq 180000`; checkpoints →
  `/home/samip/checkpoints/beliefs_pop_240/{ppo_N,belief_N}`. Per MM: **`hjb_dt: 0.5`,
  `hjb_update_freq: 50`** (adaptive whole-grid solve).   Per trader: **`arrival_lo: 0,
  arrival_hi: 60`** (randomized arrival).
- `eval_beliefs_calm.yaml`     — coordinator `f_min=f_max=1000`.
- `eval_beliefs_bull.yaml`     — coordinator `f_min=f_max=1003`.
- `eval_beliefs_strongbull.yaml` — coordinator `f_min=f_max=1005`.
- `eval_beliefs_bear.yaml`    — coordinator `f_min=f_max=997`.
- Eval: env `sim_duration 240`; traders carry the same `arrival_lo/hi` (random per episode);
  coordinator `bias_min/max ±3` (per-trader noise around fixed F); 30 eps, seeds 10000+ep,
  `episode_ticks=0`.

---

## Plots

### `plot_belief_actions.py` (NEW — the explicit ask)
Reads `belief_*_action_history.csv` (beta) + `pnl_history.csv` (inventory) across ep_0..29:
- **Panel 1:** beta(t) per trader, mean ± std band; plus a grey **fraction-of-traders-active**
  ramp (rises 0→1 as randomized arrivals fire over [0,60]). → shows arrival + conviction spike.
- **Panel 2:** inventory(t) per trader, mean. → shows fill-to-cap dynamics.
- Output `{results_dir}/belief_actions.png`. Args `results_dir n_steps`.

### `plot_belief_avg.py`
Generalise `lob_sources` and the belief-series loop from `["belief_0","belief_1"]` to
`belief_0..belief_5` (dynamic glob) so the LOB/mid/PnL plot covers all 6 traders.

---

## Execution phases

**Phase 0 — immediate value (no retrain):** write `plot_belief_actions.py`, run on the
**existing** `results_beliefs_240_bull` (2 traders). Validates the script and gives the
trader-dynamics view from data we already have. ~2 min.

**Phase 1 — code + smoke test:** edit `belief_trader.py`; write the 5 configs + 2 scripts.
Smoke-test (2-tick throwaway run): (a) factory builds `BeliefCoordinator` and `set_belief`
reaches all 6 traders; (b) `hjb_update_freq`/`hjb_dt` take effect — dump `h.shape == (481,41)`
and confirm `h` changes across a re-solve boundary; (c) randomized `active_from` differs
across two episodes for the same trader.

**Phase 2 — train:** `run_set4_beliefs_240.sh` on ZhongMa (tmux). ~25 min for 22 RL agents.
Chain: train → sed-patch eval checkpoint steps → 4 evals → all plots.

**Phase 3 — pull + read:** rsync PNGs only to `plots_beliefs_240_{calm,bull,strongbull,bear}/`.
Key read: the **discovery curve** — terminal mid vs F ∈ {997,1000,1003,1005}. If mid tracks
F, aggregation works; compare slope to the 2-trader baseline (bull closed 21%).

---

## Risks / contingencies
- **Factory rejects `BeliefCoordinator`** (untested NPC wiring) → Phase 1 smoke test catches
  it; fall back to per-trader `belief_min/max` shifted around F (no coordinator), losing the
  clean "F" but keeping the population.
- **HJB re-solve too slow / unstable** (`hjb_update_freq 50` × 15 MMs) → Phase 1 timing check;
  back off to `freq 100` / `hjb_dt 1.0` (current set3 values) — still adaptive, just coarser.
  If `h` diverges (NaN/inf in depths), disable re-solve and revisit the κ_ref limitation.
- **6 traders still fully absorbed** (ratio 0.38 too low) → dial up via `q_max 40, cash 40000`
  (q_norm uses q_max, so this needs a retrain; flag for a Set 4b). Do **not** widen MM q_max
  — that raises absorption and suppresses discovery.
- **Late-arriving traders under-trade** (arrival up to tick 60, 180 ticks left) → they reach
  ~+19 in ~10 ticks (seen in current data), so 180 ticks is ample.
