# Phase 1 onward — experiment instructions for the LLM

> Self-contained handoff. An LLM resuming this work should be able to execute Phase 1a-ii and the
> sequels from this document + the existing `PLAN_SET4_SHOCK.md`, `PLAN_SET4_G99.md`,
> `BELIEFS_EXPERIMENT.md`. **Read the GOTCHAS before touching anything.**

---

## 0. Where we are (state summary)

The project: does the market mid **discover** an informed population's latent fair value `F`, against
15 PPO HJB market makers (MMs) + 6 BeliefTraders + a BeliefCoordinator that drives beliefs.

Timeline so far:
1. **set4** (gamma=1.0, mtm-reward): static discovery works (mid ≈ F) BUT ~7/15 MMs **froze** to
   constant κ. Root cause (measured): gamma=1.0 → undiscounted returns → large un-normalized value
   loss → critic gradient blows up the **shared actor-critic trunk** under Adam (no weight decay) →
   tanh saturates → actor mean frozen. See `PLAN_SET4_G99.md`.
2. **set4-g99** (gamma=0.99): saturation **cured** (all 15 MMs 64/64 alive, critic ‖W‖ 9.9→0.85),
   BUT static discovery **weakened** (bull +0.69 vs set4's +1.93) — healthy MM book is deeper and
   absorbs informed flow. Discovery-weakening was partly the trader policy too (~21% fewer fills).
3. **Phase 1a — shock eval** (deterministic F-step 1000→1003 at t60, eval-only, on BOTH set4 and
   g99 checkpoints): dynamic discovery **partial & slow** — every arm captured only ~⅓–½ of Δ;
   frozen-vs-unfrozen gap small (set4 bull 39.6% vs g99 35.4%). **Reason: the BeliefTrader
   `q_max=20` cap exhausts informed flow (~120 lots) before the book-depth difference expresses.**
   See `PLAN_SET4_SHOCK.md` for the table.

**Conclusion of Phase 1a:** flow starvation (`q_max`) is the binding constraint on dynamic discovery.
The next experiment unchokes it.

---

## 1. The decided next experiment — Phase 1a-ii (randomize trader q_max)

**One sentence:** retrain g99 ONCE with the BeliefTrader's `q_max ~ U{20,50}` resampled each episode
(MM `q_max=50` fixed), then at eval fix trader `q_max` ∈ {20,30,40,50} to get a **discovery-vs-flow
curve** from one trained policy.

### Why this design (confirmed decisions, do not re-litigate)
- **Trader `q_max` randomized per episode** `U{20,50}` — gives a curve from one retrain; the
  `q_norm = q/q_max` obs is fractionally invariant so the policy generalizes across cap sizes.
- **MM `q_max = 50` fixed** — the HJB PDE is solved once at init on a fixed grid and **cannot move
  per episode**. 50 gives headroom for max trader flow (6×50=300 vs 15×50=750 MM capacity).
- **`phi`/`alpha` UNCHANGED** (0.001 / 0.01). Do NOT rescale. We decided: keep it simple, avoid the
  extra variable; revisit only if the MMs quote pathologically deep after retrain.
- **`initial_cash = q_max × ref_price`** (auto-scales, so `cash_norm` stays in-distribution as q_max
  varies; equals the old 20000 at q_max=20).
- **g99 only** (commit to the non-pathological MM as the baseline). set4 retained as a frozen
  reference; do NOT retrain set4 here.
- **gamma = 0.99** everywhere. (`g99` = gamma 0.99. It is NOT gae_lambda — gae_lambda stays 1.0.)

### What the curve tells us
- capture% **rises** with q_max → flow starvation was binding (we get the curve shape).
- capture% **flat** → the healthy g99 book itself caps dynamic discovery regardless of trader capacity
  → the more interesting "MM quality limits discovery" result.
Either outcome is the publishable point; the sweep is what distinguishes them.

---

## 2. Code change (ONLY `belief_trader.py` — framework-safe)

`Trading_agents/Agents_phase2/RL_Agents/ABIDES_MARL/belief_trader.py`.

Add per-episode q_max randomization, **backward-compatible** (if `q_max_lo/q_max_hi` absent → today's
fixed-`q_max` behavior). Approximate diff:

```python
# __init__ — add two params (SCALARS, not a list — see GOTCHA #4):
def __init__(self, initial_cash=20000.0, q_max=20, q_max_lo=None, q_max_hi=None,
             belief_min=None, belief_max=None, arrival_lo=0, arrival_hi=0, **kwargs):
    kwargs["obs_extra"] = 3
    super().__init__(**kwargs)
    self.initial_cash = initial_cash
    self.q_max = q_max
    self._q_max_lo = q_max_lo
    self._q_max_hi = q_max_hi
    # ... (rest unchanged)

# on_episode_start — after the existing belief-sampling block, BEFORE resetting _inventory/_cash:
    # per-episode randomized position cap (U{lo,hi}); fixed when lo==hi (eval)
    if self._q_max_lo is not None and self._q_max_hi is not None:
        self.q_max = random.randint(self._q_max_lo, self._q_max_hi)
        self.initial_cash = float(self.q_max * self.ref_price)   # keep cash_norm sane
    self._active_from = random.randint(self._arrival_lo, self._arrival_hi)
    self._inventory = 0
    self._cash = self.initial_cash
```

Notes:
- `self.ref_price` is available (inherited from `KyleInformedTrader`).
- `build_observation` already uses `self.q_max` and `self.initial_cash`, so the obs tracks the
  per-episode values automatically — no other change needed.
- `action_to_orders` already caps by `self.q_max` — fine.

---

## 3. Configs — `configs/market_reset/set4_beliefs_240_g99_rq50/`

Clone `set4_beliefs_240_g99/` (the gamma=0.99 configs) into `set4_beliefs_240_g99_rq50/` and apply:

### ALL files
- Repoint every `results_dir`/`checkpoint_dir`/`checkpoint:` path from `beliefs_pop_240_g99` →
  **`beliefs_pop_240_g99_rq50`** (see GOTCHA #2 — do NOT skip this).
- Keep `gamma: 0.99`, `gae_lambda: 1.0`, mtm reward, coordinator F∈[998,1002]/bias±3, arrival 0/60,
  `hjb_update_freq: 0`, `n_steps: 240`, `total_timesteps: 180000`.

### `train_beliefs.yaml`
- 15× PPOHjbMarketMaker: `q_min: -50`, `q_max: 50`; **`phi: 0.001`, `alpha: 0.01` unchanged**.
- 6× BeliefTrader: replace `q_max: 20` and `initial_cash: 20000.0` with **`q_max_lo: 20`,
  `q_max_hi: 50`** (initial_cash is now auto-computed in the agent; you may omit it or leave it — the
  code overrides it).
- BeliefCoordinator: `f_min/f_max: 998/1002`, `bias_min/bias_max: -3/3`, `f_mode: static`.

### Static evals `eval_beliefs_{calm,bull,strongbull,bear}.yaml` (sanity check at the max-flow end)
- MMs: `q_min: -50, q_max: 50` (phi/alpha unchanged).
- BeliefTrader: `q_max_lo: 50, q_max_hi: 50` (fixed at 50 for this sanity run).
- Coordinator: fixed `f_min=f_max` per scenario (calm 1000 / bull 1003 / strongbull 1005 / bear 997),
  `f_mode: static`.
- `checkpoint:` lines point at the trained `beliefs_pop_240_g99_rq50/*_step_180029_final.pt`.

### Shock evals `eval_step_{bull,bear}.yaml` (the sweep)
- Same MM q_min/q_max = -50/50.
- BeliefTrader: `q_max_lo: __QMAX__, q_max_hi: __QMAX__` (**placeholder** — the run script seds it per
  sweep value).
- Coordinator: `f_min: 1000, f_max: 1000`, `f_mode: step`, `step_tick: 60`, `delta: 3` (bull) /
  `delta: -3` (bear), `bias_min/bias_max: -3/3`.
- `checkpoint:` lines → `beliefs_pop_240_g99_rq50/...`.

---

## 4. Run script — `scripts/run_set4_g99_rq50.sh`

Clone `run_set4_beliefs_240_g99.sh`; set `EXP=beliefs_pop_240_g99_rq50`,
`CFGDIR=.../set4_beliefs_240_g99_rq50`. Sequence:
1. **Train** (seed 42, ~16–20 min; HJB grid 101 states — fine).
2. `sed`-patch the real `*_final.pt` step into all `eval_*.yaml` (the existing pattern).
3. **Static sanity** (calm/bull/strongbull/bear at q_max=50) → plots.
4. **Shock sweep**: `for Q in 20 30 40 50; for SCEN in step_bull step_bear:` →
   `sed "s|__QMAX__|${Q}|g" eval_${SCEN}.yaml > /tmp/run_${SCEN}_${Q}.yaml` → 30 eps (seeds 10000+ep,
   `episode_ticks=0`) → `results_shock_g99_rq50_${SCEN}_q${Q}/` → plots + `plot_shock_response.py`.
5. Print `SET4-g99-rq50 COMPLETE`.

Thread caps: `OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=NUMEXPR_NUM_THREADS=4`.
Run on **QFRCAI via `ssh -J ZhongMa QFRCAI`** (the proxy is faster than local→QFRCAI). Launch in tmux.

---

## 5. Pre-flight (DO NOT skip — catches the bugs I already hit)

After rsyncing code+configs+script to QFRCAI, BEFORE launching the full job:
1. **Import + param check**: build the BeliefCoordinator and a BeliefTrader from the new config;
   confirm `f_mode=="step"` and `q_max_lo/hi` are set (i.e. GOTCHA #1 and #4 didn't bite).
2. **1-episode smoke**: run step_bull on the g99-rq50 checkpoint at q_max=50, seed 10000. Verify
   `belief_coordinator_f_series.csv` shows `F: 1000` for t<60 and `1003` for t≥60 (i.e. the step
   fires — this is exactly the bug that bit Phase 1a, GOTCHA #1).
3. **Randomization check**: run 2 train episodes; confirm different `q_max` realizations per episode
   (grep the trader inventory caps or add a one-line print in `on_episode_start`).

---

## 6. Monitor / pull / verify

```bash
# monitor (from local)
ssh -J ZhongMa QFRCAI "tail -f ~/marl-experiments/price_discovery/logs/set4_g99_rq50.log"
# pull plots + metrics (PNG + metrics_shock.csv only — rsync rule)
#   -> plots_set4_6tr_coord_mtm-g99-rq50/{calm,bull,strongbull,bear}/
#   -> plots_set4_6tr_coord_shock-g99-rq50/{step_bull,step_bear}/q{20,30,40,50}/
```

**Verify the headline:** assemble capture% (from each `metrics_shock.csv`, the `captured_pct` column)
vs q_max for bull & bear → one curve plot. Compare to the **g99-q20 baseline (35% bull / 33% bear)**
and **set4-q20 (39.6% bull / 52.9% bear)**.

**Static sanity:** g99-rq50 mean mids (calm/bull/strongbull/bear) should track F more closely than
g99-q20 (calm 1000.07 / bull 1000.69 / strongbull 1001.19 / bear 999.02). If still pinned, the deep
book (not the cap) is the limiter.

---

## 7. Sequels (after Phase 1a-ii lands)

- **Phase 1b — Poisson shocks.** Reuse the coordinator's `f_mode: poisson` (already implemented in
  `belief_coordinator.py`): `F_t = F_0 + Σ jumps`, jumps at rate `jump_rate`, size `±U{jump_size_min..max}`.
  Run on g99-rq50 at q_max=50. Measure tracking RMSE + per-shock adjustment distribution. (May need
  stronger/larger shocks to be readable against the deep g99 book.)
- **Smooth-A (OU moving F).** `f_mode: ou` (needs adding to the coordinator) — sweep the OU timescale
  to find the tracking bandwidth.
- **Dispersed-B (information aggregation).** Each trader gets private noisy signals about F at
  staggered times; Bayesian-ish update of `v_i`. The "real" price-discovery question; most modeling
  work.
- **Phase 2 (CONDITIONAL retrain).** If even the uncapped eval-only sweep tracks poorly, retrain the
  traders WITH time-varying beliefs so they learn to handle moving `v`. (~30–40 min, mirrors g99.)
- **Per-episode q_max randomization for the curve** is already built by Phase 1a-ii. A later
  dedicated "flow-capacity sweep from one policy" can widen the range (e.g. U{20,80}) if needed.

---

## 8. GOTCHAS (read before touching code/configs)

1. **`factory.py:120` clobbers `params["mode"]`** with the top-level train/eval lifecycle value. So an
   agent param literally named `mode` is overwritten → the BeliefCoordinator's dynamics selector is
   called **`f_mode`** (values `static`/`step`/`poisson`), NOT `mode`. This bit Phase 1a (F didn't
   step until renamed).
2. **When copying configs, ALWAYS repoint `results_dir`/`checkpoint_dir`/`checkpoint:`** to the new
   experiment root. g99 originally still pointed at `beliefs_pop_240` and the train **overwrote** the
   gamma=1.0 checkpoints. Don't repeat.
3. **`plot_shock_response.py`** reads the F series from the **ep-dir top level**
   (`{epdir}/belief_coordinator_f_series.csv`), NOT under `checkpoints/`. The mid comes from
   `{epdir}/checkpoints/ppo_hjb_mm_0_lob_data.csv`.
4. **Factory list trap** (`factory._resolve_param`): a 2-element `[lo,hi]` list is resolved to ONE
   random scalar **at build time** (once per run), NOT per episode. So `q_max_lo`/`q_max_hi` MUST be
   **separate scalars**; the per-episode draw happens in the agent's `on_episode_start`.
5. **Eval is deterministic** (`kyle_base.py:303` → `sample_action(..., deterministic=True)`; PPO uses
   `z = mu_z`, no sampling). So a flat eval κ = a flat actor mean = (here) saturation, not "low noise".
6. **BeliefTrader is symmetric**: it has only `q_max` (no `q_min`); buy capped by `q_max−inventory`,
   sell by `q_max+inventory`, position ∈ `[-q_max, +q_max]`. The `q_min/q_max` pair belongs to the
   **MMs** (the HJB inventory grid) and is kept symmetric (`q_min = -q_max`).
7. **MM `q_max` cannot vary per episode** — the HJB is solved once at init on a fixed grid. Only the
   trader's cap randomizes.
8. **Framework HARD constraints**: NEVER modify `phase3_multi_agent/*` or `kyle_base.py`. All changes
   are agent code (`belief_trader.py`, `belief_coordinator.py`) + configs + scripts.
9. **Remote conventions**: QFRCAI via `ssh -J ZhongMa QFRCAI`. venv `~/marl-experiments/.venv/bin/python`.
   rsync code/configs/scripts OUT (via the proxy); PNG/metrics IN only. Set thread caps.

---

## 9. Folder/naming conventions (local `price_discovery/`)

```
configs/market_reset/set4_beliefs_240_g99_rq50/        ← this experiment (train + static + shock evals)
plots_set4_6tr_coord_mtm-g99-rq50/{calm,bull,strongbull,bear}/        ← static sanity
plots_set4_6tr_coord_shock-g99-rq50/{step_bull,step_bear}/q{20,30,40,50}/  ← shock sweep
```
Existing (do not touch):
```
plots_set4_6tr_coord_mtm-g99/{calm,bull,strongbull,bear}/     ← g99 q_max=20 (low-flow reference)
plots_set4_6tr_coord_mtm-reward/{calm,bull,strongbull,bear}/  ← set4 gamma=1.0 (frozen-MM reference)
plots_set4_6tr_coord_shock-{set4,g99}/{step_bull,step_bear}/  ← Phase 1a shock results
```
Reference docs: `PLAN_SET4_G99.md` (saturation cause + g99 result), `PLAN_SET4_SHOCK.md` (Phase 1a
shock table + findings), `BELIEFS_EXPERIMENT.md` (Set-3 era, partly stale), `PLAN_ASYMMETRY.md`
(reward-fix root cause).
