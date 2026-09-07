# Dynamic beliefs — news-shock `F_t` (Model D), eval-first

> NOTE (added after set4-g99 landed): g99 **did** unfreeze the MMs (64/64 alive, critic ‖W‖ 0.85)
> **but discovery became weaker** (mid tracks F *less*: bull +0.69 vs set4's +1.93; strongbull +1.19
> vs +4.57) because the healthy MM book is deeper and absorbs informed flow.

---

## PHASE 1a RESULT (deterministic F-step, eval-only, done)

Coordinator drives `F_t = 1000` for `t<60`, then `1000+Δ` for `t≥60`; traders' beliefs `v_i = F_t + b_i`.
Eval-only on BOTH checkpoint arms (30 eps each, seeds 10000+ep). Mid captured % of the Δ step:

| arm (MMs) | scen (Δ) | pre_mid | post_mid | F_post | **captured %** | rise50 | ss_err |
|---|---|---|---|---|---|---|---|
| set4 γ=1.0 (frozen) | bull +3 | 1000.01 | 1001.19 | 1003 | **39.6 %** | never | −1.81 |
| set4 γ=1.0 (frozen) | bear −3 | 1000.01 |  998.42 |  997 | **52.9 %** | 44 tk  | +1.42 |
| g99 γ=0.99 (unfrozen) | bull +3 | 1000.11 | 1001.18 | 1003 | **35.4 %** | never | −1.82 |
| g99 γ=0.99 (unfrozen) | bear −3 | 1000.09 |  999.10 |  997 | **33.1 %** | never | +2.10 |

### Findings
1. **Dynamic shock discovery is PARTIAL and SLOW** — every arm captures only ~⅓–½ of the ±3 step and
   does **not** converge to F (steady-state error ~1.8–2.1 ticks). Most arms never cross 50% of Δ.
2. **Frozen-vs-unfrozen gap is SMALL here** (bull 39.6 vs 35.4 %; bear 52.9 vs 33.1 %) — much smaller
   than the *static*-discovery gap (set4 strongbull 91 % vs g99 24 %). **Reason: the BeliefTrader
   `q_max=20` cap exhausts the informed flow (~120 lots total) before the MM-book-depth difference
   fully expresses.** The cap-confound flagged in the plan is the binding constraint.
3. **Symmetry side-finding:** g99 (healthy, symmetric MMs) → **symmetric** bull/bear capture
   (35.4 vs 33.1). set4 (degraded MMs, some one-sided κ) → **asymmetric** (bear 52.9 ≫ bull 39.6).
   So healthier MMs give more *symmetric* dynamic price discovery.

### Implication for next steps
- To re-expose the MM-quality effect on dynamic discovery, **loosen the trader cap** (`q_max↑`, e.g.
  50–100) so informed flow isn't exhausted — then re-run the step on both arms. (Phase 1a-ii.)
- Then Phase 1b (Poisson shocks) and the rise/settling analysis become meaningful.

### Files
- Local plots: `plots_set4_6tr_coord_shock-{set4,g99}/{step_bull,step_bear}/` (shock_response.png =
  mid-vs-F_t overlay, + eval_avg, belief_actions, ppo_hjb_avg, fills_avg, metrics_shock.csv).
- Remote results: `QFRCAI:~/marl-experiments/price_discovery/results_shock_{beliefs_pop_240,beliefs_pop_240_g99}_{step_bull,step_bear}/`.
- Code: `belief_coordinator.py` (added `f_mode` step/poisson — renamed from `mode` because
  `factory.py:120` overwrites `params["mode"]` with the train/eval lifecycle value).

---


## Recommendation (reasoned)
- **Model D (shocks) over A/B/C.** A step input is the canonical dynamical-system probe: one
  experiment yields rise time, settling time, overshoot, steady-state error. It directly tests
  shock *absorption* — and whether the frozen MMs (set4) impede it. Smooth-A (OU bandwidth sweep)
  and dispersed-B (information aggregation) are natural sequels reusing the same machinery; C
  (endogenous feedback) is deferred (herding/degeneracy risk).
- **Eval-only first, retrain conditional.** Verified in code: the trader obs already carries the
  belief via `obs[1] = mispricing = (v − touch)/ref` (`kyle_informed_trader.py:149`), and
  `order = β · edge`. So a moving `v` flows into both obs and action — the static-trained policy
  reacts to a moving belief *for free*. Eval-only is meaningful, not just cheap. Retrain only if
  tracking is poor.
- **Framework-safe.** Only `belief_coordinator.py` changes (generalize "assign once at tick 0" →
  "drive `F_t` every tick"). No `phase3_multi_agent`/`kyle_base` touched. Verified `set_belief()`
  is safe mid-episode (the `_externally_driven` flag only blocks `on_episode_start` re-sample).

---

## Phase 1 — eval-only step & shock response (the first experiment)

### 1a. Deterministic single step (cleanest probe)
`F_t = F_base` for `t < T_step`, then `F_base + Δ` for `t ≥ T_step`. Traders: `v_i,t = F_t + b_i`.
Scenarios: step-up (Δ=+3, "bull step"), step-down (Δ=−3, "bear step"), at 1–2 step times `T_step`
(e.g. 60, 120). This isolates the mid's step response with no averaging over random jumps.

### 1b. Poisson shocks (realistic)
`F_t = F_0 + Σ jumps`, jumps at rate `λ_jump`/episode, size `Δ ~ ±{3,5,7}`. Measures aggregate
tracking RMSE + per-shock adjustment distribution.

### The saturation tie-in (key scientific payoff)
Run shock-eval on **both** checkpoint sets:
- `beliefs_pop_240` (set4, gamma=1.0 — ~7/15 MMs frozen to constant κ)
- `beliefs_pop_240_g99` (set4-g99, gamma=0.99 — **CONFIRMED unfrozen**, 64/64 alive)

Original prediction (now likely INVERTED — see top note): the frozen-MM population absorbs the step
more slowly / larger steady-state error than the unfrozen one. After g99, the cleaner framing: does
MM "quality" trade off against dynamic discovery? (frozen/shallow → mid reacts more; healthy/deep →
mid absorbed.) Either outcome is publishable; the comparison is the point.

### Metrics (per step / per shock)
- rise time (ticks to 50% / 90% of Δ), settling time, overshoot, steady-state error (`mid − F_t`).
- tracking RMSE(`mid`, `F_t`) over the episode.
- Plot: `mid` vs `F_t` overlay (extend `plot_belief_avg.py` to draw the `F_t` step line from the
  coordinator's logged F series — already dumped to CSV).

### Reward/obs notes (verified, cosmetic for eval)
- `obs[1] = mispricing` carries `v` → policy sees belief changes. ✓
- `compute_reward` is mark-to-belief using *current* `v` (`kyle_informed_trader.py:175`) → at a
  shock, logged PnL revalues past positions at the new `v`. For eval this only affects logged
  PnL, not behavior (no gradient). Document; optionally also log an entry-`v` realized PnL.

### Cap-confound (must handle — sharper for g99)
Traders are `q_max=20`; they may be inventory-full when the shock hits → can't respond. Mitigations
(pick one): step the shock *early* (before caps bind), use modest Δ (3 ticks), or measure/report
each trader's inventory at the shock time. For the **g99** arm especially, the deep book + caps may
stall the mid entirely → consider a companion run with `q_max↑` (e.g. 50) or larger Δ to get a
readable signal. Document whichever.

---

## Phase 2 — retrain (CONDITIONAL on Phase 1) — deferred until Phase 1 lands
Only if eval-only tracking is poor. Train config gets the shock process; traders learn to handle
time-varying beliefs (keep room for shocks? anticipate?). Compare retrained vs static-trained
step response. ~30–40 min retrain (mirrors set4-g99).

---

## Implementation (agent code + configs + scripts only)

### `belief_coordinator.py` (the only code change)
- New params: `mode` (`static` | `step` | `poisson`), `f_base`, `step_tick`, `delta`,
  `jump_rate`, `jump_size_min/max`. Keep `static` = current behavior (back-compat).
- `on_tick`: maintain `self._F` from a seeded per-episode RNG; on change, `set_belief(F+b_i)` for
  all traders. Seed from the episode for reproducibility (eval seeds 10000+ep).
- Extend the F-series CSV to log `(sim_t, tick, F_t)` every tick (currently logs one F).

### Configs — `configs/market_reset/set4_beliefs_240_shock/`
- `eval_step_bull.yaml`, `eval_step_bear.yaml` (1a), `eval_shock_poisson.yaml` (1b).
- Same roster as set4 eval (15 MMs + 6 BeliefTrader + coordinator + noise), checkpoint lines point
  at the chosen set (set4 and/or set4-g99) via the existing `sed`-patch step.
- Coordinator `mode: step/poisson` + params; traders UNCHANGED (their `v` is driven externally).

### Scripts / plots
- `run_set4_shock_eval.sh` — eval-only: for each scenario × checkpoint-set, run 30 eps (seeds
  10000+ep, `episode_ticks=0`) → plots. No train step.
- Extend `plot_belief_avg.py` to overlay `F_t` (read coordinator `_f_series.csv`) and compute
  step-response metrics → write a small `metrics_shock.csv` per scenario.

### Folder structure (local)
```
plots_set4_6tr_coord_shock-set4/      ← shock-eval on set4 (frozen-MM) checkpoints
  {step_bull,step_bear,poisson}/
plots_set4_6tr_coord_shock-g99/       ← shock-eval on set4-g99 (unfrozen) checkpoints
  {step_bull,step_bear,poisson}/
```

### Remote
QFRCAI (via `ssh -J ZhongMa QFRCAI`), tmux, same venv/thread-cap conventions as set4-g99.

---

## Open questions to confirm before building
1. Start with **1a deterministic step** (my pick) or jump straight to **1b Poisson**?
2. Shock-eval on **both** set4 + set4-g99 checkpoints (my pick — the quality↔discovery trade-off),
   or just one?
3. For the g99 arm, run also with **loosened caps** (`q_max↑`) so the deep book doesn't stall the mid?
4. Phase 2 (retrain) deferred until Phase 1 lands? (default yes)
