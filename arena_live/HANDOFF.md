# HANDOFF — marl-experiments live-arena work

> Updated 2026-08-27: 30v30 transfer validation FAILED (§5g) with root cause
> corrected to deep-ladder exposure under async actuation (§5h); obs fix
> shipped and partially effective (9/10 calm); retraining track moved to the
> fastboxex MarlV2Env stop-the-clock simulator (Phase 0 build done, Phase 1
> trainer port in progress — §5h). Read top-to-bottom before touching
> anything. Every number below was measured from run artifacts; nothing here
> is assumed. Companion docs: `PLAN_LIVE_ARENA.md`, `README.md`,
> `NOTES_DEROUNDING.md` (this dir); `price_discovery/` docs for the sandbox
> era.

---

## 0. Hard rules (from the user, repeatedly enforced)

1. **Never train on the arena.** All training is local sandbox. Arena = eval only.
2. **Pull only plots/summaries to local** (`analysis.png`, `summary.csv`). No raw
   logs, no checkpoints to local. (Local `arena_live/runs/` still contains stage-1
   era CSVs from before this rule — cleanup pending user OK.)
3. **Don't commit to git unless asked.** Don't pollute the repo with scratch files.
4. Results must be rigorous: bear ≈ mirror of bull, calm must oscillate around F.
   Current stage-2 populations do NOT satisfy this — the asymmetry investigation
   (§5e) is the open thread.

## 1. Project arc

1. **Sandbox phase (done):** `price_discovery/` on the phase3 framework —
   15 PPO-Hjb MMs + 6 BeliefTraders + coordinator. γ=1.0 saturates MMs (frozen κ);
   γ=0.99 ("g99") cures it; dynamic-F capture only ~15–35%.
2. **Live phase (this dir):** deploy trained populations (inference-only,
   deterministic) on the real FastBoxEx exchange arena. Stage 0 = harness bug
   fixes; stage 1 = certification with g99 ckpts; stage 2 = matched-composition
   retrain (7 MM + 3 BT, λ=0.47) + live redeploy. **Current: bull/bear asymmetry
   investigation (§5e) — root cause is in training, not yet identified.**
3. **Stage 3 (deferred):** de-rounding — random per-agent decision times,
   persistent market. Design notes in `NOTES_DEROUNDING.md`. Do not start.

## 2. Environment & access (verified)

| Thing | Value |
|---|---|
| Exchange host | `172.16.22.83` (ON-CAMPUS only; auth :11001, arena :11002/:11003, obs :11005, market relay :12000) |
| Dead/stale hosts | `202.120.13.117`, `172.16.210.57` (= QFRCAI itself) — do not use |
| Deployment host | QFRCAI, via `ssh -J ZhongMa QFRCAI` |
| Arena venv on QFRCAI | `~/marl-experiments/arena_live/.venv` — py3.14 (torch 2.13 cpu, fastboxex 2.4.0) |
| Sandbox venv on QFRCAI | `~/marl-experiments/.venv` — py3.12 (NO fastboxex; use for sandbox training/eval) |
| Local venv | `~/Documents/marl-experiments/.venv` — py3.12 (torch; for mock_test/analysis) |
| Credentials | `arena_live/.env` on QFRCAI (FASTBOXEX_USER/PASS, mode 600; account "samip", arena owner) |
| Sandbox framework | QFRCAI: `~/SimExchangeSandbox/phase3_multi_agent` (runner: `run.py` / `src/runner.py::run_simulation`) |
| MarlV2 stop-the-clock env | local `~/Documents/FastboxEx-Cli` (branch `feat/marl-phase3-align`, UNCOMMITTED WIP) + `~/Documents/FastboxEx-Protocol` (v0.28.0); QFRCAI mirrors `~/FastboxEx-Cli`, `~/FastboxEx-Protocol` (rsync'd source, no git). Wheel `fastboxex-3.7.1-cp312-abi3` installed in local `.venv` AND QFRCAI sandbox `.venv` (arena venv 2.4.0 untouched). QFRCAI build venv: `~/marl-experiments/.venv_fx` (maturin only) |
| Agent code | QFRCAI `~/Trading_agents` = local `~/Documents/Trading_agents` (Agents_phase2) |
| Checkpoint roots | g99 (15+6): `~/checkpoints/beliefs_pop_240_g99/`; matched (7+3): `~/checkpoints/matched_7x3_g99/` |
| tmux on QFRCAI | `tui` (TUI, auto-restarts; orchestrator refreshes it at each session's BookOnly hold), `live` (arena chain), `set5`/`deploy` were transient |
| Foreign arena | A-12 belongs to another user (shared server) — never touch it |

## 3. Price scale & market structure (verified)

- `reference_price = from_float(1000.0)` → 10,000,000 raw ⇒ **raw_per_price = 10,000**.
- `tick_size = 100` raw (= 0.01 display); `lot_size = 100` raw ⇒ **qty_per_unit = 100**
  (1 agent unit = 100 raw shares).
- Trader endowment: `trader_inventory = 2000` raw = 20 units (asks never
  `InsufficientInventory`-rejected until −20); exchange floor −20 = sandbox q_min.
- Cash parity: 1 unit = 1e9 raw at price 1000. `trader_cash = joiner_cash = 5e10`
  raw = 50 units of buying power (sandbox `initial_cash = 20000` display ≈ 2e10 raw;
  q_max=20 needs ≤2.01e10; 5e10 never binds). The old 1e10 (=10 units) silently
  strangled buyers at inventory +10 — the original stage-0 bug.
- `circuit_breaker_bps = 2000` (800–1200), fees 0, `max_order_qty = 50000`.
- **Server caps pre-allocated trader slots at 10 per arena.** Owner cannot join own
  arena; new identities need email-verified registration.

## 4. What exists in `arena_live/` (mirrored to QFRCAI:~/marl-experiments/arena_live/)

```
Config.toml  .env(QFRCAI only)  config/scenarios.yaml  HANDOFF.md  NOTES_DEROUNDING.md
joiners/common.py        # creds, submit_orders, CycleLogger, MarketSource,
                         # reject_name (RejectReason code->name map), final_checkpoint
                         # (glob newest *_final.pt), load_scenario (defaults +
                         # optional profile: merge)
joiners/obs_mapper.py    # raw md -> sandbox md (prices /1e4, volumes /100)
joiners/mm_joiner.py     # PPOHjbMarketMaker joiner. Cycle-0 ref_anchor_warmup:
                         # first decision sees synthetic 999/1001 book so the
                         # session anchors at ref (kills the 996 first-poster
                         # artifact). mm_lambda from cfg (HJB λ prior).
joiners/belief_joiner.py # BeliefTrader joiner. Acks logged (parity with MM);
                         # n_cash_rejects surfaced. churn: --rounds --spawn-id
                         # --bias --cycle-offset(gcycle). Checkpoint = slot % n_traders
                         # (churn slots exceed trained policies).
joiners/noise_generator.py # NoiseBot: Poisson(lam)/side market orders, gated on touch.
orchestrator.py          # lifecycle: create -> suspended -> book_only (TUI refresh +
                         # hold) -> trade -> 240-cycle master loop (noise + churn
                         # manager) -> destroy (always, finally). Paired biases
                         # (draw_paired_biases / PairedBiasQueue), biases.csv,
                         # couple_pairs (± pair spawns together, same lifetime).
run_remote.sh            # LOCAL launcher: ssh -> tmux 'live' chain; refuses if running
run_scenarios_live.sh    # QFRCAI-only (refuses locally); hold defaults to 10s
status_remote.sh         # LOCAL: one-glance status (orchestrator/arena/cycle/joiners)
watch_tui.sh / watch_tui_local.sh  # TUI on QFRCAI / attach from laptop (ClearAllForwardings)
analyze_live.py          # per-session analysis: capture vs F and mean(v), first_valid_mid,
                         # max/min, impact coef, reject totals, fairness ranks,
                         # rolling-mean overlay. gcycle-aligned for churn.
kill_arenas.py  mock_test.py  probe_lifecycle.py  probe_join.py
runs/<session>/          # QFRCAI: per-agent *_cycle_log.csv + *_ack_cycle_log.csv,
                         # noise_log.csv, churn_log.csv, biases.csv, analysis.png, summary.csv
plots_set4_live_g99/     # local: stage-1 plots (calm/bull/bear)
plots_set5_live_matched/ # local: stage-2 plots (calm/bull/bear/neutral)
```

## 5. Results

### 5a. Bug era (g99 live, pre-fix): calm 996.1→996.2; bull +31.9% (buyers stuck at +10); bear −410% (net-seller seed-42 draws). All superseded.

### 5b. Stage-1 certification (arenas 23–26, g99 ckpts, λ=1, 5e10 cash, paired biases)

| scenario | pre | post | mean_v | capture vs mean(v) | rejects |
|---|---|---|---|---|---|
| calm | 997.2 | 998.3 | 1000.0 | (no F move) | 2 non-cash |
| bull | 998.0 | 1002.1 | 1003.0 | +82.3% | 1 |
| bear | 993.4 | 995.8 | 997.0 | +67.7% | 1 |
| neutral (paired+churn) | 997.6 | 1001.7 | 1000.3 | (F flat) | 13 InsufficientInventory (at ±20 floor, benign) |

Checklist all pass: inventories reached ±20 (old wall +10), zero InsufficientCash,
acks logged, breaker never close, arenas destroyed, no strays.
Note: pre_mids ≈ 996 — the cold-start anchor artifact, fixed in 5c.

### 5c. Cold-start anchor fix (arenas 29–31)

Root cause: first MM to post saw an empty book and quoted around ref with
asymmetric cold κ (e.g. kb 0.15 / ka 1.37 → bid 993 / ask 1001); all others
re-anchored via `_get_mid` (prefers observed book over ref) → session anchored
~996, deterministically. Fix `ref_anchor_warmup: true` (scenarios.yaml default):
each MM's first decision uses synthetic ref-centered md. Validated: anchor
1000.0; TUI pane live; bull 999.8→1001.9, bear 999.8→996.3 (deep overshoot to
974 mid-run = thin-book violence, recovered). TUI: orchestrator now
kills+relaunches tmux `tui` at each BookOnly hold (stale-park fix; binary has
no auto-follow flag).

### 5d. Stage-2 matched population (arenas 32–36; set5 = 7 MM + 3 BT, noise λ=0.47, MM HJB λ prior 0.47)

Ratios vs sandbox set4: noise:depth 1.01x, informed:depth 1.07x, noise:informed 0.94x.
Training: `price_discovery/configs/market_reset/set5_matched_7x3_g99/` (generated
from set4 by trimming to 7+3, λ 1.0→0.47); launched by
`price_discovery/scripts/run_set5_matched_7x3_g99.sh` (180k steps ≈ 25 min on
QFRCAI CPU; then 30-ep evals ×3 + plots). Checkpoints: `~/checkpoints/matched_7x3_g99/{ppo_0..6,belief_0..2}/*_final.pt`.

Live direction check (mm-log mid, first-10 vs last-60 cycles):

| run | F | anchor → late | move | min | verdict |
|---|---|---|---|---|---|
| stage1 bull (g99) | 1003 | 1001.3 → 1002.0 | +0.65 | 998.3 | up, weak |
| stage1 bear (g99) | 997 | 999.4 → 996.3 | −3.12 | **973.6** | down, crash |
| stage2 bull (matched) | 1003 | 1000.0 → 1002.0 | **+1.98** | 984.5 | up, 3× stronger |
| stage2 bear (matched) | 997 | 998.3 → 995.8 | **−2.49** | **993.4** | down, no crash |

matched_neutral (churn, couple_pairs): anchor 1000, end 1001.9, mean_v_end 999.2,
range [959.5, 1006.5], 11 spawns, 0 rejects. Usable but not stationary (§5e).

In-flight bugs fixed: (a) churn slot → ckpt mapping is `slot % n_traders` (slot 3
tried `belief_3` → rc=1 crashes → unpaired survivors → first neutral blew to
947–1048; fixed + rerun as arena 36); (b) analyze_live mid/bid/ask matrices must
be gcycle-aligned for churn spawns (local-row contamination warped neutral mids);
(c) YAML: `profiles:` must sit AFTER the complete `defaults:` block (splitting it
swallowed `arena:`/`ref_price:` into profiles → KeyError 'arena').

### 5e. RESOLVED (2026-08-22) — bull/bear asymmetry root cause + fix

User requirement: bear ≈ mirror(bull); calm oscillates around F.

**Root cause (established from sandbox + training data, not assumptions):**
the set5 matched retrain (7 MM + 3 BT, λ=0.47) fell into a **learned
down-drift attractor in its final ~150 episodes**. Per-75-episode training
blocks: mid_move ≈ 0 through block 6, then −0.77 → −0.97 → **−2.00**; the
population κ gap (κ_bid−κ_ask) flips from +0.2 to **−0.28** after ~step 126k.
The learned policy is momentum-amplifying: κ gap = **−0.83 when short**
(deep bid / tight ask → sells more, never raises the bid) and +0.75 when
long. Mechanism in every scenario (traced tick-by-tick): beliefs with v
below the book sell first → MMs absorb → observed-mid anchoring ratchets
down → mid falls below all v → beliefs flip to buying but **pin at the +20
cap** → MMs restock the ask at the pinned level every tick → price stuck
below fair value (calm −2.9 sandbox; bear overshoots to −7.5; bull
underreacts +1.1). Ruled out by data: cash/inventory caps (0 rejects),
engine side-asymmetry (Rust source read — symmetric sweep, no position
enforcement), runner shuffler, belief inputs (Σv ≈ ΣF, verified), aggregate
training flow (balanced ±1.6%), harness (sandbox reproduces it), init
(gap flip happened at ~ep 600+, not at init). MMs start every episode at
inventory 0 (no carry-over artifact).

**The attractor is STRUCTURAL for the 7+3 roster, not seed luck; direction
is seed/config-dependent.** Full scoreboard — sandbox evals (30 eps each,
mid_move calm / bull / bear):

| config | seed | calm | bull | bear | verdict |
|---|---|---|---|---|---|
| λ=0.47 final | 42 | −2.92 | +1.10 | −7.50 | broken (down) |
| λ=0.47 | 43 | −3.28 | +0.58 | −3.63 | broken (down) |
| λ=0.47 | 44 | −1.27 | +1.88 | −4.17 | broken (down) |
| λ=0.47, φ/α ×5 (0.005/0.05) | 42 | −1.25 | +1.60 | −6.02 | broken (down) |
| λ=0.47, reward_mid_ema=20 (C2) | 42 | +2.97 | +7.90 | −1.10 | broken (UP) |
| λ=1.0 final | 42 | −0.20 | +2.75 | −2.45 | good |
| λ=1.0 | 43 | +6.78 | +10.40 | +1.17 | broken (UP) |
| λ=1.0 | 44 | +0.25 | +6.17 | −2.47 | weak (up-tilt) |
| **λ=0.47 @ ckpt 125000 (pre-flip)** | 42 | −0.60 | +3.47 | −3.43 | **good** |
| g99 15+6 λ=1.0 (set4) | 42 | −0.08 | +0.55 | −0.88 | good, weak capture |

7/8 matched-config runs fell into a drift attractor (4 down, 3 up). 5×
inventory penalties do NOT prevent it; λ=1.0 halves the risk but seed 43
still flipped UP; EMA-marking the reward flipped the attractor UP rather
than eliminating it. **No structural lever tried robustly prevents the
attractor at 7+3 — the reliable recipe is best-of-N-seeds selection or
pre-flip checkpoint selection, scored by sandbox symmetry metrics.**
g99 (15 MM) never fell in — depth/population size may be protective, but
the arena caps us at 10 slots (7+3).

**Fix deployed:** rollback population = matched training run @ step 125000
(pre-flip; ckpts exist every 5k steps). Symlink tree
`~/checkpoints/matched_7x3_g99_125k/{ppo_i,belief_j}/*_step_125000_final.pt`
→ real ckpts; profile `matched125k` + scenarios `m125k_{calm,bull,bear,neutral}`.
**Live validation (arena 40+, 2026-08-22): calm ends 1000.07 (F=1000), bull
1002.17 (F=1003), bear 997.65 (F=997) — residuals ±0.7, mirrored; 0 rejects.
Neutral ends 1003.55 (F=1000) — still noisy, secondary.** Plots local:
`plots_m125k_live/<scen>/`.

**Pending / next:** production population = **matched @ ckpt 125000**
(live-verified 2026-08-22). If a retrain is ever needed: train N seeds,
score every 5k-step ckpt by sandbox symmetry metrics (|calm move|,
bull/bear mirror, MM/BT end inventories), deploy the best — do NOT trust
final ckpts blindly (the attractor forms late). Alternate candidate:
`~/checkpoints/matched_7x3_g99_lam1` (λ=1.0, seed 42, sandbox-good) —
would need live validation with `noise_lambda: 1.0, mm_lambda: 1.0`.
`reward_mid_ema` flag added to PPOHjbMarketMaker (default 0 = unchanged;
C2 experiment — flipped the attractor up, do not use). Pre-flip ckpt dirs
for evals: `price_discovery/configs/market_reset/set5_ckpt{115000,125000}/`,
results `results_matched_7x3_g99_ckpt{115000,125000}_*`; variant runs:
`results_matched_7x3_g99_{s43,s44,ra1,lam1,lam1_s43,lam1_s44,emamark}_*`.
Bug fix en route: orchestrator.py no longer hardcodes scenario choices
(any yaml scenario name is accepted).

Historical evidence base (superseded by the resolution above, kept for
reference): the broken final-ckpt population showed, live vs sandbox eval —
calm −1.7 / −2.92; bull +1.98 / +1.10; bear −2.49 / −7.50; κ gap −0.48 /
−0.34; MM end inv −38…−58 / −32…−37; beliefs net BUY +22…+41 / +30…+36 in
ALL scenarios (incl. bear, where informed should sell — the tell); 0 rejects.

### 5f. Arena-vs-sandbox statistical validation (2026-08-22, ckpt125k) — superseded by 5g

End-mid, arena run vs sandbox 30-ep distribution: bull z=−0.37 (pct 47),
bear z=−0.26 (pct 53) — consistent; calm z=−4.66 (pct 3) — tail draw
(sandbox calm range is [982.5, 1003.5]; arena hit the exact min). Per-tick
sandbox p5–p95 envelope coverage: calm 50%, bull 91%, bear 88%. One arena
session = one draw; validating a distribution needs ~5 sessions/scenario.

**Why arena mid is more volatile than the local plots (measured):**
1. Local plots show the 30-ep MEAN — smooth by construction (calm mid_std
   0.46 vs single-episode 1.98).
2. Baseline tick noise is IDENTICAL (median |Δmid| = 0.0 in both).
3. The excess is rare glitch jumps: arena calm had an empty one-sided book
   on 8/240 cycles; sandbox 0/7200. Arena jumps ≥5 ticks: 3.75% vs 0.26%.
   Mechanism: 10 async joiner processes cancel+repost on jittered 1s clocks
   (sandbox: atomic per-tick shuffled flush) → book transiently one-sided →
   observed mid collapses (bid 966 / ask 0) → MM observed-mid anchoring
   re-quotes around the glitched mid → multi-cycle excursion.
   **Candidate fix (joiner-level, no training change): in the MM joiner,
   when md has an empty side, substitute the last valid two-sided mid
   before build_observation (do not anchor on a one-sided book).**
Also fixed en route: churn neutral now runs exactly 3 concurrent BTs
(slots/n_parallel 4→3, couple_pairs off) — composition matches the trained
roster (was running 4 with policy 0 duplicated; slot%3 mapping).

### 5g. 30v30 transfer validation (2026-08-26, ckpt125k) — INFERENCE TRANSFER FAILS on mid dynamics

**Design.** Sandbox: 30 eps/scenario (seeds 10000–10029, reproducibility
verified ep_0 identical). Arena: 30 sessions/scenario (`val2_*`, seeds
101–130), settings matched to sandbox (iid U{−3..3} biases both sides —
arena `bias_mode: random` replaces the old paired mode for validation;
`--seed` CLI override gives independent draws per session; session_meta
records it). Cadence 0.5s+jitter 0.25 (probe-verified: gaps p95 0.686 ≤ 0.75
cap, md_ok 99.6%, 0 rejects; glitch rates match 1.0s: one-sided 2.46%,
jumps≥5 3.39%). 90 sessions ≈ 5.5h sequential. New `md_log.csv` (orch
cycle-boundary book snapshot, top-5/side + taker counters, CSV-quoted
lob_json) enables LOB-depth comparison. Metrics computed identically both
sides (`validate_transfer.py`): mid from two-sided snapshots ff/backfilled;
spread on two-sided cycles only; fill_events = our agents' fill events
(noise–MM trades count once, belief–MM twice — same convention both sides);
flow_units = Σ|Δinventory| over our 10 agents; depth from snapshots.
MW = Mann-Whitney (tie+continuity corrected, brute-force validated), δ =
Cliff's. Stats helpers are numpy-only (no scipy in either remote venv).

**Verdict (MW p ≈ 0, |δ| 0.65–0.94, arena pct ranks ≤ 30):**
- mid path NOT validated in any scenario: mid_move arena vs sandbox =
  calm −10.9 vs −0.5, bull −7.8 vs +3.0, bear −13.1 vs −2.6. Arena drifts
  DOWN in all 30+30+30 sessions (end_mid range calm [888.5, 998.5]).
- Volatility NOT validated: mid_std ≈ 7–8 vs ≈ 2 (3.5x); min_mid strongly
  lower. Activity higher: fill_events +50%, flow_units +35%.
- VALIDATED: spread_mean (p 0.03–0.73, δ ≤ 0.32), imbalance (p 0.27–0.74),
  depth_within_5 (p 0.12–0.92). Weaker: best_qty/depth_1/2 slightly low
  (δ 0.4–0.77). Report: `validation2/validation_report.csv` (local copy),
  plots `validation2/validation_{calm,bull,bear}.png`.

**Attribution (measured, not assumed):**
- NOT cadence: 1.0s pilot (n=10 calm, mm-log mid) shows the same shift
  (end_mid mean 993.2, mid_std 5–8).
- NOT bias-draw luck: iid biases both sides; and 30/30 sessions shift down.
- NOT net informed direction: MM aggregate end inventory is NEGATIVE every
  session (mean −29; sandbox −0.7) ⇒ takers net-BOUGHT (beliefs +39) while
  price fell — price falls against net buying pressure.
- Exchange `taker_volume_buy/sell` counters are INCONSISTENT with our own
  inventory logs (claim 7:1 sell:buy; MMs short ⇒ net buy). Do not use them.
- Arena-only structural difference remains: 2.46% one-sided-book cycles vs
  0/7200 sandbox; glitch-jump mechanism per §5f (observed-mid anchoring
  amplifies collapses) is the documented candidate but does NOT explain the
  systematic DOWN direction of the drift. Mechanism unresolved.

**Next candidate experiment (do not run without user OK):** joiner-level
empty-book fix (§5f candidate: substitute last valid two-sided mid before
build_observation) + rerun a 10-session calm batch to isolate the glitch
contribution; if the down-drift persists, the cause is async state/obs
  differences, not glitches.

Runs inventory: `runs/val_*_s*` = 1.0s pilot (no md_log, 10/scen);
`runs/val2_*_s*` = 0.5s validation runs (30/scen, md_log present);
`runs/val3_calm_s*` = 10 calm sessions WITH the obs fix (below);
`runs/val2_probe*` = cadence probes. Sandbox reference dirs hold 100 eps;
the first 30 (seeds 10000–10029) are the validation reference.

### 5h. Obs-fix result, corrected mechanism, and the MarlV2Env retraining track (2026-08-27)

**Obs fix deployed + gate result.** Joiner-level fix shipped (one-sided book
→ substitute last valid two-sided md for OBSERVATION only; submissions still
gate on the real book; `obs_ffill` column logged; mm_joiner.py +
belief_joiner.py; belief touch state no longer zero-corrupted; mock tests
incl. new `test_one_sided_md_substitution`, ALL PASS local+remote). Calm
gate, 10 sessions (`val3_calm_s1..s10`, 0.5s): 9/10 now end 997.4–999.8
(was: systematic ~991 drift) but s10 collapsed to 972.5 (mid_move −29.4).

**Corrected mechanism (measured; supersedes §5g "wide-spread filter" idea,
which was WRONG).** Deep-bid ladders are legitimate policy output:
`_depths` (ppo_hjb_market_maker.py:210) computes quote depth = 1/κ capped
at 50, so κ_bid ≈ 0.02–0.05 ⇒ resting bids 20–50 below mid. Facts: mm_0
submitted a 970 bid while observing mid 1002.5; κ distributions MATCH
sandbox vs arena (κ_bid p5 0.028 vs 0.021; frac<0.06: 9.1% vs 11.1% —
inference faithful); 85% of SANDBOX snapshots contain a bid >5 below mid
(deepest 54) — the ladder exists on both sides. The difference is EXPOSURE:
phase3's atomic shuffled tick refills the top before takers can reach the
ladder (jumps ≥5: 0.26% of ticks); the arena's continuous clock lets a
belief-sized sell (up to 40 units vs ~6.7 units/side displayed) land in a
cancel→repost gap, sweep the top, and rest on the ladder (s10 c125: taker
sell 6, bids 999→969, two-sided-but-wide book, mid 984.5, MMs re-anchor
~968). Down-moves persist (absorption), up-moves reverse ⇒ systematic
drift. Training env never charges for ladder risk ⇒ policies park it.
User's diagnosis confirmed: sync shared-info ticks hid this; the arena is
continuous with information asymmetry.

**Decision (user-approved plan): stop patching, retrain in a stop-the-clock
env.** Route: fastboxex `MarlV2Env` (env-owned integer clock, per-agent
periodic scheduler with first-wake phase offsets, immediate event-driven
matching, phase3 StepCollector step()) — the user's original design.

**Phase 0 — build (DONE, 2026-08-27).**
- The MarlV2Env feature is UNCOMMITTED WIP on local `FastboxEx-Cli` branch
  `feat/marl-phase3-align` (tip = main 46bd932; per `plan_marl_v2.md` the
  working agreement is no-git — changes stay uncommitted for user review;
  DO NOT COMMIT). Files: src/sim/marl_v2.rs, src/py/marl_v2.rs (untracked),
  marl.py/marl_v2.py python wrappers, tests/test_marl_v2.py, docs guide.
- Fixes made (all additive): (a) protocol repo was stale v0.20.0 —
  fast-forwarded local `FastboxEx-Protocol` to origin/main v0.28.0 (client
  46bd932 needs its fields); (b) Cargo.toml pyo3 feature `abi3-py313` →
  `abi3-py312` (3.13 ABI cannot load in the 3.12 training venvs);
  (c) bindings bug: submit_* drained acks/fills for history, breaking the
  conn.drain_* contract — fixed with monotonic per-participant
  `events_pushed` counter + `peek_events_since` + per-env
  `recorded_cursors` (`record_action_from_queue`): history observes without
  consuming (at-most-once in history, exactly-once to the agent);
  (d) added `cancel_by_tag` binding (phase3/arena cancel convention).
- Wheels built local + QFRCAI (maturin, cp312-abi3). Tests: 53 Rust + 19
  Python PASS on both. Installed into QFRCAI sandbox venv
  `~/marl-experiments/.venv` (pip via ensurepip; --no-deps, additive —
  arena venv 2.4.0 UNTOUCHED). `.venv_fx` on QFRCAI = build-only venv
  (maturin). Speed: 26.8k ticks/s ⇒ 240s episode (24k ticks, 10 agents)
  ≈ 0.9 s wall.

**Phase 1 — trainer port (IN PROGRESS; smoke passes locally).**
New package `marl-experiments/async_marlv2/` (local + synced to QFRCAI):
- `adapter.py`: `MarlV2MarketAdapter` (phase3 Market surface: env_dt=10ms s,
  tick_count, sim_t, done, get_md→display units identical to
  arena obs_mapper keys) + `ConnAdapter` (display↔raw conversion, market
  orders touch-gated = arena parity, cancel_by_tag, fills converted
  display/units). Trading_agents run UNCHANGED on top.
- `runner.py`: `_Session` builds env (arena parity: tick 100 raw, lot 100,
  ref 1e6 raw, cash 5e10, inventory 2000, max_order_qty 50000, breaker
  2000bps) + agents from a phase3-style YAML (params verbatim from
  set5_ckpt125000 configs), per-agent random phase U{0..99} ticks, callbacks
  = agent.on_tick(sim_t). Eval: fresh env+agents per episode (phase3 eval
  parity), dumps pnl_history.csv/order_events.csv + agents' own
  checkpoints/*_lob_data.csv (explicit final-dump flush added — the
  agents' market.done dump sits behind their scheduler gate and phases
  misalign with the terminal tick). Train: continuous, env.reset+agent
  reset every 240 s (phase3 episode_ticks parity).
- `gate.py` + `configs/smoke.yaml`: smoke (2×5 s eps, random init) PASSED
  locally — fills at correct display prices, phased sim_t, all dumps
  present, lob/action history flushed.
- NEXT (not yet run): attribution gate on QFRCAI — generate eval config
  from set5_ckpt125000/eval_matched_calm.yaml (agents verbatim + ckpt125k
  paths), 30 eps, seeds 10000–10029, into e.g.
  `~/marl-experiments/async_marlv2/results_gate_calm/`. GATE CRITERION:
  ckpt125k must show the ARENA signature here (down-drift, elevated jumps)
  unlike the phase3 sandbox; else the env misses the exposure → stop and
  re-investigate before any training.

**Phase 1 gate — RUN and PASSED (2026-08-27).**
- BUG FOUND + FIXED before the gate: `runner.py REF_PRICE_RAW` was 1_000_000
  (100 display) while adapter RAW_PER_PRICE=10_000 ⇒ quotes at ~10M raw sat
  10× outside the ±2000bps band around the reference ⇒ the engine's circuit
  breaker halted the market into Auction ⇒ every subsequent limit rejected
  `WrongLifecycleState` (reject_reason=1, status=4). Symptom: first quote
  rests, book then drains, one fill total, dead market. Arena truth
  (scenarios.yaml): raw_per_price 10000 ⇒ REF_PRICE_RAW=10_000_000 for
  display 1000. Fixed in runner.py. LESSON: the local smoke passed on too
  weak a criterion ("a fill happened") — always assert market stays alive
  (fills > threshold, book two-sided at end, no rejected acks).
- Gate run: 30 eps calm, seeds 10000–10029, ckpt125k, 1 s decisions,
  `async_marlv2/results_gate_calm/`. Analyzer:
  `async_marlv2/analyze_gate.py` (numpy-only; per-ep end_mid/mid_move/
  mid_std/jump_frac/mean_spread/fills from mm_0's 1 s firing grid; same
  script run on the sandbox reference for apples-to-apples).
  - gate (30 eps): move_mean −2.3, median 0.0, min −61.5 (ep_7 collapse,
    fills 1723), frac_down 0.33, end_mid [938.5, 1010.0] mean 997.7,
    mid_std 5.79, jump% 3.93, fills 680.
  - sandbox ref (100 eps, same script): move_mean −0.3, min −17.5,
    frac_down 0.40, end_mid [982.5, 1005.5] mean 999.7, mid_std 2.17,
    jump% 0.41, fills 518.
  - arena val2 calm (30 sess, §5g): move_mean −10.9, 30/30 down, mid_std
    ~3.5× sandbox, jumps 2–4%, fills +50%.
  VERDICT: PASS. The async env exposes exactly what the sync env hid:
    jumps 9.6× (into the arena's 2–4% band), mid_std 2.7×, collapse tail
    (−61.5 vs −17.5 over 100 sandbox eps), down-drift mean 7.7× stronger.
    Magnitude ~1/5 of the arena — plausibly the arena's 0.5 s cadence (2×
    actuation rate) and per-cycle phase re-jitter amplify; the training env
    needs to CHARGE for ladder exposure, not match arena magnitude.
  Misc: episode wall-time ≈ 2 s (env+agents+ckpt inference) ⇒ 30-ep gate
  ≈ 1 min. order_events.csv records fills only (phase3 parity).

**Phase 2 — retrain (RUN; from-scratch route FAILED with a clean mechanism
diagnosis, 2026-08-27).**
- Setup: `async_marlv2/train.py` CLI + configs `train_matched_s{42,43,44}`
  (from set5_matched_7x3_g99 train templates; env → sim_duration 240;
  ckpt dirs `~/checkpoints/marlv2_matched_7x3_s{42,43,44}/<agent>/`;
  everything else phase3-verbatim incl. coordinator f 998–1002). Training
  smoke passed (episode resets, ckpts, exit 0). s42 completed 180k in
  89 min; s43 killed at ~25k; s44 never started after the failure was
  diagnosed. Scorer: `async_marlv2/score_ckpts.py` (calm sweep, 10 eps per
  5k ckpt; rewrite_ckpt maps step → per-agent files; bull/bear eval
  templates also generated).
- FAILURE MODE (measured, seed-universal): the from-scratch policy PINS the
  mid at 1000 forever. s42 bull ep_0: mid first/last/tail20 = 1000.0,
  belief_1 buys its +20 cap, belief_0/2 flat, mm_3 churns 157 fills, mid
  never moves. Every s42 ckpt 5k..180k gives IDENTICAL calm evals
  (move −0.60, mid_std 0.31, jumps 0.00%, fills ~322) — behaviorally the
  same policy despite differing weights (md5s differ; deterministic κ
  trajectories bit-identical). frac κ_bid<0.06 = 0.00 at every ckpt (old
  policy: 9.1%). RANDOM-INIT control also pins bull at 1000 (κ≈1.09 tight,
  no ladders); s43@10k identical.
- MECHANISM: random init outputs κ≈1 (tight quotes, single level per side,
  7 units of depth). Total informed pressure ≤ 60 units (3 BTs × q_max 20)
  is fully absorbed by 7 MMs re-anchoring on their own pinned observed mid
  ⇒ book variance collapses ⇒ agents' observations are near-constant ⇒ no
  learning signal ⇒ PPO never leaves the init basin. In this roster, price
  discovery was ONLY ever the deep-ladder sweep (old sandbox policy's small
  κ quotes resting 20–50 deep get swept, moving the observed mid). The
  async env charges for exactly that exposure — so from-scratch training
  kills discovery along with the collapse risk. Not seed luck; structural.
- Consequence: Phase 3 (val4 live validation of a retrained population) is
  ON HOLD pending a decision on the training route. Candidate routes:
  (a) warm-start fine-tune from matched ckpt125k in MarlV2Env (start WITH
  ladders/discovery alive; env penalizes the sweep exposure; 5k-ckpt
  scoring + pre-attractor selection per §5e);
  (b) stronger informed pressure during training (more/bigger BT flow) so
  pinning is penalized via inventory — train/test population shift;
  (c) reward shaping toward F-tracking — §5e history (reward_mid_ema C2
  flipped the attractor UP) says reward hacks are fragile;
  (d) keep old policy + obs fix (current live state) and stop.
- Artifacts: `/tmp/marlv2_score_s42_calm.log` (full 36-ckpt sweep table),
  `/tmp/marlv2_{bull_5000,bull_180000,bull_randinit,bull_s43_10k}` (1-ep
  bull diagnostics), checkpoints under `~/checkpoints/marlv2_matched_7x3_*`
  (kept for inspection), `results_train_s42/` (f-series only).

**Phase 2 fix + warm-start fine-tune (2026-08-27/28) — SUCCESS.**
- CRITICAL RUNNER BUG FOUND + FIXED: my run_train reset agents at episode
  boundaries; but transitions store on the firing AFTER the action (an
  episode stores 239, never 240) and phase3 never reset agents, so the PPO
  buffer starved at 239/240 and **update() never ran in ANY prior run**
  (zero `pol=` lines; from-scratch AND first fine-tune were no-ops; md5
  differences were optimizer/step-counter payload only). First fix attempt
  hit an aliasing trap (a.reset() clears buffer IN PLACE via pos=0; buffer
  contents persist) causing 1-sample updates → NaN. Correct fix in
  runner.py run_train: preserve `buffer.pos`, `_steps_in_buffer`, and
  pending `_last_z/_last_logprob/_last_value` across the per-episode reset.
  Verified: update fires once per episode, healthy diag, no NaN.
- Fine-tune v3 (warm start ckpt125k, 60k steps, seed 42): PPO live
  throughout. Bull/calm sweeps over all 12 ckpts (10 eps):
  - 5-10k: exposure priced in fast (bull tail −36.5 → −9.5, jumps 4.6 →
    2.0) but discovery flat (+0.2..+0.4).
  - 25k: transient broken state (min −47.5).
  - 40k-60k: §5e-style UP attractor (calm +5→+23, bull +14→+31, jumps
    8-15%). Attractor forms late — exactly as §5e documents.
- Selection (30 eps × 3 scenarios, seeds 10000-10029, same as gate):
  | ckpt | calm | bull | bear | min tail |
  |---|---|---|---|---|
  | 15000 | +0.85 | +1.88 | −3.68 | −20/−17.5/−17.5 |
  | 20000 | +5.12 | +1.38 | −1.43 | −26/−30/−17.5 |
  | 35000 | +2.40 | +6.37 | −1.77 | −25.5/−13.5/−23.5 |
  Targets 0/+3/−3 (sandbox original −0.6/+3.47/−3.43). **ft@15000 wins**
  (mean |Δ| 0.9): mirror holds (bear−bull = −5.6 vs −6.9 target), bear
  spot-on, bull +1.9 under-reacts (price of killing deep ladders), tails
  and jumps far better than old policy in-env (−36..−61, 3.9-4.6%).
- SELECTED POPULATION: `marlv2_ft125k_s42 @ step 15000`. Checkpoints:
  `~/checkpoints/marlv2_ft125k_s42/{ppo_i,belief_j}/*_step_15000.pt`.
  In-env 30-ep reference for val4 = the 30-ep runs behind the table above
  (rerun into persistent dirs when generating the comparison).
- Old broken-run artifacts kept: `~/checkpoints/marlv2_ft125k_s42_noupdate`
  (fine-tune with starved buffer), `marlv2_matched_7x3_s4{2,3}` (from-scratch
  no-ops; s43 partial, s44 never ran).
- Known simplification (documented, not yet an issue): agent phases are
  fixed per episode (arena re-jitters every cycle); intra-tick callback
  order is registration order on phase collisions (rare: 10 agents, 100
  slots).

**Phase 2 (queued): retrain** matched 7+3 in async_marlv2, 3 seeds, §5e-safe
recipe (score every 5k ckpt on symmetry metrics, pre-attractor selection).
**Phase 3 (queued): live validation** val4_* 30/scenario vs in-env 30-ep
reference via validate_transfer.py.

Still-open sub-items (unrelated to the resolved asymmetry):
- `impact_coef` OLS is confounded by mean-reverting flow (negative slopes in
  calm/bull) — metric needs an event-study version.
- Fairness: belief joiners systematically arrive LATER than MMs (mean rank
  ~7.5–8.6 vs expected 4.5; MMs ~2.9–3.0). Mechanism unknown (spawn/jitter).
- matched_neutral / m125k_neutral churn config: slots=4 / n_parallel=4 with
  only 3 trained policies (slot%3 mapping) — intentional 2-pair rotation,
  but document/check whether user wants exactly 3 concurrent BTs.
- neutral plot lists 11 spawn dirs (belief_s1..s11) in the PnL panel — that
  is churn rotation, not 11 concurrent traders.
- Neutral stationarity: m125k_neutral ended +3.5 above F (range
  [957.6, 1059.2]) — better than the broken population but not stationary.

## 6. How to run things (exact recipes)

Local prep loop: edit locally → `.venv/bin/python arena_live/mock_test.py` (from
repo root; mock tests live in arena_live/) → rsync → remote mock_test → launch.

Rsync code to QFRCAI (from local `arena_live/`):
```
rsync -av --exclude 'runs/' --exclude '.env' --exclude '__pycache__' \
  --exclude '.venv/' --exclude 'plots_*' --exclude 'logs/' --exclude 'Config.toml' \
  -e "ssh -J ZhongMa" ./ QFRCAI:~/marl-experiments/arena_live/
```

Launch live scenarios from laptop:
```
cd ~/Documents/marl-experiments/arena_live
bash run_remote.sh calm bull bear neutral        # default 10s TUI hold
bash run_remote.sh 30 matched_calm matched_bull  # explicit hold
bash status_remote.sh                            # is anything running?
bash watch_tui_local.sh                          # dashboard (Ctrl-b d detaches)
```

Sandbox retrain/eval on QFRCAI (pattern):
```
cd ~/marl-experiments/price_discovery
tmux new-session -d -s set5 "bash scripts/run_set5_matched_7x3_g99.sh 2>&1 | tee logs/set5_launch.log"
```

Analysis only (remote): `cd ~/marl-experiments/arena_live && .venv/bin/python analyze_live.py runs/<session>`.

MarlV2 stop-the-clock env (async training track, §5h):
```
# rebuild wheel after editing FastboxEx-Cli (local or QFRCAI):
cd ~/FastboxEx-Cli/client-api
PYO3_PYTHON=$HOME/marl-experiments/.venv/bin/python \
  $HOME/marl-experiments/.venv_fx/bin/maturin build -i $HOME/marl-experiments/.venv/bin/python --release
~/marl-experiments/.venv/bin/python -m pip install --force-reinstall --no-deps \
  ~/FastboxEx-Cli/target/wheels/fastboxex-3.7.1-cp312-abi3-*.whl
~/marl-experiments/.venv/bin/python -m pytest ~/FastboxEx-Cli/client-api/tests/test_marl_v2.py -q

# attribution gate (QFRCAI, sandbox venv has torch+wheel):
cd ~/marl-experiments
.venv/bin/python -m async_marlv2.gate async_marlv2/configs/eval_matched_calm.yaml \
  async_marlv2/results_gate_calm 30 10000
```
Pull plots only: rsync the two files per scenario into `arena_live/plots_set5_live_matched/<scen>/`.

## 7. Gotchas (each one bit us — do not relearn)

1. **Clock domains:** `t_start` is epoch; sleep against `time.time()`, never mix
   `time.monotonic()` (56-year sleep; SIGTERM can't break — use SIGKILL).
2. **Taker orders:** `submit_market` ONLY when the opposite touch exists.
   `submit_ioc` and marketable-limit+instant-cancel both fail live.
3. **MM limit orders:** cancel-by-tag → repost each cycle is the correct pattern;
   reject names readable in `mm_*_ack_cycle_log.csv`.
4. **Market data:** arena SUB delivers ~nothing; production relay (:12000) carries
   arena assets — MarketSource falls back automatically; `md_ok` intermittent
   (~half the cycles); agents fall back to last-known book.
5. **`pkill -f` footguns:** never inside a command whose own cmdline matches; TUI
   = `pkill -x fastboxex-tui`; joiners = `pkill -9 -f "joiners\."` from plain shell.
6. **`pgrep -f` self-match:** status_remote.sh's own ssh cmdline contains the
   pattern — use `[o]rchestrator`-style bracket patterns.
7. **Joiners run with CWD = arena_live/** (Config.toml + .env read from CWD).
8. **zsh heredoc/quoting:** `echo ===FOO===` inside ssh'd zsh → "==FOO=== not
   found" glob error; avoid bare `===` separators in remote one-liners.
9. **YAML:** never insert a new top-level key in the middle of `defaults:` (see 5d).
10. **Shared server:** A-12 is foreign; sweep only own arenas (`kill_arenas.py --yes`).
11. **TUI binary:** no auto-follow; stale after arena destroy; orchestrator now
    refreshes it at each hold. TUI FILLS column double-counts (fill + ack);
    generator PnL display is garbage (marks the 1M-share float) — both artifacts,
    inventories reconcile with our drains.
12. **belief joiner churn slots:** checkpoint index is `slot % n_traders`
    (slots can exceed trained policies). gcycle column aligns spawns for analysis.
13. **Old runs:** bug-era sessions archived as `runs/*_bugera` on QFRCAI;
    stage-1 fixed runs are `runs/{calm,bull,bear,neutral}_live`; stage-2 are
    `runs/matched_*_live`. `.arena_state.json` is deleted at session end —
    `session_meta.json` persists.
14. **Local-only scripts:** run_remote.sh / watch_tui_local.sh / status_remote.sh
    run FROM the laptop; run_scenarios_live.sh refuses to run off-QFRCAI.

## 8. Sandbox-era one-liners (context only)

- set4 γ=1.0: strong static discovery but ~7/15 MMs frozen; g99 γ=0.99: all alive,
  weaker discovery (bull +0.69 sandbox). Step-shock capture 33–53%; poisson
  tracking slope 0.15–0.2.
- g99 sandbox eval is nearly symmetric (calm −0.08) — it is the matched 7+3/λ=0.47
  retrain that introduced the asymmetry (§5e).
- Phase-1a-ii (q_max randomization/retrain) was REJECTED by the user — do not revive.
