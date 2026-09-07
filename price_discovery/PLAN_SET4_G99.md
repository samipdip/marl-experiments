# Set4-g99 — gamma=0.99 retrain (MM saturation / symmetry-break test)

> Self-contained instructions for later. Read top-to-bottom to resume this thread.

---

## RESULT (filled in after the run)

**Hypothesis CONFIRMED — saturation fully cured.** gamma=1.0 → 0.99:

| metric (step 180k, 15 MMs) | set4 (γ=1.0) | g99 (γ=0.99) |
|---|---|---|
| alive layer-2 units / 64 | mean **1.2**, std 1.7 (frozen/alive split) | **64/64 every MM**, std 0 |
| critic ‖W‖ (the runaway) | ~9.9 (up to 12.0) | **0.85** (bounded, ≈ init) |
| ‖z2‖ (pre-activation) | ~6 (saturated tanh) | **~0.4** (healthy) |
| per-MM κ_bid unique values in eval | 7 MMs ≤ 8 (frozen) | **232–239 / 240 every MM** (all responsive) |

The critic stopped running away, all 15 MMs are healthy and *identical* (std 0 → symmetry preserved,
exactly as predicted: stable dynamics erase the init differences instead of amplifying them).

**BUT discovery became WEAKER** (mean mid over episode, F target in parens):

| scenario (F) | set4 γ=1.0 | g99 γ=0.99 |
|---|---|---|
| calm (1000) | 1000.20 | 1000.07 |
| bull (1003) | 1001.93 | **1000.69** |
| strongbull (1005) | 1004.57 | **1001.19** |
| bear (997) | 997.33 | 999.02 |
| bull+bear mirror | 1999.27 | 1999.71 |

Interpretation: with properly-learning MMs the book is deeper / more competitive → informed flow is
absorbed more → the mid moves less toward F. **set4's "good discovery" was partly an artifact of the
degraded/frozen MMs** (weaker liquidity → informed flow walked the mid more). Symmetry is intact
(bull+bear ≈ 2000 in both). 

Attribution (bull, F=1003, 30 eps): belief-trader fills g99 **2824** vs set4 **3572** (~21% fewer),
but the mid moved 0.69 vs 1.93 (**64% less**). So per-fill price impact roughly *halved* — the
dominant cause is the deeper/competitive MM book; secondary, the γ=0.99 BeliefTraders trade a bit
less. Practical upshot: to get price discovery with *healthy* MMs, the informed flow must be
stronger / less capped (BeliefTraders are q_max=20, trained on static F).

Plots local: `plots_set4_6tr_coord_mtm-g99/{calm,bull,strongbull,bear}/` (eval_avg, belief_actions,
ppo_hjb_avg, fills_avg). Remote checkpoints: `QFRCAI:~/checkpoints/beliefs_pop_240_g99/`.

Note on a config bug fixed mid-run: the g99 configs originally still pointed `results_dir`/
`checkpoint_dir` at `beliefs_pop_240` (only gamma was changed on copy), so the first train wrote g99
weights into the `beliefs_pop_240` dir and the run aborted before evals. Fixed by repointing paths to
`beliefs_pop_240_g99` and renaming the rescued weight dir — **no retrain needed**; weights were
valid γ=0.99. (The γ=1.0 baseline is safe on ZhongMa `~/checkpoints/beliefs_pop_240/`.)

---

## 0. Why this run exists

In set4 (gamma=1.0, mtm-reward) ~7 of the 15 PPO HJB market makers **froze to a
constant κ** in eval. Root cause found directly from the checkpoint weights (not
speculation). The fix to test: `gamma: 0.99` (config-only), retrain everything.

### The mechanism (measured from `~/checkpoints/beliefs_pop_240/` on the remotes)
- **All 15 MMs develop layer-2 tanh saturation over training.** Alive (non-saturated)
  layer-2 units, across the 15 nets:

  | step | mean | std | min–max |
  |------|------|------|---------|
  | 5k   | 64.0 | 0.0  | 64–64  (all healthy, identical) |
  | 50k  | 27.8 | 8.6  | 8–47   (spread) |
  | 180k | 1.2  | 1.7  | 0–6    (most frozen; a few retain 4–6) |

- **What grows is the weight matrices, driven by the critic.** Layer-2 bias stays
  small (`b2_max`≈0.35); `‖W2‖` 4.6→10.0; **critic-head `‖Wcrit‖` 0.79→11.99 (×15)**.
  The saturating `z2` is the `W2·h1` term, not the bias.
- **Causal chain (code: `ac_ppo.py`, `ppo_model.py`):** `gamma=1.0` → GAE returns are
  **undiscounted** sums over 240-step episodes → large value targets. Advantages are
  normalized (`ac_ppo.py:124`) but the **value loss is not** (`ac_ppo.py:158`) → big
  value-loss gradients. Actor & critic **share the trunk** (`ppo_model.py:52`) → those
  gradients flow into `shared.2`. Adam, **no weight decay** (`kyle_base.py:83`),
  grad-clip 0.5 → no explosion but monotonic drift up → tanh saturates → units die →
  actor mean frozen. **tanh is the victim, not the cause.**
- **Why some MMs and not others (the symmetry question):** the 15 nets are
  **independently initialized** — `‖W2_i − W2_j‖≈6.6 > ‖W2‖≈4.6` (near-orthogonal). The
  game is symmetric (random `OrderShuffler` kills only *time* priority; price priority
  remains, `r((κbid+κask), fills)=0.956`), but the *learners* are independent stochastic
  trajectories. Because the dynamics are **unstable/runaway (non-convergent)**, init
  differences are **amplified, not erased** → each net's units cross the saturation
  threshold at different times → some end at 0 alive (frozen), some retain ~4–6 (responsive).
  It is init-lottery × instability — **not** matching, priority, or N. Confirmed
  N-independent: 5-MM runs saturate *faster* (61/64 @ step40k) than 15-MM (6–24/64 @ step40k),
  because fewer MMs → more fills/MM → bigger undiscounted returns → faster critic growth.

### The hypothesis this run tests
`gamma=0.99` discounts the 240-step returns → bounds the value targets → removes the
critic's runaway gradient → dynamics become **convergent**. Prediction: the 15 MMs
should **not** diverge into frozen-vs-alive; they should saturate together or stay
alive together. Everything else is identical to set4 (single-variable change).

---

## 1. What changed vs set4

**Only `gamma: 1.0 → 0.99`** in every agent (15 MMs + 6 BeliefTraders = 21 occurrences).
`gae_lambda` stays 1.0. Roster, mtm-reward, coordinator, arrival, HJB-static, seeds — all
identical to `set4_beliefs_240/`. So this is a controlled single-variable test.

---

## 2. Setup (paths & names)

| thing | value |
|---|---|
| Config dir | `configs/market_reset/set4_beliefs_240_g99/` (train + 4 eval) |
| Run script | `scripts/run_set4_beliefs_240_g99.sh` (`EXP=beliefs_pop_240_g99`) |
| Pull script | `scripts/pull_set4_g99.sh` |
| Remote | **QFRCAI** (reached via `ssh -J ZhongMa QFRCAI` — faster than local→QFRCAI) |
| tmux session | `set4g99` |
| Remote checkpoints | `QFRCAI:~/checkpoints/beliefs_pop_240_g99/` |
| Remote results | `QFRCAI:~/marl-experiments/price_discovery/results_beliefs_pop_240_g99_{calm,bull,strongbull,bear}/` |
| Remote log | `QFRCAI:~/marl-experiments/price_discovery/logs/set4_g99.log` |
| Local plots (grouped) | `plots_set4_6tr_coord_mtm-g99/{calm,bull,strongbull,bear}/` |

Pre-flight passed on QFRCAI: mtm-reward fix present (`_prev_mid`), belief agents present,
all 21 gammas = 0.99, imports + config load OK, framework `OrderShuffler` (good version),
torch 2.12 / numpy 2.5.

---

## 3. Monitor (from local)

```bash
# tail the log
ssh -J ZhongMa QFRCAI "tail -f ~/marl-experiments/price_discovery/logs/set4_g99.log"
# attach to tmux (detach with Ctrl-b d)
ssh -J ZhongMa QFRCAI "tmux attach -t set4g99"
# quick status
ssh -J ZhongMa QFRCAI "tail -n 20 ~/marl-experiments/price_discovery/logs/set4_g99.log; tmux ls"
```
Done when the log prints `TRAIN + 4 EVALS + PLOTS COMPLETE`.
Timing ≈ 16 min train + ~3 min × 4 evals + plots ≈ 30–40 min total.

---

## 4. Pull plots (local, after run finishes)

```bash
bash ~/Documents/marl-experiments/price_discovery/scripts/pull_set4_g99.sh
# -> plots_set4_6tr_coord_mtm-g99/{calm,bull,strongbull,bear}/*.png
```
PNG only (rsync rule). Each scenario dir should contain: `eval_avg.png`,
`belief_actions.png`, `ppo_hjb_avg.png`, `fills_avg.png` (+ later: `eval_ep{0,1,2}.png`).

---

## 5. After it finishes — VERIFY the hypothesis (re-measure saturation)

This is the whole point. Recompute alive-units across the 15 MMs on the **g99**
checkpoints and compare to set4 (gamma=1.0). Run on QFRCAI via proxy:

```bash
ssh -J ZhongMa QFRCAI '~/marl-experiments/.venv/bin/python -c "
import torch
CK=\"/home/samip/checkpoints/beliefs_pop_240_g99\"
base=torch.tensor([0.0,0.5,1,1,0,0.0,0.5,0.8,0.8],dtype=torch.float32)
import numpy as np
A=[]
for i in range(15):
    m=torch.load(f\"{CK}/ppo_{i}/ppo_hjb_mm_{i}_step_180000.pt\",map_location=\"cpu\",weights_only=False)[\"model_state_dict\"]
    h1=torch.tanh(m[\"shared.0.weight\"]@base+m[\"shared.0.bias\"])
    h2=torch.tanh(m[\"shared.2.weight\"]@h1+m[\"shared.2.bias\"])
    A.append(int((h2.abs()<0.99).sum()))
import numpy as np
A=np.array(A)
print(\"alive-units g99:\",list(A))
print(\"mean %.1f  std %.1f  min %d max %d\"%(A.mean(),A.std(),A.min(),A.max()))
print(\"critic ||Wcrit|| per net:\", [round(float(torch.load(f\"{CK}/ppo_{i}/ppo_hjb_mm_{i}_step_180000.pt\",map_location=\"cpu\",weights_only=False)[\"model_state_dict\"][\"critic_head.weight\"].norm()),2) for i in range(15)])
"'
```

**set4 (gamma=1.0) baseline for comparison:** alive-units mean **1.2**, std **1.7**, min 0 max 6;
`‖Wcrit‖` up to ~12.0.

**Interpretation:**
- **Hypothesis CONFIRMED** if g99 has substantially more alive-units (mean ≫ 1.2) AND
  much lower across-net spread (std ≪ 1.7) AND `‖Wcrit‖` stays small — i.e. the 15 MMs
  behave similarly (no frozen/alive split).
- **Also check discovery still works:** eval mids should still track F (calm ≈1000,
  bull ≈1002, strongbull ≈1004, bear ≈997), mirroring set4 (bull+bear ≈1999.3).
  If gamma=0.99 fixes saturation but breaks discovery, that's a different (interesting) finding.
- **Hypothesis REJECTED** if g99 still saturates like set4 → the runaway isn't (only)
  the return scale; pursue the deeper fixes below.

---

## 6. If gamma=0.99 does NOT fix it — deeper fixes

All of these are in **allowed** files (NOT `phase3_multi_agent/`, NOT `kyle_base.py`):

1. **Normalize value targets** in `ac_ppo.py:158` (standardize `mb_ret` like `adv` is
   standardized at line 124). Most likely single fix — directly removes the oversized
   critic gradient.
2. **Split actor/critic trunks** in `ppo_model.py` so the critic's gradient can't grow
   the actor's saturating layer.
3. **Lower LR** (`lr: 0.0003 → 0.0001`, config-only) — slows the drift.

(Weight decay would need `kyle_base.py:83`, which is **off-limits** — skip.)

Re-run the §5 probe after any of these.

---

## 7. Folder structure (local `price_discovery/`)

```
plots_set4_6tr_coord_mtm-g99/        ← THIS run (gamma=0.99), grouped
  {calm,bull,strongbull,bear}/
plots_set4_6tr_coord_mtm-reward/     ← set4 gamma=1.0 baseline (the frozen-MM run), grouped
  {calm,bull,strongbull,bear}/
configs/market_reset/set4_beliefs_240_g99/   ← g99 configs
configs/market_reset/set4_beliefs_240/       ← original gamma=1.0 configs
scripts/run_set4_beliefs_240_g99.sh          ← g99 run (train+evals+plots)
scripts/pull_set4_g99.sh                     ← g99 plot puller (proxy)
```
