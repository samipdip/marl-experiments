# Plan: 2D Kappa PPO HJB MM + Full Experiment Matrix

## Overview

Modify `PPOHjbMarketMaker` to learn **two independent kappas** (`kappa_bid`, `kappa_ask`) instead of one. This matches the analytical `HjbMarketMaker` structure (separate `kappa_plus` / `kappa_minus`). The PPO learns an additive adjustment `1/kappa` on each side of the depth formula, on top of the frozen HJB value function `h(t,q)`.

All experiments are re-run from scratch because old checkpoints are architecturally incompatible (actor head changes from `Linear(64,1)` to `Linear(64,2)`).

---

## Part 1: Code Changes

### 1a. `ppo_hjb_market_maker.py`

**`__init__` changes:**

| Parameter | Old (1D) | New (2D) |
|-----------|----------|----------|
| `action_dim` | `1` | `2` |
| `obs_dim` | `7` | `8` |
| `action_min` | `kappa_min` (scalar) | `[kappa_min, kappa_min]` (list) |
| `action_max` | `kappa_max` (scalar) | `[kappa_max, kappa_max]` (list) |

Forward these to `KyleRLAgent.__init__`:
```python
KyleRLAgent.__init__(
    self, name=name,
    obs_dim=obs_dim,                  # 8
    action_dim=action_dim,            # 2
    action_min=[kappa_min]*action_dim,
    action_max=[kappa_max]*action_dim,
    ...
)
```

Track two separate previous kappas:
```python
self._last_kappa_bid = kappa_ref
self._last_kappa_ask = kappa_ref
```

**`_depths` method — accept two kappas:**
```python
def _depths(self, step, q, kappa_bid, kappa_ask):
    ...
    if q < self.q_max:
        bid_depth = min(1.0 / kappa_bid + hq - h_next, 50.0)
    if q > self.q_min:
        ask_depth = min(1.0 / kappa_ask + hq - h_prev, 50.0)
    return bid_depth, ask_depth
```

**`action_to_orders` — unpack 2D action:**
```python
def action_to_orders(self, action, sim_t, md):
    if hasattr(action, '__len__'):
        kappa_bid = max(self.kappa_min, min(float(action[0]), self.kappa_max))
        kappa_ask = max(self.kappa_min, min(float(action[1]), self.kappa_max))
    else:
        kappa_bid = kappa_ask = max(self.kappa_min, min(float(action), self.kappa_max))
    self._last_kappa_bid = kappa_bid
    self._last_kappa_ask = kappa_ask
    bid_depth, ask_depth = self._depths(self._decision_step, q_lots, kappa_bid, kappa_ask)
    ...
```

**`build_observation` — 8-dim (add `kappa_ask_prev`):**
```python
obs = np.array([
    float(q_norm),        # 1
    float(spread_norm),   # 2
    float(lambda_ratio),  # 3
    float(mid_vol),       # 4
    float(imbalance),     # 5
    float(t_norm),        # 6
    float(self._last_kappa_bid),  # 7 (was kappa_prev)
    float(self._last_kappa_ask),  # 8 (NEW)
], dtype=np.float32)
```

**`get_metrics` — return both kappas:**
```python
def get_metrics(self):
    return {
        ...
        "kappa_bid": self._last_kappa_bid,
        "kappa_ask": self._last_kappa_ask,
        "pnl_history": self._pnl_history,
        "order_events": self._order_events,
    }
```

**Override `_save_action_history` — write 2 columns:**
```python
def _save_action_history(self):
    ckpt_dir = self.checkpoint_dir if self.checkpoint_dir else os.path.join(self.results_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    path = os.path.join(ckpt_dir, f"{self.name}_action_history.csv")
    with open(path, "w") as f:
        f.write("step,kappa_bid,kappa_ask\n")
        for step, action in self._action_history:
            if hasattr(action, '__len__'):
                f.write(f"{step},{float(action[0])},{float(action[1])}\n")
            else:
                f.write(f"{step},{float(action)},{float(action)}\n")
    print(f"  action history saved: {path}")
```

**`reset` — reset both kappas:**
```python
def reset(self):
    ...
    self._last_kappa_bid = self.kappa_ref
    self._last_kappa_ask = self.kappa_ref
```

**`_solve_hjb` — NO CHANGE.** Still solved once at init with single `kappa_ref`. The learned kappas only enter the depth formula (additive adjustment interpretation).

### 1b. `plot_ppo_hjb_avg.py` — update for 2D kappa

Read `kappa_bid` and `kappa_ask` columns:
```python
for row in csv.DictReader(f):
    t = int(row["step"])
    if t < n_steps:
        kappa_bid[ep, t] = float(row["kappa_bid"])
        kappa_ask[ep, t] = float(row["kappa_ask"])
```

Plot both with ±1σ bands (left panel), beta or theta on right panel.

### 1c. Files NOT changed

- `kyle_base.py` (KyleRLAgent) — DO NOT MODIFY
- `ppo_model.py` (KyleActorCritic) — already supports multi-dim actions
- `kyle_liquidity_trader.py` — already has sell_only support
- `kyle_informed_trader.py` — already has buy_only/fair_value
- `hjb_market_maker.py` (analytical) — already has kappa_plus/kappa_minus
- Factory / runner — no changes needed

---

## Part 2: Experiment Matrix

All experiments use: 3 analytical HJB MMs + 3 PPO HJB MMs (2D) + Poisson noise.

| # | Experiment | IT | LT | PPO HJB | Fair Value | Checkpoint Dir |
|---|-----------|----|----|---------|------------|----------------|
| 1 | IT vs MMs | Buy (train) | — | Train (2D) | v=1003 | `~/checkpoints/exp2d_it_v1003/` |
| 2 | LT vs MMs | — | Sell (train) | Train (2D) | — | `~/checkpoints/exp2d_lt/` |
| 3 | IT+LT opposite | Buy (train) | Sell (train) | Train (2D) | v=1003 | `~/checkpoints/exp2d_opp_v1003/` |
| 4 | IT+LT same | Sell (train) | Sell (train) | Train (2D) | v=997 | `~/checkpoints/exp2d_same_v997/` |

### Experiment Details

**Exp 1: IT vs MMs (v=1003)** — baseline validation
- IT: `buy_only=true, fair_value=1003, policy=linear`
- LT: not present
- Purpose: validate 2D architecture reproduces known dip-then-rise pattern. Compare kappa_bid vs kappa_ask divergence (ask should widen more as IT attacks).

**Exp 2: LT vs MMs** — sell-side pressure
- IT: not present
- LT: `sell_only=true, q_target=20`
- Purpose: test asymmetric quoting under sell pressure. Expect kappa_bid to drop (wider bid, protect against LT) while kappa_ask stays tight.

**Exp 3: IT+LT opposite (v=1003)** — tug-of-war
- IT: `buy_only=true, fair_value=1003` (buys, pushes mid up)
- LT: `sell_only=true, q_target=20` (sells, pushes mid down)
- Purpose: opposing pressures. Does mid discover v=1003? Do MMs learn differentiated kappas (ask widens for IT, bid widens for LT)?

**Exp 4: IT+LT same direction (v=997)** — amplified sell pressure
- IT: `buy_only=false, fair_value=997` (sells when mid > 997)
- LT: `sell_only=true, q_target=20` (sells)
- Purpose: both push mid down. MMs face sell pressure from both informed and liquidity traders. Expect both kappa_bid and kappa_ask to drop (defensive on both sides), but kappa_bid more so.

---

## Part 3: Config Files

All configs follow the existing naming convention in `price_discovery/`.

### Training configs

| File | Agents (trainable marked *) |
|------|---------------------------|
| `train_exp2d_it_v1003.yaml` | noise + 3 HJB + 3 PPO HJB* + IT* |
| `train_exp2d_lt.yaml` | noise + 3 HJB + 3 PPO HJB* + LT* |
| `train_exp2d_opp_v1003.yaml` | noise + 3 HJB + 3 PPO HJB* + IT* + LT* |
| `train_exp2d_same_v997.yaml` | noise + 3 HJB + 3 PPO HJB* + IT* + LT* |

### Eval configs

| File | Mode |
|------|------|
| `eval_exp2d_it_v1003.yaml` | All eval, load checkpoints |
| `eval_exp2d_lt.yaml` | All eval, load checkpoints |
| `eval_exp2d_opp_v1003.yaml` | All eval, load checkpoints |
| `eval_exp2d_same_v997.yaml` | All eval, load checkpoints |

### PPO HJB MM params (in all configs)

```yaml
- id: ppo_hjb_mm_0
  type: PPOHjbMarketMaker
  train: true
  eval: true
  params:
    # ... same as before ...
    obs_dim: 8           # was 7
    action_dim: 2        # was 1 (implicit)
    kappa_min: 0.01
    kappa_max: 2.0
    kappa_ref: 0.8       # PDE solved with this
    entropy_coef: 0.02   # was 0.01 (doubled for 2D exploration)
    n_steps: 40
    total_timesteps: 20000
    checkpoint_freq: 5000
    # ...
```

### Training hyperparameters

| Parameter | Old (1D) | New (2D) | Reason |
|-----------|----------|----------|--------|
| `entropy_coef` | 0.01 | 0.02 | Compensate for halved per-dim entropy bonus |
| `total_timesteps` | 20000 | 20000 | Keep same; monitor convergence |
| `sim_duration` | 25000.0 | 25000.0 | Keep same |
| `n_steps` | 40 | 40 | Keep same |
| `lr` | 3e-4 | 3e-4 | Keep same |

If convergence is poor after first experiment, bump `total_timesteps` to 30000.

---

## Part 4: Plot Scripts

### Updated `plot_ppo_hjb_avg.py`

Read 2D action history:
```python
kappa_bid = np.full((n_episodes, n_steps), np.nan)
kappa_ask = np.full((n_episodes, n_steps), np.nan)

for row in csv.DictReader(f):
    t = int(row["step"])
    if t < n_steps:
        kappa_bid[ep, t] = float(row["kappa_bid"])
        kappa_ask[ep, t] = float(row["kappa_ask"])
```

Plot:
- **Left panel**: `kappa_bid` (blue) and `kappa_ask` (red) with ±1σ bands
- **Right panel**: IT beta or LT theta (depending on experiment)

### `plot_eval_avg.py`

Already updated for LT support (orange line). No further changes needed — it reads from `pnl_history.csv` which already includes all agents.

### `plot_lt_avg.py`

Already created. No changes needed.

### `plot_beta_avg.py`

No changes needed (reads IT action history, unaffected by MM architecture).

---

## Part 5: Execution Workflow

For each experiment, follow the established pattern:

### Step 1: Rsync to remote

```bash
# Agent code
rsync -avz -e ssh \
  ~/Documents/Trading_agents/Agents_phase2/RL_Agents/ABIDES_MARL/ppo_hjb_market_maker.py \
  QFRCAI_Local:~/Trading_agents/Agents_phase2/RL_Agents/ABIDES_MARL/

# Configs
rsync -avz -e ssh \
  ~/Documents/marl-experiments/price_discovery/train_exp2d_*.yaml \
  ~/Documents/marl-experiments/price_discovery/eval_exp2d_*.yaml \
  QFRCAI_Local:~/marl-experiments/price_discovery/

# Plot scripts
rsync -avz -e ssh \
  ~/Documents/marl-experiments/price_discovery/plot_ppo_hjb_avg.py \
  ~/Documents/marl-experiments/price_discovery/plot_eval_avg.py \
  ~/Documents/marl-experiments/price_discovery/plot_lt_avg.py \
  ~/Documents/marl-experiments/price_discovery/plot_beta_avg.py \
  QFRCAI_Local:~/marl-experiments/price_discovery/
```

### Step 2: Train on remote

```bash
ssh QFRCAI_Local "cd ~/marl-experiments && nohup .venv/bin/python -c \"
import sys, os
sys.path.insert(0, os.path.expanduser('~/SimExchangeSandbox'))
sys.path.insert(0, os.path.expanduser('~/Trading_agents'))
from phase3_multi_agent.src.runner import run_simulation
run_simulation('price_discovery/train_exp2d_it_v1003.yaml', device='cpu', seed=42, n_ticks=25000)
\" > ~/marl-experiments/price_discovery/train_exp2d_it_v1003.log 2>&1 & echo pid=\$!"
```

Wait for `=== Done ===` in log.

### Step 3: Eval on remote (30 episodes)

```bash
ssh QFRCAI_Local "cd ~/marl-experiments && for ep in \$(seq 0 29); do
  seed=\$((10000 + ep))
  dir=~/price_discovery/results_exp2d_it_v1003/ep_\${ep}
  mkdir -p \$dir
  .venv/bin/python -c \"
import sys, os
sys.path.insert(0, os.path.expanduser('~/SimExchangeSandbox'))
sys.path.insert(0, os.path.expanduser('~/Trading_agents'))
from phase3_multi_agent.src.runner import run_simulation
run_simulation('price_discovery/eval_exp2d_it_v1003.yaml', device='cpu', seed=\${seed}, results_dir='\${dir}')
\"
done"
```

### Step 4: Plot on remote

```bash
ssh QFRCAI_Local "cd ~/marl-experiments && .venv/bin/python plot_eval_avg.py ~/price_discovery/results_exp2d_it_v1003/ 40"
ssh QFRCAI_Local "cd ~/marl-experiments && .venv/bin/python plot_ppo_hjb_avg.py ~/price_discovery/results_exp2d_it_v1003/ 40"
# Plus plot_beta_avg.py if IT present, plot_lt_avg.py if LT present
```

### Step 5: Pull PNGs only

```bash
mkdir -p ~/Documents/marl-experiments/price_discovery/results_exp2d_it_v1003
rsync -avz -e ssh \
  QFRCAI_Local:~/price_discovery/results_exp2d_it_v1003/*.png \
  ~/Documents/marl-experiments/price_discovery/results_exp2d_it_v1003/
```

### Execution Order

1. **Exp 1 (IT vs MMs)** — run first to validate 2D architecture
2. Check results: does kappa_ask diverge from kappa_bid? Does dip-then-rise pattern persist?
3. **Exp 2 (LT vs MMs)** — test asymmetric quoting under sell pressure
4. **Exp 3 (IT+LT opposite)** — tug-of-war
5. **Exp 4 (IT+LT same)** — amplified sell pressure

Each experiment takes ~5 min training + ~10 min eval + ~1 min plotting = ~16 min total.

---

## Part 6: Expected Results

### Exp 1: IT vs MMs (validation)
- `kappa_ask` should DECREASE over time (MM widens ask to protect against IT buying)
- `kappa_bid` should stay near `kappa_ref` or slightly increase (no adverse selection on bid)
- Dip-then-rise pattern should persist (driven by PDE's `h(q)`, not kappa)
- Comparison with 1D results: 2D should show TIGHTER spreads (bid stays competitive)

### Exp 2: LT vs MMs
- `kappa_bid` should DECREASE (MM widens bid to protect against LT selling)
- `kappa_ask` should stay near `kappa_ref` or slightly increase
- Mid should dip slightly below 1000 (LT sell pressure)
- LT should achieve ~95% completion (19/20 shares)

### Exp 3: IT+LT opposite
- BOTH `kappa_bid` and `kappa_ask` should decrease (both sides face adverse flow)
- But `kappa_ask` more so early (IT starts buying immediately), `kappa_bid` catches up (LT sells steadily)
- Mid should stabilize near v=1003 (IT pulls up, LT pulls down, net effect ~zero)
- This is the "price discovery" experiment: does the mid find the fair value?

### Exp 4: IT+LT same direction
- BOTH `kappa_bid` and `kappa_ask` should decrease significantly
- `kappa_bid` more aggressive (LT hits bid + IT sells when mid > 997)
- Mid should drop below 997 (both pushing down)
- MMs accumulate long inventory (buying from both IT and LT) → `h(q)` adjusts depths further

---

## Part 7: Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| 2D PPO doesn't converge | Increase `total_timesteps` to 30000, `entropy_coef` to 0.03 |
| `kappa_bid` and `kappa_ask` collapse to same value | Increase `entropy_coef`, check observation includes both prev kappas |
| Old checkpoint loaded by mistake | New configs use different checkpoint paths (`exp2d_*`) |
| Plot script crashes on missing columns | Check CSV header before parsing |
| `action_to_orders` receives scalar instead of array | Handle both cases with `hasattr(action, '__len__')` |

---

## Part 8: File Summary

### New/Modified Files

| File | Action |
|------|--------|
| `ppo_hjb_market_maker.py` | Modify: 2D action, 8-dim obs, dual kappa depths |
| `plot_ppo_hjb_avg.py` | Modify: read `kappa_bid`, `kappa_ask` columns |
| `train_exp2d_it_v1003.yaml` | New: Exp 1 training |
| `eval_exp2d_it_v1003.yaml` | New: Exp 1 eval |
| `train_exp2d_lt.yaml` | New: Exp 2 training |
| `eval_exp2d_lt.yaml` | New: Exp 2 eval |
| `train_exp2d_opp_v1003.yaml` | New: Exp 3 training |
| `eval_exp2d_opp_v1003.yaml` | New: Exp 3 eval |
| `train_exp2d_same_v997.yaml` | New: Exp 4 training |
| `eval_exp2d_same_v997.yaml` | New: Exp 4 eval |

### Unchanged Files

- `kyle_base.py`, `ppo_model.py`, `kyle_liquidity_trader.py`, `kyle_informed_trader.py`
- `hjb_market_maker.py`, `uniform_market_makers.py`, `poisson_noise_trader.py`
- `plot_eval_avg.py` (already updated), `plot_lt_avg.py` (already created), `plot_beta_avg.py`
- All phase3 framework code
