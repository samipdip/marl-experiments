# Price Discovery — Experiment Automation Guide

## The Phase3 Multi-Agent Framework

All experiments run through `~/SimExchangeSandbox/phase3_multi_agent/` — a Rust-backed limit-order-book exchange (`_sim_exchange`) with a Python agent lifecycle loop (`runner.py` → `factory.py` → `market.py` + agents). The **runner calls `on_tick()` on every agent every tick**, agents submit orders through a shared `OrderShuffler` that interleaves all orders randomly before dispatch to the exchange.

**THIS FRAMEWORK MUST NEVER BE MODIFIED.** No changes to `runner.py`, `factory.py`, `env.py`, `market.py`, or any file inside `phase3_multi_agent/`. All experiment work happens by:
- Editing **config YAMLs** (which agents to load, their parameters, train/eval mode)
- Editing **agent code** in `~/Trading_agents/Agents_phase2/RL_Agents/ABIDES_MARL/` (the agent implementations are separate from the framework)

## Agent Roles

| Agent | Type | Behavior |
|-------|------|----------|
| **Uniform MM** (×5) | Fixed | Posts limit orders at random depths 1-5, ref_price=1000 |
| **Poisson Noise** | Fixed | Random market orders, λ=1.0 each side |
| **HJB MM** (analytical) | Fixed | Posts depths from HJB PDE solved with κ=0.8 |
| **PPO HJB MM** | **Trainable** | Learns κ via PPO; depth = 1/κ + h(q)-h(q±1) where h is from PDE |
| **Informed Trader** | **Trainable** (or fixed via checkpoint) | Linear policy: action=β, order_qty=β×(v−ask), β∈[0,10] |

## Remote Server

```
Host QFRCAI_Local
  hostname 172.16.210.57
  User samip
```

**Key paths on remote (`/home/samip/`):**

| Path | Content |
|------|---------|
| `~/marl-experiments/` | Configs YAMLs, experiment scripts, results |
| `~/marl-experiments/price_discovery/` | Experiment YAMLs, analysis scripts, results |
| `~/SimExchangeSandbox/` | Phase3 multi-agent framework (DO NOT MODIFY) |
| `~/Trading_agents/` | Agent implementations (informed trader, HJB MM, PPO HJB MM, etc.) |
| `~/checkpoints/` | Training checkpoints |
| `~/price_discovery/results/` | Eval result directories |

**Python environment:**
```bash
~/marl-experiments/.venv/bin/python
```
This venv has `_sim_exchange`, `torch`, `numpy`, etc. installed.

**Important sys.path setup when running Python:**
```python
import sys, os
sys.path.insert(0, os.path.expanduser("~/SimExchangeSandbox"))
sys.path.insert(0, os.path.expanduser("~/Trading_agents"))
```

---

## Constraints (DO NOT VIOLATE)

1. **NEVER modify `phase3_multi_agent` source code.** The framework is in `~/SimExchangeSandbox/phase3_multi_agent/`. No edits to `factory.py`, `runner.py`, `env.py`, `market.py`, or any file inside that directory.
2. **NEVER rsync messy files to remote repos.** Only rsync specific files that need updating:
   - Config YAMLs to `~/marl-experiments/price_discovery/`
   - Trading agent files to `~/Trading_agents/...` (only when explicitly modifying agent code)
3. **Always use the venv:** `~/marl-experiments/.venv/bin/python`
4. **Eval episodes must each use a fresh `run_simulation` call** with different seed to get independent results. Use seeds 10000+ep.

---

## Config Files

All in `~/marl-experiments/price_discovery/` (remote) or `~/Documents/marl-experiments/price_discovery/` (local).

### `price_discovery_train.yaml` — Train Informed Trader only

- Full agent lineup: UniformMM(5) + PoissonNoise + HjbMarketMaker(analytical) + KyleInformedTrader(PPO)
- Environment: `sim_duration: 25000.0` (25000 ticks)
- IT: `n_steps: 40`, `total_timesteps: 20000`, `checkpoint_freq: 5000`
- IT: `buy_only: true`, `policy: linear`, `action_min: 0.0`, `action_max: 10.0`
- Checkpoints go to `/home/samip/checkpoints/price_discovery_train_60/`
- HJB MM: `sim_duration: 60.0`, `episode_steps: 60`, `hjb_dt: 1.0`

### `price_discovery_eval.yaml` — Eval Informed Trader

- Same agent lineup (all fixed, no training)
- Environment: `sim_duration: 60.0`
- IT checkpoint loaded from `price_discovery_train_60/informed_step_20029_final.pt`
- All agents in eval mode

### `ppo_hjb_train.yaml` — Train PPO HJB MM + Analytical HJB MM simultaneously

- Full lineup: UniformMM(5) + PoissonNoise + **HjbMarketMaker**(analytical, fixed) + **PPOHjbMarketMaker**(trainable) + KyleInformedTrader(fixed eval)
- PPO MM params: `kappa_min: 0.01, kappa_max: 2.0, kappa_ref: 0.8`, `alpha: 0.01, phi: 0.001`
- PPO MM: `n_steps: 40`, `total_timesteps: 20000`, `checkpoint_freq: 5000`
- PPO MM has terminal penalty in reward (alpha * q² on last step)
- IT: `mode: eval, total_timesteps: 999999, eval_freq: 999999, buy_only: true`
- IT checkpoint: `/home/samip/checkpoints/price_discovery_train/informed_step_20029_final.pt`
- Checkpoints go to `/home/samip/checkpoints/ppo_hjb_train/`

### `ppo_hjb_eval.yaml` — Eval PPO HJB MM + Analytical HJB MM

- Same lineup, all in eval mode
- PPO MM checkpoint: `/home/samip/checkpoints/ppo_hjb_train/ppo_hjb_mm_step_20029_final.pt`
- IT checkpoint: `/home/samip/checkpoints/price_discovery_train/informed_step_20029_final.pt`
- Environment: `sim_duration: 40.0`

### Adding `fair_value` for experiments

To run experiments with different fair values (v), add `fair_value: <VALUE>` to the informed trader's params in both train and eval YAMLs:

```yaml
  - id: informed
    ...
    params:
      ...
      buy_only: true
      fair_value: 1003   # <-- add this
```

This is handled by the modified `KyleInformedTrader` which checks `self._fair_value` in `on_episode_start` and uses it to set `self._v` instead of the default `mid + 6 * tick`.

---

## Complete Workflow (copy-paste chain)

All commands below assume your **local working directory** is:
```bash
cd ~/Documents/marl-experiments/price_discovery
```

### 1. Edit config locally → rsync to remote

```bash
# Configs (train + eval)
rsync -avz -e ssh \
  price_discovery/ppo_hjb_train.yaml \
  price_discovery/ppo_hjb_eval.yaml \
  QFRCAI_Local:~/marl-experiments/price_discovery/

# Agent code (only if you modified Trading_agents)
rsync -avz -e ssh \
  ~/Documents/Trading_agents/Agents_phase2/RL_Agents/ABIDES_MARL/ppo_hjb_market_maker.py \
  QFRCAI_Local:~/Trading_agents/Agents_phase2/RL_Agents/ABIDES_MARL/

rsync -avz -e ssh \
  ~/Documents/Trading_agents/Agents_phase2/RL_Agents/ABIDES_MARL/kyle_informed_trader.py \
  QFRCAI_Local:~/Trading_agents/Agents_phase2/RL_Agents/ABIDES_MARL/
```

### 2. Run training on remote

```bash
ssh QFRCAI_Local "cd ~/marl-experiments && nohup .venv/bin/python -c \"
import sys, os
sys.path.insert(0, os.path.expanduser('~/SimExchangeSandbox'))
sys.path.insert(0, os.path.expanduser('~/Trading_agents'))
from phase3_multi_agent.src.runner import run_simulation
run_simulation('price_discovery/ppo_hjb_train.yaml', device='cpu', seed=42, n_ticks=25000)
\" > ~/marl-experiments/price_discovery/results/train.log 2>&1 & echo pid=\$!"
```

**Check progress:**
```bash
ssh QFRCAI_Local "tail -10 ~/marl-experiments/price_discovery/results/train.log"
```

Training is done when `=== Done ===` appears. Final checkpoint at `~/checkpoints/ppo_hjb_train/ppo_hjb_mm_step_20029_final.pt`.

### 3. Run eval episodes on remote

**30 episodes:**
```bash
ssh QFRCAI_Local "cd ~/marl-experiments && for ep in \$(seq 0 29); do
  seed=\$((10000 + ep))
  dir=~/price_discovery/results/eval_dir/ep_\${ep}
  mkdir -p \$dir
  .venv/bin/python -c \"
import sys, os
sys.path.insert(0, os.path.expanduser('~/SimExchangeSandbox'))
sys.path.insert(0, os.path.expanduser('~/Trading_agents'))
from phase3_multi_agent.src.runner import run_simulation
run_simulation('price_discovery/ppo_hjb_eval.yaml', device='cpu', seed=\${seed}, results_dir='\${dir}')
\"
done"
```

**100 episodes** — same command, replace `seq 0 29` with `seq 0 99`.

### 4. Pull results to local

```bash
# Create local results dir
mkdir -p results/eval_dir

# Pull from remote
rsync -avz -e ssh \
  QFRCAI_Local:~/price_discovery/results/eval_dir/ \
  results/eval_dir/
```

### 5. Generate plots

```bash
cd ~/Documents/marl-experiments/price_discovery

# LOB heatmap + inventory + PnL (3 agents)
python3 plot_eval_avg.py results/eval_dir/ 40

# Beta + fill VWAP with ±1σ
python3 plot_beta_avg.py results/eval_dir/ 40

# Kappa + beta side-by-side with ±1σ
python3 plot_ppo_hjb_avg.py results/eval_dir/ 40
```

Plots are saved as `eval_avg.png`, `beta_avg.png`, `ppo_hjb_avg.png` inside `results/eval_dir/`.

### Full chain in one shot

```bash
# 1. rsync
rsync -avz -e ssh price_discovery/ppo_hjb_train.yaml QFRCAI_Local:~/marl-experiments/price_discovery/
rsync -avz -e ssh price_discovery/ppo_hjb_eval.yaml QFRCAI_Local:~/marl-experiments/price_discovery/

# 2. train
ssh QFRCAI_Local "cd ~/marl-experiments && nohup .venv/bin/python -c \"
import sys, os
sys.path.insert(0, os.path.expanduser('~/SimExchangeSandbox'))
sys.path.insert(0, os.path.expanduser('~/Trading_agents'))
from phase3_multi_agent.src.runner import run_simulation
run_simulation('price_discovery/ppo_hjb_train.yaml', device='cpu', seed=42, n_ticks=25000)
\" > ~/marl-experiments/price_discovery/results/train.log 2>&1 &"

# 3. wait for training to finish, then eval
ssh QFRCAI_Local "cd ~/marl-experiments && for ep in \$(seq 0 29); do
  seed=\$((10000 + ep));
  dir=~/price_discovery/results/eval_dir/ep_\${ep};
  mkdir -p \$dir;
  .venv/bin/python -c \"
import sys, os
sys.path.insert(0, os.path.expanduser('~/SimExchangeSandbox'))
sys.path.insert(0, os.path.expanduser('~/Trading_agents'))
from phase3_multi_agent.src.runner import run_simulation
run_simulation('price_discovery/ppo_hjb_eval.yaml', device='cpu', seed=\${seed}, results_dir='\${dir}')
\";
done"

# 4. pull
rsync -avz -e ssh QFRCAI_Local:~/price_discovery/results/eval_dir/ results/eval_dir/

# 5. plot
python3 plot_eval_avg.py results/eval_dir/ 40
python3 plot_beta_avg.py results/eval_dir/ 40
python3 plot_ppo_hjb_avg.py results/eval_dir/ 40
```

---
cd ~/Documents/marl-experiments/price_discovery
python3 plot_eval_avg.py results/eval_dir/ 40   # LOB heatmap + inv + PnL
python3 plot_beta_avg.py results/eval_dir/ 40   # beta + fill VWAP
python3 plot_ppo_hjb_avg.py results/eval_dir/ 40  # kappa + beta
```

---

## Agent Code Locations (Trading_agents)

### `~/Trading_agents/Agents_phase2/RL_Agents/ABIDES_MARL/ppo_hjb_market_maker.py`
**Class:** `PPOHjbMarketMaker(KyleRLAgent, PositionAgent)`

Key overrides:
- `build_observation()` — 7-dim obs (q, spread, lambda_ratio, mid_vol, imbalance, t_norm, prev_kappa)
- `compute_reward()` — `cash_change - phi*q²*dt` with terminal penalty `-alpha*q²` on last step
- `action_to_orders()` — cancels old bid/ask, places new limit orders at depths from HJB PDE
- `get_metrics()` — returns inventory, cash, mid, total_pnl, lambda_est, kappa, q_lots

**Important fix applied:** In `build_observation()`, `_last_best_bid`/`_last_best_ask` are patched with fallback values (`mid ± 1`) when both are zero (window between cancel and replace). This prevents PnL from exploding to `cash + inventory * 0 = cash`.

**Important fix applied:** `kappa_ref` can be set via YAML param. Defaults to `(kappa_min + kappa_max)/2` if None.

### `~/Trading_agents/Agents_phase2/RL_Agents/ABIDES_MARL/kyle_informed_trader.py`
**Class:** `KyleInformedTrader(KyleRLAgent, PositionAgent)`

**Modification:** Added `fair_value: float | None = None` parameter. When set, `on_episode_start` uses `self._fair_value` directly for `self._v` instead of `mid + 6 * tick`.

### `~/Trading_agents/Agents_phase2/RL_Agents/ABIDES_MARL/kyle_base.py`
**Class:** `KyleRLAgent` — PPO training lifecycle. **DO NOT MODIFY** (it's effectively phase3 framework).

Key `on_tick()` flow:
1. Drain acks/fills
2. Read market data → `_last_best_bid/ask`
3. `build_observation(sim_t, md)` → agent override
4. Compute mid from `_last_best_bid/ask` (falls to 0 if both missing)
5. `compute_reward()` → stores in buffer
6. If buffer full → `on_episode_end()` → `update()` (PPO) → `on_episode_start()`
7. `sample_action()` → `action_to_orders()` → submit via shuffler

### `~/Trading_agents/Agents_phase2/RL_Agents/ABIDES_MARL/hjb_market_maker.py`
**Class:** `HjbMarketMaker` — standalone (non-PPO), analytical HJB. Used as baseline. Has its own PnL tracking.

---

## Plot Scripts (local, in `price_discovery/`)

### `plot_eval_avg.py`
- Args: `results_base_dir n_steps`
- Reads 30 episodes × n_steps from `{base}/ep_{ep}/pnl_history.csv` + `informed_lob_data.csv`
- Plots: LOB volume heatmap + mid/bid/ask lines + inventory (3 agents) + PnL (3 agents)
- Colors: informed=blue (#1f77b4), HJB MM=red (#d62728), PPO HJB MM=green (#2ca02c)

### `plot_beta_avg.py`
- Args: `results_base_dir n_steps [out_path] [title]`
- Reads informed action history + order events
- Left panel: beta line with ±1σ band (step 0 removed)
- Right panel: fill qty bars + VWAP line

### `plot_ppo_hjb_avg.py`
- Args: `results_base_dir n_steps`
- Reads PPO HJB MM + informed action histories
- Left: kappa with ±1σ band
- Right: beta with ±1σ band (step 0 removed)

---

## Multi-Value Experiment Runner

For running 5 experiments (v=1001..1005) with different fair values:

1. Create temp YAMLs per v with `fair_value` and separate checkpoint dirs:
   ```bash
   sed "s|fair_value:.*|fair_value: $v|" ppo_hjb_train.yaml > /tmp/train_v$v.yaml
   ```
2. Override checkpoint dirs: `sed -i "s|ppo_hjb_train|ppo_hjb_train_v$v|g" /tmp/*.yaml`
3. Run training + eval for each v.
4. Pull all results locally: `rsync -avz -e ssh QFRCAI_Local:~/price_discovery/results/ .`

---

## Known Issues

1. **eval mode never resets inventory/cash** — `KyleRLAgent.on_tick()` never calls `on_episode_end()` in eval mode (since `phase != "rollout"`), so `_inventory` and `_cash` carry over across eval episodes. This is fine for single-episode eval runs but means multi-episode runs accumulate state.
2. **PPnL spike when bid/ask both 0** — FIXED: PPOHjbMM patches `_last_best_bid/ask` in `build_observation()`.
3. **Shell quoting in SSH commands** — Use heredocs or write scripts to remote to avoid quoting hell with nested Python.
4. **Checkpoint naming** — `_final` suffix only added when `_training_done` triggers at step 20029 (not 20000). Eval config must match the exact filename.
