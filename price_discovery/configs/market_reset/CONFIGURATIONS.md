# RL Training & Testing Configuration

All trainable agents across every experiment share the **same** RL configuration
regardless of agent type (PPOHjbMarketMaker, KyleInformedTrader, KyleLiquidityTrader).

---

## Training Configuration

### Schedule

| Parameter | Value |
|-----------|-------|
| `total_timesteps` | 40000 |
| `n_steps` (rollout buffer) | 80 |
| PPO updates | 500 (40000 / 80) |
| `eval_freq` | 40000 (eval once at end) |
| `eval_episodes` | 30 |
| `checkpoint_freq` | 5000 |
| `seed` | 42 |

### PPO Hyperparameters (identical for all agents)

| Parameter | Value |
|-----------|-------|
| `lr` | 0.0003 |
| `gamma` | 1.0 |
| `gae_lambda` | 1.0 |
| `clip_epsilon` | 0.2 |
| `n_epochs` | 10 |
| `batch_size` | 64 |

### Episode / Simulation

| Parameter | Training | Eval |
|-----------|----------|------|
| `sim_duration` (environment) | 40000.0 | 80.0 |
| `sim_duration` (PDE / per-agent) | 80.0 | 80.0 |
| `N` (PDE grid points) | 80 | 80 |
| `hjb_dt` | 1.0 | 1.0 |
| Episode length | 80 ticks | 80 ticks |

---

## Testing (Eval) Configuration

| Parameter | Value |
|-----------|-------|
| Episodes | 30 |
| Episode seed | `10000 + ep` |
| `sim_duration` | 80.0 (80 ticks) |
| `mode` | eval |
| `total_timesteps` | 0 |
| `eval_freq` | 0 |
| `checkpoint_freq` | 0 |
| Checkpoint | `step_40029_final.pt` |

Note: IT and LT eval configs use `sim_duration: 40000.0` (NOT 80.0). This is
deliberate — `sim_duration` only feeds the `t_norm = sim_t/sim_duration`
observation feature, and training saw t_norm ≈ 0 (episode-local clock /
40000). Eval at 40000 keeps t_norm ≈ 0, matching training. All other agents
(PPO HJB MM, HJB MM) correctly use `sim_duration: 80.0` (PDE horizon =
episode length, same as training).

---

## Per-Agent Differences (NOT RL config)

The only parameters that differ across agents are their **domain-specific** params,
not their RL training/eval params:

| Agent | RL-trained? | Unique params |
|-------|-------------|---------------|
| PPOHjbMarketMaker (1D) | Yes | `obs_dim: 7, action_dim: 1, entropy_coef: 0.01` |
| PPOHjbMarketMaker (2D) | Yes | `obs_dim: 8, action_dim: 2, entropy_coef: 0.02` |
| KyleInformedTrader | Yes | `policy: linear, max_order: 500, entropy_coef: 0.01` |
| KyleLiquidityTrader | Yes | `q_target: 20, beta_penalty: 10.0, entropy_coef: 0.01` |
| HjbMarketMaker | No (analytical) | `kappa_plus: 0.8, kappa_minus: 0.8` |
| PoissonNoiseTrader | No | `lambda_plus: 1.0, lambda_minus: 1.0, qty: 1` |

Note: `entropy_coef` is the **only** PPO hyperparameter that differs — it is 0.02
for the 2D PPO HJB MM (to compensate for halved per-dimension entropy) and 0.01
for all other trainable agents. Everything else (`lr`, `gamma`, `gae_lambda`,
`clip_epsilon`, `n_epochs`, `batch_size`, `total_timesteps`, `n_steps`) is
identical across all agents.
