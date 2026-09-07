# RL Training & Testing Configuration — No-Reset Experiments

All trainable agents share the **same** PPO hyperparameters.

**Phase:** PPO HJB Market Maker training — 40-step episodes, IT pre-trained and frozen

---

## PPO Hyperparameters (identical for all agents)

| Parameter | Value |
|-----------|-------|
| `lr` | 0.0003 |
| `gamma` | 1.0 |
| `gae_lambda` | 1.0 |
| `clip_epsilon` | 0.2 |
| `entropy_coef` | 0.01 |
| `n_epochs` | 10 |
| `batch_size` | 64 |
| `seed` | 42 |

---

## Training Schedule

| Parameter | Value |
|-----------|-------|
| `total_timesteps` | 20000 |
| `n_steps` | 40 |
| PPO updates | 500 (20000 / 40) |
| `eval_freq` | 20000 |
| `eval_episodes` | 30 |
| `checkpoint_freq` | 5000 |
| `seed` | 42 |

## Episode / Simulation

| Parameter | Training | Eval |
|-----------|----------|------|
| `sim_duration` (environment) | 25000.0 | 40.0 |
| `sim_duration` (PPO HJB agent) | 40.0 | 40.0 |
| `N` (PDE grid) | 40 | 40 |
| Episode length | 40 ticks | 40 ticks |
| Market reset | No | No |

---

## Agent Roster

| Agent | Type | Trained? | Key Params |
|-------|------|----------|------------|
| uniform_mm | UniformMarketMakers | No | n_makers=5, depth 1-5 |
| noise | PoissonNoiseTrader | No | lambda=1.0, qty=1 |
| hjb_mm | HjbMarketMaker | No | kappa=0.8, N=40, sim_duration=40 |
| ppo_hjb_mm | PPOHjbMarketMaker | **Yes** | kappa_ref=0.8, kappa_min=0.01, kappa_max=2.0, N=40 |
| informed | KyleInformedTrader | No (frozen) | checkpoint pre-trained, buy_only=true |

PPO HJB checkpoint: `ppo_hjb_train/ppo_hjb_mm_step_20029_final.pt`
IT checkpoint (frozen): `price_discovery_train/informed_step_20029_final.pt`

---

## Eval Protocol

| Parameter | Value |
|-----------|-------|
| Episodes (standard) | 30 |
| Episodes (extended) | 100 |
| Checkpoint step | `step_20029_final` |

---

## Experiments Run (No-Reset)

### PPO HJB Eval (v=1001 to 1005)

| Experiment | fair_value | Episodes | Results Dir |
|------------|------------|----------|-------------|
| ppo_hjb_v1001 | 1001 | 100 | `results-1001-1005/ppo_hjb_v1001/` |
| ppo_hjb_v1002 | 1002 | 100 | `results-1001-1005/ppo_hjb_v1002/` |
| ppo_hjb_v1003 | 1003 | 100 | `results-1001-1005/ppo_hjb_v1003/` |
| ppo_hjb_v1004 | 1004 | 100 | `results-1001-1005/ppo_hjb_v1004/` |
| ppo_hjb_v1005 | 1005 | 100 | `results-1001-1005/ppo_hjb_v1005/` |
| ppo_hjb_eval_30ep | 1003 | 30 | `results-1001-1005/ppo_hjb_eval_30ep/` |

### Variant Experiments (v1003 & v1006)

Standalone runs with different agent compositions. Configs were edited in-place
from the base configs.

| Experiment | MM Composition | fair_value | Results Dir |
|------------|---------------|------------|-------------|
| v1003_ppo | PPO HJB | 1003 | `results_v1003_ppo/` |
| v1003_mixed_3mms | HJB + PPO HJB | 1003 | `results_v1003_mixed_3mms/` |
| v1003_hjb | HJB only (analytical) | 1003 | `results_v1003_hjb/` |
| v1006_ppo | PPO HJB | 1006 | `results_v1006_ppo/` |
| v1006_mixed_3mms | HJB + PPO HJB | 1006 | `results_v1006_mixed_3mms/` |
| v1006_hjb | HJB only (analytical) | 1006 | `results_v1006_hjb/` |

---

## Calibration Configs (No RL)

| Config | Purpose | Agents |
|--------|---------|--------|
| `kappa_cal_config.yaml` | Kappa calibration | UniformMM(5) + Noise |
| `hjb_test_config.yaml` | HJB MM validation | UniformMM(5) + Noise + HJB(1) |
