# Experiment Plan: Market Reset + Randomized v

## Code Change

**`KyleInformedTrader.on_episode_start()`**: support `fair_value` as a `[lo, hi]` list.
If list, sample `random.randint(lo, hi)` each episode. Int stays as-is.
```python
def on_episode_start(self, md=None):
    if isinstance(self._fair_value, (list, tuple)):
        self._v = random.randint(self._fair_value[0], self._fair_value[1])
    elif self._fair_value is not None:
        self._v = self._fair_value
    elif md is not None:
        self._v = md.get("mid", 0) + 6 * self.tick
```

---

## Experiment Matrix (8 train + 10 eval)

### Set 1 — Original, fixed v
| Train | Agents | v |
|-------|--------|---|
| `train_v1003_ppo` | 3 PPO HJB MM + IT buy | 1003 |
| `train_v1003_mixed` | 3 HJB MM + 3 PPO HJB MM + IT buy | 1003 |
| `train_v1006_ppo` | 3 PPO HJB MM + IT buy | 1006 |
| `train_v1006_mixed` | 3 HJB MM + 3 PPO HJB MM + IT buy | 1006 |

### Set 1R — Original, v ~ U[1002, 1006]
| Train | Agents | v |
|-------|--------|---|
| `train_vrand_ppo` | 3 PPO HJB MM + IT buy | 1002–1006 |
| `train_vrand_mixed` | 3 HJB MM + 3 PPO HJB MM + IT buy | 1002–1006 |

### Set 2 — 2D kappa, fixed v
| Train | Agents | v |
|-------|--------|---|
| `train_2d_it_v1003` | 3 HJB + 3 PPO HJB 2D MM + IT buy | 1003 |
| `train_2d_lt` | 3 HJB + 3 PPO HJB 2D MM + LT sell | — |
| `train_2d_opp_v1003` | 3 HJB + 3 PPO HJB 2D MM + IT buy + LT sell | 1003 |
| `train_2d_same_v997` | 3 HJB + 3 PPO HJB 2D MM + IT sell + LT sell | 997 |

### Set 2R — 2D kappa, v ~ U[1002, 1006]
| Train | Agents | v |
|-------|--------|---|
| `train_2d_it_vrand` | 3 HJB + 3 PPO HJB 2D MM + IT buy | 1002–1006 |
| `train_2d_opp_vrand` | 3 HJB + 3 PPO HJB 2D MM + IT buy + LT sell | 1002–1006 |
| `train_2d_same_vrand` | 3 HJB + 3 PPO HJB 2D MM + IT sell + LT sell | 1002–1006 |

### Eval configs
Each trained checkpoint is evaluated at 30 episodes × 40 ticks.
For randomized-v experiments, eval at **all 5** v values:

| Eval | For checkpoint from |
|------|-------------------|
| `eval_v1003_hjb` | (HJB analytical — no retrain needed) |
| `eval_v1003_ppo` | `train_v1003_ppo` |
| `eval_v1003_mixed` | `train_v1003_mixed` |
| `eval_v1006_ppo` | `train_v1006_ppo` |
| `eval_v1006_mixed` | `train_v1006_mixed` |
| `eval_v1002_ppo` / `_v1003_ppo` / `_v1004_ppo` / `_v1005_ppo` / `_v1006_ppo` | `train_vrand_ppo` |
| `eval_v1002_mixed` / `_v1003_mixed` / `_v1004_mixed` / `_v1005_mixed` / `_v1006_mixed` | `train_vrand_mixed` |
| `eval_2d_it_v1003` | `train_2d_it_v1003` |
| `eval_2d_lt` | `train_2d_lt` |
| `eval_2d_opp_v1003` | `train_2d_opp_v1003` |
| `eval_2d_same_v997` | `train_2d_same_v997` |
| (eval at 1002–1006 for each 2D vrand) | `train_2d_it_vrand` / `train_2d_opp_vrand` / `train_2d_same_vrand` |

---

## Configs to create (new or copied from existing)

### New: fixed-v copies from exp2d into configs/
- `train_2d_*.yaml` + `eval_2d_*.yaml` — copy from `configs/exp2d/*` into `configs/` for consistency

### New: randomized-v configs
- `train_vrand_ppo.yaml` — copy `train_v1003_ppo.yaml`, change `fair_value: 1003` → `fair_value: [1002, 1006]`
- `train_vrand_mixed.yaml` — same
- `train_2d_it_vrand.yaml` — copy `train_2d_it_v1003.yaml`, change fair_value
- `train_2d_opp_vrand.yaml` — copy `train_2d_opp_v1003.yaml`, change fair_value
- `train_2d_same_vrand.yaml` — copy `train_2d_same_v997.yaml`, change fair_value

### New: eval at each v for randomized experiments
- `eval_v1002_ppo.yaml` through `eval_v1006_ppo.yaml` (5 files)
- `eval_vrand_mixed_v1002.yaml` through `eval_vrand_mixed_v1006.yaml` (5 files)
- (and so on for each 2D vrand experiment)

---

## Total
- **Train runs:** 4 (Set 1) + 2 (Set 1R) + 4 (Set 2) + 3 (Set 2R) = **13**
- **Eval runs:** 4 + 2×5 + 4 + 3×5 = **33** (across all v values)
- Estimated time: ~3-4 hours on QFRCAI_Local
