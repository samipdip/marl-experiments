# Plan: Re-run All Reset Experiments with n_steps=80

## Why 80 Steps

The root of Framework 2's failure (IT collapse, erratic kappa, overshoot) is that 40-step episodes contain **only discovery-phase states** (ask < v, IT buying). The IT never sees equilibrium states, so its PPO policy collapses to constant β=9.6 (proven from training logs: entropy dives to -1.65, β std=0 across all steps).

Extending to **80 steps** ensures each episode contains:
- Discovery phase: first ~20-30 ticks (IT buys, mid rises toward v)
- Equilibrium phase: remaining ~50-60 ticks (mid ≈ v, IT quiet, MM captures spread)

This exposes the IT to diverse states, preventing policy collapse.

## What Changes

| Parameter | Old | New | Where |
|-----------|-----|-----|-------|
| `sim_duration` (training horizon) | `25000.0` | `40000.0` | environment, IT, LT |
| `sim_duration` (PDE + eval horizon) | `40.0` | `80.0` | PPO HJB MM, analytical HJB MM, environment (eval) |
| `N` (PDE grid) | `40` | `80` | PPO HJB MM, analytical HJB MM |
| `n_steps` (PPO buffer) | `40` | `80` | PPO HJB MM, IT, LT |
| `episode_steps` (analytical HJB) | `40` | `80` | analytical HJB MM |
| `total_timesteps` | `20000` | `40000` | PPO HJB MM, IT, LT, top-level `train:` |
| `eval_freq` | `20000` | `40000` | PPO HJB MM, IT, LT, top-level |

`episode_ticks` is auto-detected from `min(n_steps)` → becomes 80.
PPO updates: 40000/80 = **500** (same count as before: 20000/40 = 500).

## Files to Modify

### Configs (46 files total)

**Training configs (13):**
```
train_v1003_ppo.yaml      train_v1003_mixed.yaml
train_v1006_ppo.yaml      train_v1006_mixed.yaml
train_vrand_ppo.yaml      train_vrand_mixed.yaml
train_2d_it_v1003.yaml    train_2d_lt.yaml
train_2d_opp_v1003.yaml   train_2d_same_v997.yaml
train_2d_it_vrand.yaml    train_2d_opp_vrand.yaml
train_2d_same_vrand.yaml
```

**Eval configs (33):**
```
eval_v1003_ppo.yaml           eval_v1003_mixed.yaml
eval_v1006_ppo.yaml           eval_v1006_mixed.yaml
eval_vrand_ppo_v{1002..1006}.yaml  (5 files)
eval_vrand_mixed_v{1002..1006}.yaml (5 files)
eval_2d_it_v1003.yaml         eval_2d_lt.yaml
eval_2d_opp_v1003.yaml        eval_2d_same_v997.yaml
eval_2d_it_vrand_v{1002..1006}.yaml  (5 files)
eval_2d_opp_vrand_v{1002..1006}.yaml (5 files)
eval_2d_same_vrand_v{1002..1006}.yaml (5 files)
```

### `scripts/run_all_sequential.sh`

- `gen_plots()`: change `40` → `80` in all 3 plot script calls
- Add `patch_checkpoint_steps()` function (described below)
- Interleave checkpoint patching between `run_train` and `run_eval`

## Execution Plan

### Phase 1: Modify Configs Locally (sed batch)

```bash
# In price_discovery/ directory:
CFG=configs

# 1. sim_duration: 25000.0 → 40000.0 (training horizon — environment + IT + LT)
for f in $CFG/train_*.yaml; do
  sed -i 's/sim_duration: 25000\.0/sim_duration: 40000.0/g' "$f"
done

# 2. sim_duration: 40.0 → 80.0 (PDE/eval horizon — all agents in all configs)
for f in $CFG/train_*.yaml $CFG/eval_*.yaml; do
  sed -i 's/sim_duration: 40\.0/sim_duration: 80.0/g' "$f"
done

# 3. N: 40 → 80 (PDE grid — all agents in all configs)
for f in $CFG/train_*.yaml $CFG/eval_*.yaml; do
  sed -i 's/^\([[:space:]]*\)N: 40/\1N: 80/' "$f"
done

# 4. n_steps: 40 → 80 (PPO buffer — all agents in all configs)
for f in $CFG/train_*.yaml $CFG/eval_*.yaml; do
  sed -i 's/n_steps: 40/n_steps: 80/g' "$f"
done

# 5. episode_steps: 40 → 80 (analytical HJB only — all configs)
for f in $CFG/train_*.yaml $CFG/eval_*.yaml; do
  sed -i 's/episode_steps: 40/episode_steps: 80/g' "$f"
done

# 6. total_timesteps: 20000 → 40000 (training configs only)
for f in $CFG/train_*.yaml; do
  sed -i 's/total_timesteps: 20000/total_timesteps: 40000/g' "$f"
done

# 7. eval_freq: 20000 → 40000 (training configs only)
for f in $CFG/train_*.yaml; do
  sed -i 's/eval_freq: 20000/eval_freq: 40000/g' "$f"
done
```

### Phase 2: Update run_all_sequential.sh

**Add checkpoint patching function:**
```bash
patch_checkpoint_step() {
  local TRAIN_NAME=$1
  # Discover the checkpoint step number from any _final.pt file
  local CKPT_DIR="/home/samip/checkpoints/${TRAIN_NAME}"
  local STEP=$(find "$CKPT_DIR" -name "*_final.pt" -print -quit 2>/dev/null | grep -oP 'step_\K\d+(?=_final)')
  if [ -z "$STEP" ]; then
    echo "  ERROR: No final checkpoint found in ${CKPT_DIR}!"
    return 1
  fi
  echo "  Discovered checkpoint step: ${STEP}"
  for cfg in ${CFG}/eval_*${TRAIN_NAME}*.yaml; do
    [ -f "$cfg" ] || continue
    sed -i "s/step_[0-9]*_final/step_${STEP}_final/g" "$cfg"
    echo "  Patched: $(basename $cfg)"
  done
}
```

**Update gen_plots:**
```bash
gen_plots() {
  local DIR=$1
  cd ${RESULTS}
  $VENV scripts/plot_eval_avg.py ${DIR} 80 2>/dev/null
  $VENV scripts/plot_ppo_hjb_avg.py ${DIR} 80 2>/dev/null
  $VENV scripts/plot_fills_avg.py ${DIR} 80 2>/dev/null
  echo "  PLOTS: ${DIR}"
}
```

**Update main loop to add checkpoint patching:**
```bash
for exp in v1003_ppo v1003_mixed v1006_ppo v1006_mixed; do
  run_train $exp                     # trains with new params
  patch_checkpoint_step $exp         # discovers step number, patches eval YAMLs
  run_eval eval_${exp}.yaml ${RESULTS}/results_${exp}_reset
  gen_plots results_${exp}_reset
done
```

### Phase 3: Clear Old Checkpoints

Before running, clear all old checkpoint directories to prevent stale files:

```bash
for dir in v1003_ppo v1003_mixed v1006_ppo v1006_mixed \
           vrand_ppo vrand_mixed \
           2d_it_v1003 2d_lt 2d_opp_v1003 2d_same_v997 \
           2d_it_vrand 2d_opp_vrand 2d_same_vrand; do
  rm -rf /home/samip/checkpoints/${dir}
done
```

Add this as the first action in `run_all_sequential.sh`.

### Phase 4: Rsync to Remote

```bash
rsync -avz -e ssh \
  price_discovery/configs/train_v1003_ppo.yaml \
  price_discovery/configs/train_v1003_mixed.yaml \
  price_discovery/configs/train_v1006_ppo.yaml \
  price_discovery/configs/train_v1006_mixed.yaml \
  price_discovery/configs/train_vrand_ppo.yaml \
  price_discovery/configs/train_vrand_mixed.yaml \
  price_discovery/configs/train_2d_it_v1003.yaml \
  price_discovery/configs/train_2d_lt.yaml \
  price_discovery/configs/train_2d_opp_v1003.yaml \
  price_discovery/configs/train_2d_same_v997.yaml \
  price_discovery/configs/train_2d_it_vrand.yaml \
  price_discovery/configs/train_2d_opp_vrand.yaml \
  price_discovery/configs/train_2d_same_vrand.yaml \
  price_discovery/configs/eval_v*.yaml \
  price_discovery/configs/eval_2d_*.yaml \
  QFRCAI:~/marl-experiments/price_discovery/configs/

rsync -avz -e ssh \
  price_discovery/scripts/run_all_sequential.sh \
  price_discovery/scripts/plot_eval_avg.py \
  price_discovery/scripts/plot_ppo_hjb_avg.py \
  price_discovery/scripts/plot_fills_avg.py \
  QFRCAI:~/marl-experiments/price_discovery/scripts/
```

### Phase 5: Execute on Remote (SEQUENTIALLY — CRITICAL)

**Training runs MUST be sequential. Running parallel trainings on QFRCAI breaks the server. Each training uses 100% CPU and parallel runs cause thrashing.**

```bash
ssh QFRCAI "cd ~/marl-experiments && nohup bash price_discovery/scripts/run_all_sequential.sh \
  > price_discovery/logs/run_all_80steps.log 2>&1 & echo PID=\$!"
```

The `run_all_sequential.sh` already runs experiments sequentially (no `&` in background). The `nohup` ensures the process survives SSH disconnection.

### Phase 6: Monitor Progress

```bash
# Check overall progress
ssh QFRCAI "tail -5 ~/marl-experiments/price_discovery/logs/run_all_80steps.log"

# Check specific training log
ssh QFRCAI "tail -10 ~/marl-experiments/price_discovery/logs/v1003_ppo.log"

# Check if process is still running
ssh QFRCAI "ps aux | grep run_all | grep -v grep"

# Check training is learning properly (IT beta should NOT be constant ~9.6)
# Look for varying value loss and positive entropy:
ssh QFRCAI "grep '\[informed\]' ~/marl-experiments/price_discovery/logs/v1003_ppo.log | head -20"
```

### Phase 7: Pull Results to Local

After all experiments complete:

```bash
# Pull results from remote
mkdir -p price_discovery/plots_reset_80steps
ssh QFRCAI "find ~/marl-experiments/price_discovery -maxdepth 1 -type d -name 'results_*' | sort" \
  | while read dir; do
  name=$(basename "$dir")
  mkdir -p "price_discovery/plots_reset_80steps/${name}"
  rsync -avz -e ssh "QFRCAI:${dir}/*.png" "price_discovery/plots_reset_80steps/${name}/"
done

# Also pull logs
rsync -avz -e ssh \
  QFRCAI:~/marl-experiments/price_discovery/logs/ \
  price_discovery/logs_80steps/
```

## Time Estimate (Sequential)

| Phase | Count | Per-unit | Sequential Total |
|-------|-------|----------|-----------------|
| Training | 13 | ~10 min | ~130 min |
| Eval | 33 × 30 ep × 80 ticks | ~5 sec/ep | ~165 min |
| Plotting | 33 | ~1 min | ~33 min |
| **Sequential total** | | | **~5.5 hours** |

Run overnight. Start at end of day, results ready next morning.

## Verification After Run

1. **IT beta NOT constant**: Run analysis on remote — IT beta should vary across steps (not β=9.6 at every step)
2. **MM kappa std lowered**: Should drop from 0.52-0.65 to 0.20-0.35
3. **Value loss stabilized**: MM value loss should drop below 2.0 and stay there in the second half of training
4. **Mid trajectory**: Should show both discovery and equilibrium phases within episodes
