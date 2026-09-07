#!/usr/bin/env bash
set -e

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

VENV=$HOME/marl-experiments/.venv/bin/python
CFG=$HOME/marl-experiments/price_discovery/configs
RESULTS=$HOME/marl-experiments/price_discovery

run_train() {
  local NAME=$1
  local CFG_PATH=$(find "${CFG}" -maxdepth 3 -name "train_${NAME}.yaml" -print -quit)
  if [ -z "$CFG_PATH" ]; then
    echo "  ERROR: Config train_${NAME}.yaml not found!"
    return 1
  fi
  echo "=== TRAIN ${NAME} ($(date +%H:%M:%S)) ==="
  cd $HOME/marl-experiments
  rm -rf /home/samip/checkpoints/${NAME}
  PYTHONUNBUFFERED=1 $VENV -c "
import sys, os
sys.path.insert(0, os.path.expanduser('~/SimExchangeSandbox'))
sys.path.insert(0, os.path.expanduser('~/Trading_agents'))
from phase3_multi_agent.src.runner import run_simulation
run_simulation('${CFG_PATH}', device='cpu', seed=42)
" > ${RESULTS}/logs/${NAME}.log 2>&1
  echo "  DONE: ${NAME} ($(date +%H:%M:%S))"
}

patch_checkpoint_step() {
  local TRAIN_NAME=$1
  local CKPT_DIR="/home/samip/checkpoints/${TRAIN_NAME}"
  local STEP=$(find "$CKPT_DIR" -name "*_final.pt" -print -quit 2>/dev/null | grep -oP 'step_\K\d+(?=_final)')
  if [ -z "$STEP" ]; then
    echo "  ERROR: No final checkpoint found in ${CKPT_DIR}!"
    return 1
  fi
  echo "  Checkpoint step: ${STEP}"
  for cfg in $(find "${CFG}" -maxdepth 3 -name "eval_*${TRAIN_NAME}*.yaml"); do
    [ -f "$cfg" ] || continue
    sed -i "s/step_[0-9]*_final/step_${STEP}_final/g" "$cfg"
    echo "  Patched: $(basename $cfg)"
  done
}

run_eval() {
  local CFG_NAME=$1
  local OUTDIR=$2
  local CFG_PATH=$(find "${CFG}" -maxdepth 3 -name "${CFG_NAME}" -print -quit)
  if [ -z "$CFG_PATH" ]; then
    echo "  ERROR: Config ${CFG_NAME} not found!"
    return 1
  fi
  echo "=== EVAL ${CFG_NAME} ($(date +%H:%M:%S)) ==="
  rm -rf "$OUTDIR"
  mkdir -p "$OUTDIR"
  for ep in $(seq 0 29); do
    seed=$((10000 + ep))
    epdir="${OUTDIR}/ep_${ep}"
    mkdir -p "$epdir"
    PYTHONUNBUFFERED=1 $VENV -c "
import sys, os
sys.path.insert(0, os.path.expanduser('~/SimExchangeSandbox'))
sys.path.insert(0, os.path.expanduser('~/Trading_agents'))
from phase3_multi_agent.src.runner import run_simulation
run_simulation('${CFG_PATH}', device='cpu', seed=${seed},
               results_dir='${epdir}', episode_ticks=0)
" 2>&1 | tail -1
  done
  echo "  DONE: ${CFG_NAME} ($(date +%H:%M:%S))"
}

gen_plots() {
  local DIR=$1
  cd ${RESULTS}
  $VENV scripts/plot_eval_avg.py ${DIR} 80 2>/dev/null
  $VENV scripts/plot_ppo_hjb_avg.py ${DIR} 80 2>/dev/null
  $VENV scripts/plot_fills_avg.py ${DIR} 80 2>/dev/null
  echo "  PLOTS: ${DIR}"
}

echo "=================================================="
echo "  Bug-E-fixed reruns: all PPO-MM experiments"
echo "  Host: $(hostname)   Sequential — NO parallel runs"
echo "=================================================="
mkdir -p ${RESULTS}/logs

# ── SET 1: fixed v, 1D kappa ──
for exp in v1003_ppo v1003_mixed v1006_ppo v1006_mixed; do
  run_train $exp
  patch_checkpoint_step $exp
  run_eval eval_${exp}.yaml ${RESULTS}/results_${exp}_reset
  gen_plots results_${exp}_reset
done

# ── SET 1R: randomized v, 1D kappa ──
for exp in vrand_ppo vrand_mixed; do
  run_train $exp
  patch_checkpoint_step $exp
  for v in 1002 1003 1004 1005 1006; do
    run_eval eval_${exp}_v${v}.yaml ${RESULTS}/results_${exp}_v${v}
    gen_plots results_${exp}_v${v}
  done
done

# ── SET 2: fixed v, 2D kappa (same-direction removed) ──
for exp in 2d_it_v1003 2d_lt 2d_opp_v1003; do
  run_train $exp
  patch_checkpoint_step $exp
  run_eval eval_${exp}.yaml ${RESULTS}/results_${exp}
  gen_plots results_${exp}
done

# ── SET 2R: randomized v, 2D kappa (same-direction removed) ──
for exp in 2d_it_vrand 2d_opp_vrand; do
  run_train $exp
  patch_checkpoint_step $exp
  for v in 1002 1003 1004 1005 1006; do
    run_eval eval_${exp}_v${v}.yaml ${RESULTS}/results_${exp}_v${v}
    gen_plots results_${exp}_v${v}
  done
done

# ── 2D PPO-only (5 MMs, fixed v) ──
for exp in 2dppo_it_v1003 2dppo_lt 2dppo_opp_v1003; do
  run_train $exp
  patch_checkpoint_step $exp
  run_eval eval_${exp}.yaml ${RESULTS}/results_${exp}
  gen_plots results_${exp}
done

# ── 2D PPO-only (5 MMs, randomized v) ──
for exp in 2dppo_it_vrand 2dppo_opp_vrand; do
  run_train $exp
  patch_checkpoint_step $exp
  for v in 1002 1003 1004 1005 1006; do
    run_eval eval_${exp}_v${v}.yaml ${RESULTS}/results_${exp}_v${v}
    gen_plots results_${exp}_v${v}
  done
done

echo "=================================================="
echo "ALL BUG-E-FIXED RERUNS COMPLETE ($(date))"
echo "=================================================="
