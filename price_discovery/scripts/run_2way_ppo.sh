#!/usr/bin/env bash
set -e

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
  echo "=== TRAIN ${NAME} (${CFG_PATH}) ==="
  cd $HOME/marl-experiments
  rm -rf /home/samip/checkpoints/${NAME}
  PYTHONUNBUFFERED=1 $VENV -c "
import sys, os
sys.path.insert(0, os.path.expanduser('~/SimExchangeSandbox'))
sys.path.insert(0, os.path.expanduser('~/Trading_agents'))
from phase3_multi_agent.src.runner import run_simulation
run_simulation('${CFG_PATH}', device='cpu', seed=42)
" > ${RESULTS}/logs/${NAME}.log 2>&1
  echo "  DONE: ${NAME}"
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
  echo "=== EVAL ${CFG_NAME} ==="
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
  echo "  DONE: ${CFG_NAME}"
}

gen_plots() {
  local DIR=$1
  cd ${RESULTS}
  $VENV scripts/plot_eval_avg.py ${DIR} 120 2>/dev/null
  $VENV scripts/plot_ppo_hjb_avg.py ${DIR} 120 2>/dev/null
  $VENV scripts/plot_fills_avg.py ${DIR} 120 2>/dev/null
  echo "  PLOTS: ${DIR}"
}

echo "============================================"
echo "  2-way IT vs 15 PPO 2D MMs (vrand)"
echo "  Sequential — NO parallel runs"
echo "============================================"
mkdir -p ${RESULTS}/logs

EXP=2way_it_vrand

run_train $EXP
patch_checkpoint_step $EXP

for v in 1003 1006; do
  run_eval eval_${EXP}_v${v}.yaml ${RESULTS}/results_${EXP}_v${v}
  gen_plots results_${EXP}_v${v}
done

echo "========================================"
echo "2-WAY IT TRAINING + EVAL + PLOTS COMPLETE"
echo "========================================"
