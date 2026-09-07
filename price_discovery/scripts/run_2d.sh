#!/usr/bin/env bash
set -e

VENV=~/marl-experiments/.venv/bin/python
CFG=~/marl-experiments/price_discovery/configs
RESULTS=~/marl-experiments/price_discovery
CKPT=~/checkpoints

RUN_TRAIN() {
  local NAME=$1
  local CFG_PATH=$(find "${CFG}" -maxdepth 3 -name "train_${NAME}.yaml" -print -quit)
  if [ -z "$CFG_PATH" ]; then
    echo "  ERROR: Config train_${NAME}.yaml not found!"
    return 1
  fi
  echo "=== Train ${NAME} ==="
  cd ~/marl-experiments
  PYTHONUNBUFFERED=1 $VENV -c "
import sys, os
sys.path.insert(0, os.path.expanduser('~/SimExchangeSandbox'))
sys.path.insert(0, os.path.expanduser('~/Trading_agents'))
from phase3_multi_agent.src.runner import run_simulation
run_simulation('${CFG_PATH}', device='cpu', seed=42)
" > ${RESULTS}/logs/${NAME}.log 2>&1
  echo "  Done: ${NAME}"
}

RUN_EVAL() {
  local CFG_NAME=$1
  local OUTDIR=$2
  local CFG_PATH=$(find "${CFG}" -maxdepth 3 -name "${CFG_NAME}" -print -quit)
  if [ -z "$CFG_PATH" ]; then
    echo "  ERROR: Config ${CFG_NAME} not found!"
    return 1
  fi
  echo "=== Eval ${CFG_NAME} → ${OUTDIR} ==="
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
  echo "  Done eval: ${CFG_NAME}"
}

mkdir -p ${RESULTS}/logs

# ── Set 2 (fixed v, 2D) ──
RUN_TRAIN 2d_it_v1003
RUN_TRAIN 2d_lt
RUN_TRAIN 2d_opp_v1003
RUN_TRAIN 2d_same_v997

# ── Set 2R (random v, 2D) ──
RUN_TRAIN 2d_it_vrand
RUN_TRAIN 2d_opp_vrand
RUN_TRAIN 2d_same_vrand

echo "=== All Set 2 training done ==="

# ── Eval: Set 2 ──
RUN_EVAL eval_2d_it_v1003.yaml ${RESULTS}/results_2d_it_v1003
RUN_EVAL eval_2d_lt.yaml ${RESULTS}/results_2d_lt
RUN_EVAL eval_2d_opp_v1003.yaml ${RESULTS}/results_2d_opp_v1003
RUN_EVAL eval_2d_same_v997.yaml ${RESULTS}/results_2d_same_v997

# ── Eval: Set 2R (test at v=1002..1006) ──
for exp in it opp same; do
  for v in 1002 1003 1004 1005 1006; do
    RUN_EVAL eval_2d_${exp}_vrand_v${v}.yaml ${RESULTS}/results_2d_${exp}_vrand_v${v}
  done
done

echo "=== All Set 2 evals done ==="

# ── Generate plots ──
for d in results_2d_it_v1003 results_2d_lt results_2d_opp_v1003 results_2d_same_v997; do
  echo "Plotting: ${d}"
  cd ${RESULTS}
  $VENV scripts/plot_eval_avg.py ${d} 40
  $VENV scripts/plot_ppo_hjb_avg.py ${d} 40 2>/dev/null || true
done

for exp in it opp same; do
  for v in 1002 1003 1004 1005 1006; do
    d="results_2d_${exp}_vrand_v${v}"
    echo "Plotting: ${d}"
    cd ${RESULTS}
    $VENV scripts/plot_eval_avg.py ${d} 40
    $VENV scripts/plot_ppo_hjb_avg.py ${d} 40 2>/dev/null || true
  done
done

echo "=== All Set 2 plots generated ==="
