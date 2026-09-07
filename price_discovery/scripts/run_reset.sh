#!/usr/bin/env bash
# Run Set 1 + Set 1R with market reset patch on QFRCAI_Local
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

# ── Train: Set 1 (fixed v) sequentially ──
RUN_TRAIN v1003_ppo
RUN_TRAIN v1003_mixed
RUN_TRAIN v1006_ppo
RUN_TRAIN v1006_mixed
RUN_TRAIN vrand_ppo
RUN_TRAIN vrand_mixed

echo "=== All training done ==="

# ── Eval: Set 1 (fixed v) ──
RUN_EVAL eval_v1003_ppo.yaml ${RESULTS}/results_v1003_ppo_reset
RUN_EVAL eval_v1003_mixed.yaml ${RESULTS}/results_v1003_mixed_reset
RUN_EVAL eval_v1006_ppo.yaml ${RESULTS}/results_v1006_ppo_reset
RUN_EVAL eval_v1006_mixed.yaml ${RESULTS}/results_v1006_mixed_reset

# ── Eval: Set 1R (random v, test each v) ──
for v in 1002 1003 1004 1005 1006; do
  RUN_EVAL eval_vrand_ppo_v${v}.yaml ${RESULTS}/results_vrand_ppo_v${v}
  RUN_EVAL eval_vrand_mixed_v${v}.yaml ${RESULTS}/results_vrand_mixed_v${v}
done

echo "=== All evals done ==="

# ── Generate plots ──
cd ${RESULTS}
for DIR in results_v1003_ppo_reset results_v1003_mixed_reset \
           results_v1006_ppo_reset results_v1006_mixed_reset; do
  echo "Plotting: ${DIR}"
  $VENV scripts/plot_eval_avg.py ${DIR} 40
  $VENV scripts/plot_ppo_hjb_avg.py ${DIR} 40 2>/dev/null || true
done

for EXP in vrand_ppo vrand_mixed; do
  for v in 1002 1003 1004 1005 1006; do
    DIR="${RESULTS}/results_${EXP}_v${v}"
    echo "Plotting: ${DIR}"
    $VENV scripts/plot_eval_avg.py ${DIR} 40
    $VENV scripts/plot_ppo_hjb_avg.py ${DIR} 40 2>/dev/null || true
  done
done

echo "=== All plots generated ==="
