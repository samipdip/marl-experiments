#!/usr/bin/env bash
set -e
RESULTS=$HOME/marl-experiments/price_discovery
CFG=$RESULTS/configs
VENV=$HOME/marl-experiments/.venv/bin/python

run_eval() {
  local CFG_NAME=$1
  local OUTDIR=$2
  local CFG_PATH=$(find "${CFG}" -maxdepth 3 -name "${CFG_NAME}" -print -quit)
  if [ -z "$CFG_PATH" ]; then
    echo "  ERROR: Config ${CFG_NAME} not found!"
    return 1
  fi
  echo "=== Eval $CFG_NAME → $OUTDIR ==="
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
run_simulation('${CFG_PATH}', device='cpu', seed=${seed}, results_dir='${epdir}', episode_ticks=0)
" 2>&1 | tail -1
  done
  echo "  Done eval: $CFG_NAME"
}

run_eval eval_2d_lt.yaml $RESULTS/results_2d_lt
for v in 1002 1003 1004 1005 1006; do
  run_eval eval_2d_same_vrand_v${v}.yaml $RESULTS/results_2d_same_vrand_v${v}
done
echo "=== DONE ==="
