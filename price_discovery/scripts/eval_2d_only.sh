#!/usr/bin/env bash
set -e
VENV=~/marl-experiments/.venv/bin/python
CFG=~/marl-experiments/price_discovery/configs
RESULTS=~/marl-experiments/price_discovery

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
run_simulation('${CFG_PATH}', device='cpu', seed=${seed},
               results_dir='${epdir}', episode_ticks=0)
" 2>&1 | tail -1
  done
  echo "  Done eval: $CFG_NAME"
}

# Set 2
run_eval eval_2d_it_v1003.yaml ${RESULTS}/results_2d_it_v1003
run_eval eval_2d_lt.yaml ${RESULTS}/results_2d_lt
run_eval eval_2d_opp_v1003.yaml ${RESULTS}/results_2d_opp_v1003
run_eval eval_2d_same_v997.yaml ${RESULTS}/results_2d_same_v997

# Set 2R
for exp in it opp same; do
  for v in 1002 1003 1004 1005 1006; do
    run_eval eval_2d_${exp}_vrand_v${v}.yaml ${RESULTS}/results_2d_${exp}_vrand_v${v}
  done
done

echo "=== All evals done ==="

# Plots
for d in results_2d_it_v1003 results_2d_lt results_2d_opp_v1003 results_2d_same_v997; do
  cd ${RESULTS}
  $VENV scripts/plot_eval_avg.py ${d} 40
  $VENV scripts/plot_ppo_hjb_avg.py ${d} 40 2>/dev/null || true
done

for exp in it opp same; do
  for v in 1002 1003 1004 1005 1006; do
    d="results_2d_${exp}_vrand_v${v}"
    cd ${RESULTS}
    $VENV scripts/plot_eval_avg.py ${d} 40
    $VENV scripts/plot_ppo_hjb_avg.py ${d} 40 2>/dev/null || true
  done
done

echo "=== All plots done ==="
