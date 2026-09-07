#!/usr/bin/env bash
set -e

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

VENV=$HOME/marl-experiments/.venv/bin/python
RESULTS=$HOME/marl-experiments/price_discovery
CFGDIR=$RESULTS/configs/market_reset/set2R_vrand_ppo_unif

run_eval() {
  local CFG_NAME=$1
  local OUTDIR=$2
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
run_simulation('${CFGDIR}/${CFG_NAME}', device='cpu', seed=${seed},
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
echo "  set2R_vrand_ppo + UniformMM evals"
echo "  Host: $(hostname)   Sequential"
echo "=================================================="

for side in it opp; do
  for v in 1002 1003 1004 1005 1006; do
    run_eval eval_2dppo_unif_${side}_vrand_v${v}.yaml ${RESULTS}/results_2dppo_unif_${side}_vrand_v${v}
    gen_plots results_2dppo_unif_${side}_vrand_v${v}
  done
done

echo "=================================================="
echo "ALL UNIF EVALS COMPLETE ($(date))"
echo "=================================================="
