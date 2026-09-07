#!/usr/bin/env bash
set -e

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

VENV=$HOME/marl-experiments/.venv/bin/python
CFGDIR=$HOME/marl-experiments/price_discovery/configs/market_reset/set2R_2way_ppo_240_direct
RESULTS=$HOME/marl-experiments/price_discovery
EXP=2way_it_vrand_240_direct

for v in 1003 1006; do
  CFG_PATH="${CFGDIR}/eval_2way_it_vrand_v${v}.yaml"
  OUTDIR="${RESULTS}/results_${EXP}_v${v}"
  echo "=== EVAL v${v} ==="
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
  echo "  DONE v${v}"

  cd ${RESULTS}
  $VENV scripts/plot_eval_avg.py results_${EXP}_v${v} 240 2>/dev/null
  $VENV scripts/plot_ppo_hjb_avg.py results_${EXP}_v${v} 240 2>/dev/null
  $VENV scripts/plot_fills_avg.py results_${EXP}_v${v} 240 2>/dev/null
  echo "  PLOTS v${v}"
done

echo "========================================"
echo "EVAL + PLOTS COMPLETE"
echo "========================================"
