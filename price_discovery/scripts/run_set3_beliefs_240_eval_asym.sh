#!/usr/bin/env bash
set -e

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

VENV=$HOME/marl-experiments/.venv/bin/python
CFGDIR=$HOME/marl-experiments/price_discovery/configs/market_reset/set3_beliefs_240
RESULTS=$HOME/marl-experiments/price_discovery

echo "============================================"
echo "  EVAL ONLY: asymmetric fixed beliefs, 240-step"
echo "  bull (1003/1005), flat (1000/1006)"
echo "============================================"

for EXP in bull flat; do
  CFG_PATH="${CFGDIR}/eval_beliefs_${EXP}.yaml"
  OUTDIR="${RESULTS}/results_beliefs_240_${EXP}"
  echo "=== EVAL beliefs_240_${EXP} ==="
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
  echo "  DONE eval ${EXP}"

  cd ${RESULTS}
  $VENV scripts/plot_belief_avg.py results_beliefs_240_${EXP} 240 15 2>/dev/null
  $VENV scripts/plot_ppo_hjb_avg.py results_beliefs_240_${EXP} 240 2>/dev/null
  $VENV scripts/plot_fills_avg.py results_beliefs_240_${EXP} 240 2>/dev/null
  echo "  PLOTS ${EXP}"
done

echo "========================================"
echo "EVAL + PLOTS COMPLETE"
echo "========================================"
