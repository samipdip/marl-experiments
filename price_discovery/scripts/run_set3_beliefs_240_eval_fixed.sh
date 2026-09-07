#!/usr/bin/env bash
set -e

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

VENV=$HOME/marl-experiments/.venv/bin/python
CFGDIR=$HOME/marl-experiments/price_discovery/configs/market_reset/set3_beliefs_240
RESULTS=$HOME/marl-experiments/price_discovery
EXP=beliefs_240_fixed

echo "============================================"
echo "  EVAL ONLY: beliefs fixed 1003 / 997, 240-step"
echo "============================================"

CFG_PATH="${CFGDIR}/eval_beliefs_fixed.yaml"
OUTDIR="${RESULTS}/results_${EXP}"
echo "=== EVAL ${EXP} ==="
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
echo "  DONE eval"

cd ${RESULTS}
$VENV scripts/plot_belief_avg.py results_${EXP} 240 15 2>/dev/null
$VENV scripts/plot_ppo_hjb_avg.py results_${EXP} 240 2>/dev/null
$VENV scripts/plot_fills_avg.py results_${EXP} 240 2>/dev/null
echo "  PLOTS done"

echo "========================================"
echo "EVAL + PLOTS COMPLETE"
echo "========================================"
