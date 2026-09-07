#!/usr/bin/env bash
# Set4-g99 EVAL ONLY. Train already completed (gamma=0.99); checkpoints rescued to
# ~/checkpoints/beliefs_pop_240_g99/. This runs the 4 eval scenarios + plots.
set -e

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

VENV=$HOME/marl-experiments/.venv/bin/python
CFGDIR=$HOME/marl-experiments/price_discovery/configs/market_reset/set4_beliefs_240_g99
RESULTS=$HOME/marl-experiments/price_discovery
EXP=beliefs_pop_240_g99

echo "============================================"
echo "  Set4-g99 EVAL-ONLY (gamma=0.99; train already done)"
echo "============================================"
mkdir -p ${RESULTS}/logs

for SCEN in calm bull strongbull bear; do
  CFG_PATH="${CFGDIR}/eval_beliefs_${SCEN}.yaml"
  OUTDIR="${RESULTS}/results_${EXP}_${SCEN}"
  echo "=== EVAL ${SCEN} ==="
  rm -rf "$OUTDIR"; mkdir -p "$OUTDIR"
  for ep in $(seq 0 29); do
    seed=$((10000 + ep))
    epdir="${OUTDIR}/ep_${ep}"; mkdir -p "$epdir"
    PYTHONUNBUFFERED=1 $VENV -c "
import sys, os
sys.path.insert(0, os.path.expanduser('~/SimExchangeSandbox'))
sys.path.insert(0, os.path.expanduser('~/Trading_agents'))
from phase3_multi_agent.src.runner import run_simulation
run_simulation('${CFG_PATH}', device='cpu', seed=${seed},
               results_dir='${epdir}', episode_ticks=0)
" 2>&1 | tail -1
  done
  echo "  DONE eval ${SCEN}"

  cd ${RESULTS}
  $VENV scripts/plot_belief_avg.py     results_${EXP}_${SCEN} 240 15    2>/dev/null || true
  $VENV scripts/plot_belief_actions.py results_${EXP}_${SCEN} 240       2>/dev/null || true
  $VENV scripts/plot_ppo_hjb_avg.py    results_${EXP}_${SCEN} 240       2>/dev/null || true
  $VENV scripts/plot_fills_avg.py      results_${EXP}_${SCEN} 240       2>/dev/null || true
  echo "  PLOTS done ${SCEN}"
done

echo "========================================"
echo "SET4-G99 EVAL + PLOTS COMPLETE"
echo "========================================"
