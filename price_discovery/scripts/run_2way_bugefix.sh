#!/usr/bin/env bash
set -e
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
VENV=$HOME/marl-experiments/.venv/bin/python
CFGDIR=$HOME/marl-experiments/price_discovery/configs/market_reset/set2R_2way_ppo_bugefix
RESULTS=$HOME/marl-experiments/price_discovery
EXP=2way_it_vrand_bugefix

echo "=== TRAIN ==="
cd $HOME/marl-experiments
rm -rf /home/samip/checkpoints/${EXP}
PYTHONUNBUFFERED=1 $VENV -c "
import sys, os
sys.path.insert(0, os.path.expanduser('~/SimExchangeSandbox'))
sys.path.insert(0, os.path.expanduser('~/Trading_agents'))
from phase3_multi_agent.src.runner import run_simulation
run_simulation('${CFGDIR}/train_2way_it_vrand.yaml', device='cpu', seed=42)
" > ${RESULTS}/logs/${EXP}.log 2>&1
echo "  DONE"

STEP=$(find /home/samip/checkpoints/${EXP} -name "*_final.pt" -print -quit | grep -oP 'step_\K\d+(?=_final)')
for cfg in ${CFGDIR}/eval_*.yaml; do sed -i "s/step_[0-9]*_final/step_${STEP}_final/g" "$cfg"; done

for v in 1003 1006; do
  OUTDIR="${RESULTS}/results_${EXP}_v${v}"
  rm -rf "$OUTDIR"; mkdir -p "$OUTDIR"
  for ep in $(seq 0 29); do
    seed=$((10000+ep)); epdir="${OUTDIR}/ep_${ep}"; mkdir -p "$epdir"
    PYTHONUNBUFFERED=1 $VENV -c "
import sys, os
sys.path.insert(0, os.path.expanduser('~/SimExchangeSandbox'))
sys.path.insert(0, os.path.expanduser('~/Trading_agents'))
from phase3_multi_agent.src.runner import run_simulation
run_simulation('${CFGDIR}/eval_2way_it_vrand_v${v}.yaml', device='cpu', seed=${seed},
               results_dir='${epdir}', episode_ticks=0)
" 2>&1 | tail -1
  done
  cd ${RESULTS}
  $VENV scripts/plot_eval_avg.py results_${EXP}_v${v} 120 2>/dev/null
  $VENV scripts/plot_ppo_hjb_avg.py results_${EXP}_v${v} 120 2>/dev/null
  $VENV scripts/plot_fills_avg.py results_${EXP}_v${v} 120 2>/dev/null
  echo "  PLOTS v${v}"
done
echo "ALL DONE"
