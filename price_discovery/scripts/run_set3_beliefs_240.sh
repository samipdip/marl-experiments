#!/usr/bin/env bash
set -e

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

VENV=$HOME/marl-experiments/.venv/bin/python
CFGDIR=$HOME/marl-experiments/price_discovery/configs/market_reset/set3_beliefs_240
RESULTS=$HOME/marl-experiments/price_discovery
EXP=beliefs_240

echo "============================================"
echo "  2 BeliefTraders (v ~ U[994,1006]) vs 15 PPO 2D MMs, 240-step"
echo "  Sequential - NO parallel runs"
echo "============================================"
mkdir -p ${RESULTS}/logs

TRAIN_CFG="${CFGDIR}/train_beliefs.yaml"
echo "=== TRAIN ${EXP} (${TRAIN_CFG}) ==="
cd $HOME/marl-experiments
rm -rf /home/samip/checkpoints/${EXP}
PYTHONUNBUFFERED=1 $VENV -c "
import sys, os
sys.path.insert(0, os.path.expanduser('~/SimExchangeSandbox'))
sys.path.insert(0, os.path.expanduser('~/Trading_agents'))
from phase3_multi_agent.src.runner import run_simulation
run_simulation('${TRAIN_CFG}', device='cpu', seed=42)
" > ${RESULTS}/logs/${EXP}.log 2>&1
echo "  DONE: ${EXP}"

CKPT_DIR="/home/samip/checkpoints/${EXP}"
STEP=$(find "$CKPT_DIR" -name "*_final.pt" -print -quit 2>/dev/null | grep -oP 'step_\K\d+(?=_final)')
if [ -z "$STEP" ]; then
  echo "  ERROR: No final checkpoint found in ${CKPT_DIR}!"
  exit 1
fi
echo "  Checkpoint step: ${STEP}"
for cfg in "${CFGDIR}"/eval_*.yaml; do
  [ -f "$cfg" ] || continue
  sed -i "s/step_[0-9]*_final/step_${STEP}_final/g" "$cfg"
  echo "  Patched: $(basename $cfg)"
done

echo "========================================"
echo "TRAIN COMPLETE"
echo "========================================"

CFG_PATH="${CFGDIR}/eval_beliefs.yaml"
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
$VENV scripts/plot_belief_avg.py results_${EXP} 240 2>/dev/null
$VENV scripts/plot_ppo_hjb_avg.py results_${EXP} 240 2>/dev/null
$VENV scripts/plot_fills_avg.py results_${EXP} 240 2>/dev/null
echo "  PLOTS done"

echo "========================================"
echo "TRAIN + EVAL + PLOTS COMPLETE"
echo "========================================"
