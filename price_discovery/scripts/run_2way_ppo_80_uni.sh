#!/usr/bin/env bash
set -e

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

VENV=$HOME/marl-experiments/.venv/bin/python
CFGDIR=$HOME/marl-experiments/price_discovery/configs/market_reset/set2R_2way_ppo_80_uni
RESULTS=$HOME/marl-experiments/price_discovery
EXP=2way_it_vrand_80_uni

echo "============================================"
echo "  2-way IT 80-step vs 15 PPO 2D MMs (v ~ U[1000,1006])"
echo "  Sequential - NO parallel runs"
echo "============================================"
mkdir -p ${RESULTS}/logs

TRAIN_CFG="${CFGDIR}/train_2way_it_vrand.yaml"
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
