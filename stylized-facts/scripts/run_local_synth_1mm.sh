#!/usr/bin/env bash
# Local synthetic-market experiment: 1 PPO HJB MM + 5 Uniform MMs + 2 Poisson noise.
# Train (short), patch eval checkpoint step, eval 30 episodes, dump CSVs for the notebook.
set -e

VENV=$HOME/Documents/marl-experiments/.venv/bin/python
REPO=$HOME/Documents/marl-experiments
CFGDIR=$REPO/price_discovery/configs/local_synth_1mm
EXP=local_synth_1mm
CKPT=$REPO/price_discovery/checkpoints_local/${EXP}
OUT=$REPO/price_discovery/results_${EXP}

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

mkdir -p $REPO/price_discovery/logs
cd $REPO

echo "=== TRAIN ${EXP} ==="
rm -rf "$CKPT"
$VENV -c "
import sys, os
sys.path.insert(0, os.path.expanduser('~/Documents/SimExchangeSandbox'))
sys.path.insert(0, os.path.expanduser('~/Documents/Trading_agents'))
from phase3_multi_agent.src.runner import run_simulation
run_simulation('${CFGDIR}/train.yaml', device='cpu', seed=42)
" > $REPO/price_discovery/logs/${EXP}_train.log 2>&1
echo "  train done"

STEP=$(find "$CKPT" -name "*_final.pt" -print -quit 2>/dev/null | grep -oP 'step_\K\d+(?=_final)')
if [ -z "$STEP" ]; then
  echo "  ERROR: no final checkpoint in $CKPT"; exit 1
fi
echo "  final step: $STEP"
sed -i "s/step_[0-9]*_final/step_${STEP}_final/g" "${CFGDIR}/eval.yaml"

echo "=== EVAL ${EXP} (30 episodes) ==="
rm -rf "$OUT"; mkdir -p "$OUT"
for ep in $(seq 0 29); do
  seed=$((10000 + ep))
  epdir="${OUT}/ep_${ep}"; mkdir -p "$epdir"
  $VENV -c "
import sys, os
sys.path.insert(0, os.path.expanduser('~/Documents/SimExchangeSandbox'))
sys.path.insert(0, os.path.expanduser('~/Documents/Trading_agents'))
from phase3_multi_agent.src.runner import run_simulation
run_simulation('${CFGDIR}/eval.yaml', device='cpu', seed=${seed},
               results_dir='${epdir}', episode_ticks=0)
" > "${epdir}/run.log" 2>&1
  echo "  ep_${ep} done"
done

echo "=== DONE: results in ${OUT} ==="
