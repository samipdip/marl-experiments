#!/usr/bin/env bash
set -e

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

VENV=$HOME/marl-experiments/.venv/bin/python
CFGDIR=$HOME/marl-experiments/price_discovery/configs/market_reset/set5_matched_7x3_g99
RESULTS=$HOME/marl-experiments/price_discovery
EXP=matched_7x3_g99

echo "============================================"
echo "  Set5-matched: 7 PPO-MM + 3 BeliefTrader, noise lambda=0.47 (stage-2)"
echo "  Ratio-matched to sandbox set4 (15MM+6BT,lam=1): noise:depth 1.01x, informed:depth 1.07x"
echo "============================================"
mkdir -p ${RESULTS}/logs

# ── TRAIN ──────────────────────────────────────────────────────────────
TRAIN_CFG="${CFGDIR}/train_matched.yaml"
echo "=== TRAIN ${EXP} ==="
cd $HOME/marl-experiments
rm -rf /home/samip/checkpoints/${EXP}
PYTHONUNBUFFERED=1 $VENV -c "
import sys, os
sys.path.insert(0, os.path.expanduser('~/SimExchangeSandbox'))
sys.path.insert(0, os.path.expanduser('~/Trading_agents'))
from phase3_multi_agent.src.runner import run_simulation
run_simulation('${TRAIN_CFG}', device='cpu', seed=42)
" > ${RESULTS}/logs/${EXP}_train.log 2>&1
echo "  DONE: train"

# ── patch eval configs with the real final step ────────────────────────
CKPT_DIR="/home/samip/checkpoints/${EXP}"
STEP=$(find "$CKPT_DIR" -name "*_final.pt" -print -quit 2>/dev/null | grep -oP 'step_\K\d+(?=_final)')
if [ -z "$STEP" ]; then
  echo "  ERROR: No final checkpoint found in ${CKPT_DIR}!"; exit 1
fi
echo "  Checkpoint step: ${STEP}"
for cfg in "${CFGDIR}"/eval_*.yaml; do
  [ -f "$cfg" ] || continue
  sed -i "s/step_[A-Za-z0-9]*_final/step_${STEP}_final/g" "$cfg"
done
echo "========================================"
echo "TRAIN COMPLETE"
echo "========================================"

# ── EVAL (calm, bull, bear) ────────────────────────────────────────────
for SCEN in calm bull bear; do
  CFG_PATH="${CFGDIR}/eval_matched_${SCEN}.yaml"
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
  $VENV scripts/plot_belief_avg.py    results_${EXP}_${SCEN} 240 7     2>/dev/null || true
  $VENV scripts/plot_belief_actions.py results_${EXP}_${SCEN} 240      2>/dev/null || true
  $VENV scripts/plot_ppo_hjb_avg.py   results_${EXP}_${SCEN} 240      2>/dev/null || true
  $VENV scripts/plot_fills_avg.py     results_${EXP}_${SCEN} 240      2>/dev/null || true
  echo "  PLOTS done ${SCEN}"
done

echo "========================================"
echo "SET5 TRAIN + 3 EVALS + PLOTS COMPLETE"
echo "========================================"
