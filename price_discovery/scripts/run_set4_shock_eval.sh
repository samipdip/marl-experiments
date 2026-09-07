#!/usr/bin/env bash
# Set4 shock-eval (Phase 1a): deterministic step in F_t, eval-only, on BOTH checkpoint arms.
#   arm beliefs_pop_240     = set4 gamma=1.0  (frozen MMs)
#   arm beliefs_pop_240_g99 = set4 gamma=0.99 (unfrozen MMs)
# Configs carry a __CKPT_ROOT__ placeholder sed-patched per arm.
set -e

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

VENV=$HOME/marl-experiments/.venv/bin/python
CFGDIR=$HOME/marl-experiments/price_discovery/configs/market_reset/set4_beliefs_240_shock
RESULTS=$HOME/marl-experiments/price_discovery

mkdir -p ${RESULTS}/logs
echo "============================================"
echo "  Set4 SHOCK eval (Phase 1a): deterministic F-step, both arms"
echo "============================================"

for ARM in beliefs_pop_240 beliefs_pop_240_g99; do
  for SCEN in step_bull step_bear; do
    CFG_TEMPLATE="${CFGDIR}/eval_${SCEN}.yaml"
    CFG_RUN="/tmp/eval_${SCEN}_${ARM}.yaml"
    sed "s|__CKPT_ROOT__|${ARM}|g" "${CFG_TEMPLATE}" > "${CFG_RUN}"
    OUTDIR="${RESULTS}/results_shock_${ARM}_${SCEN}"
    echo "=== ARM=${ARM}  SCEN=${SCEN} ==="
    rm -rf "$OUTDIR"; mkdir -p "$OUTDIR"
    for ep in $(seq 0 29); do
      seed=$((10000 + ep))
      epdir="${OUTDIR}/ep_${ep}"; mkdir -p "$epdir"
      PYTHONUNBUFFERED=1 $VENV -c "
import sys, os
sys.path.insert(0, os.path.expanduser('~/SimExchangeSandbox'))
sys.path.insert(0, os.path.expanduser('~/Trading_agents'))
from phase3_multi_agent.src.runner import run_simulation
run_simulation('${CFG_RUN}', device='cpu', seed=${seed},
               results_dir='${epdir}', episode_ticks=0)
" 2>&1 | tail -1
    done
    echo "  DONE eval ${ARM} ${SCEN}"
    cd ${RESULTS}
    for p in plot_belief_avg plot_belief_actions plot_ppo_hjb_avg plot_fills_avg plot_shock_response; do
      $VENV scripts/${p}.py results_shock_${ARM}_${SCEN} 240 15 >/dev/null 2>&1 || true
    done
    echo "  PLOTS done ${ARM} ${SCEN}"
  done
done

echo "========================================"
echo "SET4 SHOCK EVAL (Phase 1a) COMPLETE"
echo "========================================"
