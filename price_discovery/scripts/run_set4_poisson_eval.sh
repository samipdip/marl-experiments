#!/usr/bin/env bash
# Set4 Poisson-shock eval (Phase 1b): random jumps in F_t, eval-only, g99 only.
#   arm beliefs_pop_240_g99 = set4 gamma=0.99 (unfrozen MMs)
# Coordinator drives F_t as a jump random walk (f_mode: poisson). Each tick, with
# prob jump_rate, F_t += +/-U{jump_size_min..jump_size_max}; beliefs v_i = F_t + b_i.
# Config carries __CKPT_ROOT__ and __JR__ placeholders sed-patched per arm/rate.
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
echo "  Set4 POISSON eval (Phase 1b): random F jumps, g99 only"
echo "============================================"

ARM=beliefs_pop_240_g99
for JR in 0.01 0.02 0.04; do
  CFG_TEMPLATE="${CFGDIR}/eval_poisson.yaml"
  CFG_RUN="/tmp/eval_poisson_${ARM}_jr${JR}.yaml"
  sed -e "s|__CKPT_ROOT__|${ARM}|g" -e "s|__JR__|${JR}|g" "${CFG_TEMPLATE}" > "${CFG_RUN}"
  OUTDIR="${RESULTS}/results_poisson_${ARM}_jr${JR}"
  echo "=== ARM=${ARM}  JR=${JR} ==="
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
  echo "  DONE eval ${ARM} jr${JR}"
  cd ${RESULTS}
  for p in plot_belief_avg plot_belief_actions plot_ppo_hjb_avg plot_fills_avg plot_poisson_response; do
    $VENV scripts/${p}.py results_poisson_${ARM}_jr${JR} 240 15 >/dev/null 2>&1 || true
  done
  echo "  PLOTS done ${ARM} jr${JR}"
done

echo "========================================"
echo "SET4 POISSON EVAL (Phase 1b) COMPLETE"
echo "========================================"
