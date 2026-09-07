#!/usr/bin/env bash
set -e
RESULTS=$HOME/marl-experiments/price_discovery
VENV=$HOME/marl-experiments/.venv/bin/python
cd $RESULTS

# All experiment result directories
ALL_DIRS=(
  # Set 1 (fixed v)
  results_v1003_ppo_reset
  results_v1003_mixed_reset
  results_v1006_ppo_reset
  results_v1006_mixed_reset
  # Set 1R (random v)
  results_vrand_ppo_v1002
  results_vrand_ppo_v1003
  results_vrand_ppo_v1004
  results_vrand_ppo_v1005
  results_vrand_ppo_v1006
  results_vrand_mixed_v1002
  results_vrand_mixed_v1003
  results_vrand_mixed_v1004
  results_vrand_mixed_v1005
  results_vrand_mixed_v1006
  # Set 2 (fixed v, 2D)
  results_2d_it_v1003
  results_2d_lt
  results_2d_opp_v1003
  results_2d_same_v997
  # Set 2R (random v, 2D)
  results_2d_it_vrand_v1002
  results_2d_it_vrand_v1003
  results_2d_it_vrand_v1004
  results_2d_it_vrand_v1005
  results_2d_it_vrand_v1006
  results_2d_opp_vrand_v1002
  results_2d_opp_vrand_v1003
  results_2d_opp_vrand_v1004
  results_2d_opp_vrand_v1005
  results_2d_opp_vrand_v1006
  results_2d_same_vrand_v1002
  results_2d_same_vrand_v1003
  results_2d_same_vrand_v1004
  results_2d_same_vrand_v1005
  results_2d_same_vrand_v1006
)

for d in "${ALL_DIRS[@]}"; do
  if [ ! -d "$d" ]; then
    echo "MISSING: $d"
    continue
  fi
  # Check if pnl_history exists
  if [ ! -f "$d/ep_0/pnl_history.csv" ]; then
    echo "NO PNL: $d"
    continue
  fi
  echo -n "Plotting $d ... "
  $VENV scripts/plot_eval_avg.py "$d" 40 2>&1 | tail -1
  $VENV scripts/plot_ppo_hjb_avg.py "$d" 40 2>&1 | tail -1
  $VENV scripts/plot_fills_avg.py "$d" 40 2>&1 | tail -1
done

echo "=== ALL PLOTS DONE ==="
