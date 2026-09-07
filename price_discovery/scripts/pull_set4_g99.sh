#!/usr/bin/env bash
# Pull set4-g99 plots (PNG only) from QFRCAI -> local, grouped under one parent dir.
# Reaches QFRCAI via the ZhongMa jump host (faster than local->QFRCAI).
set -e

REMOTE=QFRCAI
PREFIX=$HOME/Documents/marl-experiments/price_discovery
EXP=beliefs_pop_240_g99
LOCAL_PARENT="${PREFIX}/plots_set4_6tr_coord_mtm-g99"

for SCEN in calm bull strongbull bear; do
  SRC="${REMOTE}:~/marl-experiments/price_discovery/results_${EXP}_${SCEN}/"
  DST="${LOCAL_PARENT}/${SCEN}/"
  mkdir -p "${DST}"
  echo "=== pulling ${SCEN} ==="
  rsync -avz -e "ssh -J ZhongMa" \
    --include="*.png" --exclude="*" \
    "${SRC}" "${DST}"
done

echo "DONE: plots under ${LOCAL_PARENT}/{calm,bull,strongbull,bear}/"
