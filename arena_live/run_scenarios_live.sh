#!/usr/bin/env bash
# Run live arena scenarios sequentially, with a pre-start hold so you can
# attach the TUI dashboard and watch the session live.
#
# Run ON QFRCAI, from arena_live/ (typically inside tmux):
#     bash run_scenarios_live.sh calm bull bear          # default 10s hold
#     bash run_scenarios_live.sh 45 bull                 # explicit hold
#
# [hold_secs]  seconds in BookOnly before trading starts (time to attach TUI);
#              optional numeric first arg, defaults to 10.
# Sessions are named <scenario>_live; each is analyzed at the end.
#
# Watch live (second terminal):  bash watch_tui.sh   ->  tmux attach -t tui
# Monitor text logs:             tail -f logs/scenarios_live.log

set -e
cd "$(dirname "$0")"

# QFRCAI-only: the joiners need the fastboxex venv + .env present there.
if [ ! -x ".venv/bin/python" ]; then
    echo "ERROR: .venv/bin/python not found - this script runs ON QFRCAI." >&2
    echo "From the local machine use:  bash run_remote.sh [hold_secs] <scenario...>" >&2
    exit 1
fi

HOLD=10
if [[ "${1:-}" =~ ^[0-9]+$ ]]; then
    HOLD="$1"
    shift
fi
[ $# -gt 0 ] || { echo "no scenarios given (calm|bull|bear|neutral|biased)"; exit 1; }

PY=.venv/bin/python
mkdir -p logs

for S in "$@"; do
    echo "=========================================="
    echo "  LIVE scenario: $S   (hold ${HOLD}s)"
    echo "=========================================="
    $PY -u orchestrator.py --scenario "$S" --session "${S}_live" --hold-secs "$HOLD" \
        2>&1 | tee -a logs/scenarios_live.log
    $PY analyze_live.py "runs/${S}_live" || true
    echo "  -> results: runs/${S}_live/ (analysis.png, summary.csv)"
done

echo "ALL LIVE SCENARIOS COMPLETE"
