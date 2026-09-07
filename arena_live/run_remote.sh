#!/usr/bin/env bash
# LOCAL launcher: run live-arena scenarios on QFRCAI inside tmux 'live'.
#
# Usage (from local machine, anywhere in arena_live/):
#     bash run_remote.sh calm bull bear neutral    # default 10s TUI hold
#     bash run_remote.sh 30 calm                   # explicit hold secs
#
# Detach from the chain with Ctrl-b d (run continues on QFRCAI).
# Watch the dashboard:  bash watch_tui_local.sh   (separate terminal)
# Progress:             bash status_remote.sh

set -e
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

HOLD=10
if [[ "${1:-}" =~ ^[0-9]+$ ]]; then
    HOLD="$1"
    shift
fi
[ $# -gt 0 ] || { echo "usage: run_remote.sh [hold_secs] <scenario...> (calm|bull|bear|neutral|biased)"; exit 1; }
SCENARIOS="$*"

REMOTE_CMD="
set -e
cd ~/marl-experiments/arena_live
mkdir -p logs
if tmux has-session -t live 2>/dev/null; then
    pane_pid=\$(tmux list-panes -t live -F '#{pane_pid}' 2>/dev/null || true)
    if [ -n \"\$pane_pid\" ] && ps --ppid \"\$pane_pid\" -o comm= 2>/dev/null | grep -qE 'orchestrator|run_scenarios|tee'; then
        echo 'REFUSED: a live chain is already running in tmux session live.'
        echo '  watch it: ssh -J ZhongMa QFRCAI -t \"tmux attach -t live\"'
        exit 1
    fi
    tmux kill-session -t live
fi
tmux new-session -d -s live 'bash run_scenarios_live.sh $HOLD $SCENARIOS 2>&1 | tee logs/live.log'
echo \"chain started in tmux 'live' (hold ${HOLD}s): $SCENARIOS\"
"

ssh -J ZhongMa QFRCAI -o ClearAllForwardings=yes "$REMOTE_CMD"
echo
echo "attach to the chain:   ssh -J ZhongMa QFRCAI -t 'tmux attach -t live'  (Ctrl-b d detaches)"
echo "watch the dashboard:   bash watch_tui_local.sh"
