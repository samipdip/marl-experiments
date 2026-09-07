#!/usr/bin/env bash
# Launch the FastBoxEx TUI dashboard in tmux, restarting it forever until you
# kill the tmux session. If the TUI exits (crash, 'Q' pressed, exchange drop),
# it relaunches after 3s — the dashboard is always available.
#
# Run ON QFRCAI, from arena_live/:
#     bash watch_tui.sh            (session name "tui")
#     bash watch_tui.sh myview     (custom tmux session name)
#
# Attach from any terminal:
#     tmux attach -t tui           (detach with Ctrl-b d)
#
# STOP it (this is the only way — pressing Q inside just restarts it):
#     tmux kill-session -t tui
#
# First launch asks for the account password inside the TUI (username comes
# from Config.toml / .env). As the arena owner you see ALL participants
# (joiners' portfolios, orders, fills) — the "professor view".

set -e
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

TUI="$DIR/bin/fastboxex-tui"
SESSION="${1:-tui}"

if [ ! -x "$TUI" ]; then
    echo "TUI binary missing at $TUI" >&2
    echo "copy it: cp ~/Workspace/fastboxex-client/fastboxex-tui bin/" >&2
    exit 1
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "tmux session '$SESSION' already running — attach with: tmux attach -t $SESSION"
    exit 0
fi

tmux new-session -d -s "$SESSION" "
  while true; do
    '$TUI' -c '$DIR/Config.toml'
    echo ''
    echo '=== TUI exited -- restarting in 3s ==='
    echo '=== (to STOP the dashboard: tmux kill-session -t $SESSION) ==='
    sleep 3
  done
"
sleep 1
echo "TUI started in tmux session '$SESSION' (auto-restarts forever)."
echo "  attach:     tmux attach -t $SESSION"
echo "  stop:       tmux kill-session -t $SESSION"
