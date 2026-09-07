#!/usr/bin/env bash
# LOCAL: one-glance status of the live-arena chain on QFRCAI.
#
# Usage:  bash status_remote.sh
# Prints: orchestrator running?, active scenario/arena, current cycle,
# joiners alive, last log lines.

DIR="$(cd "$(dirname "$0")" && pwd)"

ssh -J ZhongMa QFRCAI -o ClearAllForwardings=yes '
cd ~/marl-experiments/arena_live 2>/dev/null || { echo "arena_live not found"; exit 1; }
echo "== tmux sessions =="
tmux ls 2>/dev/null || echo "(none)"
echo
echo "== orchestrator =="
if pgrep -f "[o]rchestrator.py" >/dev/null; then
    pgrep -af "[o]rchestrator.py" | head -2
else
    echo "not running (an arena exists only during a session)"
fi
echo
if pgrep -f "[o]rchestrator.py" >/dev/null; then
    echo "== active session state =="
    state=$(ls -t runs/*/.arena_state.json 2>/dev/null | head -1)
    if [ -n "$state" ]; then
        sed "s/\"/ /g; s/,/\n   /g" "$state" | tr -d " {}" | sed "s/^/   /"
    fi
    echo
    echo "== noise progress (current cycle) =="
    noise=$(ls -t runs/*/noise_log.csv 2>/dev/null | head -1)
    [ -n "$noise" ] && echo "   $(basename $(dirname $noise)): $(tail -1 "$noise" | cut -d, -f1-3)"
else
    meta=$(ls -t runs/*/session_meta.json 2>/dev/null | head -1)
    [ -n "$meta" ] && echo "last completed session: $meta"
fi
echo
echo "== joiner processes =="
echo "   mm=$(pgrep -fc "[j]oiners.mm_joiner" ) belief=$(pgrep -fc "[j]oiners.belief_joiner" )"
echo
echo "== live.log tail =="
tail -5 logs/live.log 2>/dev/null || tail -5 logs/matched_live.log 2>/dev/null || echo "(no live log yet)"
'
