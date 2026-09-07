#!/usr/bin/env bash
# From your LOCAL machine: attach to (or start) the TUI dashboard on QFRCAI.
#
# Usage:  bash watch_tui_local.sh          (detach with Ctrl-b d)
#
# Requires ~/.ssh/config entries: ZhongMa (jump) and QFRCAI.

exec ssh -J ZhongMa QFRCAI -o ClearAllForwardings=yes -t '
  cd ~/marl-experiments/arena_live
  tmux has-session -t tui 2>/dev/null || bash watch_tui.sh
  tmux attach -t tui
'
