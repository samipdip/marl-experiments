#!/usr/bin/env bash
# scripts/commit_runs.sh — Commit all new run summaries to git.
#
# Adds summary.json + metadata.json from runs/ (heavy data files are gitignored).
# Run after a strategy session to checkpoint your results.

set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

COUNT=$(find runs/ -name "summary.json" 2>/dev/null | wc -l | tr -d ' ')

if [ "$COUNT" -eq 0 ]; then
    echo "No runs/ summaries found — nothing to commit."
    exit 0
fi

git add runs/
git commit -m "runs: checkpoint $COUNT run summaries" \
    && echo "✓ Committed $COUNT run summaries." \
    || echo "Nothing new to commit."
