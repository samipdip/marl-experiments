#!/usr/bin/env bash
# scripts/tunnel.sh — SSH port-forward tunnel to the exchange via QFRCAI01.
#
# How it works:
#   - SSH connects to <gateway> (your .ssh/config alias for QFRCAI01).
#   - Each exchange port is forwarded: localhost:<p> → QFRCAI01:127.0.0.1:<p>
#   - QFRCAI01's OUTPUT DNAT rules redirect those to the exchange.
#
# Note: "channel N: open failed: Connection refused" messages are normal for
# port offsets that have no active asset yet (exchange opens ports on demand).
# They are suppressed below — the tunnel remains fully functional.
#
# Prerequisites:
#   ~/.ssh/config must have an entry for <gateway>, e.g.:
#     Host QFRCAI01
#         HostName <ip>
#         User <your-user>
#
# Usage:
#   bash scripts/tunnel.sh QFRCAI01
#   # Keep this terminal open — Ctrl+C to close.

set -e

GATEWAY="${1:?Usage: bash scripts/tunnel.sh <ssh-config-alias>}"

PORTS=(
  8880   # QFRCAI01 nginx: dashboard + wheel downloads + docs
  11001  # Auth (login)
  11003  # Arena service
  11004  # Exchange log PUB
  11005  # Fill broadcast PUB (server-dashboard FillWorker)
  11050  # Asset XPUB relay
  11051  # Arena PUB
  11052  # Arena shared ROUTER
)
for p in $(seq 12001 12009); do PORTS+=("$p"); done  # market data  (one per asset slot)
for p in $(seq 13001 13009); do PORTS+=("$p"); done  # order ingress (one per asset slot)

FORWARDS=""
for p in "${PORTS[@]}"; do
  FORWARDS="$FORWARDS -L ${p}:127.0.0.1:${p}"
done

echo "Opening tunnel via ${GATEWAY} (${#PORTS[@]} ports)..."
echo "  Dashboard: http://localhost:8880/"
echo "  Docs:      http://localhost:8880/docs/"
echo "  Wheels:    http://localhost:8880/wheels/"
echo "  Auth:      tcp://127.0.0.1:11001"
echo "  Relay:     tcp://127.0.0.1:11050"
echo ""
echo "Press Ctrl+C to close."
echo ""

# Suppress "channel N: open failed: connect failed: Connection refused"
# These appear for port offsets the exchange hasn't opened yet (normal).
# shellcheck disable=SC2086
ssh -o ExitOnForwardFailure=no -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
    -N $FORWARDS "$GATEWAY" \
    2> >(grep -v "open failed: connect failed" >&2)

