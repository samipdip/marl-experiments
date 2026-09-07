# NOTES — De-rounding (stage 3, DEFERRED — do not implement yet)

> User decision 2026-08-15: keep this design for later. Do not start. Stages 0–2
> of the arena-transition plan run first (see
> `~/.plannotator/plans/arena-transition-plan-v2-certi-2026-08-15-approved.md`).

## Problem

Agents act in lockstep with a global round clock:

- **Live joiners** (`arena_live/joiners/*`): decisions gated to cycle boundaries
  (`sleep to t0 + (cycle+1)*cycle_secs + U(0, jitter)`), one decision per cycle.
- **Sandbox** `PeriodicScheduler` (`Agents_phase2/base.py:22-38`):
  `tick_count % period_ticks == phase` — periodic, phase-locked.

Connections are NOT the problem — agents stay connected the whole time. What is
un-market-like is *when* they trade: markets have no global round structure.

## Target model (agreed with user)

Every agent is persistently connected and **trades at random times**:

- Each agent driven by its own independent stochastic decision clock —
  Poisson inter-decision times, per-agent rate, preserving that agent's cadence
  in expectation (e.g. rate 1/s for MMs = today's 1 decision/cycle on average).
- Optional event-driven triggers: on own fill, on md change.
- No global round structure; no episode resets.

## Work items (when this is picked up)

1. **Scheduler rework** — two places must change together:
   - local sandbox runner (`SimExchangeSandbox/phase3_multi_agent/src/`):
     replace periodic phase-locked scheduling with randomized per-agent
     decision clocks;
   - live joiner loop: decisions fire off the agent's own clock, not the cycle
     counter. Engine clock and md streaming are already continuous — only the
     decision gating changes.
2. **Persistence** — no market reset between episodes, no 240-cycle boundaries;
   book/cash/inventory carry forward; analysis shifts from per-round episodes
   to sliding windows (drift, spread, depth, capture on rolling horizons).
3. **Local retrain under randomized timing** — current g99 checkpoints were
   trained on periodic cadence; random decision clocks are a timing-distribution
   shift. Retrain locally (stage-2 roster/λ) BEFORE redeploying to the arena.
   The arena is never a training environment.

## Sequencing rationale

Stay round-based through stages 1–2: single-variable discipline — stage 1
certifies the harness and stage 2 tests composition/λ transfer with the same
timing structure the checkpoints were trained on. Folding de-rounding in earlier
would confound the transfer claim. Design doc for this stage gets written after
stage-2 results are in.
