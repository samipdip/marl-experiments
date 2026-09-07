# SyntheticMarket — Design Document

Status: **planning draft — nothing is fixed.** Single source of truth for the
experiment design. All prior docs merged here.

---

## 1. Goal

**Populate the FastboxEx virtual exchange with many agents so the arena produces a
realistic synthetic market** — an event stream statistically similar to real market
data. The matching engine handles arbitrary book depth; there is no depth
limitation to design around.

Market generation goal, not agent performance. Success is measured on the market's
output, never on any agent's PnL.

The feedback loop already exists end-to-end: **arena records its event stream →
stats compare against real market data → adjust population → repeat.**

---

## 2. Core philosophy (agreed stance)

1. **Train everyone together; run them together; study what happens.** Pure
   co-training. The object of study is the *trajectory of the ecology*, not a final
   converged policy. Training never "ends" — checkpoints are snapshots along the
   trajectory; "running the market" is eval-mode on a snapshot.

2. **Bounded rationality = striving under constraints.** Agents don't reach
   optimality, but they *try* — everyone wants profit/goal completion, and the
   striving is the dynamics. RL captures exactly this. "Bounded" is engineered
   through *constraints* — what an agent observes, how often it may act, what
   actions cost, inventory limits — not through weaker algorithms.

3. **Heterogeneity is the diversity engine.** Diverse RL algos, reward structures,
   and training configurations across (and within) species. Different inductive
   biases fall into different local optima → genuine behavioral diversity. The
   framework already supports this: each learner is self-contained (own network,
   buffer, reward, schedule); the market only ever sees orders.

4. **CDA equilibrium ≠ Kyle equilibrium.** Do not expect mid → v. In Kyle, a
   neutral competitive pricer sets p = E[v|flow], so convergence to v is baked in.
   In a double auction, MMs are strategic inventory-constrained quoters defending
   against adverse selection; the stationary state is *zero informed profit* —
   quotes walk until the IT's edge closes. That state can sit anywhere, including
   past v (price_discovery's "overshoot" was an equilibrium completing its defense,
   not a failure).

5. **Ecology composition over individual sophistication.** Realism lives in species
   mix, proportions, who supplies/takes liquidity and why, who knows what. The
   market is a mechanical majority with an adaptive minority.

---

## 3. Two environments, one agent codebase

| | **Sim — training ground** | **Live — deployment** |
|---|---|---|
| Engine | `_sim_exchange` via `SimExchangeSandbox/phase3_multi_agent` | FastboxEx server (arena, ZMQ) |
| Clock | Tick grid; `env_dt` is ours to define (the engine has **no time notion — we introduce it**) | Wall-clock (`time.sleep` cooperative loops, ns `OrderScheduler`) |
| Connection | `SimConnection` (`submit_limit/market`, `cancel_by_tag`, `drain_acks/fills`, `get_md`) | `PyConnection` + `PyMarketState` (near-identical verbs, over ZMQ) |
| Speed | Fast-forward (40k ticks in minutes) — the only place PPO is feasible | Real-time |
| Recording | Self-logged CSVs | Exchange event recording (same schema as real market data) |

**Adapter layer.** Agents are written against the minimal connection+market-data
interface both environments provide. A `FastboxAdapter` wraps
`PyConnection`/`PyMarketState` behind the same verbs, so the same agent class
trains in sim and deploys live unchanged.

**Time mapping.** `1 sim tick = δ wall seconds` (e.g. δ=0.1s, matching the existing
`LOOP_SECS` convention). Agent event rates calibrated from data (events/sec per
type → per-tick probabilities). Jittered schedulers parameterized in ticks in sim,
milliseconds live — same distribution, different unit.

**Scale mapping.** For a realism claim, match the instrument's conventions instead
of the abstract sim scale (e.g. a typical equity: tick 0.01 = 100 raw units, lot=100
shares; live examples use `REF=100.00, TICK=100, LOT=100`). Config-level change.

---

## 4. Timing model: jittered scheduling

Current sim agents fire deterministically (`tick % period == phase`): synchronized
requoting, no arrival uncertainty. Real markets have stochastic arrivals.

**`JitteredScheduler`** (agent-side, `Agents_phase2/base.py`; framework untouched):

- After each fire, resample: `interval = max(1, round(period · (1 + jitter·N(0,1))))`
  — literally *t ± dt*. Bernoulli/Poisson variants behind a `dist=` flag later.
- Randomized **entry time**: first fire ~ `Uniform(entry_window)` — staggered
  market entry.
- Same interface as `PeriodicScheduler` (`should_fire(tick_count)`); global RNG
  keeps runs reproducible.
- Live equivalent: jittered wall-clock timers (same distribution, ms units).

---

## 5. Continuous market (no resets)

No market resets, in sim or live.

- **γ < 1** (0.99–0.999): continuing task; PPO buffer boundaries are not episodes.
- **Per-agent virtual episodes**: LT tasks cycle (acquire Q over my next N
  decisions, restart); MM penalties run continuously (running φq², no terminal
  penalty).
- **v-jump process**: latent fundamental `v ← v + N(0, σ_v)` at Poisson times.
  ITs observe (noisy) current v; repeated discovery-then-equilibrium cycles with
  no resets. Jump intensity/size calibratable from large mid moves in the data.
- **Windowed eval**: metrics on rolling windows, not episodes.

---

## 6. Agent taxonomy

Reused from `Trading_agents/Agents_phase2/RL_Agents/ABIDES_MARL/`:
`PPOHjbMarketMaker` (learns κ_bid/κ_ask), `KyleInformedTrader` (β policy),
`KyleLiquidityTrader` (θ execution), `HjbMarketMaker` (analytical),
`PoissonNoiseTrader`.

New types (priority order):

1. **Fundamental/value trader** — noisy signal `v̂ = v + ε`; trades toward it.
2. **Mean-reversion / arbitrageur** — trades mid-vs-reference deviations.
3. **Momentum trader** — follows recent returns.
4. **Trainable retail/noise** — small random-demand market orders.

**Heterogeneous algorithms welcome.** Any learner satisfying
`on_tick/on_fill/get_metrics` can join: PPO, DQN, bandits, rule-based adaptive
(ZIP/GD-class). Mixing algos across species is a feature (diverse inductive
biases → diverse local optima).

---

## 7. Population design

| Layer | Contents | Role | Ablated |
|---|---|---|---|
| **Floor** | 1× `ArenaGenerator` (OU two-ring MM — proven live) | Keeps a sane book shape | Last |
| **Flow** | Calibrated takers (Poisson/Hawkes, sizes from data); optionally MNLIM bot | Realistic arrival burstiness | Late |
| **Adaptive core** | Many instances per species: PPO MMs, ITs, LTs, value, arb, momentum — mixed algos/configs | The striving population | Never |

- Start **15–25 agents** (sim CPU budget); scale after realism metrics hold.
- **Many instances per species**, not one-of-each: smooths the flow each agent
  faces (queue-position luck averages out) and enables within-species speciation
  (as the 5 PPO MMs demonstrated).
- All parameters drawn from `[lo, hi]` ranges (factory `_resolve_param` in sim;
  launcher draws live).

---

## 8. Co-training dynamics (ecology governance)

The framework needs no structural change for heterogeneous co-training. What must
be *designed* is governance:

1. **Timescales are a design knob, not a bug.** A bandit adapts in ~10 steps, PPO
   in ~10⁴. Fast learners optimize against the current population; slow learners
   finish inside a regime that's gone. Pathology for convergence, *feature for
   realism* (HFT vs pension fund). Decide deliberately which species are nimble
   and which sluggish; set configs accordingly.
2. **Cold-start ownership.** Whoever learns first owns the regime in which everyone
   else's data is generated. Worst case: fast ITs exploit naive MMs until MM value
   nets break before defense emerges. Mitigation = **incubation**: entropy floors,
   annealed reward subsidies for vulnerable species, or staggered starts.
3. **Reward-scale comparability within species.** Rewards are agent-local and need
   never be compared — but adaptation *speed* depends on gradient scale, so set
   scales intentionally per species.
4. **Path dependence.** A co-trained ecology is a complex dynamical system; one
   seed says nothing. The ensemble over seeds is the result. Budget multiple seeds
   from the start.
5. **Measurement protocol** (the real deliverable of each run):
   - rolling market stats along the trajectory
   - per-species PnL accounting (who profits from whom)
   - exploitability probes: does a simple predatory strategy print money at any
     point? (weak-efficiency check)
   - checkpoint snapshots at fixed intervals → each is a runnable market

---

## 9. Realism metrics (success criteria)

Primary — market level (vs real market stats):
spread/depth/book-shape distributions; inter-event times by order type; order-size
distributions; cancel rates; queue dynamics; signature plot; impact curves;
distributional distance metrics.

Secondary — species plausibility: MMs earn spread but lose to informed; execution
tracks schedules; informed profit exists early after v-jumps and decays as quotes
adjust.

Tertiary — weak efficiency: no trivially exploitable free lunch survives.

Never: agent-level "optimality", or mid = v.

---

## 10. Path: training → deployment

1. **Stage 0 — calibrate** sim from real market data (arrival rates, sizes, cancels,
   depth profile, κ, v-jump estimates).
2. **Stage 1 — dumb population live.** OU floor + simple jittered takers in one
   arena; verify the arena → recording → stats loop end-to-end.
3. **Stage 2 — co-train the adaptive core in sim.** All species together,
   continuous market, jittered schedules, incubation governance (§8). Multiple
   seeds; snapshot checkpoints along the trajectory.
4. **Stage 3 — deploy a snapshot to the arena** via the adapter (floor + flow +
   adaptive core, wall-clock, frozen). Record the event stream.
5. **Stage 4 — measure** (§9) vs real data → iterate population config / rewards /
   rates. Fix the worst mismatch first (likely arrival rates or sizes, not the RL).
6. **Stage 5 — growth.** Scale population; ablate the OU floor last; optional live
   fine-tuning (small LR) so the ecology keeps adapting on the arena.

---

## 11. Repo layout (SyntheticMarket)

```
SyntheticMarket/
  Experiment-design/     # this document
  configs/               # sim configs (YAML) + live bot manifests
  populations/           # population specs (counts + param ranges)
   scripts/               # calibration (data→params), config generator, runners
  adapter/               # SimConnection ⇄ PyConnection adapter layer
  results/               # gitignored
```

Constraints: `phase3_multi_agent` frozen; agent code in `Trading_agents`;
exchange/SDK untouched.

---

## 12. Open decisions (in order)

1. Species list + counts for the first co-trained population (§6, §7).
2. v-jump process: fixed schedule vs calibrated from data?
3. Incubation mechanism: entropy floors vs annealed subsidies vs staggered starts?
4. Stage-2 background: parametric calibrated flow vs HawkesNN/MNLIM generator?
5. Instrument mapping: abstract sim scale vs real instrument conventions?
6. Adapter boundary: exact minimal verb set both environments satisfy.
7. Metric suite ownership: reuse existing eval tooling vs fresh stats?
