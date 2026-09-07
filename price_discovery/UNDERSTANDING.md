# Framework Understanding

## Core Mechanism: The Loop Waits for the Agent

Every tick, `Simulation.run()` calls `a.on_tick(t)` on every agent and **blocks** until all return. The clock does not advance until every agent's `on_tick()` has finished.

## Agent's on_tick (base.py:130-140)

```
on_tick(sim_t):
  if scheduler.should_fire(tick_count):  # agent decides when to act
    drain acks
    drain fills
    read market data
    orders = step(sim_t, md)             # subclasses implement this
    execute_orders(conn, orders)
  else:
    return                                # not decision time, do nothing
```

The `PeriodicScheduler` (base.py:22-38) is purely agent-local: `tick_count % period_ticks == phase`. The agent controls its own decision cadence via `decision_dt` and `decision_phase` in the YAML.

## Architecture: Order Flow

```
                                     sim_t
                                       │
                    ┌──────────────────┴──────────────────┐
                    │                                     │
            OrderShuffler.on_tick(t)           HjbMarketMaker.on_tick(t)
                    │                                     │
            step() collects orders               step() returns orders
            from UniformMM, Poisson              (but doesn't submit)
                    │                                     │
                    └──────────────┬──────────────────────┘
                                   │
                           collect HJB's orders
                           + its own sub-agent orders
                                   │
                            shuffle ALL
                                   │
                      dispatch each order
                      through its agent's conn
                      (OrderShuffler's conn for
                       UniformMM/Poisson,
                       HJB's conn for HJB orders)
                                   │
                                   ▼
                            SimExchange
```

### Key points

- Runner calls `on_tick()` on both agents independently, shuffled each tick
- HJB gets its own `conn` for correct fill routing and PNL tracking
- HJB's `step()` returns orders but doesn't submit — it hands them (plus its conn) to OrderShuffler
- OrderShuffler collects everything, shuffles, dispatches through each agent's conn
- Phase3 runner is untouched — OrderShuffler's `on_tick()` returns `[]`

## Two Perspectives, Same Mechanism

- **Agent's view**: when the scheduler fires, the agent does compute while the runner **waits**. Agent "freezes the clock" for its own work.
- **Runner's view**: dispatches `on_tick()` to all agents and waits. Has no say in what happens inside. Clock advances only after all return.

Both describe the same synchronous dispatch loop.

## OrderShuffler (order_coordinator.py)

Meta-agent that wraps sub-agents AND collects orders from independent agents (HJB, IT).

**Decision tick flow:**
1. `step()` calls `step()` on each sub-agent (UniformMM, Poisson) → collects their orders
2. Collects orders from independent agents (HJB, IT) along with their conns
3. `random.shuffle()` interleaves all orders together
4. `_dump_metrics()` reads fill counts from sub-agents via `get_metrics()`, appends a row to CSV
5. Dispatches each order through its originating agent's conn via `execute_orders()`
6. Returns `[]` — parent `on_tick()` submits nothing

**Sub-agent loading**: sub_agents in YAML can be dicts with `class` key — OrderShuffler resolves from `Agents_phase2` and instantiates. Sub-agents have no `conn` or `market`.

**Independent agents**: registered normally in the YAML agent list, have their own `conn` and scheduler. They implement a custom `on_tick()` that sends orders + conn to OrderShuffler instead of submitting directly.

**CSV logging**: `output_path` param — appends a row on each decision tick with cumulative fill counts per depth `fill_bid_d1..d5, fill_ask_d1..d5`. Header written on first write (check `os.path.exists`).

## Phase3 Multiagent Framework (runner.py)

```
run.py → run_simulation(config_path, n_ticks)
  → factory.build(config)
      → SimEnv(market + generators)
      → Agent instances (attached to market via conn)
  → Simulation(env.market, all_agents)
  → sim.run(duration, n_ticks)
      → while not market.done:
          market.tick()
          shuffle(agents)           # random order each tick
          for a in agents: a.on_tick(sim_t)
```

The runner is **agent-agnostic**: no train/eval modes, no metrics collection, no checkpointing. Each agent self-logs its own data.

### Random Sequencing

Agents are shuffled each tick so execution order is uniformly random — no agent consistently gets priority at the exchange. This replaces the phase-ordering approach (`decision_phase`) used in earlier configs. If an agent must not be reordered relative to its sub-agents (e.g. OrderShuffler), that grouping is handled inside its own `step()` — the runner only sees the meta-agent and shuffles at that level.

The only bug fix needed in factory.py: `_resolve_param` treated any list-of-2 as a random range, but `sub_agents` is a list of 2 dicts. Fixed by checking `isinstance(val[0], (int, float))` before sampling.

## Kappa Calibration Method

Poisson noise trader with arrival rate λ. Expected fill rate at depth d:

    fill_rate(d) = count(d) / (λ · sim_duration · n_episodes)
    log(fill_rate(d)) = intercept - κ · d + ε

κ estimated by OLS. Needs ≥3 depth levels with positive counts.

## What's Been Done (Step 1)

- **kappa_cal_config.yaml**: OrderShuffler wrapping 5× UniformMarketMakers (depths 1-5) + PoissonNoiseTrader (λ=1.0). 10,000 ticks.
- **estimate_kappa.py**: reads last row of `per_step.csv`, runs OLS, saves `kappa_estimate.json` + `kappa_fit_plot.png`.
- **Result**: κ⁺=0.81, κ⁻=0.80, R²≈0.98 in `results/kappa_cal_v8/`.

