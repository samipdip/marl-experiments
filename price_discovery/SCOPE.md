# Price Discovery — Project Scope

Validate `phase3_multiagent` in `SimExchangeSandbox` as a platform for multi-agent market
simulation with RL potential. Supports UniformMarketMakers, PoissonNoiseTrader,
HjbMarketMaker, KyleInformedTrader.

## Framework

- `run.py` → `runner.run_simulation()` → `Simulation(market, agents)`
- Runner is agent-agnostic: advances time, calls `on_tick`/`on_fill`, collects nothing
- Each agent self-logs its own CSV metrics (OrderShuffler pattern)
- Factory parses YAML → builds env + agents; `_resolve_param` treats list-of-2 as random range unless elements are numeric

## Steps

1. **Calibrate kappa on Uniform MM + Noise Tracker** (DONE)
   - OrderShuffler wraps 5x UniformMarketMakers + 1x PoissonNoiseTrader
   - 10,000 ticks, fill-vs-depth CSV → OLS: κ⁺=0.81, κ⁻=0.80
   - Config: `kappa_cal_config.yaml`, results: `results/kappa_cal_v8/`

2. **Add HJB MM → calibrate its kappa vs Uniform MM + Noise** (DONE)
   - Same Step 1 + standalone HjbMarketMaker (analytical, non-trainable)
   - HJB posts at optimal depths (not fixed 1..5), so tracking must capture its variable-depth fills

3. **Add trainable Informed Trader (RL)**
   - Uniform MMs + Poisson + HJB MM + KyleInformedTrader (PPO)
   - Train IT only; HJB MM and uniform MMs are fixed

4. **Adaptive HJB MM (lambda, kappa update)**
   - HJB MM re-estimates κ/λ online from fills, or via small RL policy
   - Multi-agent RL with IT + adaptive HJB MM

## Key Files

| File | What |
|---|---|
| `phase3_multiagent/src/runner.py` | Agent-agnostic tick loop |
| `phase3_multiagent/src/factory.py` | YAML → env + agents |
| `order_coordinator.py` | OrderShuffler meta-agent (shuffles sub-agents, logs CSV) |
| `hjb_market_maker.py` | HjbMarketMaker (PDE solver) |
| `uniform_market_makers.py` | Posts limit orders at random depths 1..5 |
| `poisson_noise_trader.py` | Market orders at Poisson rate |
| `configs/exp1_price_discovery/exp1_hjb_train.yaml` | Reference: HJB MM + KyleNoiseTrader + KyleInformedTrader |
| `price_discovery/estimate_kappa.py` | OLS on log(fill_rate) ~ depth → κ |

## Conventions

- `decision_dt` + `decision_phase`: HJB MM at phase 0 (dt=2.0), IT at phase 1 (dt=2.0)
- CSV self-logged by each agent, not the runner
- Results in `results/<exp_name>_v<N>/` with `run_meta.json`
- Estimation: `fill_rate(d) = count(d) / (λ · duration)`, κ = -slope of log(rate) ~ depth
