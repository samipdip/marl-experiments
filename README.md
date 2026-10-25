MARL Gymnasium
## Libraries (installed as path dependencies)

- `SimExchangeSandbox` — CDA matching engine, SimEnv, factory, runner
- `Trading_agents` — Agent base classes + RL agents

## Setup

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e ../SimExchangeSandbox
uv pip install -e ../Trading_agents
uv pip install -e .
```

## Running

```bash
# Experiment 1: Price Discovery
python experiments/run_exp1.py --config configs/exp1_price_discovery/kyle_exp1_linear_20mm.yaml

# Analyze + plot
python analysis/generate_all.py --results-dir results/exp1_run_...
```
