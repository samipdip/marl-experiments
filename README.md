# ABIDES-MARL Experiments

Recreation of three experiments from Cheridiko, Dupret & Wu (arXiv:2511.02016) using the SimExchangeSandbox CDA matching engine.

## Structure

```
abides-marl-experiments/
  configs/           Experiment YAML configs (one per sweep config)
  experiments/       Orchestration scripts (train -> eval -> analyze -> plot)
  analysis/          Plotting + table generation
  results/           Output (gitignored): checkpoints, CSVs, figures, tables
```

## Libraries (installed as path dependencies)

- `SimExchangeSandbox` — CDA matching engine, SimEnv, factory, runner
- `Trading_agents` — Agent base classes + Kyle RL agents (ABIDES_MARL/)

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
