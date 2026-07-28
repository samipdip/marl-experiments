# stylized-facts

Verification of canonical **stylized facts of financial markets** on a small
synthetic market produced by the multi-agent price-discovery stack.

## What was run

A short local experiment (`local_synth_1mm`):

- **1 trainable PPO-HJB market maker** — learns quote depth κ_bid/κ_ask (γ=0.99).
- **5 Uniform market makers** — limit orders at random depths 1–5.
- **2 Poisson noise traders** — market orders, λ = 1.0 and 0.5 per side.
- Market resets at ref = 1000, tick = 1, 240 ticks/episode.
- Train: 30k steps (seed 42). Eval: **30 episodes** (seeds 10000–10029).

Success is judged on the market's output, never on any agent's PnL.

## Files

```
stylized-facts/
  synthetic_market_stylized_facts.ipynb   # the analysis notebook (executed, outputs embedded)
  configs/train.yaml                      # experiment roster (train)
  configs/eval.yaml                       # experiment roster (eval)
  scripts/run_local_synth_1mm.sh          # train -> patch checkpoint -> 30-ep eval
  figures/                                # rendered figures (also embedded in the notebook)
```

## Scorecard (from the 30-episode run)

| Stylized fact | Observed | Verdict |
|---|---|---|
| Returns heavy-tailed (excess kurtosis) | 1.47 | YES |
| Return autocorr ≈ 0 (non-zero lags) | acf1 = −0.024 | YES |
| Negative 1-tick autocorr (bid–ask bounce) | −0.024 | YES |
| Volatility clustering (ACF(\|r\|) ≫ ACF(r)) | 0.087 vs −0.024 | YES |
| Aggregational Gaussianity (kurt → 0) | 1.47 → 0.27 | YES |
| Signature plot / variance ratio < 1 | VR(2) = 0.98 | NO |
| Spread positive & right-skewed | mean 2.35, skew 1.13 | YES |
| Order-book depth concave in distance | d1=2.0, d5=9.8, d10=10.4 | YES |
| Price impact of trades (finite, signed) | ≈ 0 | weak |

**Reading:** most canonical facts are reproduced by this thin market. The main
realism gap is the **upward-trending variance ratio** (VR rises with Δt, i.e.
positive long-horizon autocorrelation / trending) rather than the
microstructure-noise signature (VR < 1 at short horizons) typical of real books.

## Reproducing

```bash
# from the marl-experiments working repo, with its .venv:
bash price_discovery/scripts/run_local_synth_1mm.sh
```

Then execute the notebook (it auto-discovers the results directory
`results_local_synth_1mm/`). Data artifacts (`.csv`, `*.pt`, raw results) are
gitignored and not shipped — the notebook embeds all outputs/figures.

## Caveats

- Only the adaptive MM logs `pnl_history` / `order_events`; noise/uniform order
  flow is observed only indirectly through the book and fills.
- Episodes are short (240 ticks) and reset, so long-memory / tail estimates are
  read on **pooled** returns and are noisy.
- A single trainable MM is a *thin* adaptive core — scale the population before
  drawing strong conclusions (see `SyntheticMarket/Experiment-design/SCOPE.md §9`).
