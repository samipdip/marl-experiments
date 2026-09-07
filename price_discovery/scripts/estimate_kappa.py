"""estimate_kappa.py — Estimate kappa+ and kappa- from fill counts CSV.

Usage:
    python price_discovery/estimate_kappa.py <results_dir> [--lambda 0.7] [--duration 2000] [--episodes 1]
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson as dw_stat, jarque_bera as jb_stat
from statsmodels.stats.stattools import omni_normtest as omnibus_stat
import matplotlib.pyplot as plt


def load_fill_counts(results_dir: str, max_depth: int = 5) -> dict:
    """Read cumulative fill counts from the LAST row of per_step.csv."""
    csv_path = Path(results_dir) / "per_step.csv"
    if not csv_path.exists():
        return {}

    last_row = None
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            last_row = row

    if last_row is None:
        return {}

    agg = {}
    for side_label, side in [("bid", 0), ("ask", 1)]:
        for d in range(1, max_depth + 1):
            key = f"fill_{side_label}_d{d}"
            val = int(last_row.get(key, 0))
            if val > 0:
                agg[(side, d)] = val
    return agg


def estimate_kappa(
    agg_fills: dict,
    lambda_: float,
    sim_duration: float,
    n_episodes: int = 1,
    max_depth: int = 5,
):
    n_arrivals = lambda_ * sim_duration * n_episodes
    results = {}

    for side_label, side in [("kappa+ (ask)", 1), ("kappa- (bid)", 0)]:
        depths = []
        log_rates = []
        rates = []
        for d in range(1, max_depth + 1):
            cnt = agg_fills.get((side, d), 0)
            rate = cnt / n_arrivals if n_arrivals > 0 else 0.0
            if rate > 0:
                depths.append(d)
                rates.append(rate)
                log_rates.append(np.log(rate))

        entry = {"depths": depths, "fill_rates": rates, "log_rates": log_rates}

        if len(depths) >= 3:
            X = sm.add_constant(depths)
            model = sm.OLS(log_rates, X).fit()
            kappa = -model.params[1]

            conf_int = model.conf_int()
            ci_depth_lower = float(conf_int[1, 0])
            ci_depth_upper = float(conf_int[1, 1])
            ci_const_lower = float(conf_int[0, 0])
            ci_const_upper = float(conf_int[0, 1])

            entry.update({
                "kappa": float(kappa),
                "intercept": float(model.params[0]),
                "r_squared": float(model.rsquared),
                "adj_r_squared": float(model.rsquared_adj),
                "f_statistic": float(model.fvalue),
                "f_p_value": float(model.f_pvalue),
                "aic": float(model.aic),
                "bic": float(model.bic),
                "nobs": int(model.nobs),
                "df_model": int(model.df_model),
                "df_resid": int(model.df_resid),
                "coef_depth": {
                    "estimate": float(model.params[1]),
                    "std_err": float(model.bse[1]),
                    "t_statistic": float(model.tvalues[1]),
                    "p_value": float(model.pvalues[1]),
                    "ci_95_lower": ci_depth_lower,
                    "ci_95_upper": ci_depth_upper,
                },
                "coef_intercept": {
                    "estimate": float(model.params[0]),
                    "std_err": float(model.bse[0]),
                    "t_statistic": float(model.tvalues[0]),
                    "p_value": float(model.pvalues[0]),
                    "ci_95_lower": ci_const_lower,
                    "ci_95_upper": ci_const_upper,
                },
                "residual_std_err": float(np.sqrt(model.mse_resid)),
                "dw_statistic": float(dw_stat(model.resid)),
                "jarque_bera": {
                    "statistic": float(jb_stat(model.resid)[0]),
                    "p_value": float(jb_stat(model.resid)[1]),
                },
                "omnibus": {
                    "statistic": float(omnibus_stat(model.resid)[0]),
                    "p_value": float(omnibus_stat(model.resid)[1]),
                },
                "condition_number": float(model.condition_number),
            })

            x_fit = np.linspace(min(depths) - 0.5, max(depths) + 0.5, 100)
            y_fit = np.exp(float(model.params[0]) + float(model.params[1]) * x_fit)
            entry["x_fit"] = x_fit.tolist()
            entry["y_fit"] = y_fit.tolist()
            entry["prediction"] = model.get_prediction(X).summary_frame().to_dict("list")

        elif len(depths) == 2:
            slope, intercept = np.linalg.lstsq(
                np.vstack([depths, np.ones_like(depths)]).T, log_rates, rcond=None
            )[0]
            kappa = -float(slope)
            x_fit = np.linspace(min(depths) - 0.5, max(depths) + 0.5, 100)
            y_fit = np.exp(intercept + slope * x_fit)
            entry.update({
                "kappa": kappa,
                "intercept": float(intercept),
                "r_squared": None,
                "n_points": 2,
                "x_fit": x_fit.tolist(),
                "y_fit": y_fit.tolist(),
                "warning": "Only 2 depths — insufficient for full stats",
            })
        else:
            entry["kappa"] = None

        results[side_label] = entry

    return results


def plot_fill_rates(results: dict, results_dir: str):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, (label, data) in zip(axes, results.items()):
        depths = data["depths"]
        rates = data["fill_rates"]
        x_fit = data.get("x_fit")
        y_fit = data.get("y_fit")
        kappa = data.get("kappa")
        ax.bar(depths, rates, width=0.4, color="steelblue", edgecolor="black", alpha=0.7, label="Observed")
        if x_fit is not None and y_fit is not None and kappa is not None:
            ax.plot(x_fit, y_fit, "r-", linewidth=2, label=f"exp(-{kappa:.4f} · d)")
        ax.set_xlabel("Depth (ticks)")
        ax.set_ylabel("Fill rate")
        ax.set_title(label)
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = Path(results_dir) / "kappa_fit_plot.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Plot saved to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Estimate kappa from fill counts")
    parser.add_argument("results_dir", type=str)
    parser.add_argument("--lambda", type=float, default=0.7, dest="lambda_")
    parser.add_argument("--duration", type=float, default=2000.0)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-depth", type=int, default=5)
    args = parser.parse_args()

    agg = load_fill_counts(args.results_dir, args.max_depth)
    if not agg:
        print("No fill data found.")
        return

    print(f"Fill counts: {dict(agg)}")

    results = estimate_kappa(agg, args.lambda_, args.duration, args.episodes, args.max_depth)

    lines = []
    lines.append("")
    lines.append("====================== Kappa Estimation ======================")
    for label, r in results.items():
        if r["kappa"] is None:
            lines.append(f"\n  {label}: insufficient data")
            continue
        lines.append(f"\n  {label}")
        lines.append(f"  {'─' * 50}")
        lines.append(f"  kappa           = {r['kappa']:.6f}")
        lines.append(f"  intercept       = {r['intercept']:.6f}")
        lines.append(f"  R²              = {r.get('r_squared', 'N/A')}")
        if "adj_r_squared" in r:
            lines.append(f"  Adj R²          = {r['adj_r_squared']:.6f}")
            lines.append(f"  F-statistic     = {r['f_statistic']:.6f}")
            lines.append(f"  F p-value       = {r['f_p_value']:.6e}")
            lines.append(f"  AIC             = {r['aic']:.2f}")
            lines.append(f"  BIC             = {r['bic']:.2f}")
            lines.append(f"  Nobs            = {r['nobs']}")
            lines.append(f"  Durbin-Watson   = {r['dw_statistic']:.4f}")
            lines.append(f"  Cond. number    = {r['condition_number']:.2f}")
            lines.append(f"\n  Depth coefficient:")
            lines.append(f"    Estimate      = {r['coef_depth']['estimate']:.6f}")
            lines.append(f"    Std err       = {r['coef_depth']['std_err']:.6f}")
            lines.append(f"    t-stat        = {r['coef_depth']['t_statistic']:.4f}")
            lines.append(f"    p-value       = {r['coef_depth']['p_value']:.6e}")
            lines.append(f"    95% CI        = [{r['coef_depth']['ci_95_lower']:.6f}, {r['coef_depth']['ci_95_upper']:.6f}]")
            lines.append(f"\n  Residual diagnostics:")
            lines.append(f"    Residual std  = {r['residual_std_err']:.6f}")
            lines.append(f"    JB statistic  = {r['jarque_bera']['statistic']:.4f}")
            lines.append(f"    JB p-value    = {r['jarque_bera']['p_value']:.4f}")
            lines.append(f"    Omnibus       = {r['omnibus']['statistic']:.4f}")
            lines.append(f"    Omnibus p     = {r['omnibus']['p_value']:.4f}")
        lines.append("")

    report = "\n".join(lines)
    print(report)

    out_path = Path(args.results_dir) / "kappa_estimate.txt"
    with open(out_path, "w") as f:
        f.write(report)
    print(f"Summary saved to {out_path}")

    plot_fill_rates(results, args.results_dir)


if __name__ == "__main__":
    main()
