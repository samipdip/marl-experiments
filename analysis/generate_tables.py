"""generate_tables.py -- Generate paper-style tables (CSV + LaTeX + Markdown).

Usage:
    python -m analysis.generate_tables \
        --exp1-results results/exp1_all_configs.json \
        --exp3-results results/exp3_all_configs.json \
        --output-dir results/tables
"""

import argparse
import csv
import json
import os


def generate_table1(metrics_by_config, output_dir):
    """Table 1: Price discovery metrics (paper format)."""
    os.makedirs(output_dir, exist_ok=True)

    # CSV
    csv_path = os.path.join(output_dir, "table1_price_discovery.csv")
    fieldnames = ["act_type", "n_mm", "mode", "phi", "p_phi",
                  "half_life", "kyle_lambda", "p_lambda", "kurt", "ad_pval", "arch_lm_pval"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for cfg_name, m in sorted(metrics_by_config.items()):
            w.writerow(m)
    print(f"Saved: {csv_path}")

    # Markdown
    md_path = os.path.join(output_dir, "table1_price_discovery.md")
    with open(md_path, "w") as f:
        f.write("| Act. Type | N MM | Mode | φ | p(φ) | Half-life | λ | p(λ) | Kurt. | AD p-val | ARCH-LM p-val |\n")
        f.write("|-----------|------|------|---|------|-----------|---|------|-------|----------|---------------|\n")
        for cfg_name, m in sorted(metrics_by_config.items()):
            f.write(f"| {m['act_type']} | {m['n_mm']} | {m['mode']} "
                    f"| {m['phi']} | {m['p_phi']} | {m['half_life']} "
                    f"| {m['kyle_lambda']} | {m['p_lambda']} "
                    f"| {m['kurt']} | {m['ad_pval']} | {m['arch_lm_pval']} |\n")
    print(f"Saved: {md_path}")

    # LaTeX
    tex_path = os.path.join(output_dir, "table1_price_discovery.tex")
    with open(tex_path, "w") as f:
        f.write("\\begin{table}[htbp]\n\\centering\n\\caption{Price discovery experiment results.}\n\\label{tab:price_discovery}\n")
        f.write("\\small\n\\begin{tabular}{llcrrrrrrrrr}\n\\toprule\n")
        f.write("Act. Type & N MM & Mode & $\\varphi$ & $p(\\varphi)$ & Half-life & $\\lambda$ & $p(\\lambda)$ & Kurt. & AD $p$-val & ARCH-LM $p$-val \\\\\n")
        f.write("\\midrule\n")
        for cfg_name, m in sorted(metrics_by_config.items()):
            f.write(f"{m['act_type']} & {m['n_mm']} & {m['mode']} "
                    f"& {m['phi']} & {m['p_phi']} & {m['half_life']} "
                    f"& {m['kyle_lambda']} & {m['p_lambda']} "
                    f"& {m['kurt']} & {m['ad_pval']} & {m['arch_lm_pval']} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    print(f"Saved: {tex_path}")


def generate_is_table(is_by_config, output_dir, filename, title, caption):
    """Tables 2-3: Implementation Shortfall comparison."""
    os.makedirs(output_dir, exist_ok=True)

    strategies = ["ppo", "vwap", "twap", "analytical", "ppo_single"]
    strat_labels = ["PPO", "VWAP", "TWAP", "Analytical", "PPO-Single"]

    # CSV
    csv_path = os.path.join(output_dir, filename + ".csv")
    fieldnames = ["act_type", "phi_risk", "mode"] + [f"{s}_mean" for s in strategies] + [f"{s}_std" for s in strategies]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for cfg_name, d in sorted(is_by_config.items()):
            row = {"act_type": d.get("act_type", ""),
                   "phi_risk": d.get("phi_risk", ""), "mode": d.get("mode", "")}
            for s in strategies:
                v = d.get(s, {})
                row[f"{s}_mean"] = v.get("mean", "")
                row[f"{s}_std"] = v.get("std", "")
            w.writerow(row)
    print(f"Saved: {csv_path}")

    # Markdown
    md_path = os.path.join(output_dir, filename + ".md")
    with open(md_path, "w") as f:
        header = "| Act. Type | φ | Mode | " + " | ".join(strat_labels) + " |\n"
        sep = "|-----------|---|------|" + "|".join(["---"] * len(strat_labels)) + "|\n"
        f.write(header)
        f.write(sep)
        for cfg_name, d in sorted(is_by_config.items()):
            vals = []
            for s in strategies:
                v = d.get(s, {})
                m, sd = v.get("mean", ""), v.get("std", "")
                if m != "":
                    vals.append(f"{m:.3f} ± {sd:.3f}")
                else:
                    vals.append("N/A")
            f.write(f"| {d.get('act_type','')} | {d.get('phi_risk','')} | {d.get('mode','')} | " + " | ".join(vals) + " |\n")
    print(f"Saved: {md_path}")

    # LaTeX
    tex_path = os.path.join(output_dir, filename + ".tex")
    with open(tex_path, "w") as f:
        f.write(f"\\begin{{table}}[htbp]\n\\centering\n\\caption{{{caption}}}\n\\label{{tab:{filename}}}\n")
        f.write("\\small\n\\begin{tabular}{llclrrrrr}\n\\toprule\n")
        f.write("Act. Type & $\\varphi$ & Mode & " + " & ".join(strat_labels) + " \\\\\n")
        f.write("\\midrule\n")
        for cfg_name, d in sorted(is_by_config.items()):
            vals = []
            for s in strategies:
                v = d.get(s, {})
                m, sd = v.get("mean", ""), v.get("std", "")
                if m != "":
                    vals.append(f"{m:.3f} $\\pm$ {sd:.3f}")
                else:
                    vals.append("N/A")
            f.write(f"{d.get('act_type','')} & {d.get('phi_risk','')} & {d.get('mode','')} & " + " & ".join(vals) + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    print(f"Saved: {tex_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp1-results", help="JSON file with exp1 metrics per config")
    parser.add_argument("--exp3-results", help="JSON file with exp3 IS per config")
    parser.add_argument("--output-dir", default="results/tables")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.exp1_results and os.path.exists(args.exp1_results):
        with open(args.exp1_results) as f:
            metrics = json.load(f)
        generate_table1(metrics, args.output_dir)

    if args.exp3_results and os.path.exists(args.exp3_results):
        with open(args.exp3_results) as f:
            is_data = json.load(f)
        linear_configs = {k: v for k, v in is_data.items() if v.get("act_type") == "linear"}
        nonlinear_configs = {k: v for k, v in is_data.items() if v.get("act_type") == "nonlinear"}
        if linear_configs:
            generate_is_table(linear_configs, args.output_dir, "table2_is_linear",
                              "IS comparison (linear)", "Implementation shortfall (IS) under linear policy.")
        if nonlinear_configs:
            generate_is_table(nonlinear_configs, args.output_dir, "table3_is_nonlinear",
                              "IS comparison (nonlinear)", "Implementation shortfall (IS) under non-linear policy.")
