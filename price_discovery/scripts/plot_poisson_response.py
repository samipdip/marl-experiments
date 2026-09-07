"""Poisson-shock tracking plot + metrics for set4 dynamic-belief eval (Phase 1b).

Reads, per episode:
  - {epdir}/belief_coordinator_f_series.csv   (tick, F_t)
  - {epdir}/checkpoints/ppo_hjb_mm_0_lob_data.csv  (step, mid)

F_t is a random walk here (not a single step), so the step plotter's
capture%/rise metrics do not apply. Instead: per-episode RMSE(mid,F), MAE,
corr(mid,F), OLS slope of mid~F (tracking coefficient), and best cross-correlation
lag. Pools across episodes; writes poisson_response.png + metrics_poisson.csv.

Usage: plot_poisson_response.py <results_dir> [n_steps]
"""
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

results_base = sys.argv[1]
n_steps = int(sys.argv[2]) if len(sys.argv) > 2 else 240
n_eps = 30

mids, Fs, njumps = [], [], []
for ep in range(n_eps):
    epdir = os.path.join(results_base, f"ep_{ep}")
    fpath = os.path.join(epdir, "belief_coordinator_f_series.csv")
    mpath = os.path.join(epdir, "checkpoints", "ppo_hjb_mm_0_lob_data.csv")
    if not (os.path.exists(fpath) and os.path.exists(mpath)):
        continue
    f_by_tick = {int(r["tick"]): float(r["F"]) for r in csv.DictReader(open(fpath))}
    m_by_step = {int(r["step"]): float(r["mid"]) for r in csv.DictReader(open(mpath))}
    ticks = sorted(t for t in set(f_by_tick) & set(m_by_step) if t < n_steps)
    if not ticks:
        continue
    Ft = np.array([f_by_tick[t] for t in ticks])
    mt = np.array([m_by_step[t] for t in ticks])
    mids.append(mt)
    Fs.append(Ft)
    njumps.append(int(np.sum(np.abs(np.diff(Ft)) > 1e-9)))

if not mids:
    print(f"no data in {results_base}")
    sys.exit(0)

# per-episode metrics
rmses, maes, corrs, slopes, lags = [], [], [], [], []
for mt, Ft in zip(mids, Fs):
    err = mt - Ft
    rmses.append(float(np.sqrt(np.mean(err ** 2))))
    maes.append(float(np.mean(np.abs(err))))
    if Ft.std() > 1e-9 and mt.std() > 1e-9:
        corrs.append(float(np.corrcoef(mt, Ft)[0, 1]))
        slopes.append(float(np.polyfit(Ft, mt, 1)[0]))
        best = 0.0
        bestk = 0
        for k in range(-20, 21):
            if k >= 0:
                a = Ft[: len(Ft) - k]
                b = mt[k:]
            else:
                a = Ft[-k:]
                b = mt[: len(mt) + k]
            if len(a) > 5 and a.std() > 1e-9 and b.std() > 1e-9:
                c = abs(float(np.corrcoef(a, b)[0, 1]))
                if c > best:
                    best = c
                    bestk = k
        lags.append(bestk)
    else:
        corrs.append(float("nan"))
        slopes.append(float("nan"))
        lags.append(0)

# pooled arrays for overlay / scatter
n = min(len(m) for m in mids)
M = np.array([m[:n] for m in mids])
F = np.array([f[:n] for f in Fs])
mmean, mstd = M.mean(0), M.std(0)
fmean, fstd = F.mean(0), F.std(0)
x = np.arange(n)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
ax1.plot(x, mmean, color="#1f77b4", label="mid (mean)")
ax1.fill_between(x, mmean - mstd, mmean + mstd, color="#1f77b4", alpha=0.2)
ax1.plot(x, fmean, color="#d62728", linestyle="--", linewidth=2, label="F_t (mean)")
ax1.fill_between(x, fmean - fstd, fmean + fstd, color="#d62728", alpha=0.1)
ax1.set_xlabel("tick")
ax1.set_ylabel("price")
ax1.legend()
ax1.grid(alpha=0.3)
ax1.set_title(os.path.basename(results_base) + " (mean +/- std)")

Ms = M.ravel()
Fp = F.ravel()
ax2.scatter(Fp, Ms, s=3, alpha=0.15)
lo = min(Fp.min(), Ms.min())
hi = max(Fp.max(), Ms.max())
ax2.plot([lo, hi], [lo, hi], color="gray", linestyle="--", label="y = F (perfect)")
if len(Fp) > 5 and Fp.std() > 1e-9:
    b = float(np.polyfit(Fp, Ms, 1)[0])
    xx = np.array([lo, hi])
    ax2.plot(xx, b * xx, color="#d62728", linewidth=2, label=f"OLS slope={b:.2f}")
ax2.set_xlabel("F_t")
ax2.set_ylabel("mid_t")
ax2.legend()
ax2.grid(alpha=0.3)
ax2.set_title("mid vs F (pooled, all ticks x eps)")
plt.tight_layout()
out = os.path.join(results_base, "poisson_response.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"saved {out}")


def _m(a):
    a = [v for v in a if v == v]  # drop nan
    return round(float(np.mean(a)), 3) if a else "nan"


metrics = dict(
    scenario=os.path.basename(results_base),
    n_episodes=len(mids),
    mean_jumps_per_ep=round(float(np.mean(njumps)), 2),
    rmse_mid_F=_m(rmses),
    mae_mid_F=_m(maes),
    corr_mid_F=_m(corrs),
    ols_slope=_m(slopes),
    best_lag_ticks=_m(lags),
    F_std_mean=round(float(np.mean([f.std() for f in Fs])), 3),
)
mpath = os.path.join(results_base, "metrics_poisson.csv")
with open(mpath, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(metrics))
    w.writeheader()
    w.writerow(metrics)
print("metrics:", metrics)
