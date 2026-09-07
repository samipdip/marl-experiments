"""Shock-response plot + metrics for set4 dynamic-belief eval.

Reads, per episode:
  - {epdir}/belief_coordinator_f_series.csv   (tick, F_t)
  - {epdir}/checkpoints/ppo_hjb_mm_0_lob_data.csv  (step, mid)
Plots mean mid (±std) vs F_t and writes shock_response.png + metrics_shock.csv.

Usage: plot_shock_response.py <results_dir> [n_steps] [_clip_ticks]
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

mids, Fs = [], []
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
    mids.append([m_by_step[t] for t in ticks])
    Fs.append([f_by_tick[t] for t in ticks])

if not mids:
    print(f"no data in {results_base}")
    sys.exit(0)

n = min(len(m) for m in mids)
M = np.array([m[:n] for m in mids])
F = np.array([f[:n] for f in Fs])
mmean, mstd = M.mean(0), M.std(0)
fmean = F.mean(0)
x = np.arange(n)

# step detection + metrics
diff = np.diff(fmean)
step_idx = int(np.argmax(np.abs(diff))) if len(diff) else 0
delta = float(fmean[step_idx + 1] - fmean[step_idx]) if n > step_idx + 1 else 0.0
pre = float(mmean[max(0, step_idx - 30):step_idx].mean()) if step_idx > 0 else float(mmean[0])
post = float(mmean[step_idx + 1:].mean()) if step_idx + 1 < n else float(mmean[-1])
captured = (post - pre) / delta if delta else float("nan")
sign = 1 if delta >= 0 else -1


def first_cross(arr, target, s):
    for i, v in enumerate(arr):
        if s * (v - target) >= 0:
            return i
    return None


after = mmean[step_idx + 1:]
r50 = first_cross(after, pre + 0.5 * delta, sign)
r90 = first_cross(after, pre + 0.9 * delta, sign)
fpost = float(fmean[step_idx + 1:].mean()) if step_idx + 1 < n else float(fmean[-1])
ss_err = post - fpost
peak = float(np.max(sign * (after - pre))) if len(after) else 0.0
overshoot = max(0.0, sign * peak - abs(delta)) / abs(delta) if delta else float("nan")

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(x, mmean, color="#1f77b4", label="mid (mean over eps)")
ax.fill_between(x, mmean - mstd, mmean + mstd, color="#1f77b4", alpha=0.2)
ax.plot(x, fmean, color="#d62728", linestyle="--", linewidth=2, label="F_t")
ax.axvline(step_idx, color="gray", linestyle=":", alpha=0.6)
ax.set_xlabel("tick"); ax.set_ylabel("price")
ax.set_title(os.path.basename(results_base)); ax.legend(); ax.grid(alpha=0.3)
out = os.path.join(results_base, "shock_response.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"saved {out}")

metrics = dict(
    scenario=os.path.basename(results_base),
    step_tick=step_idx, delta=delta,
    pre_mid=round(pre, 3), post_mid=round(post, 3), F_post=round(fpost, 3),
    captured_pct=round(captured * 100, 1) if captured == captured else "nan",
    rise50_ticks=r50, rise90_ticks=r90,
    ss_err=round(ss_err, 3),
    overshoot_pct=round(overshoot * 100, 1) if overshoot == overshoot else "nan",
)
mpath = os.path.join(results_base, "metrics_shock.csv")
with open(mpath, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(metrics)); w.writeheader(); w.writerow(metrics)
print("metrics:", metrics)
