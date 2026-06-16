"""
Smooth posterior trajectory by phase × workout status.

The GPs f_workout(t) and f_workout_delta(t) are smooth functions of
session_in_phase, so the systematic component IS smooth.  Within a group the
only thing that varies across days is the additive baseline (intercept +
period random walk + period random slope).  We collapse that to a single
per-group scalar and let the GP do the heavy lifting on the t axis.

Per posterior sample s:
  baseline_obs_s   = mu_obs_s / exp(f_workout_s(t_obs)*w_obs
                                    + f_delta_s(t_obs)*phaseB_w_obs)
  c_group_s        = arithmetic mean of baseline_obs_s over the group's obs
  traj_group_s(t)  = c_group_s * exp(f_workout_s(t)*w_group
                                     + f_delta_s(t)*phaseB_w_group)

Each f sample is a smooth function of t (GP), so traj_group_s(t) is too.
Arithmetic mean is used to match the absolute-scale binned plot.
"""

import arviz as az
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys; sys.path.insert(0, ".")
from utils import load_observed_data

import sys; sys.path.insert(0, ".")
import os
from utils import load_observed_data

if len(sys.argv) > 1:
    mode = sys.argv[1]
else:
    from model import MODE as mode
    if mode not in ["bsts_daily_gp"]:
        mode = "bsts_daily_gp"

trace_path = f"results/{mode}/fit_trace.nc"
if not os.path.exists(trace_path):
    print(f"Error: {trace_path} not found.")
    sys.exit(1)

# ── Load trace ────────────────────────────────────────────────────────────────
print(f"Loading trace from {trace_path}...")
idata = az.from_netcdf(trace_path)
post  = idata.posterior

if "f_workout" not in post:
    print(f"Warning: 'f_workout' not found in trace. gp_trajectory_plot only applies to models with daily GP components (e.g. bsts_daily_gp). skipping.")
    sys.exit(0)

S = post["mu"].shape[0] * post["mu"].shape[1]
mu_flat = post["mu"].values.reshape(S, -1)            # (S, n_obs)
fw_flat = post["f_workout"].values.reshape(S, -1)
has_delta = "f_workout_delta" in post
fd_flat = (post["f_workout_delta"].values.reshape(S, -1)
           if has_delta else np.zeros_like(fw_flat))

# ── Load data ─────────────────────────────────────────────────────────────────
df_obs = load_observed_data()
df_obs['phase_occurrence'] = df_obs['phase'] + df_obs['period'].astype(int).astype(str)
df_obs = df_obs.sort_values('datetime').reset_index(drop=True)
df_obs['session_in_phase'] = df_obs.groupby('phase_occurrence')['datetime_of_experiment'].transform(
    lambda x: (x - x.min()).dt.total_seconds() / (24 * 3600))
rt = pd.read_csv("data/resistance_training_dates.csv")
rt['date'] = pd.to_datetime(rt['date']).dt.date
df_obs['date'] = df_obs['datetime'].dt.date
df_obs['is_workout'] = df_obs['date'].isin(rt['date'].values).astype(int)

t_obs        = df_obs['session_in_phase'].values
is_phaseA    = (df_obs['phase'] == 'A').values
is_phaseB    = ~is_phaseA
is_workout   = df_obs['is_workout'].values.astype(bool)
w_obs        = is_workout.astype(float)
phaseB_w_obs = (is_phaseB & is_workout).astype(float)

# ── Smooth GP curves on a fine t grid (dedupe by session_in_phase) ───────────
# Every obs's f_workout value depends only on its session_in_phase (the GP
# input).  Dedup + sort by t, then linearly interpolate to a fine grid.
t_unique, first_idx = np.unique(t_obs, return_index=True)
order      = np.argsort(t_unique)
t_unique   = t_unique[order]
first_idx  = first_idx[order]
fw_unique  = fw_flat[:, first_idx]                   # (S, n_unique)
fd_unique  = fd_flat[:, first_idx]

t_fine = np.linspace(t_obs.min(), t_obs.max(), 400)

def interp_samples(y_unique):
    idx = np.clip(np.searchsorted(t_unique, t_fine) - 1, 0, len(t_unique) - 2)
    t0, t1 = t_unique[idx], t_unique[idx + 1]
    w = np.clip((t_fine - t0) / np.maximum(t1 - t0, 1e-12), 0.0, 1.0)
    return y_unique[:, idx] * (1 - w) + y_unique[:, idx + 1] * w

fw_fine = interp_samples(fw_unique)                  # (S, n_fine), smooth
fd_fine = interp_samples(fd_unique)

# ── Per-group baseline scalar (arithmetic mean across that group's obs) ──────
def baseline_per_sample(mask):
    gp_contrib = fw_flat[:, mask] * w_obs[mask] + fd_flat[:, mask] * phaseB_w_obs[mask]
    return (mu_flat[:, mask] / np.exp(gp_contrib)).mean(axis=1)   # (S,)

def trajectory(mask, w_group, phaseB_w_group):
    c = baseline_per_sample(mask)[:, None]
    return c * np.exp(fw_fine * w_group + fd_fine * phaseB_w_group)

traj_A_nw = trajectory(is_phaseA & ~is_workout, 0.0, 0.0)
traj_B_nw = trajectory(is_phaseB & ~is_workout, 0.0, 0.0)
traj_A_wo = trajectory(is_phaseA &  is_workout, 1.0, 0.0)
traj_B_wo = trajectory(is_phaseB &  is_workout, 1.0, 1.0)

print("Per-group baseline c (arithmetic μg/m³ at GP=0):")
for label, mask in [("Phase A non-workout", is_phaseA & ~is_workout),
                     ("Phase B non-workout", is_phaseB & ~is_workout),
                     ("Phase A workout",     is_phaseA &  is_workout),
                     ("Phase B workout",     is_phaseB &  is_workout)]:
    print(f"  {label}: {baseline_per_sample(mask).mean():.3f}")

def band(arr, lo_pct=5, hi_pct=95):
    return arr.mean(axis=0), np.percentile(arr, lo_pct, axis=0), np.percentile(arr, hi_pct, axis=0)

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 5))
cfg = [
    (traj_A_nw, 'salmon',    '--', 1.5, 'Phase A — non-workout (purifier OFF)'),
    (traj_B_nw, 'lightblue', '--', 1.5, 'Phase B — non-workout (purifier ON)'),
    (traj_A_wo, 'firebrick', '-',  2.5, 'Phase A — workout (purifier OFF)'),
    (traj_B_wo, 'steelblue', '-',  2.5, 'Phase B — workout (purifier ON)'),
]
for traj, color, ls, lw, label in cfg:
    mn, lo, hi = band(traj)
    ax.fill_between(t_fine, lo, hi, alpha=0.15, color=color)
    ax.plot(t_fine, mn, color=color, lw=lw, ls=ls, label=label)

ax.axvline(0.75, ls=':', color='black', alpha=0.5, label='t≈0.75 (workout end)')
ax.set_xlabel('session_in_phase (days)')
ax.set_ylabel('Predicted mean PM₂.₅ (μg/m³)')
ax.set_title('Smooth posterior PM₂.₅ trajectory by phase × workout status')
ax.legend(fontsize=9, loc='upper left')
plt.tight_layout()
os.makedirs(f"results/{mode}", exist_ok=True)
plt.savefig(f"results/{mode}/gp_trajectory_smooth.png", dpi=150)
print(f"Saved results/{mode}/gp_trajectory_smooth.png")
