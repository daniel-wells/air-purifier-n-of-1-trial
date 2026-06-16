"""
bsts_ite_plot.py  —  Individual Treatment Effect counterfactual plot

For a chosen period, compares:
  • Phase A (purifier OFF) — observed latent mean
  • Phase B (purifier ON)  — observed latent mean
  • Phase B counterfactual — what PM2.5 would have been if the purifier had
    stayed off, computed as  λ_cf = λ_obs × exp(γ_j)

The washout gap between phases is bridged by linear interpolation of the
posterior median and 80% CI so the counterfactual line reads continuously.
Individual posterior draws are shown as spaghetti to convey variability.

Usage:
    uv run python bsts_ite_plot.py            # default period 21 (Feb 26-27)
    uv run python bsts_ite_plot.py --period 34 # Mar 24-25 (highest contrast)
    uv run python bsts_ite_plot.py --period 5  # Jan 25-26
    uv run python bsts_ite_plot.py --n-spag 60 --period 26
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import arviz as az
from utils import load_observed_data

# ── CLI ────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--period",  type=int, default=21,
                    help="Period index to plot (default: 21 = Feb 26-27)")
parser.add_argument("--n-spag",  type=int, default=40,
                    help="Number of individual posterior draws to overlay (default: 40)")
parser.add_argument("--lo-pct",  type=float, default=10,
                    help="Lower percentile for CI band (default: 10)")
parser.add_argument("--hi-pct",  type=float, default=90,
                    help="Upper percentile for CI band (default: 90)")
parser.add_argument("--mode",    type=str,  default="bsts",
                    help="Model mode subdirectory under results/ (default: bsts)")
args = parser.parse_args()

TARGET_PERIOD = args.period
N_SPAG        = args.n_spag
LO_PCT        = args.lo_pct
HI_PCT        = args.hi_pct
MODE          = args.mode
TRACE_PATH    = f"results/{MODE}/fit_trace.nc"
OUT_PATH      = f"results/{MODE}/ite_counterfactual_period{TARGET_PERIOD}.png"

# ── Load data ─────────────────────────────────────────────────────────────────
print(f"Loading data and trace for period {TARGET_PERIOD}...")
df = load_observed_data()
df['y'] = df['PM2p5_censored'].astype(int)
df['phase_occurrence'] = df['phase'] + df['period'].astype(int).astype(str)
df = df.sort_values('datetime').reset_index(drop=True)

session_A = f'A{TARGET_PERIOD}'
session_B = f'B{TARGET_PERIOD}'

obs_A = df.index[df['phase_occurrence'] == session_A].tolist()
obs_B = df.index[df['phase_occurrence'] == session_B].tolist()

if not obs_A or not obs_B:
    raise ValueError(f"Period {TARGET_PERIOD} not found. "
                     f"Available: {sorted(df['period'].unique().astype(int))}")

tA = pd.to_datetime(df.loc[obs_A, 'datetime'].values)
tB = pd.to_datetime(df.loc[obs_B, 'datetime'].values)
yA = df.loc[obs_A, 'y'].values
yB = df.loc[obs_B, 'y'].values

# ── Load trace ────────────────────────────────────────────────────────────────
idata = az.from_netcdf(TRACE_PATH)
period_list = list(idata.posterior.coords["period"].values)

if TARGET_PERIOD not in period_list:
    raise ValueError(f"Period {TARGET_PERIOD} not in trace. Available: {period_list}")
b_period_idx = period_list.index(TARGET_PERIOD)

n_obs = len(df)
mu_flat = idata.posterior["mu"].values.reshape(-1, n_obs)          # (D, N)
pn_flat = idata.posterior["phase_numeric"].values.reshape(-1, len(period_list))
gamma   = pn_flat[:, b_period_idx]                                  # (D,)
n_draws = mu_flat.shape[0]

mu_A_post = mu_flat[:, obs_A]                                       # (D, T)
mu_B_obs  = mu_flat[:, obs_B]
mu_B_cf   = mu_B_obs * np.exp(gamma[:, None])                       # counterfactual

# ── Posterior bands ───────────────────────────────────────────────────────────
def bands(arr):
    return (np.median(arr, 0),
            np.percentile(arr, LO_PCT, 0),
            np.percentile(arr, HI_PCT, 0))

med_A,  lo_A,  hi_A  = bands(mu_A_post)
med_Bo, lo_Bo, hi_Bo = bands(mu_B_obs)
med_Bc, lo_Bc, hi_Bc = bands(mu_B_cf)

# Cap extreme counterfactual CI at 2× max observed Phase A
y_cap  = max(yA) * 2.2
hi_Bc  = np.clip(hi_Bc,  None, y_cap)
med_Bc = np.clip(med_Bc, None, y_cap)

# ── Washout bridge (linear interpolation across the gap) ─────────────────────
t_wash = pd.date_range(tA[-1] + pd.Timedelta('15min'),
                       tB[0]  - pd.Timedelta('15min'), freq='15min')
n_w = len(t_wash) + 2   # including anchor endpoints

def lerp(a, b, n):
    return a + np.linspace(0, 1, n) * (b - a)

t_cf   = np.concatenate([[tA[-1]], t_wash, tB])
# Median bridge: hold Phase A's last value constant across the washout.
# We have no observations there, so interpolating toward the Phase B
# counterfactual start would imply knowledge we don't have.
med_cf = np.concatenate([[med_A[-1]], np.full(len(t_wash), med_A[-1]), med_Bc])
# CI bands widen linearly through the washout to reflect growing uncertainty.
lo_cf  = np.concatenate([[lo_A[-1]],  lerp(lo_A[-1],  lo_Bc[0],  n_w)[1:-1], lo_Bc])
hi_cf  = np.concatenate([[hi_A[-1]],  lerp(hi_A[-1],  hi_Bc[0],  n_w)[1:-1], hi_Bc])

def lerp_draw(a_val, b_arr):
    """Bridge a single draw: hold last-A-value constant across the washout."""
    bridge = np.full(len(t_wash), a_val)
    return np.concatenate([[a_val], bridge, b_arr])

# ── ITE summary ───────────────────────────────────────────────────────────────
ite_draws = 1 - mu_B_obs.mean(1) / mu_B_cf.mean(1)
ite_pct   = np.mean(ite_draws) * 100
ite_lo    = np.percentile(ite_draws,  5) * 100
ite_hi    = np.percentile(ite_draws, 95) * 100

# ── Figure ────────────────────────────────────────────────────────────────────
# Color convention: RED = purifier OFF, BLUE = purifier ON
RED    = '#c0392b'
BLUE   = '#2980b9'
ORANGE = '#e67e22'

fig, ax = plt.subplots(figsize=(13, 5.2))
fig.patch.set_facecolor('white')
ax.set_facecolor('#fafafa')

# Washout shading
ax.axvspan(tA[-1], tB[0], color='#ddd', alpha=0.55, zorder=0)
ax.text(tA[-1] + (tB[0] - tA[-1]) / 2, 0.98, 'washout\n(3.5 h)',
        transform=ax.get_xaxis_transform(),
        ha='center', va='top', fontsize=8, color='#666')

# Individual posterior draws (spaghetti)
rng      = np.random.default_rng(42)
spag_idx = rng.choice(n_draws, size=min(N_SPAG, n_draws), replace=False)
for i in spag_idx:
    ax.plot(tA, mu_A_post[i],  color=RED,  lw=0.45, alpha=0.15, zorder=1)
    ax.plot(tB, mu_B_obs[i],   color=BLUE, lw=0.45, alpha=0.15, zorder=1)
    cf_draw = np.clip(mu_B_cf[i], None, y_cap)
    ax.plot(t_cf, lerp_draw(mu_A_post[i, -1], cf_draw),
            color=RED, lw=0.45, alpha=0.15, ls='--', zorder=1)

# Posterior CI bands
ax.fill_between(tA,   lo_A,  hi_A,  alpha=0.15, color=RED,  zorder=2)
ax.fill_between(tB,   lo_Bo, hi_Bo, alpha=0.15, color=BLUE, zorder=2)
ax.fill_between(t_cf, lo_cf, hi_cf, alpha=0.12, color=RED,  zorder=2)

# Posterior medians
ax.plot(tA,   med_A,  color=RED,  lw=2.2,        zorder=3)
ax.plot(tB,   med_Bo, color=BLUE, lw=2.2,        zorder=3)
ax.plot(t_cf, med_cf, color=RED,  lw=2.2, ls='--', zorder=3)

# Observed data scatter
ax.scatter(tA, yA, s=14, color=RED,  alpha=0.60, zorder=5)
ax.scatter(tB, yB, s=14, color=BLUE, alpha=0.60, zorder=5)

# ITE fill (orange, between observed and counterfactual median on Phase B)
ax.fill_between(tB, med_Bo, np.clip(med_Bc, med_Bo, None),
                alpha=0.28, color=ORANGE, zorder=2)

# Phase-boundary lines and day labels
for t in (tA[-1], tB[0]):
    ax.axvline(t, color='#888', lw=0.8, ls=':', zorder=4)

ax.text(tA[len(tA) // 2], y_cap * 0.97,
        f'Phase A — purifier OFF',
        ha='center', va='top', fontsize=9.5, color=RED, fontweight='bold')
ax.text(tB[len(tB) // 2], y_cap * 0.97,
        f'Phase B — purifier ON',
        ha='center', va='top', fontsize=9.5, color=BLUE, fontweight='bold')

# Legend
handles = [
    Line2D([0],[0], color=RED,    lw=2.2,
           label='Purifier OFF — posterior median latent mean'),
    Line2D([0],[0], color=RED,    lw=2.2,  ls='--',
           label='Purifier OFF — counterfactual (if purifier had stayed off)'),
    Line2D([0],[0], color=BLUE,   lw=2.2,
           label='Purifier ON  — posterior median latent mean'),
    Line2D([0],[0], color=RED,    lw=0.9,  alpha=0.45,
           label=f'Individual posterior draws ({min(N_SPAG, n_draws)} of {n_draws})'),
    Patch(facecolor=RED,    alpha=0.25,
          label=f'{int(HI_PCT - LO_PCT)}% credible interval'),
    Patch(facecolor=ORANGE, alpha=0.45,
          label=f'ITE — PM2.5 removed by purifier  '
                f'({ite_pct:.0f}%  90% CI [{ite_lo:.0f}–{ite_hi:.0f}%])'),
]
ax.legend(handles=handles, fontsize=8.5, ncol=2,
          loc='upper center', bbox_to_anchor=(0.5, -0.20), frameon=True)

# Axes formatting
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d\n%H:%M'))
ax.xaxis.set_major_locator(mdates.HourLocator(interval=4))
ax.set_xlim(tA[0], tB[-1])
ax.set_ylim(0, y_cap)
ax.set_xlabel('Date / Time', fontsize=11, labelpad=8)
ax.set_ylabel('PM2.5 (μg/m³)', fontsize=11)

date_A = tA[0].strftime('%b %-d')
date_B = tB[0].strftime('%b %-d')
ax.set_title(
    f'Individual Treatment Effect — Period {TARGET_PERIOD}  '
    f'({date_A}–{date_B} 2022)\n'
    'Counterfactual: what would PM2.5 have been if the purifier had not switched on?',
    fontsize=11, pad=8)

plt.tight_layout()
fig.subplots_adjust(bottom=0.30)
plt.savefig(OUT_PATH, dpi=150, bbox_inches='tight')
print(f"Saved  {OUT_PATH}")
print(f"ITE:   {ite_pct:.1f}%  90% CI [{ite_lo:.1f}%, {ite_hi:.1f}%]")
print(f"Phase A observed mean: {yA.mean():.2f} μg/m³  "
      f"Phase B observed mean: {yB.mean():.2f} μg/m³  "
      f"Counterfactual median mean: {np.mean(med_Bc):.2f} μg/m³")
