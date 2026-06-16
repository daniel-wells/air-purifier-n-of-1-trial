import pandas as pd
import numpy as np
import arviz as az
import plotnine as pn
from utils import load_observed_data
import sys
if len(sys.argv) > 1:
    mode = sys.argv[1]
else:
    from model import MODE as mode

import bambi as bmb
import matplotlib.pyplot as plt

print("Loading data and trace...")
df = load_observed_data()
df['y'] = df['PM2p5_censored'].astype(int)
df['phase_numeric'] = df['phase'].map({'A': 0, 'B': -1})
df['phase_occurrence'] = df['phase'] + df['period'].astype(int).astype(str)
df = df.sort_values('datetime')
df['session_in_phase'] = df.groupby('phase_occurrence')['datetime_of_experiment'].transform(
    lambda x: (x - x.min()).dt.total_seconds() / (24 * 3600)
)

trace = az.from_netcdf(f"results/{mode}/fit_trace.nc")

print("Calculating posterior mean (expected value)...")
# mu is often stored in trace.posterior for some Bambi versions/families
if "mu" in trace.posterior:
    mu = trace.posterior["mu"].values
else:
    # Fallback: manual reconstruction if needed, but let's assume it's there
    print("Warning: 'mu' not in posterior. Checking posterior_predictive...")
    mu = trace.posterior_predictive["y"].values # Last resort: use PPS mean

n_obs = mu.shape[-1]
mu_flat = mu.reshape(-1, n_obs)

pred_mean = mu_flat.mean(axis=0)
hdi_mu = az.hdi(mu_flat, hdi_prob=0.95)

# Calculate PPS from posterior_predictive (already there)
pps = trace.posterior_predictive["y"].values
pps_flat = pps.reshape(-1, n_obs)
hdi_pps = az.hdi(pps_flat, hdi_prob=0.95)

df['pred_mean'] = pred_mean
df['mu_low'] = hdi_mu[:, 0]
df['mu_high'] = hdi_mu[:, 1]
df['pps_low'] = hdi_pps[:, 0]
df['pps_high'] = hdi_pps[:, 1]

print("Generating timeline overlay plot...")

# Plot observed as thin line, model mean as thicker line
p = (
    pn.ggplot(df, pn.aes(x='datetime', group='phase_occurrence'))
    # Observed data (thin gray-ish lines)
    + pn.geom_line(pn.aes(y='PM2p5_censored', color='phase'), alpha=0.3, size=0.3)
    # Model's expected value (the smooth mu)
    + pn.geom_line(pn.aes(y='pred_mean', colour='phase'), size=1.0)
    # Uncertainty in the mean (Confidence interval)
    + pn.geom_ribbon(pn.aes(ymin='mu_low', ymax='mu_high', fill='phase'), alpha=0.4, colour=None)
    # Prediction interval (the full volatility) - very faint
    # + pn.geom_ribbon(pn.aes(ymin='pps_low', ymax='pps_high', fill='phase'), alpha=0.1, colour=None)

    + pn.scale_x_datetime(date_labels="%b %d")
    + pn.scale_colour_manual(values={'A': '#E41A1C', 'B': '#377EB8'})
    + pn.scale_fill_manual(values={'A': '#E41A1C', 'B': '#377EB8'})
    + pn.labs(x="Datetime", y="PM2.5 (ug/m^3)", title="Model Timeline Overlay (Observed vs Predicted Mean)")
    + pn.theme_minimal()
    + pn.theme(figure_size=(12, 6))
)

p.save(f"results/{mode}/fit_timeline_overlay.png", dpi=300)
print(f"Saved results/{mode}/fit_timeline_overlay.png")

# Zoomed-in version (Feb 20 - Mar 1)
import datetime
zoom_start = datetime.datetime(2022, 2, 20)
zoom_end = datetime.datetime(2022, 3, 1)
df_zoom = df[(df['datetime'] >= zoom_start) & (df['datetime'] <= zoom_end)]

print("Generating zoomed timeline overlay plot (Feb 20 - Mar 1)...")
p_zoom = (
    pn.ggplot(df_zoom, pn.aes(x='datetime', group='phase_occurrence'))
    + pn.geom_line(pn.aes(y='PM2p5_censored', color='phase'), alpha=0.3, size=0.3)
    + pn.geom_line(pn.aes(y='pred_mean', colour='phase'), size=1.0)
    + pn.geom_ribbon(pn.aes(ymin='mu_low', ymax='mu_high', fill='phase'), alpha=0.4, colour=None)
    + pn.scale_x_datetime(date_labels="%b %d")
    + pn.scale_colour_manual(values={'A': '#E41A1C', 'B': '#377EB8'})
    + pn.scale_fill_manual(values={'A': '#E41A1C', 'B': '#377EB8'})
    + pn.labs(x="Datetime", y="PM2.5 (ug/m^3)", title="Model Timeline Overlay — Zoomed (Feb 20 – Mar 1)")
    + pn.theme_minimal()
    + pn.theme(figure_size=(12, 6))
)

p_zoom.save(f"results/{mode}/fit_timeline_overlay_zoom.png", dpi=300)
print(f"Saved results/{mode}/fit_timeline_overlay_zoom.png")
