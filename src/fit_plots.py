import pandas as pd
import numpy as np
import arviz as az
from scipy import stats
import matplotlib.pyplot as plt
import plotnine as pn
import sys
if len(sys.argv) > 1:
    mode = sys.argv[1]
else:
    from model import MODE as mode

from utils import load_observed_data
from plots import (
    plot_jitter_comparison, 
    plot_distribution_comparison, 
    plot_ecdf_comparison,
    plot_summary_histogram,
    plot_retrodictive_distribution_step,
    plot_retrodictive_ecdf_step
)

print("Loading observed data and model trace...")
df = load_observed_data()
df['y'] = df['PM2p5_censored'].astype(int)
df['phase_occurrence'] = df['phase'] + df['period'].astype(int).astype(str)

# For ECDF/Dist observed data matching
df_obs_plotting = df.rename(columns={'PM2p5_censored': 'y_simulated'})

trace = az.from_netcdf(f"results/{mode}/fit_trace.nc")
pps = trace.posterior_predictive["y"].values
# Flatten chains and draws -> (2000, 5766)
all_y = pps.reshape(-1, pps.shape[-1])
n_draws = all_y.shape[0]

idx_a = (df['phase'] == 'A').values
idx_b = (df['phase'] == 'B').values

# Calculate observed modes and 95th percentiles
obs_a_vals = df[idx_a]['y']
obs_b_vals = df[idx_b]['y']

means_obs = pd.DataFrame({'phase': ['A', 'B'], 'mean': [obs_a_vals.mean(), obs_b_vals.mean()]})
df_p95_obs = pd.DataFrame({'phase': ['A', 'B'], 'p95': [np.percentile(obs_a_vals, 95), np.percentile(obs_b_vals, 95)]})

obs_mode_a = stats.mode(obs_a_vals, keepdims=True).mode[0]
obs_mode_b = stats.mode(obs_b_vals, keepdims=True).mode[0]
df_modes_obs = pd.DataFrame({'phase': ['A', 'B'], 'mode': [obs_mode_a, obs_mode_b]})

# Calculate simulated summaries across all 2000 draws
means_sim = pd.DataFrame({
    'mean': np.concatenate([all_y[:, idx_a].mean(axis=1), all_y[:, idx_b].mean(axis=1)]),
    'phase': ['A']*n_draws + ['B']*n_draws
})

p95_sim = pd.DataFrame({
    'p95': np.concatenate([np.percentile(all_y[:, idx_a], 95, axis=1), np.percentile(all_y[:, idx_b], 95, axis=1)]),
    'phase': ['A']*n_draws + ['B']*n_draws
})

global_modes_a = stats.mode(all_y[:, idx_a], axis=1, keepdims=True).mode.flatten()
global_modes_b = stats.mode(all_y[:, idx_b], axis=1, keepdims=True).mode.flatten()
modes_sim = pd.DataFrame({
    'mode': np.concatenate([global_modes_a, global_modes_b]),
    'phase': ['A']*n_draws + ['B']*n_draws
})

# Select 50 random draws for ECDF (too many will crash the plot)
np.random.seed(42)
n_draws_to_show = 50
selected_draws = np.random.choice(n_draws, n_draws_to_show, replace=False)

df_all_draws = []
for d in selected_draws:
    temp_df = df[['phase', 'phase_occurrence']].copy()
    temp_df['y_simulated'] = all_y[d]
    temp_df['draw'] = d
    df_all_draws.append(temp_df)
df_all_draws = pd.concat(df_all_draws)

print("Saving fit_pm2p5_dist_retrodictive.png...")
# Use the new step-function density plot!
sim_a = all_y[:, idx_a]
sim_b = all_y[:, idx_b]
p_dist_retro = plot_retrodictive_distribution_step(sim_a, obs_a_vals, sim_b, obs_b_vals, max_bin=40)
p_dist_retro.save(f"results/{mode}/fit_pm2p5_dist_retrodictive.png", width=8, height=6, dpi=300)

print("Saving fit_pm2p5_dist_lines.png...")
p_dist = plot_distribution_comparison(
    df_all_draws, "y_simulated", "phase", draw_col="draw", 
    x_lim=(0, 40), title="Posterior Predictive Distribution (Lines) vs Observed", observed_df=df_obs_plotting
)
p_dist.save(f"results/{mode}/fit_pm2p5_dist_lines.png", width=7, height=5, dpi=300)

print("Saving fit_pm2p5_ecdf_retrodictive.png...")
p_ecdf_retro = plot_retrodictive_ecdf_step(sim_a, obs_a_vals, sim_b, obs_b_vals, max_bin=40)
p_ecdf_retro.save(f"results/{mode}/fit_pm2p5_ecdf_retrodictive.png", width=8, height=6, dpi=300)

print("Saving fit_pm2p5_ecdf_lines.png...")
p_ecdf = plot_ecdf_comparison(
    df_all_draws, "y_simulated", "phase", draw_col="draw", 
    limit=5, x_lim=(0, 40), title="Posterior Predictive ECDF (Lines) vs Observed", observed_df=df_obs_plotting
)
p_ecdf.save(f"results/{mode}/fit_pm2p5_ecdf_lines.png", width=7, height=5, dpi=300)

print("Saving fit_post_mean_hist.png...")
p_mean_hist = plot_summary_histogram(
    means_sim, means_obs, "mean", "phase", 
    title="Posterior Predictive: Mean PM2.5 per Phase (Log Scale)", use_log_scale=True
)
p_mean_hist.save(f"results/{mode}/fit_post_mean_hist.png", width=8, height=4, dpi=300)

print("Saving fit_post_p95_hist.png...")
p_p95_hist = plot_summary_histogram(
    p95_sim, df_p95_obs, "p95", "phase", 
    title="Posterior Predictive: 95th Percentile of Counts (Log Scale)", use_log_scale=True
)
p_p95_hist.save(f"results/{mode}/fit_post_p95_hist.png", width=8, height=4, dpi=300)

print("Saving fit_post_mode_hist.png...")
p_mode_hist = plot_summary_histogram(
    modes_sim, df_modes_obs, "mode", "phase", 
    title="Posterior Predictive: Most Frequent Integer Count (Mode)", x_lim=(0, 10), discrete_integers=True
)
p_mode_hist.save(f"results/{mode}/fit_post_mode_hist.png", width=8, height=4, dpi=300)

print("Done! Check the results/ directory.")

print("\n--- Generating Trace & Distribution Plots ---")

# Trace Plot (Chain health)
# az.plot_trace(trace)  # Commented out because it takes too long with many random effects
# plt.tight_layout()
# plt.savefig("results/fit_trace.png")
# plt.clf()

# Forest plot of the multiplier (The main effect)

posterior_intercept = trace.posterior["Intercept"].values.flatten()
posterior_phase_effect = trace.posterior["mean_phase_effect"].values.flatten()

# Calculate derived metrics
mean_a_samples = np.exp(posterior_intercept)
mean_b_samples = np.exp(posterior_intercept - posterior_phase_effect)
reduction_samples = (1 - np.exp(-posterior_phase_effect)) * 100

# 1. Plot Mean A vs Mean B Distribution
plt.figure(figsize=(10, 5))
az.plot_dist(mean_a_samples, color="red", label="Phase A (OFF) Mean")
az.plot_dist(mean_b_samples, color="blue", label="Phase B (ON) Mean")
plt.title("Posterior Distributions: Expected PM2.5 Count")
plt.xlabel("PM2.5 Count (Particles)")
plt.legend()
plt.savefig(f"results/{mode}/fit_means_comparison.png")
plt.clf()

# 2. Plot Percent Reduction distribution
df_red = pd.DataFrame({'reduction': reduction_samples})
mean_red = np.mean(reduction_samples)
median_red = np.median(reduction_samples)
hdi_red_tmp = az.hdi(reduction_samples)

p_red = (
    pn.ggplot(df_red, pn.aes(x='reduction'))
    + pn.geom_density(fill='#377EB8', alpha=0.5)
    + pn.geom_vline(xintercept=mean_red, linetype='dashed', color='black', size=1)
    + pn.geom_vline(xintercept=median_red, linetype='solid', color='black', size=1)
    + pn.annotate('segment', x=hdi_red_tmp[0], xend=hdi_red_tmp[1], y=0.01, yend=0.01, size=4, color='black')
    + pn.labs(title="Posterior: Percent Reduction in Pollution (Phase B vs A)", 
              x="Reduction (%)", y="Density",
              caption=f"Solid: Median ({median_red:.1f}%), Dashed: Mean ({mean_red:.1f}%), Base: HDI ({hdi_red_tmp[0]:.1f}%-{hdi_red_tmp[1]:.1f}%)")
    + pn.scale_x_continuous(limits=(None, 100))
    + pn.theme_minimal()
)
p_red.save(f"results/{mode}/fit_reduction_percentage.png", width=8, height=4, dpi=300)

# 2b. Marginal reduction: E_j[exp(-gamma_j)] averaged over per-period draws
gamma_j = trace.posterior["phase_numeric"].values  # (chains, draws, periods)
marginal_remaining = np.exp(-gamma_j).mean(axis=-1).flatten()  # mean over periods per draw
marginal_reduction_samples = (1 - marginal_remaining) * 100

df_compare = pd.DataFrame({
    'reduction': np.concatenate([reduction_samples, marginal_reduction_samples]),
    'estimand': (
        ['Conditional\n$1 - \\exp(-\\exp(\\mu_{\\log\\gamma}))$'] * len(reduction_samples) +
        ['Marginal\n$1 - E_j[\\exp(-\\gamma_j)]$'] * len(marginal_reduction_samples)
    )
})

mean_marg = np.mean(marginal_reduction_samples)
median_marg = np.median(marginal_reduction_samples)
hdi_marg = az.hdi(marginal_reduction_samples)

p_compare = (
    pn.ggplot(df_compare, pn.aes(x='reduction', fill='estimand', colour='estimand'))
    + pn.geom_density(alpha=0.4)
    + pn.geom_vline(xintercept=np.mean(reduction_samples), linetype='dashed', color='#377EB8', size=0.8)
    + pn.geom_vline(xintercept=mean_marg, linetype='dashed', color='#E41A1C', size=0.8)
    + pn.scale_fill_manual(values=['#377EB8', '#E41A1C'])
    + pn.scale_colour_manual(values=['#377EB8', '#E41A1C'])
    + pn.labs(
        title="Conditional vs Marginal Reduction Posterior",
        x="Reduction (%)", y="Density", fill="Estimand", colour="Estimand",
        caption=(
            f"Conditional: mean {np.mean(reduction_samples):.1f}%, HDI [{hdi_red_tmp[0]:.1f}%, {hdi_red_tmp[1]:.1f}%]\n"
            f"Marginal:     mean {mean_marg:.1f}%, HDI [{hdi_marg[0]:.1f}%, {hdi_marg[1]:.1f}%]"
        )
    )
    + pn.scale_x_continuous(limits=(None, 100))
    + pn.theme_minimal()
    + pn.theme(legend_position='left')
)
p_compare.save(f"results/{mode}/fit_reduction_comparison.png", width=9, height=4, dpi=300)
print(f"Saved fit_reduction_comparison.png  (marginal mean: {mean_marg:.1f}%, HDI: [{hdi_marg[0]:.1f}%, {hdi_marg[1]:.1f}%])")

# Save key metrics for automated report (loaded by index.qmd)
key_metrics = pd.DataFrame([{
    'mean_phase_effect': float(np.mean(posterior_phase_effect)),
    'cond_mean':   float(np.mean(reduction_samples)),
    'cond_hdi_lo': float(az.hdi(reduction_samples)[0]),
    'cond_hdi_hi': float(az.hdi(reduction_samples)[1]),
    'marg_mean':   float(mean_marg),
    'marg_hdi_lo': float(hdi_marg[0]),
    'marg_hdi_hi': float(hdi_marg[1]),
}])
key_metrics.to_csv(f"results/{mode}/key_metrics.csv", index=False)
print(f"Saved key_metrics.csv")

# 3. Print summary metrics
hdi_a = az.hdi(mean_a_samples)
hdi_b = az.hdi(mean_b_samples)
hdi_red = az.hdi(reduction_samples)

print("\n--- Modelled Means (95% HDI) ---")
print(f"Phase A Typical Mean: {np.mean(mean_a_samples):.2f} (HDI: {hdi_a[0]:.2f} to {hdi_a[1]:.2f})")
print(f"Phase B Typical Mean: {np.mean(mean_b_samples):.2f} (HDI: {hdi_b[0]:.2f} to {hdi_b[1]:.2f})")
print(f"Percent Reduction: {np.mean(reduction_samples):.1f}% (HDI: {hdi_red[0]:.1f}% to {hdi_red[1]:.1f}%)")

full_summary = az.summary(trace)
full_summary.to_csv(f"results/{mode}/fit_summary.csv")

# Also save a cleaned version for reporting (excluding thousand of indexed parameters)
summary_df = full_summary[~full_summary.index.str.contains(r'\[.*\]')]
summary_df.to_csv(f"results/{mode}/fit_summary_clean.csv")

print("\n--- Summary of Global Parameters (Ignoring Point Estimates) ---")
print(summary_df)
