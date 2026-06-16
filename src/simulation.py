import pandas as pd
import numpy as np
import plotnine as pn
import arviz as az
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import sys
if len(sys.argv) > 1:
    mode = sys.argv[1]
else:
    from model import MODE as mode

from model import get_modelKEEP, get_pymc_bsts_model
import pymc as pm
import os

# 1. Setup Simulation Data
phase_sequence = ["A", "B"] * 10
n_phases = len(phase_sequence)
sessions_per_phase = 40
total_obs = n_phases * sessions_per_phase

df_list = []
for i, p_type in enumerate(phase_sequence):
    temp_df = pd.DataFrame({
        "phase": [p_type] * sessions_per_phase,
        "phase_occurrence": [str(i)] * sessions_per_phase,
        # Real data session_in_phase is measured in fractional DAYS.
        # 40 sessions over 21 hours is ~0.875 days.
        "session_in_phase": np.linspace(0, 0.875, sessions_per_phase)
    })
    df_list.append(temp_df)

df = pd.concat(df_list).reset_index(drop=True)
df["session"] = df.index
df["y"] = 0  # Dummy integer, required for model graph

# 2. Define the Model in Bambi (Imported from shared file)
df["phase_numeric"] = df["phase"].map({"A": 0, "B": -1})
df["period"] = df["phase_occurrence"].astype(int)  # each phase occurrence is its own period
if mode == "bsts":
    model = get_pymc_bsts_model(df)
    print("\nBSTS PyMC Model instantiated.")
    with model:
        prior_samples_idata = pm.sample_prior_predictive(samples=1000, random_seed=42)
else:
    model = get_modelKEEP(df)
    model.build()
    print("\nBambi Model Terms (with custom priors):")
    print(model)
    prior_samples_idata = model.prior_predictive(draws=1000, random_seed=42) 

# Ensure output directory exists
os.makedirs(f"results/{mode}/", exist_ok=True)

# (Parameter distribution plots skipped — used old model structure with
#  scalar phase_numeric / daily_offset; outputs not referenced in index.qmd)

# 11b. LATENT MEAN LOGGING (mu)
all_mu = prior_samples_idata.prior["mu"].values.reshape(1000, -1)

from plots import plot_jitter_comparison, plot_distribution_comparison, plot_ecdf_comparison, plot_timeline_area, plot_timeline_with_mu, plot_summary_histogram, plot_session_patterns
from utils import load_observed_data

# 7. PICK ONE DRAW FOR DATA VISUALIZATION
# The shape is (chain, draw, observation). We take chain 0, draw 0.
# 8. LOAD OBSERVED DATA FOR OVERLAY
print("\nLoading observed data for comparison...")
df_obs = load_observed_data()
df_obs['phase_occurrence'] = df_obs['phase'] + df_obs['period'].astype(int).astype(str)
df_obs_plotting = df_obs.rename(columns={'PM2p5_censored': 'y_simulated'})

all_y = prior_samples_idata.prior_predictive["y"].values.reshape(1000, -1)

# Full summary computation logic
def compute_summaries(data_df, value_col='y'):
    import pandas as pd
    import numpy as np
    from scipy import stats
    means = data_df.groupby('phase')[value_col].mean().reset_index().rename(columns={value_col: 'mean'})
    p95 = data_df.groupby('phase')[value_col].apply(lambda x: np.percentile(x, 95)).reset_index().rename(columns={value_col: 'p95'})
    modes = data_df.groupby('phase')[value_col].apply(lambda x: stats.mode(x, keepdims=True).mode[0]).reset_index().rename(columns={value_col: 'mode'})
    zf = data_df.groupby('phase')[value_col].apply(lambda x: (x == 0).mean()).reset_index().rename(columns={value_col: 'zero_frac'})
    ac = data_df.groupby('phase')[value_col].apply(lambda x: x.autocorr(lag=1)).reset_index().rename(columns={value_col: 'autocorr'})
    
    occ_stats = data_df.groupby(['phase', 'phase_occurrence'])[value_col].agg(['mean', 'var', 'max']).reset_index()
    between_var = occ_stats.groupby('phase')['mean'].var(ddof=1).reset_index().rename(columns={'mean': 'variance_of_means'})
    within_var = occ_stats.groupby('phase')['var'].mean().reset_index().rename(columns={'var': 'mean_within_variance'})
    
    occ_stats['burst'] = occ_stats['max'] / (occ_stats['mean'] + 1e-6)
    burst = occ_stats.groupby('phase')['burst'].mean().reset_index().rename(columns={'burst': 'burst_index'})
    
    return {
        'mean': means, 'p95': p95, 'mode': modes, 'zero_frac': zf, 'autocorr': ac, 
        'variance_of_means': between_var, 'mean_within_variance': within_var, 'burst_index': burst
    }

obs_summaries = compute_summaries(df_obs, value_col='PM2p5_censored')
sim_results = {k: [] for k in obs_summaries.keys()}
# Also track latent mu summaries
mu_sim_results = []

for i in range(1000):
    df_temp_sim = df.copy()
    df_temp_sim['y_sim_temp'] = all_y[i]
    df_temp_sim['mu_sim_temp'] = all_mu[i]
    
    draw_summaries = compute_summaries(df_temp_sim, value_col='y_sim_temp')
    for k in sim_results.keys(): sim_results[k].append(draw_summaries[k])
    
    # Latent mu summary (Mean per phase)
    mu_means = df_temp_sim.groupby('phase')['mu_sim_temp'].mean().reset_index().rename(columns={'mu_sim_temp': 'mu'})
    mu_sim_results.append(mu_means)

# Plot all y summaries
for k in sim_results.keys():
    df_sim_concat = pd.concat(sim_results[k], ignore_index=True)
    use_log = k in ['mean', 'p95', 'variance_of_means', 'mean_within_variance']
    # ensure autocorr and zero_frac are linear
    if k in ['autocorr', 'zero_frac']: use_log = False
    p_hist = plot_summary_histogram(df_sim_concat, obs_summaries[k], k, "phase", title=f"Prior Predictive: {k}", use_log_scale=use_log)
    p_hist.save(f"results/{mode}/sim_prior_{k}_hist.png", width=5, height=5, dpi=300)

# Plot latent mu summary (MATCHING STYLE)
print("Saving sim_prior_mu_hist.png...")
df_mu_concat = pd.concat(mu_sim_results, ignore_index=True)
# Reference line is the observed empirical mean (rename 'mean' to 'mu' for the plot function)
obs_mu_ref = obs_summaries['mean'].rename(columns={'mean': 'mu'})
p_mu_hist = plot_summary_histogram(df_mu_concat, obs_mu_ref, 'mu', 'phase', title="Prior Predictive: Latent Mean (mu)", use_log_scale=True)
p_mu_hist.save(f"results/{mode}/sim_prior_mu_hist.png", width=5, height=5, dpi=300)

# 10. SELECT THREE REPRESENTATIVE DRAWS
total_within_var = []
for res in sim_results['mean_within_variance']:
    total_within_var.append(res['mean_within_variance'].sum())
total_within_var = np.array(total_within_var)

q25, q50, q75 = np.percentile(total_within_var, [25, 50, 75])
representative_draws = {
    "low": np.abs(total_within_var - q25).argmin(),
    "med": np.abs(total_within_var - q50).argmin(),
    "high": np.abs(total_within_var - q75).argmin()
}

# 11. CONSTRUCT DATASET FOR MULTIPLE DRAWS (ECDF/Dist uncertainty)
n_draws_to_show = 30
df_all_draws = []
for d in range(n_draws_to_show):
    temp_df = df[['phase', 'phase_occurrence', 'session']].copy()
    temp_df['y_simulated'] = all_y[d]
    temp_df['draw'] = d
    df_all_draws.append(temp_df)
df_all_draws = pd.concat(df_all_draws)

# 13. Comprehensive Visualizations (using the shared module)
print("Saving sim_pm2p5_dist_uncertainty.png...")
p_dist = plot_distribution_comparison(df_all_draws, "y_simulated", "phase", draw_col="draw", x_lim=(0, 40), title="Prior Predictive Distribution vs Observed (Solid)", observed_df=df_obs_plotting)
p_dist.save(f"results/{mode}/sim_pm2p5_dist_uncertainty.png", width=5, height=5, dpi=300)

print("Saving sim_pm2p5_ecdf_uncertainty.png...")
p_ecdf = plot_ecdf_comparison(df_all_draws, "y_simulated", "phase", draw_col="draw", limit=5, x_lim=(0, 40), title="Prior Predictive ECDF vs Observed (Solid)", observed_df=df_obs_plotting)
p_ecdf.save(f"results/{mode}/sim_pm2p5_ecdf_uncertainty.png", width=5, height=5, dpi=300)

# 14. GENERATE PLOTS FOR THE THREE REPRESENTATIVE DRAWS
for label, idx in representative_draws.items():
    print(f"\nGenerating plots for {label} variance draw (Idx: {idx})...")
    df_temp = df.copy()
    df_temp["y_simulated"] = all_y[idx]
    df_temp["mu_simulated"] = all_mu[idx]
    
    p_timeline_mu = plot_timeline_with_mu(df_temp, 'session', 'y_simulated', 'mu_simulated', 'phase', title=f"Simulated: Timeline ({label.capitalize()} Var Draw {idx})")
    p_timeline_mu.save(f"results/{mode}/sim_pm2p5_timeline_mu_{label}.png", width=12, height=4, dpi=300)
    
    p_timeline = plot_timeline_area(df_temp, 'session', 'y_simulated', 'phase', phase_col='phase_occurrence', title=f"Simulated: Timeline ({label.capitalize()} Var Draw {idx})")
    p_timeline.save(f"results/{mode}/sim_pm2p5_timeline_area_{label}.png", width=10, height=4, dpi=300)
    
    # Session Patterns
    p_session = plot_session_patterns(df_temp, "session_in_phase", "y_simulated", "phase_occurrence", "phase", title=f"Simulated: Internal Trends ({label.capitalize()} Var Draw {idx})")
    p_session.save(f"results/{mode}/sim_pm2p5_session_patterns_{label}.png", width=5, height=5, dpi=300)

pct_mu_over_1000 = (np.array(all_mu) > 1000).mean() * 100
print(f"\nPercentage of Prior Mu values > 1000: {pct_mu_over_1000:.2f}%")

print(f"\nGround truth simulations complete. Diagnostic plots saved to results/{mode}/")
