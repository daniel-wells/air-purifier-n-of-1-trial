import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pymc as pm
import os
from model import get_pymc_bsts_model
from utils import load_observed_data

def generate_pushforward_plots():
    n_samples = 10_000
    df_obs = load_observed_data()
    
    # Pre-process data
    df_obs['phase_occurrence'] = df_obs['phase'] + df_obs['period'].astype(int).astype(str)
    df_obs['phase_numeric'] = df_obs['phase'].map({'A': 0, 'B': -1})
    df_obs = df_obs.sort_values('datetime')
    df_obs['session_in_phase'] = df_obs.groupby('phase_occurrence')['datetime_of_experiment'].transform(
        lambda x: (x - x.min()).dt.total_seconds() / (24 * 3600)
    )
    df_obs['y'] = df_obs['PM2p5_censored'].fillna(0).astype(int)

    # 1. Instantiate the live model
    model = get_pymc_bsts_model(df_obs)
    
    # 2. Sample from priors (native PyMC way)
    print(f"Sampling {n_samples} prior points from the live model...")
    with model:
        prior_idata = pm.sample_prior_predictive(samples=n_samples, random_seed=42)
    
    prior = prior_idata.prior.stack(sample=("chain", "draw"))
    
    # 3. Component Extraction
    components = {}
    components['Global Baseline [exp(mu)]'] = np.exp(prior['Intercept'].values)
    components['Purifier Efficiency [exp(-gamma)]'] = np.exp(-prior['phase_numeric'].values.flatten())
    components['Session Init [exp(init)]'] = np.exp(prior['session_init'].isel(session=0).values)
    
    # 4. Local Random Walk Drift at multiple timepoints (t=index)
    # Reversing order so narrow peaks (t=1) are plotted last (on top)
    t_points = [82, 16, 1]
    labels = ["20h", "4h", "15m"]
    colors = ['#1f77b4', '#2ca02c', '#d62728'] # Bold Tab10 Blue, Green, Red
    
    innovations = prior['innovations'].values
    sigma_level = prior['sigma_level'].values
    
    # 5. Plotting with Matplotlib for mixed scales
    fig = plt.figure(figsize=(13, 11))
    
    # Grid: 0,0 Baseline | 0,1 Efficiency
    # Grid: 1,0 Offset   | 1,1 Drift (Overlay)
    
    # -- 1. Global Baseline
    ax1 = fig.add_subplot(221)
    d = components['Global Baseline [exp(mu)]']
    ax1.set_xscale('log')
    ax1.hist(d, bins=np.logspace(np.log10(d.min()), np.log10(d.max()), 50), edgecolor='white', alpha=0.8)
    ax1.set_title('Global Baseline [exp(mu)]', fontweight='bold')
    ax1.set_xlabel('Multiplier Value')
    ax1.grid(True, alpha=0.3)
    
    # -- 2. Purifier Efficiency
    ax2 = fig.add_subplot(222)
    d = components['Purifier Efficiency [exp(-gamma)]']
    ax2.hist(d, bins=50, edgecolor='white', alpha=0.8)
    ax2.set_xlim(-0.05, 1.05) # Fixed padding
    ax2.set_title('Purifier Efficiency [exp(-gamma)]', fontweight='bold')
    ax2.set_xlabel('Fraction Remaining (0-1)')
    ax2.grid(True, alpha=0.3)
    
    # -- 3. Daily Offset
    ax3 = fig.add_subplot(223)
    d = components['Session Init [exp(init)]']
    ax3.set_xscale('log')
    ax3.hist(d, bins=np.logspace(np.log10(d.min()), np.log10(d.max()), 50), edgecolor='white', alpha=0.8)
    ax3.set_title('Session Init [exp(init)]', fontweight='bold')
    ax3.set_xlabel('Multiplier Value')
    ax3.grid(True, alpha=0.3)
    
    # -- 4. Local Random Walk (Evolution over time)
    ax4 = fig.add_subplot(224)
    # Using transparency layers to show widening
    for t, label, color in zip(t_points, labels, colors):
        rw_sum = np.sum(innovations[:t, :], axis=0) * sigma_level
        rw_mult = np.exp(rw_sum)
        
        bins = np.logspace(-4, 4, 120)
        ax4.hist(rw_mult, bins=bins, histtype='stepfilled', alpha=0.6, color=color, label=f"t={t} ({label})")
        ax4.hist(rw_mult, bins=bins, histtype='step', color=color, alpha=1.0, lw=2.0)

    ax4.set_xscale('log')
    ax4.set_xlim(0.001, 1000)
    ax4.set_title('Local Drift Evolution [exp(L_ij)]', fontweight='bold')
    ax4.set_xlabel('Multiplier Value (Diffusion over time)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.suptitle(f'BSTS Dynamic Prior Pushforwards (N={n_samples})', 
                 fontsize=14, y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    output_path = 'results/bsts/sim_prior_pushforward_components.png'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    print(f"SUCCESS: Evolving pushforward plots saved to {output_path}")

    # --- GP component pushforward (only when daily GP was built) ---
    if 'f_base' in prior:
        _gp_prior_pushforward(prior, df_obs, output_path.replace('_components', '_gp'))


def _gp_prior_pushforward(prior, df_obs, output_path):
    """Plot prior distributions for the HSGP hyperparameters and function values."""
    import warnings

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle('HSGP Daily-GP Prior Pushforward', fontsize=14)

    # ---- Lengthscales (in days) ----------------------------------------
    for ax, key, title in zip(
        [axes[0, 0], axes[0, 1]],
        ['ell_base', 'ell_workout'],
        ['ell_base (days)', 'ell_workout (days)'],
    ):
        vals = prior[key].values.flatten()
        ax.hist(vals, bins=60, edgecolor='white', alpha=0.85)
        ax.axvline(np.median(vals), color='red', lw=1.5, label=f'median={np.median(vals):.2f}')
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel('Lengthscale (days)')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    # ---- Amplitudes -------------------------------------------------------
    for ax, key, title in zip(
        [axes[1, 0], axes[1, 1]],
        ['eta_base', 'eta_workout'],
        ['eta_base (log-scale SD)', 'eta_workout (log-scale SD)'],
    ):
        vals = prior[key].values.flatten()
        ax.hist(vals, bins=60, edgecolor='white', alpha=0.85)
        ax.axvline(np.median(vals), color='red', lw=1.5, label=f'median={np.median(vals):.2f}')
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel('Amplitude (log scale)')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    # ---- f_base across a subsample of observations -----------------------
    ax_fb = axes[0, 2]
    f_base_vals = prior['f_base'].values  # (n_obs, n_samples) after stacking
    # Draw 200 random prior curves (one per sample → one value per obs)
    rng = np.random.default_rng(42)
    n_samples_plot = min(200, f_base_vals.shape[-1])
    idx_s = rng.choice(f_base_vals.shape[-1], size=n_samples_plot, replace=False)
    t_raw = df_obs['session_in_phase'].values.astype(float)
    sort_order = np.argsort(t_raw)
    for i in idx_s:
        ax_fb.plot(t_raw[sort_order], f_base_vals[sort_order, i],
                   color='steelblue', alpha=0.05, lw=0.8)
    ax_fb.set_title('f_base prior sample curves', fontweight='bold')
    ax_fb.set_xlabel('Time within phase (days)')
    ax_fb.set_ylabel('f_base (log scale)')
    ax_fb.grid(True, alpha=0.3)

    # ---- f_workout across a subsample of observations -------------------
    ax_fw = axes[1, 2]
    f_workout_vals = prior['f_workout'].values
    workout_mask = (df_obs['resistance_training'] == 'Yes').values
    # Show workout vs non-workout distribution as overlaid histograms
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        fw_flat_workout    = f_workout_vals[workout_mask, :].flatten()
        fw_flat_nonworkout = f_workout_vals[~workout_mask, :].flatten()
    ax_fw.hist(fw_flat_nonworkout, bins=80, alpha=0.5, label='Non-workout', color='grey')
    ax_fw.hist(fw_flat_workout,    bins=80, alpha=0.6, label='Workout',     color='orangered')
    ax_fw.set_title('f_workout × indicator distribution', fontweight='bold')
    ax_fw.set_xlabel('f_workout (log scale)')
    ax_fw.legend(fontsize=9)
    ax_fw.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"SUCCESS: GP prior pushforward saved to {output_path}")


if __name__ == "__main__":
    generate_pushforward_plots()
