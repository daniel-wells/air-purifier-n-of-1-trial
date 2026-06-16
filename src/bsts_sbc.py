import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
from model import get_pymc_bsts_model
from utils import load_observed_data
import os
import argparse
import subprocess

# Parse CLI arguments
parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str, default=None, choices=["bsts", "bsts_daily_gp"],
                    help="Model configuration mode: 'bsts' or 'bsts_daily_gp'")
parser.add_argument("--name", type=str, default=None,
                    help="Directory name under results/ for saving SBC results (defaults to mode)")
args, _ = parser.parse_known_args()

import model
mode = args.mode if args.mode is not None else model.MODE
model_name = args.name if args.name is not None else mode
use_daily_gp = (mode == "bsts_daily_gp")

# --- SBC SETTINGS ---
N_GOAL_DRAWS = 250  
N_SAMPLES_POSTERIOR = 1000 # Samples per posterior fit
TUNING = 1000
CSV_PATH = f"results/{model_name}/sbc/sbc_ranks.csv"
STATS_CSV_PATH = f"results/{model_name}/sbc/sbc_stats.csv"

# 1. Load data and create a strictly BLINDED structural template
print("Loading data and creating structural template (Blinded)...")
df_full = load_observed_data()
df_full['phase_occurrence'] = df_full['phase'] + df_full['period'].astype(int).astype(str)

template_columns = ['datetime', 'phase', 'period', 'phase_occurrence']
df_template = df_full[template_columns].copy()
df_template['phase_numeric'] = df_template['phase'].map({'A': 0, 'B': -1})
df_template = df_template.sort_values('datetime')
df_template['y'] = 0 

selected_occurrences = df_template['phase_occurrence'].unique()[:4]
df_subset = df_template[df_template['phase_occurrence'].isin(selected_occurrences)].copy()

params_to_track = ["Intercept", "mean_phase_effect", "sigma_log_phase", "sigma_level", "alpha"]

# 2. Check for existing progress
if os.path.exists(CSV_PATH):
    df_existing = pd.read_csv(CSV_PATH)
    start_iteration = len(df_existing)
    ranks = df_existing.to_dict('list')
    print(f"\nResuming from iteration {start_iteration+1}/{N_GOAL_DRAWS} (Found {start_iteration} existing ranks).")
else:
    start_iteration = 0
    ranks = {p: [] for p in params_to_track}
    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
    print(f"\nStarting fresh SBC run. Goal: {N_GOAL_DRAWS} iterations.")

# Load existing stats if available
if os.path.exists(STATS_CSV_PATH):
    stats_list = pd.read_csv(STATS_CSV_PATH).to_dict('records')
else:
    stats_list = []

# 3. Build model and sample prior pool
model_base = get_pymc_bsts_model(df_subset, use_daily_gp=use_daily_gp)
with model_base:
    prior_idata = pm.sample_prior_predictive(samples=N_GOAL_DRAWS, random_seed=42)

# Synchronize prior SDs dynamically from the prior samples to ensure contraction is always correct
prior_sds = {p: float(np.std(prior_idata.prior[p].values)) for p in params_to_track}
print(f"Dynamically calculated Prior SDs: {prior_sds}")

# 4. SBC Loop
print(f"\nContinuing SBC loop (Collecting full stats for Eye Chart)...")

for i in range(start_iteration, N_GOAL_DRAWS):
    print(f"Iteration {i+1}/{N_GOAL_DRAWS}")
    
    true_vals = {p: float(prior_idata.prior[p].values[0, i]) for p in params_to_track}
    sim_y = prior_idata.prior_predictive["y"].values[0, i]
    
    sbc_model = get_pymc_bsts_model(df_subset, observed=sim_y, use_daily_gp=use_daily_gp)
    
    try:
        with sbc_model:
            trace = pm.sample(draws=N_SAMPLES_POSTERIOR, tune=TUNING, chains=1, cores=1,
                               progressbar=False, random_seed=i,
                               target_accept=0.9, shutdown_on_error=False)
            
        for p in params_to_track:
            posterior_samples = trace.posterior[p].values.flatten()
            rank = int(np.sum(posterior_samples < true_vals[p]))
            ranks[p].append(rank)
            
            post_mean = float(np.mean(posterior_samples))
            post_sd = float(np.std(posterior_samples))
            
            stats_list.append({
                'sim_id': i,
                'variable': p,
                'rank': rank,
                'true_value': true_vals[p],
                'post_mean': post_mean,
                'post_sd': post_sd,
                'prior_sd': prior_sds[p],
                'max_rank': N_SAMPLES_POSTERIOR - 1
            })
            
        # 5. Save Progress Incrementally
        pd.DataFrame(ranks).to_csv(CSV_PATH, index=False)
        pd.DataFrame(stats_list).to_csv(STATS_CSV_PATH, index=False)
        
        # 6. Trigger Replotters
        subprocess.run(["python", "bsts_sbc_replot.py", model_name], check=False)
        # 5. Trigger Plotting (R script + Python ECDF) every N iterations
        if (i + 1) >= 5 and (i + 1) % 5 == 0:
            print("\nRefreshing diagnostic plots (Eye Chart & ECDF)...")
            subprocess.run(["Rscript", "sbc_bridge.r", model_name], check=False)
            subprocess.run(["uv", "run", "python", "bsts_sbc_ecdf.py", model_name], check=False)
        
    except Exception as e:
        print(f"Error in iteration {i+1}: {e}. Skipping...")
        continue

print("\nSBC stats collection loop finished.")
