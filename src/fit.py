import argparse
import os
import pandas as pd
import numpy as np
import arviz as az
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from utils import load_observed_data
from model import get_modelKEEP

# Parse flags: default=3 for M-series Mac (numpyro backend avoids fork deadlocks)
parser = argparse.ArgumentParser()
parser.add_argument("--cores", type=int, default=3,
    help="Number of parallel sampling cores (default: 3). Ignored for numpyro backend (uses chain_method=vectorized).")
parser.add_argument("--chains", type=int, default=3,
    help="Number of MCMC chains (default: 3). Set to 4 to saturate a 4-core server.")
parser.add_argument("--sampler", type=str, default="numpyro",
    help="NUTS backend: 'numpyro' (fast JAX, recommended for M-series) or 'pymc' (default PyTensor).")
parser.add_argument("--mode", type=str, default=None, choices=["hsgp", "bsts", "bsts_daily_gp"],
    help="Model mode: 'hsgp', 'bsts', or 'bsts_daily_gp'. If not specified, reads from model.py.")
parser.add_argument("--name", type=str, default=None,
    help="Directory name under results/ to save trace/summaries (defaults to mode).")
args, _ = parser.parse_known_args()

# 1. LOAD AND PREPARE OBSERVED DATA
print("Loading observed data...")
df_obs = load_observed_data()

# Ensure we have the same schema as simulation
df_obs['y'] = df_obs['PM2p5_censored'].astype(int) # Poisson needs integers
df_obs['phase_numeric'] = df_obs['phase'].map({'A': 0, 'B': -1})
df_obs['phase_occurrence'] = df_obs['phase'] + df_obs['period'].astype(int).astype(str)

# Calculate session_in_phase (days since start of that specific period/phase block)
# Groups by phase_occurrence and subtracts the minimum time in that group
df_obs = df_obs.sort_values('datetime')
df_obs['session_in_phase'] = df_obs.groupby('phase_occurrence')['datetime_of_experiment'].transform(
    lambda x: (x - x.min()).dt.total_seconds() / (24 * 3600)
)

print(f"Data ready. Observations: {len(df_obs)}")
print(df_obs[['phase_occurrence', 'y', 'phase_numeric', 'session_in_phase']].head())

# 2. DEFINE THE MODEL (Imported from shared file)
import model
mode = args.mode if args.mode is not None else model.MODE
model_name = args.name if args.name is not None else mode
use_daily_gp = (mode == "bsts_daily_gp")

if mode == "hsgp":
    model = model.get_modelKEEP(df_obs)
else:
    import pymc as pm
    model = model.get_pymc_bsts_model(df_obs, use_daily_gp=use_daily_gp)

if __name__ == "__main__":
    # 3. FIT THE MODEL
    print(f"\nFitting model in {mode.upper()} mode (MCMC), output directory: results/{model_name}...")
    
    print(f"Sampling with chains={args.chains}, sampler={args.sampler}")
    if mode == "hsgp":
        # HSGP uses Bambi's fit(); keep cores=1 to avoid multiprocessing fork issues with BLAS
        trace = model.fit(draws=1000, tune=1000, chains=args.chains, cores=1, target_accept=0.9, random_seed=42)
    else:
        # BSTS: use numpyro JAX backend by default — JIT-compiles the gradient and
        # runs all chains simultaneously via jax.vmap (no multiprocessing, M-series friendly).
        with model:
            if args.sampler == "numpyro":
                target_accept_val = 0.95 if use_daily_gp else 0.9
                trace = pm.sample(
                    draws=1000, tune=1000, chains=args.chains,
                    nuts_sampler="numpyro",
                    nuts_sampler_kwargs={"chain_method": "vectorized"},
                    target_accept=target_accept_val, random_seed=42,
                )
            else:
                trace = pm.sample(
                    draws=1000, tune=1000, chains=args.chains, cores=args.cores,
                    target_accept=0.9, random_seed=42,
                )
            # Need to manually add posterior predictive if we want them
            print("Generating posterior predictive samples...")
            pm.sample_posterior_predictive(trace, extend_inferencedata=True)

    # Ensure target directory exists
    os.makedirs(f"results/{model_name}", exist_ok=True)

    # Posterior Summary
    summary = az.summary(trace)
    print("\n--- Model Fit Summary ---")
    print(summary)
    summary.to_csv(f"results/{model_name}/fit_summary.csv")

    # In BSTS mode, we need to save the trace to netcdf manually if it's not already handled
    print(f"\nSaving full trace to results/{model_name}/fit_trace.nc...")
    trace.to_netcdf(f"results/{model_name}/fit_trace.nc")

    print(f"\nFit complete. To generate plots, run 'uv run python fit_plots.py {model_name}'.")
