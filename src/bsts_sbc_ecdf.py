import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import stats
from scipy.optimize import minimize_scalar

import sys
import model

if len(sys.argv) > 1:
    model_name = sys.argv[1]
else:
    model_name = model.MODE

# Settings
N_SAMPLES_POSTERIOR = 1000 # This is max_rank + 1 (L in the R code sense of range)
CSV_PATH = f"results/{model_name}/sbc/sbc_ranks.csv"
OUTPUT_DIR = f"results/{model_name}/sbc/"

def p_interior(p_int, x1, x2, z1, z2, N):
    """Transition probability for a binomial path staying within bounds"""
    if z1 == z2:
        # No progress in time, just filter x1 to match x2
        mask = np.isin(x1, x2)
        new_p = np.zeros_like(x2, dtype=float)
        # Match indices (simplified since we only call this when z2 > z1 in the loop)
        return p_int, x1
        
    z_tilde = (z2 - z1) / (1.0 - z1)
    
    # Prob(x2 | x1) = Binomial(x2 - x1; N - x1, z_tilde)
    # Vectorized: rows are x2, columns are x1
    n_tilde = N - x1 # (len_x1,)
    x_diff = x2[:, None] - x1[None, :] # (len_x2, len_x1)
    
    # Calculate binomial pmf
    # We use log-space for stability if needed, but for small N pmf is fine
    pmfs = stats.binom.pmf(x_diff, n_tilde, z_tilde)
    
    new_p = np.dot(pmfs, p_int)
    return new_p, x2

def calculate_gamma(N, K, conf_level=0.95):
    """Finds the gamma parameter for simultaneous confidence intervals"""
    z_all = np.linspace(0, 1, K + 1)
    
    def target(gamma):
        # Upper and lower bounds (Pointwise)
        # Note: Rev logic for upper to handle symmetry as in R
        low = stats.binom.ppf(gamma / 2, N, z_all)
        high = N - stats.binom.ppf(gamma / 2, N, 1 - z_all)
        
        p_int = np.array([1.0])
        x1 = np.array([0])
        
        for i in range(1, K + 1):
            z1, z2 = z_all[i-1], z_all[i]
            x2_lower = int(max(x1[0], low[i]))
            x2_upper = int(high[i])
            
            if x2_lower > x2_upper:
                return 1.0 # Impossible path
            
            x2 = np.arange(x2_lower, x2_upper + 1)
            p_int, x1 = p_interior(p_int, x1, x2, z1, z2, N)
            
            if len(p_int) == 0 or np.sum(p_int) == 0:
                return 1.0
                
        return abs(conf_level - np.sum(p_int))

    # Search for gamma in [0, 1 - conf_level]
    res = minimize_scalar(target, bounds=(0, 1 - conf_level), method='bounded', options={'xatol': 1e-5})
    return res.x

def get_ecdf_data(ranks, max_rank, prob=0.95):
    N = len(ranks)
    # Match SBC package exactly: K = min(max_rank + 1, N)
    K = min(max_rank + 1, N)
    
    gamma = calculate_gamma(N, K, conf_level=prob)
    z = np.linspace(0, 1, K + 1)
    
    # Bounds
    low = stats.binom.ppf(gamma / 2, N, z)
    high = N - stats.binom.ppf(gamma / 2, N, 1 - z)
    
    # Points for step plots (repeated as in R)
    # R: z_twice <- c(0, rep(z[2:(K + 1)], each = 2))
    z_step = np.concatenate([[0], np.repeat(z[1:], 2)])
    
    # R: lims$lower <- c(rep(lims$lower[1:K], each=2), lims$lower[K + 1])
    low_step = np.concatenate([np.repeat(low[:-1], 2), [low[-1]]])
    high_step = np.concatenate([np.repeat(high[:-1], 2), [high[-1]]])
    
    # The uniform line (diagonal) at the same step points
    # R: uniform_val = c(rep(z[1:K], each = 2), 1)
    uniform_step = np.concatenate([np.repeat(z[:-1], 2), [1]])
    
    # Empirical ECDF
    # base_vals = floor((0:K) * (max_rank + 1) / K)
    base_vals = np.floor(np.arange(K + 1) * (max_rank + 1) / K)
    ecdf_vals = []
    for bv in base_vals:
        ecdf_vals.append(np.mean(ranks < bv))
    
    return {
        'z': z,
        'ecdf': np.array(ecdf_vals),
        'z_step': z_step,
        'low_step': low_step / N,
        'high_step': high_step / N,
        'uniform_step': uniform_step,
        'N': N, 'K': K, 'gamma': gamma
    }

def plot_ecdf(results, param_name, output_path):
    plt.figure(figsize=(8, 8))
    # CI Band (Theoretical)
    plt.fill_between(results['z_step'], results['low_step'], results['high_step'], 
                     color='skyblue', alpha=0.3, label='Theoretical CDF (95% CI)')
    # Diagonal line
    # Note: Using the step points for consistency
    # plt.plot(results['z_step'], results['uniform_step'], color='red', linestyle='--', alpha=0.3)
    
    # Observed ECDF
    plt.step(results['z'], results['ecdf'], where='post', color='black', label='Sample ECDF', linewidth=1.5)
    
    plt.title(f"SBC ECDF: {param_name} (N={results['N']})")
    plt.xlabel("Rank (Scaled)")
    plt.ylabel("Probability")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_ecdf_diff(results, param_name, output_path):
    # ECDF Diff: observed - theoretical
    # Theoretical evaluated at z is simply z
    diff = results['ecdf'] - results['z']
    
    # Limits diff
    # In R: upper - uniform_val
    # uniform_val at the evaluation points z is z
    # Since we repeat for steps:
    low_diff = results['low_step'] - results['uniform_step']
    high_diff = results['high_step'] - results['uniform_step']
    
    plt.figure(figsize=(10, 6))
    # Confidence Band
    plt.fill_between(results['z_step'], low_diff, high_diff, color='skyblue', alpha=0.3, label='Theoretical CDF (95% CI)')
    
    # Zero line
    plt.axhline(0, color='red', linestyle='--', alpha=0.4)
    
    # Differences
    plt.step(results['z'], diff, where='post', color='black', linewidth=1.2, label='Sample ECDF Diff')
    
    plt.title(f"SBC ECDF Difference: {param_name} (N={results['N']})")
    plt.xlabel("Rank (Scaled)")
    plt.ylabel("Difference (Sample - Theoretical)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_path, dpi=300)
    plt.close()

# Main
if __name__ == "__main__":
    df_ranks = pd.read_csv(CSV_PATH)
    max_rank = N_SAMPLES_POSTERIOR - 1 # ranks are 0 to 999
    
    for param in df_ranks.columns:
        print(f"Calculating and plotting ECDF for {param}...")
        ranks = df_ranks[param].values
        res = get_ecdf_data(ranks, max_rank)
        
        plot_ecdf(res, param, f"{OUTPUT_DIR}sbc_ecdf_{param}_matched.png")
        plot_ecdf_diff(res, param, f"{OUTPUT_DIR}sbc_ecdf_diff_{param}_matched.png")

    print("\nMatched ECDF plots saved to results/bsts/sbc/")
