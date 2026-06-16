import numpy as np
import pandas as pd
import arviz as az
from utils import load_observed_data, bootstrap_ci
import sys
if len(sys.argv) > 1:
    mode = sys.argv[1]
else:
    from model import MODE as mode

# 1. Load Data
df = load_observed_data()
df['y'] = df['PM2p5_censored'].astype(int)

# 2. Point-based Bootstrap for % Reduction
def point_bootstrap_reduction(df, n_samples=1000):
    idx_a = (df['phase'] == 'A').values
    idx_b = (df['phase'] == 'B').values
    data_a = df[idx_a]['y'].values
    data_b = df[idx_b]['y'].values
    
    reductions = []
    for _ in range(n_samples):
        # Sample points independently for A and B
        sample_a = np.random.choice(data_a, size=len(data_a), replace=True)
        sample_b = np.random.choice(data_b, size=len(data_b), replace=True)
        m_a = np.mean(sample_a)
        m_b = np.mean(sample_b)
        reductions.append((1 - m_b / m_a) * 100)
    
    reductions.sort()
    return np.mean(reductions), reductions[int(0.025*n_samples)], reductions[int(0.975*n_samples)]

# 3. Cluster-based Bootstrap for % Reduction
def cluster_bootstrap_reduction(df, n_samples=1000):
    periods = df['period'].unique()
    idx_a = df['phase'] == 'A'
    idx_b = df['phase'] == 'B'
    
    reductions = []
    for _ in range(n_samples):
        # Sample periods (days) with replacement
        sample_pers = np.random.choice(periods, size=len(periods), replace=True)
        # Reconstruct the dataset from sampled days
        sampled_df = pd.concat([df[df['period'] == p] for p in sample_pers])
        
        m_a = sampled_df[sampled_df['phase'] == 'A']['y'].mean()
        m_b = sampled_df[sampled_df['phase'] == 'B']['y'].mean()
        
        if m_a > 0:
            reductions.append((1 - m_b / m_a) * 100)
    
    reductions.sort()
    return np.mean(reductions), reductions[int(0.025*n_samples)], reductions[int(0.975*n_samples)]

# 5. Paired Per-Period Ratio Bootstrap (arithmetic mean of per-period B/A ratios).
# Includes periods where mean_B = 0 (complete purification → ratio = 0), unlike PLR.
# Estimand: 1 - mean_j(mean_B_j / mean_A_j)
def paired_ratio_bootstrap(df, n_samples=1000):
    periods = df['period'].unique()

    ratios = []
    for p in periods:
        pdata = df[df['period'] == p]
        m_a = pdata[pdata['phase'] == 'A']['y'].mean()
        m_b = pdata[pdata['phase'] == 'B']['y'].mean()
        if m_a > 0:
            ratios.append(m_b / m_a)  # 0 when mean_B = 0 (100% reduction that period)
    ratios = np.array(ratios)
    n = len(ratios)

    point_est = (1 - ratios.mean()) * 100

    idxs = np.random.randint(0, n, size=(n_samples, n))
    boot_means = ratios[idxs].mean(axis=1)
    boot_reductions = np.sort((1 - boot_means) * 100)

    lo = boot_reductions[int(0.025 * n_samples)]
    hi = boot_reductions[int(0.975 * n_samples)]
    return float(point_est), float(lo), float(hi)

# 6. Efron-Morris heteroscedastic Empirical Bayes
# Shrinks each per-period estimate toward the grand mean proportionally to
# its within-period noise: B_j = tau2 / (tau2 + sigma2_j).
# tau2 is estimated from the data (between-period variance minus noise).
# The CI is a percentile bootstrap of the grand mean of the shrunk estimates.
def efron_morris_eb(df, n_boot=10000, n_within=2000, seed=42):
    rng = np.random.default_rng(seed)
    periods = df['period'].unique()

    theta_obs = []
    sigma2_j  = []
    for p in periods:
        pdata = df[df['period'] == p]
        g_a = pdata[pdata['phase'] == 'A']['y'].values.astype(float)
        g_b = pdata[pdata['phase'] == 'B']['y'].values.astype(float)
        m_a = g_a.mean()
        if m_a <= 0:
            continue
        theta_obs.append((1 - g_b.mean() / m_a) * 100)
        # Within-period bootstrap variance for this period's estimate
        boot = []
        for _ in range(n_within):
            a_ = rng.choice(g_a, len(g_a))
            b_ = rng.choice(g_b, len(g_b))
            if a_.mean() > 0:
                boot.append((1 - b_.mean() / a_.mean()) * 100)
        sigma2_j.append(np.var(boot, ddof=1))

    theta = np.array(theta_obs)
    sigma2 = np.array(sigma2_j)
    ok = np.isfinite(theta) & np.isfinite(sigma2)
    theta, sigma2 = theta[ok], sigma2[ok]
    n = len(theta)

    theta_bar = theta.mean()
    SS = np.sum((theta - theta_bar) ** 2)
    tau2 = max(0.0, (SS - np.sum(sigma2)) / n)
    shrink = 1.0 - sigma2 / (sigma2 + tau2)
    theta_eb = theta_bar + shrink * (theta - theta_bar)

    point_est = theta_eb.mean()
    idxs = rng.integers(0, n, size=(n_boot, n))
    boot_means = np.sort(theta_eb[idxs].mean(axis=1))
    lo = boot_means[int(0.025 * n_boot)]
    hi = boot_means[int(0.975 * n_boot)]
    return float(point_est), float(lo), float(hi)

# 7. BSTS % Reduction (from the trace) — conditional and marginal
trace = az.from_netcdf(f"results/{mode}/fit_trace.nc")

# Conditional: 1 - exp(-exp(mu_log_gamma))  — efficiency at the median day (gamma_j = 0)
posterior_phase_effect = trace.posterior["mean_phase_effect"].values.flatten()
reduction_samples = (1 - np.exp(-posterior_phase_effect)) * 100
mean_bambi = np.mean(reduction_samples)
hdi_bambi  = az.hdi(reduction_samples, hdi_prob=0.95)

# Marginal: 1 - E_j[exp(-gamma_j)]  — expected efficiency averaged over all periods
gamma_j = trace.posterior["phase_numeric"].values  # (chain, draw, period)
marginal_remaining = np.exp(-gamma_j).mean(axis=-1).flatten()   # mean over periods per draw
marginal_reduction_samples = (1 - marginal_remaining) * 100
mean_marg = np.mean(marginal_reduction_samples)
hdi_marg  = az.hdi(marginal_reduction_samples, hdi_prob=0.95)

# --- SAVE RESULTS ---
m_pt,  l_pt,  u_pt  = point_bootstrap_reduction(df)
m_cl,  l_cl,  u_cl  = cluster_bootstrap_reduction(df)
m_ppr, l_ppr, u_ppr = paired_ratio_bootstrap(df)
m_eb,  l_eb,  u_eb  = efron_morris_eb(df)

records = [
    {
        "Method": "Point-wise Bootstrap",
        "Mean (%)": round(m_pt, 2),
        "Lower CI (%)": round(l_pt, 2),
        "Upper CI (%)": round(u_pt, 2),
        "Width (%)": round(u_pt - l_pt, 2)
    },
    {
        "Method": "Cluster-based Bootstrap",
        "Mean (%)": round(m_cl, 2),
        "Lower CI (%)": round(l_cl, 2),
        "Upper CI (%)": round(u_cl, 2),
        "Width (%)": round(u_cl - l_cl, 2)
    },
    {
        "Method": "Paired Ratio Bootstrap",
        "Mean (%)": round(m_ppr, 2),
        "Lower CI (%)": round(l_ppr, 2),
        "Upper CI (%)": round(u_ppr, 2),
        "Width (%)": round(u_ppr - l_ppr, 2)
    },
    {
        "Method": "Efron-Morris EB",
        "Mean (%)": round(m_eb, 2),
        "Lower CI (%)": round(l_eb, 2),
        "Upper CI (%)": round(u_eb, 2),
        "Width (%)": round(u_eb - l_eb, 2)
    },
    {
        "Method": "BSTS Conditional HDI",
        "Mean (%)": round(mean_bambi, 2),
        "Lower CI (%)": round(hdi_bambi[0], 2),
        "Upper CI (%)": round(hdi_bambi[1], 2),
        "Width (%)": round(hdi_bambi[1] - hdi_bambi[0], 2)
    },
    {
        "Method": "BSTS Marginal HDI",
        "Mean (%)": round(mean_marg, 2),
        "Lower CI (%)": round(hdi_marg[0], 2),
        "Upper CI (%)": round(hdi_marg[1], 2),
        "Width (%)": round(hdi_marg[1] - hdi_marg[0], 2)
    }
]

df_res = pd.DataFrame(records)
df_res.to_csv(f"results/{mode}/ci_comparison.csv", index=False)
print(f"Saved comparison results to results/{mode}/ci_comparison.csv")
