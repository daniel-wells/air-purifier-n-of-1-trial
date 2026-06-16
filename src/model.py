import bambi as bmb
import numpy as np
import pymc as pm

# --- CONFIGURATION ---
# Options: 'hsgp' (Global GP - stable), 'bsts' (Local Level per session - flexible), 'bsts_daily_gp'
MODE = "bsts"

# Set True to add a shared daily-pattern HSGP (f_workout) plus a Phase-B
# correction HSGP (f_workout_delta) on top of the per-period random walk.
# Requires 'session_in_phase' and 'resistance_training' columns in the data
# (both present when using fit.py).
DAILY_GP = (MODE == "bsts_daily_gp")


def get_modelKEEP(df):
    """
    Returns the stable Global HSGP + Random Slopes model.
    Captures overall room air physics with independent daily drifts.
    """
    formula = "y ~ phase_numeric + (1|phase_occurrence) + (session_in_phase|phase_occurrence) + hsgp(session_in_phase, m=15, c=1.5)"
    priors = {
        "Intercept": bmb.Prior("Normal", mu=np.log(5.0), sigma=np.log(2)),
        "phase_numeric": bmb.Prior("Exponential", lam=0.75),
        "1|phase_occurrence": bmb.Prior("Normal", mu=0, sigma=bmb.Prior("HalfNormal", sigma=1.0)),
        "session_in_phase|phase_occurrence": bmb.Prior("Normal", mu=0, sigma=bmb.Prior("HalfNormal", sigma=0.5)),
        "hsgp(session_in_phase, m=15, c=1.5)": {
            "ell": bmb.Prior("InverseGamma", alpha=3.0, beta=0.8),
            "sigma": bmb.Prior("HalfNormal", sigma=1.0)
        }
    }
    model = bmb.Model(formula, df, family="negativebinomial")
    model.set_priors(priors=priors)
    return model

def get_pymc_bsts_model(df, observed=None, use_daily_gp=None):
    """
    Direct PyMC implementation of a BSTS (Bayesian Structural Time Series).
    Completely vectorized to avoid 'Scratchpad' compilation errors.

    use_daily_gp: if None, falls back to the module-level DAILY_GP flag.
                  Set True to add the shared workout-spike HSGP (f_workout)
                  and the Phase-B correction HSGP (f_workout_delta).
    """
    # 1. Coordinate setup
    sessions = df['phase_occurrence'].unique()
    session_map = {s: i for i, s in enumerate(sessions)}
    session_idx = df['phase_occurrence'].map(session_map).values

    # Period coordinates for hierarchical treatment effect
    periods = sorted(df['period'].unique())
    period_map = {p: i for i, p in enumerate(periods)}
    period_idx = df['period'].map(period_map).values

    # Identify the start index of each session for the cumsum reset
    # Assumes df is sorted by datetime (as in fit.py)
    df_reset = df.reset_index(drop=True)
    first_obs_indices = df_reset.groupby('phase_occurrence').head(1).index.values
    obs_to_session_start_idx = df_reset.groupby('phase_occurrence')['y'].transform(lambda x: x.index[0]).values
    
    if use_daily_gp is None:
        use_daily_gp = DAILY_GP

    coords = {
        "obs": np.arange(len(df)),
        "session": sessions,
        "period": periods,
    }
    
    with pm.Model(coords=coords) as model:
        # 1. Priors
        # Hierarchical treatment effect: each period has its own log-ratio.
        # Log-normal parameterisation keeps every beta_k > 0 (purifier always reduces PM2.5).
        # Non-centred form improves sampler geometry.
        mu_log_phase = pm.Normal("mu_log_phase", -1.0, 1.0)
        sigma_log_phase = pm.HalfNormal("sigma_log_phase", 0.5)
        log_phase_raw = pm.Normal("log_phase_raw", 0.0, 1.0, dims="period")
        phase_numeric = pm.Deterministic(
            "phase_numeric",
            pm.math.exp(mu_log_phase + sigma_log_phase * log_phase_raw),
            dims="period",
        )
        # Scalar mean effect for reporting and downstream CI scripts
        mean_phase_effect = pm.Deterministic("mean_phase_effect", pm.math.exp(mu_log_phase))

        phase_data = pm.Data("phase_data", df['phase_numeric'].values, dims="obs")
        Intercept = pm.Normal("Intercept", mu=np.log(5.0), sigma=np.log(2))
        
        # 2. Local Level Components
        # When daily_gp is active the GP owns smooth within-session variation;
        # the random walk only needs to handle residual session-level drift.
        # Tighter prior (0.1) reduces the ridge between GP and RW posteriors.
        sigma_level_sigma = 0.1 if use_daily_gp else 0.25
        sigma_level = pm.HalfNormal("sigma_level", sigma_level_sigma)
        
        # Per-session baseline offset (starting state of the random walk).
        # Widened from 0.1 to 0.5 now that daily_offset is removed.
        session_init = pm.Normal("session_init", 0, 0.5, dims="session")
        
        # Innovations for every single time step
        innovations = pm.Normal("innovations", 0, 1.0, dims="obs")
        
        # Vectorized Random Walk:
        # For each index i, we want sum(innovations[start_of_session[i]:i+1])
        # This is full_cumsum[i] - full_cumsum[start_of_session[i]-1]
        full_cumsum = pm.math.cumsum(innovations * sigma_level)
        
        # We need an array where each index i contains the cumsum value at (start_of_session[i] - 1)
        # For session starts, the offset is 0.
        padded_cumsum = pm.math.concatenate([np.array([0.0]), full_cumsum])
        # Map each observation to the index of the cumsum BEFORE its session starts
        # (Using obs_to_session_start_idx which is already precomputed above)
        cumsum_offsets = padded_cumsum[obs_to_session_start_idx]
        
        # The BSTS mu is: session_init + current_innovations_sum
        mu = session_init[session_idx] + (full_cumsum - cumsum_offsets)

        # 3. Optional daily-pattern components (toggle via DAILY_GP / use_daily_gp)
        # GP components for within-session workout effect.
        # f_workout:       spike shape shared across ALL workout days (A and B).
        # f_workout_delta: additive modification for phase-B workout days only.
        #   Hypothesis: purifier accelerates post-spike recovery → f_workout_delta
        #   should show a negative slope after t≈0.75 on phase-B workout days.
        gp_offset = 0.0
        if use_daily_gp:
            t_raw = df['session_in_phase'].values.astype(float)
            t_mid = t_raw.max() / 2.0
            X_t = (t_raw - t_mid)[:, None]      # centred ≈ [-0.5, 0.5], shape (n_obs, 1)

            workout_obs = (df['resistance_training'] == 'Yes').astype(float).values
            workout_data = pm.Data("workout_data", workout_obs, dims="obs")

            # Phase-B workout indicator (1 only when resistance_training AND phase B)
            phaseB_workout_obs = (
                (df['resistance_training'] == 'Yes') & (df['phase'] == 'B')
            ).astype(float).values
            phaseB_workout_data = pm.Data("phaseB_workout_data", phaseB_workout_obs, dims="obs")

            # Shared spike: common to all workout days.
            # InverseGamma(3, 0.2) → median ~0.1 days (~2.5h).
            ell_workout = pm.InverseGamma("ell_workout", alpha=3.0, beta=0.2)
            eta_workout = pm.HalfNormal(  "eta_workout", sigma=0.8)
            cov_workout = eta_workout**2 * pm.gp.cov.ExpQuad(1, ls=ell_workout)
            gp_workout  = pm.gp.HSGP(m=[32], c=1.5, cov_func=cov_workout)
            f_workout   = gp_workout.prior("f_workout", X=X_t)

            # Phase-B deviation: how the purifier modifies the spike shape.
            # Same lengthscale family (allows similar temporal resolution).
            # Tighter eta prior (0.5) since this is a correction, not a full spike.
            ell_delta = pm.InverseGamma("ell_delta", alpha=3.0, beta=0.2)
            eta_delta = pm.HalfNormal(  "eta_delta", sigma=0.5)
            cov_delta = eta_delta**2 * pm.gp.cov.ExpQuad(1, ls=ell_delta)
            gp_delta  = pm.gp.HSGP(m=[32], c=1.5, cov_func=cov_delta)
            f_workout_delta = gp_delta.prior("f_workout_delta", X=X_t)

            gp_offset = f_workout * workout_data + f_workout_delta * phaseB_workout_data

        # 4. Combine Components
        log_lambda = Intercept + (phase_numeric[period_idx] * phase_data) + mu + gp_offset
        
        # 5. Likelihood
        # Gamma(mu=5, sigma=5) puts most prior mass on alpha in [1, 20],
        # typical for count data with moderate overdispersion.
        # HalfCauchy had unbounded heavy tails causing alpha >> 1000 (near-Poisson drift).
        alpha = pm.Gamma("alpha", mu=5.0, sigma=5.0)
        
        # Track the latent mean as a deterministic variable for prior/posterior checks
        mu_latent = pm.Deterministic("mu", pm.math.exp(log_lambda))
        
        if observed is None:
            observed = df['y'].values
            
        pm.NegativeBinomial("y", mu=mu_latent, alpha=alpha, observed=observed, dims="obs")
        
    return model
