import pandas as pd
import plotnine as pn
import numpy as np
from utils import bootstrap_lci, bootstrap_uci

def plot_jitter_comparison(df, value_col, group_col, stats=None, limit=None, limit_label="Limit", title=None):
    """Jitter plot with group means and bootstrap CIs.
    If 'stats' is provided, it must have [group_col, value_col, 'lci', 'uci'] columns.
    """
    if stats is None:
        stats = df.dropna(subset=[value_col]).groupby(group_col).agg(
            mean_val = (value_col, np.mean),
            lci = (value_col, bootstrap_lci),
            uci = (value_col, bootstrap_uci)
        ).reset_index().rename(columns={'mean_val': value_col})

    p = (
        pn.ggplot(df, pn.aes(x=group_col, y=value_col))
        + pn.geom_jitter(alpha=0.2, height=0, stroke=0)
        + pn.geom_crossbar(data=stats, mapping=pn.aes(ymax='uci', ymin='lci'), 
                           width=0.8, colour='red', size=0.2)
    )

    if limit is not None:
        p += pn.geom_hline(yintercept=limit, linetype='dashed', colour='blue')
        p += pn.annotate("text", x=0.5, y=limit + 0.5, label=limit_label, 
                         color="blue", ha="left")

    if title:
        p += pn.ggtitle(title)

    return p

def plot_distribution_comparison(df, value_col, group_col, draw_col=None, binwidth=1, x_lim=(0, 20), title=None, observed_df=None):
    """Frequency polygon distribution for comparing groups, optionally showing individual draws and observed data"""
    # Performance safety: Clip extremely large values to prevent bin explosion with geom_freqpoly(binwidth=1)
    # We clip to 2x the x_lim maximum
    max_val = x_lim[1] * 2
    df_plot = df.copy()
    df_plot[value_col] = df_plot[value_col].clip(upper=max_val)
    df_plot = df_plot.dropna(subset=[value_col])
    
    df_plot['_plot_group'] = df_plot[group_col].astype(str)
    if observed_df is not None:
        df_plot['_plot_group'] = df_plot['_plot_group'] + " (Sim)"
        obs_df = observed_df.copy()
        obs_df['_plot_group'] = obs_df[group_col].astype(str) + " (Obs)"
    
    # Global aesthetics (shared by all layers)
    aes_base = pn.aes(x=value_col, y=pn.after_stat('density'), colour='_plot_group')
    
    if draw_col:
        df_plot['___group_draw'] = df_plot[group_col].astype(str) + "_" + df_plot[draw_col].astype(str)
        # Layer-specific mapping for the simulation
        aes_sim = pn.aes(group='___group_draw')
        alpha = 0.25
        size = 0.5 
    else:
        aes_sim = pn.aes(group='_plot_group') if observed_df is not None else pn.aes()
        alpha = 1.0
        size = 1
    private_colors = {
        'A': '#E41A1C', 'B': '#377EB8',
        'False': '#E41A1C', 'True': '#377EB8',
        'A (Obs)': '#E41A1C', 'B (Obs)': '#377EB8',
        'A (Sim)': '#FF7F00', 'B (Sim)': '#984EA3',
        'False (Obs)': '#E41A1C', 'True (Obs)': '#377EB8',
        'False (Sim)': '#FF7F00', 'True (Sim)': '#984EA3'
    }

    p = (
        pn.ggplot(df_plot, aes_base)
        + pn.geom_freqpoly(mapping=aes_sim, binwidth=binwidth, alpha=alpha, size=size)
        + pn.scale_colour_manual(values=private_colors)
        + pn.labs(x=value_col, y="Density", colour=group_col)
        + pn.theme_minimal()
        + pn.theme(legend_position='bottom', aspect_ratio=1)
        + pn.coord_cartesian(xlim=x_lim)
    )

    if observed_df is not None:
        p += pn.geom_freqpoly(data=obs_df, mapping=pn.aes(group='_plot_group'), 
                            binwidth=binwidth, alpha=1, size=1.0)

    if title:
        p += pn.ggtitle(title)
    return p

def plot_ecdf_comparison(df, value_col, group_col, draw_col=None, limit=None, x_lim=(0, 20), title=None, observed_df=None):
    """Empirical Cumulative Distribution Function plot, optionally showing individual draws and observed data"""
    df_plot = df.copy()
    df_plot['_plot_group'] = df_plot[group_col].astype(str)
    if observed_df is not None:
        df_plot['_plot_group'] = df_plot['_plot_group'] + " (Sim)"
        obs_df = observed_df.copy()
        obs_df['_plot_group'] = obs_df[group_col].astype(str) + " (Obs)"
        
    aes_base = pn.aes(x=value_col, colour='_plot_group')
    
    if draw_col:
        df_plot['___group_draw'] = df_plot[group_col].astype(str) + "_" + df_plot[draw_col].astype(str)
        aes_sim = pn.aes(group='___group_draw')
        alpha = 0.25
        size = 0.5 # Increased from 0.2
    else:
        aes_sim = pn.aes(group='_plot_group') if observed_df is not None else pn.aes()
        alpha = 1.0
        size = 1

    private_colors = {
        'A': '#E41A1C', 'B': '#377EB8',
        'False': '#E41A1C', 'True': '#377EB8',
        'A (Obs)': '#E41A1C', 'B (Obs)': '#377EB8',
        'A (Sim)': '#FF7F00', 'B (Sim)': '#984EA3',
        'False (Obs)': '#E41A1C', 'True (Obs)': '#377EB8',
        'False (Sim)': '#FF7F00', 'True (Sim)': '#984EA3'
    }

    p = (
        pn.ggplot(df_plot, aes_base)
        + pn.stat_ecdf(mapping=aes_sim, alpha=alpha, size=size)
        + pn.scale_colour_manual(values=private_colors)
        + pn.labs(x=value_col, y="Cumulative Proportion", colour=group_col)
        + pn.theme_minimal()
        + pn.theme(legend_position='bottom', aspect_ratio=1)
        + pn.coord_cartesian(xlim=x_lim)
    )

    if observed_df is not None:
        p += pn.stat_ecdf(data=obs_df, mapping=pn.aes(group='_plot_group'), alpha=1, size=1.0)
    if limit is not None:
        p += pn.geom_vline(xintercept=limit, linetype='dashed', colour='blue')
    if title:
        p += pn.ggtitle(title)
    return p

def plot_timeline_area(df, time_col, value_col, group_col, phase_col=None, title=None):
    """Area plot for timelines with clean phase boundaries.
    Automatically detects contiguous phases to prevent 'bleeding'.
    """
    plot_df = df.copy().sort_values(time_col)
    
    if phase_col is None:
        plot_df['__phase_id'] = (plot_df[group_col] != plot_df[group_col].shift()).cumsum()
        phase_col = '__phase_id'
        
    # We create a unique key for every contiguous block of a single color
    plot_df['__group_phase_id'] = plot_df[group_col].astype(str) + "_" + plot_df[phase_col].astype(str)
    
    boundaries = []
    # Identify unique ribbons
    unique_ids = plot_df['__group_phase_id'].unique()
    
    for uid in unique_ids:
        chunk = plot_df[plot_df['__group_phase_id'] == uid]
        if chunk.empty: continue
        
        t_min = chunk[time_col].min()
        t_max = chunk[time_col].max()
        p_type = chunk[group_col].iloc[0]
        p_id = chunk[phase_col].iloc[0]
        
        # Force a 0-drop at both ends of the ribbon
        boundaries.append(pd.DataFrame({
            time_col: [t_min, t_max],
            value_col: [0, 0],
            group_col: [p_type, p_type],
            phase_col: [p_id, p_id],
            '__group_phase_id': [uid, uid]
        }))
            
    plot_df = pd.concat([plot_df] + boundaries).sort_values(time_col)

    p = (
        pn.ggplot(plot_df, pn.aes(x=time_col, y=value_col, fill=group_col, group='__group_phase_id'))
        + pn.geom_area(alpha=0.6, position="identity") 
        + pn.scale_fill_brewer(type="qual", palette="Set1")
        + pn.theme_minimal()
        + pn.theme(legend_position='bottom')
    )
    
    if title:
        p += pn.ggtitle(title)
        
    return p

def plot_timeline_with_mu(df, time_col, y_col, mu_col, group_col, phase_col=None, title=None):
    """Layered Area + Line plot comparing observed counts (area) and latent means (colored line).
    Ensures phase-based coloring and clean boundaries.
    """
    plot_df = df.copy().sort_values(time_col)
    
    if phase_col is None:
        plot_df['__phase_id'] = (plot_df[group_col] != plot_df[group_col].shift()).cumsum()
        phase_col = '__phase_id'
        
    plot_df['__group_phase_id'] = plot_df[group_col].astype(str) + "_" + plot_df[phase_col].astype(str)
    
    # Phase colors for consistency
    phase_colors = {'A': '#E41A1C', 'B': '#377EB8'}
    
    p = (
        pn.ggplot(plot_df, pn.aes(x=time_col, group='__group_phase_id'))
        + pn.geom_area(pn.aes(y=y_col, fill=group_col), alpha=0.3, position="identity")
        + pn.geom_line(pn.aes(y=mu_col, color=group_col), alpha=1.0, size=1.0)
        + pn.scale_fill_manual(values=phase_colors, name="Observed Counts (Area)")
        + pn.scale_color_manual(values=phase_colors, name="Latent Mean (Line)")
        + pn.theme_minimal()
        + pn.theme(legend_position='bottom', legend_box='vertical')
        + pn.labs(x="Session Index", y="PM2.5 (ug/m^3)")
    )
    
    if title:
        p += pn.ggtitle(title)
        
    return p

def plot_summary_histogram(sim_df, obs_df, value_col, phase_col, title, use_log_scale=False, x_lim=None, discrete_integers=False, bins=100):
    """
    Plots a histogram of simulated summaries (e.g. means) per phase, 
    with different colors on the same axis.
    """
    if discrete_integers:
        # One bar per integer, centered exactly on the integer
        geom_hist = pn.geom_histogram(pn.aes(fill=phase_col), binwidth=1, color="white", boundary=-0.5, position="identity", alpha=0.5)
    else:
        geom_hist = pn.geom_histogram(pn.aes(fill=phase_col), bins=bins, color="white", position="identity", alpha=0.5)

    p = (pn.ggplot(sim_df, pn.aes(x=value_col))
         + geom_hist
         + pn.geom_vline(pn.aes(xintercept=value_col, color=phase_col), data=obs_df, size=0.5, linetype="dashed")
         + pn.scale_fill_brewer(type="qual", palette="Set1")
         + pn.scale_color_brewer(type="qual", palette="Set1")
         + pn.theme_minimal()
         + pn.theme(aspect_ratio=1, legend_position="bottom")
         + pn.labs(title=title, x=value_col, y="Frequency", fill="Phase", color="Observed Mean"))
    
    if use_log_scale:
        from copy import deepcopy
        sim_df = sim_df.copy(); sim_df[value_col] = sim_df[value_col] + 1e-6
        p = (p + pn.scale_x_log10(labels=lambda l: [f"{x:g}" for x in l]))
    if x_lim: p = p + pn.coord_cartesian(xlim=x_lim)
    return p

def plot_session_patterns(df, x_col, y_col, group_col, facet_col, title=None):
    """Plot trends within a session/phase, with one line per occurrence (like pm2p5_day.png)"""
    p = (
        pn.ggplot(df, pn.aes(x=x_col, y=y_col, colour=facet_col, group=group_col))
        + pn.geom_line(alpha=0.1)
        + pn.scale_colour_brewer(type="qual", palette="Set1")
        + pn.theme_minimal()
        + pn.theme(aspect_ratio=1, legend_position="bottom")
        + pn.labs(x="Minutes in Phase", y="PM2.5 Count", colour="Phase")
    )
    if title:
        p += pn.ggtitle(title)
    return p

def _calc_retrodictive_data_for_phase(sim_matrix, obs_array, phase, max_bin):
    bins = np.arange(0, max_bin + 2) - 0.5
    obs_counts, _ = np.histogram(obs_array, bins=bins)
    
    n_draws = sim_matrix.shape[0]
    sim_counts = np.zeros((n_draws, max_bin + 1))
    for i in range(n_draws):
        sim_counts[i, :] = np.histogram(sim_matrix[i], bins=bins)[0]
        
    probs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    quantiles = np.quantile(sim_counts, probs, axis=0)
    
    records = []
    for b in range(max_bin + 1):
        x = b
        for idx_pair, prob_val in [(0, '10-90%'), (1, '20-80%'), (2, '30-70%'), (3, '40-60%')]:
            ymin = quantiles[idx_pair, b]
            ymax = quantiles[8 - idx_pair, b]
            records.append({
                'phase': phase,
                'x': x,
                'xmin': x - 0.5,
                'xmax': x + 0.5,
                'ymin': ymin,
                'ymax': ymax,
                'interval': prob_val
            })
    df_rects = pd.DataFrame(records)
    line_records = []
    for b in range(max_bin + 1):
        line_records.append({'phase': phase, 'x': b - 0.5, 'y': obs_counts[b], 'type': 'obs'})
        line_records.append({'phase': phase, 'x': b + 0.5, 'y': obs_counts[b], 'type': 'obs'})
        
        med = quantiles[4, b]
        line_records.append({'phase': phase, 'x': b - 0.5, 'y': med, 'type': 'median'})
        line_records.append({'phase': phase, 'x': b + 0.5, 'y': med, 'type': 'median'})

    df_lines = pd.DataFrame(line_records)
    return df_rects, df_lines

def plot_retrodictive_distribution_step(sim_matrix_a, obs_a, sim_matrix_b, obs_b, max_bin=40):
    """
    Plots a posterior retrodictive distribution mimicking Michael Betancourt's style.
    Uses layered credible intervals explicitly computed as step functions across integer counts.
    """
    df_rects_a, df_lines_a = _calc_retrodictive_data_for_phase(sim_matrix_a, obs_a, 'Phase A (OFF)', max_bin)
    df_rects_b, df_lines_b = _calc_retrodictive_data_for_phase(sim_matrix_b, obs_b, 'Phase B (ON)', max_bin)
    
    df_rects = pd.concat([df_rects_a, df_rects_b])
    df_lines = pd.concat([df_lines_a, df_lines_b])
    
    df_rects['interval'] = pd.Categorical(df_rects['interval'], categories=['10-90%', '20-80%', '30-70%', '40-60%'], ordered=True)
    
    custom_colors = {'Phase A (OFF)': '#E41A1C', 'Phase B (ON)': '#377EB8'}
    custom_median = {'Phase A (OFF)': '#990000', 'Phase B (ON)': '#003366'}
    
    p = (
        pn.ggplot()
        + pn.geom_rect(
            df_rects,
            pn.aes(xmin='xmin', xmax='xmax', ymin='ymin', ymax='ymax', fill='phase', alpha='interval')
        )
        + pn.geom_path(
            df_lines[df_lines['type'] == 'median'],
            pn.aes(x='x', y='y', color='phase'),
            size=0.5
        )
        + pn.geom_path(
            df_lines[df_lines['type'] == 'obs'],
            pn.aes(x='x', y='y'),
            color='black',
            size=0.5
        )
        + pn.scale_fill_manual(values=custom_colors)
        + pn.scale_color_manual(values=custom_median)
        + pn.scale_alpha_manual(values={'10-90%': 0.25, '20-80%': 0.45, '30-70%': 0.65, '40-60%': 0.85})
        + pn.facet_wrap('~phase', ncol=1, scales='free_y')
        + pn.labs(x="PM2.5 Count (Particles)", y="Number of Observations", title="Posterior Retrodictive Check")
        + pn.theme_minimal()
        + pn.theme(legend_position='right')
        + pn.coord_cartesian(xlim=(-0.5, max_bin + 0.5))
    )
    return p

def _calc_retrodictive_ecdf_data_for_phase(sim_matrix, obs_array, phase, max_bin):
    bins = np.arange(0, max_bin + 2) - 0.5
    obs_counts, _ = np.histogram(obs_array, bins=bins)
    obs_ecdf = np.cumsum(obs_counts) / len(obs_array)
    
    n_draws = sim_matrix.shape[0]
    sim_ecdf = np.zeros((n_draws, max_bin + 1))
    for i in range(n_draws):
        sim_counts = np.histogram(sim_matrix[i], bins=bins)[0]
        sim_ecdf[i, :] = np.cumsum(sim_counts) / len(sim_matrix[i])
        
    probs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    quantiles = np.quantile(sim_ecdf, probs, axis=0)
    
    records = []
    for b in range(max_bin + 1):
        x = b
        for idx_pair, prob_val in [(0, '10-90%'), (1, '20-80%'), (2, '30-70%'), (3, '40-60%')]:
            ymin = quantiles[idx_pair, b]
            ymax = quantiles[8 - idx_pair, b]
            records.append({
                'phase': phase,
                'x': x,
                'xmin': x - 0.5,
                'xmax': x + 0.5,
                'ymin': ymin,
                'ymax': ymax,
                'interval': prob_val
            })
    df_rects = pd.DataFrame(records)
    
    line_records = []
    for b in range(max_bin + 1):
        line_records.append({'phase': phase, 'x': b - 0.5, 'y': obs_ecdf[b], 'type': 'obs'})
        line_records.append({'phase': phase, 'x': b + 0.5, 'y': obs_ecdf[b], 'type': 'obs'})
        
        med = quantiles[4, b]
        line_records.append({'phase': phase, 'x': b - 0.5, 'y': med, 'type': 'median'})
        line_records.append({'phase': phase, 'x': b + 0.5, 'y': med, 'type': 'median'})

    df_lines = pd.DataFrame(line_records)
    return df_rects, df_lines

def plot_retrodictive_ecdf_step(sim_matrix_a, obs_a, sim_matrix_b, obs_b, max_bin=40):
    df_rects_a, df_lines_a = _calc_retrodictive_ecdf_data_for_phase(sim_matrix_a, obs_a, 'Phase A (OFF)', max_bin)
    df_rects_b, df_lines_b = _calc_retrodictive_ecdf_data_for_phase(sim_matrix_b, obs_b, 'Phase B (ON)', max_bin)
    
    df_rects = pd.concat([df_rects_a, df_rects_b])
    df_lines = pd.concat([df_lines_a, df_lines_b])
    
    df_rects['interval'] = pd.Categorical(df_rects['interval'], categories=['10-90%', '20-80%', '30-70%', '40-60%'], ordered=True)
    
    custom_colors = {'Phase A (OFF)': '#E41A1C', 'Phase B (ON)': '#377EB8'}
    custom_median = {'Phase A (OFF)': '#990000', 'Phase B (ON)': '#003366'}
    
    p = (
        pn.ggplot()
        + pn.geom_rect(
            df_rects,
            pn.aes(xmin='xmin', xmax='xmax', ymin='ymin', ymax='ymax', fill='phase', alpha='interval')
        )
        + pn.geom_path(
            df_lines[df_lines['type'] == 'median'],
            pn.aes(x='x', y='y', color='phase'),
            size=0.5
        )
        + pn.geom_path(
            df_lines[df_lines['type'] == 'obs'],
            pn.aes(x='x', y='y'),
            color='black',
            size=0.5
        )
        + pn.scale_fill_manual(values=custom_colors)
        + pn.scale_color_manual(values=custom_median)
        + pn.scale_alpha_manual(values={'10-90%': 0.25, '20-80%': 0.45, '30-70%': 0.65, '40-60%': 0.85})
        + pn.facet_wrap('~phase', ncol=1, scales='free_y')
        + pn.labs(x="PM2.5 Count (Particles)", y="Cumulative Proportion", title="Posterior Retrodictive ECDF")
        + pn.theme_minimal()
        + pn.theme(legend_position='right')
        + pn.coord_cartesian(xlim=(-0.5, max_bin + 0.5))
    )
    return p
