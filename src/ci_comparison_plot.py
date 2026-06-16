"""
Forest plot comparing all estimators for % PM2.5 reduction.

Rows: each method from ci_comparison.csv, ordered from simplest to most complex.
Horizontal: % reduction with 95% CI / HDI error bars.
Colour: three groups — bootstrap (red), EB (orange), BSTS (blue).

Usage: uv run python ci_comparison_plot.py [mode]
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

if len(sys.argv) > 1:
    mode = sys.argv[1]
else:
    from model import MODE as mode

# ── Data ────────────────────────────────────────────────────────────────────
df = pd.read_csv(f"results/{mode}/ci_comparison.csv")

# Display order top-to-bottom: most complex first (y=0 is top row when inverted)
order = [
    "BSTS Conditional HDI",
    "BSTS Marginal HDI",
    "Efron-Morris EB",
    "Paired Ratio Bootstrap",
    "Cluster-based Bootstrap",
    "Point-wise Bootstrap",
]
df["_order"] = df["Method"].map({m: i for i, m in enumerate(order)})
df = df.sort_values("_order").reset_index(drop=True)

# Assign groups for colour
def method_group(m):
    if "BSTS" in m:
        return "bsts"
    if "EB" in m:
        return "eb"
    return "bootstrap"

df["group"] = df["Method"].map(method_group)

palette = {
    "bootstrap": "#c0392b",  # red
    "eb":        "#e67e22",  # orange
    "bsts":      "#2980b9",  # blue
}

# ── Plot ─────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))

n = len(df)
y_positions = np.arange(n)

for i, row in df.iterrows():
    y = y_positions[i]
    col = palette[row["group"]]
    lo, hi = row["Lower CI (%)"], row["Upper CI (%)"]
    mean = row["Mean (%)"]

    # CI line
    ax.hlines(y, lo, hi, colors=col, linewidth=1.2, zorder=2)
    # end caps
    ax.vlines([lo, hi], y - 0.12, y + 0.12, colors=col, linewidth=1.2, zorder=2)
    # mean diamond
    diamond_x = [mean, mean - 0.3, mean, mean + 0.3, mean]
    diamond_y = [y + 0.15, y,      y - 0.15, y,     y + 0.15]
    ax.fill(diamond_x, diamond_y, color=col, zorder=3)

# Vertical reference line at BSTS Marginal mean
bsts_marg = df.loc[df["Method"] == "BSTS Marginal HDI", "Mean (%)"].values[0]
ax.axvline(bsts_marg, color="#2980b9", linewidth=0.8, linestyle="--", alpha=0.45, zorder=1)

# Axis labels and ticks
ax.set_yticks(y_positions)
ax.set_yticklabels(df["Method"], fontsize=10)
ax.invert_yaxis()
ax.set_xlabel("PM2.5 reduction (%)", fontsize=10)
ax.set_xlim(72, 102)
ax.xaxis.set_minor_locator(plt.MultipleLocator(1))
ax.tick_params(axis="x", which="both", direction="in")

# Legend
patches = [
    mpatches.Patch(color=palette["bootstrap"], label="Bootstrap"),
    mpatches.Patch(color=palette["eb"],        label="Empirical Bayes"),
    mpatches.Patch(color=palette["bsts"],      label="BSTS (Bayesian)"),
]
ax.legend(handles=patches, fontsize=9, framealpha=0.7, loc="lower left")

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_title("Estimator comparison: 95% CI / HDI", fontsize=11, pad=10)

fig.tight_layout()
out = f"results/{mode}/ci_comparison_forest.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved {out}")
