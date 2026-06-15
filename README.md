# SCED Air Purification Analysis

This repository contains the analysis for my N-of-1 trial testing the real-world reduction of PM2.5 particles using an air purifier.

## How to regenerate `index.html`

To run the analysis from scratch and generate all the plots and the final HTML report, execute the following commands in order. The project relies on `uv` for python dependency management and `quarto` for building the final document. Note that rendering Python code blocks in Quarto requires `jupyter` to be installed (`uv pip install jupyter`).

To activate the environment with all the dependencies ready:
```bash
nix develop
```

### 0. Rule-of-thumb power plot (R)
Generates the paired vs two-sample power comparison plot.
```bash
Rscript paired_power.r
```

### 1. Exploratory Data Analysis & Base Plots
Generate the initial data visualizations (timelines, density plots, ECDFs, etc.) and standard baseline statistics.
```bash
uv run python air_purification.py
```

### 2. Fit the Bayesian Hierarchical Model
Fit the BSTS NegBinomial model using PyMC + NumPyro. Saves the posterior trace and summary.
*(Note: This step takes ~20 minutes as it relies on MCMC).*
```bash
uv run python bambi_fit.py --mode bsts_daily_gp
uv run python bambi_fit.py --mode bsts

```

### 3. Generate Posterior Plots & Key Metrics
Use the trace to create posterior distributions, retrodictive checks, reduction posteriors, and
`plots/bsts/key_metrics.csv` (loaded by `index.qmd` for inline numbers).
```bash
uv run python bambi_fit_plots.py bsts
uv run python bambi_fit_plots.py bsts_daily_gp

```

### 4. Generate Timeline Overlay
Overlays raw PM2.5 levels with the model's continuous latent mean estimate.
```bash
uv run python bambi_timeline_plot.py bsts
```

### 5. Prior Predictive Simulation Plots
Generates `sim_prior_*_hist.png`, `sim_pm2p5_*.png` — prior predictive checks comparing
simulated data summaries to observed data.
```bash
uv run python bambi_simulation.py bsts
```

### 6. Prior Pushforward Components
Generates `sim_prior_pushforward_components.png` — evolving pushforward of the BSTS components.
```bash
uv run python bsts_prior_pushforward.py
```

### 7. Plate Diagram
Generates `plots/bsts/plate_diagram.png` (used in blog) and the PyMC auto-generated reference.
```bash
uv run python bsts_plate_diagram.py
```


uv run python gp_trajectory_plot.py


### 8. PSIS-LOO Diagnostic (k-hat)
Generates `plots/bsts/fit_khat_diagnostic.png`.
```bash
uv run python bsts_khat_plot.py
```

### 9. Individual Treatment Effect (ITE) Counterfactual Plot
For a chosen period, compares the observed Phase B latent mean (purifier ON) against
a synthetic counterfactual trajectory (purifier stayed OFF) computed from the posterior.
Outputs `plots/bsts/ite_counterfactual_period{N}.png`.
```bash
uv run python bsts_ite_plot.py --period 21   # Feb 26-27 (default)
uv run python bsts_ite_plot.py --period 34   # Mar 24-25 (highest contrast)
```

### 10. CI Comparison Table
Computes the bootstrap vs model CI comparison and saves `plots/bsts/ci_comparison.csv`
(loaded by the Quarto data table chunk).
```bash
uv run python ci_comparison_reduction.py bsts
uv run python ci_comparison_plot.py
```

### 10. Bootstrap Coverage Plots
Runs 100 simulated datasets to evaluate bootstrap interval coverage; saves `plots/bootstrap_coverage*.png`.
```bash
uv run python coverage_plot.py bsts
```

### 11. SBC Calibration (optional — slow, run on server)
Simulation-based calibration for the BSTS model. Requires `bsts_sbc.py` to have been run first
(typically on the remote server). Then replot from saved ranks:
```bash
# Run on server (takes hours):
uv run python bsts_sbc.py
# Replot locally once plots/bsts/sbc/sbc_ranks.csv exists:
uv run python bsts_sbc_ecdf.py
```

### 12. Render the Quarto Report
Compiles `index.qmd` into `index.html`. Inline Python numbers are pulled automatically
from `plots/bsts/key_metrics.csv` and `plots/bsts/fit_summary_clean.csv`.
```bash
uv run quarto render index.qmd
```

---

## Remote Server Setup (Linux / Hetzner)

The MCMC fits and SBC calibration loops are computationally expensive. A remote Linux server can be used to run these in the background without tying up your local machine.

**Server spec**: Ubuntu 24.04, 4 vCPUs, 8 GB RAM  
**Login**: `ssh -i ~/.ssh/hertz1 root@2a01:4f9:c013:9cf9::1`

### One-time setup

The server does not have internet access, so all dependencies are pushed from the local machine.

```bash
# 1. Download the Linux uv binary on your Mac and push it to the server
curl -LsSf "https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-unknown-linux-gnu.tar.gz" \
  -o /tmp/uv-linux.tar.gz
tar -xzf /tmp/uv-linux.tar.gz -C /tmp/
scp -i ~/.ssh/hertz1 /tmp/uv-x86_64-unknown-linux-gnu/uv \
  "root@[2a01:4f9:c013:9cf9::1]:/usr/local/bin/uv"
ssh -i ~/.ssh/hertz1 root@2a01:4f9:c013:9cf9::1 "chmod +x /usr/local/bin/uv"

# 2. Install g++ and Python dev headers (required for PyTensor C compilation)
ssh -i ~/.ssh/hertz1 root@2a01:4f9:c013:9cf9::1 "apt-get update -qq && apt-get install -y g++ python3.12-dev"

# 3. Sync the project (excluding .venv, caches and large trace files)
rsync -avz \
  -e "ssh -i ~/.ssh/hertz1" \
  --exclude='.venv' --exclude='__pycache__' --exclude='*.nc' \
  --exclude='plots/bsts/sbc/*.png' --exclude='plots/bsts/sbc/*.csv' \
  --exclude='.git' --exclude='.gemini' \
  /Users/d.wells/Dropbox/Github/sced/ \
  "root@[2a01:4f9:c013:9cf9::1]:/root/sced/"

# 4. Install Python dependencies on the server
ssh -i ~/.ssh/hertz1 root@2a01:4f9:c013:9cf9::1 "cd /root/sced && uv sync"
```

### Syncing changes and running jobs

```bash
# Re-sync local edits to the server (safe to run repeatedly)
rsync -avz \
  -e "ssh -i ~/.ssh/hertz1" \
  --exclude='.venv' --exclude='__pycache__' --exclude='*.nc' \
  --exclude='.git' --exclude='.gemini' \
  /Users/d.wells/Dropbox/Github/sced/ \
  "root@[2a01:4f9:c013:9cf9::1]:/root/sced/"

# Run the SBC calibration loop in the background on the server
ssh -i ~/.ssh/hertz1 root@2a01:4f9:c013:9cf9::1 \
  "cd /root/sced && nohup uv run python bsts_sbc.py > logs/sbc.log 2>&1 &"

# Run the model fit in the background
ssh -i ~/.ssh/hertz1 root@2a01:4f9:c013:9cf9::1 \
  "cd /root/sced && nohup uv run python bambi_fit.py > logs/fit.log 2>&1 &"

# Tail logs remotely
ssh -i ~/.ssh/hertz1 root@2a01:4f9:c013:9cf9::1 "tail -f /root/sced/logs/sbc.log"

# Sync results back to local machine
rsync -avz \
  -e "ssh -i ~/.ssh/hertz1" \
  "root@[2a01:4f9:c013:9cf9::1]:/root/sced/plots/" \
  /Users/d.wells/Dropbox/Github/sced/plots/
```

---

## Nix Quick Start

If you are on Nix, this repo now includes a `flake.nix` so you can keep using the same `uv run ...` workflow.

### 1. Enter the dev shell

```bash
nix develop
```

This shell provides Python 3.12, uv, Quarto, and a C compiler toolchain for packages that need native compilation.

When entering this shell, the uv project environment is redirected outside the repo to:

```bash
$HOME/.cache/uv/venvs/sced
```

This keeps `.venv` out of your Dropbox-synced project folder.

### 2. Sync dependencies

```bash
uv sync
```

### 3. Run scripts exactly as before

```bash
uv run python air_purification.py
uv run python bambi_fit.py
uv run python bambi_fit_plots.py
uv run quarto render index.qmd
```

### Troubleshooting: `ModuleNotFoundError` during `quarto render`

If Quarto reports missing Python modules (for example `No module named 'pandas'`), run:

```bash
uv sync --locked
uv run python -c "import pandas, sys; print(sys.executable, pandas.__version__)"
uv run quarto render index.qmd
```

In the Nix dev shell, `QUARTO_PYTHON` is set automatically to the uv environment Python so Quarto uses the same interpreter as `uv run python`.

You do not need to set up uv2nix to use this project productively. The flake gives you a stable toolchain, and uv continues to manage the Python environment.
