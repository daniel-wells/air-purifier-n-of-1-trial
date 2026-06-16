# Bridge R script to use the SBC package locally for the Bayesian Eye Chart
library(ggplot2)
library(dplyr)
library(tidyr)

if (dir.exists("/Users/d.wells/Dropbox/Github/SBC")) {
  message("Found local SBC repo. Attempting to load...")
  if (requireNamespace("devtools", quietly = TRUE)) {
    devtools::load_all("/Users/d.wells/Dropbox/Github/SBC")
  } else {
    r_files <- list.files("/Users/d.wells/Dropbox/Github/SBC/R", pattern = "\\.R$", full.names = TRUE)
    for (f in r_files) source(f)
  }
} else {
  stop("Local SBC repo not found at /Users/d.wells/Dropbox/Github/SBC")
}

# Parse CLI arguments for model directory name
args <- commandArgs(trailingOnly = TRUE)
model_name <- if (length(args) > 0) args[1] else "bsts"

# 1. Load the stats generated in Python
stats_path <- paste0("results/", model_name, "/sbc/sbc_stats.csv")
if (!file.exists(stats_path)) {
  stop(paste("Waiting for stats file to be generated at", stats_path, "... (Python loop is running)"))
}

stats_raw <- read.csv(stats_path)

# 2. Check if we have enough data (at least 1 iteration)
if (nrow(stats_raw) == 0) {
  stop("No stats recorded yet. Wait for SBC to finish at least one iteration.")
}

# 3. Augment with z_score and contraction for filtering
# Note: SBC package calculates contraction internally for plotting, but we filter here
stats_final <- stats_raw %>%
  rename(sd = post_sd) %>%
  mutate(
    z_score = (post_mean - true_value) / sd,
    contraction = 1 - (sd^2 / prior_sd^2)
  ) %>%
  filter(contraction >= -3.0)

# 4. Generate the Bayesian Eye Chart
message("Generating Bayesian Eye Chart (Prior-Posterior Contraction)...")

# Need to provide prior_sd as a named vector to plot_contraction
prior_sd_vec <- stats_final %>% 
  group_by(variable) %>% 
  summarize(p_sd = first(prior_sd)) %>% 
  { setNames(.$p_sd, .$variable) }

p_eye <- plot_contraction(stats_final, prior_sd = prior_sd_vec) +
  geom_hline(yintercept = 0, color = "red", linetype = "dashed", alpha = 0.5) +
  ylim(-4, 4) +
  labs(title = "The Bayesian Eye Chart: Contraction vs Z-score",
       subtitle = "High contraction (right) = data informs the parameter. Zero z-score = unbiased estimation.")

chart_path <- paste0("results/", model_name, "/sbc/sbc_bayesian_eye_chart.png")
ggsave(chart_path, p_eye, width = 10, height = 7, dpi = 300)

message(paste("SUCCESS: Bayesian Eye Chart generated in", chart_path))
