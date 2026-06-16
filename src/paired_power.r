library(pwr)
library(ggplot2)
# reshape2

pwr.t.test(
  n = NULL,
  d = 0.2,
  sig.level = 0.05,
  type = "two.sample",
  alternative = "two.sided",
  power = 0.8
)

# paired vs two sample, S to noise ratio 0.5
power.t.test(delta = 0.1, sd = 0.2, sig.level = 0.05, power = 0.8, type = "paired")
power.t.test(delta = 0.1, sd = 0.2, sig.level = 0.05, power = 0.8, type = "two.sample")


# paired == one sample
power.t.test(delta = 2, sd = 3, sig.level = 0.05, n = 25, type = "paired", alternative = "one.sided")
power.t.test(delta = 2, sd = 3, sig.level = 0.05, n = 25, type = "one.sample", alternative = "one.sided")


# if no correlation, very similar power
power.t.test(delta = 5, sd = sqrt(10^2 + 10^2 - 0.0 * 10 * 10), sig.level = 0.05, power = 0.8, type = "paired", alternative = "two.sided")
power.t.test(delta = 5, sd = 10, sig.level = 0.05, power = 0.8, type = "two.sample", alternative = "two.sided")

# max correlation, about half
power.t.test(delta = 5, sd = sqrt(10^2 + 10^2 - 1.0 * 10 * 10), sig.level = 0.05, power = 0.8, type = "paired", alternative = "two.sided")
power.t.test(delta = 5, sd = 10, sig.level = 0.05, power = 0.8, type = "two.sample", alternative = "two.sided")


d <- 5
sd <- 10
power.t.test(delta = d, sd = sqrt(sd^2 + sd^2 - 0.99 * sd * sd), sig.level = 0.05, power = 0.8, type = "paired", alternative = "two.sided")
power.t.test(delta = d, sd = sd, sig.level = 0.05, power = 0.8, type = "two.sample", alternative = "two.sided")



power.t.test(n = 60, delta = 5, sd = 10, sig.level = 0.05, type = "two.sample")

power.t.test(n = 60, delta = 5, sd = sqrt(10^2 + 10^2), sig.level = 0.05, type = "paired")

one_sample_t_test_power <- function(n, d, sd, num_samples = 1) {
  df <- num_samples * n - num_samples
  c <- qt(1 - 0.05 / 2, df)
  ncp <- sqrt(num_samples * n) * d / sd / num_samples
  1 - pt(c, df, ncp)
}

n <- 30
d <- 5
sd <- 10
sd_diff <- sqrt(10^2 + 10^2)

one_sample_t_test_power(n, d, sd_diff)
power.t.test(delta = d, sd = sd_diff, n = n, sig.level = 0.05, type = "paired", alternative = "two.sided")

one_sample_t_test_power(n, d, sd, num_samples = 2)
power.t.test(delta = d, sd = sd, n = n, sig.level = 0.05, type = "two.sample", alternative = "two.sided")


# check rule of thumb

z_a2 <- qnorm(0.975) # 1.96
z_b <- qnorm(0.8) # 0.84
n_numerator <- 2 * (z_a2 + z_b)^2
n <- (n_numerator) / (d / sd)^2
# Example usage
d <- 5
sd <- 10

power.t.test(delta = d, sd = sd, power = 0.8, sig.level = 0.05, type = "two.sample", alternative = "two.sided")



# Define the functions
zscore_test <- function(d, sd, sig.level = 0.05, power = 0.8) {
  z_b <- qnorm(power)
  z_a2 <- qnorm(1 - sig.level / 2)
  n <- 2 * (z_a2 + z_b)^2 / (d / sd)^2
  return(n)
}

rule_of_thumb_power_test <- function(d, sd) {
  16 / (d / sd)^2
}

t_distribution_power_test <- function(d, sd, power = 0.8, sig.level = 0.05) {
  result <- power.t.test(
    delta = d, sd = sd, power = power, sig.level = sig.level,
    type = "two.sample", alternative = "two.sided"
  )
  result$n
}

# Generate data for different signal-to-noise ratios
signal_to_noise_ratios <- seq(0.05, 2, by = 0.01) # Range of d/sd values
results <- data.frame(
  SignalToNoise = signal_to_noise_ratios,
  rule_of_16 = sapply(signal_to_noise_ratios, function(ratio) rule_of_thumb_power_test(d = ratio, sd = 1)),
  power.t.test = sapply(signal_to_noise_ratios, function(ratio) t_distribution_power_test(d = ratio, sd = 1))
)

# Reshape data for ggplot
results_long <- reshape2::melt(results,
  id.vars = "SignalToNoise",
  variable.name = "Method", value.name = "SampleSize"
)

# Create the plot
p <- ggplot(results_long, aes(x = SignalToNoise, y = SampleSize, color = Method)) +
  geom_line() +
  labs(
    title = "Sample Size requirements vs Cohen's d",
    subtitle = "For a power of 80% at significance level of 5% using a two-sided test.",
    x = "Cohen's d (difference/SD)",
    y = "Sample Size per group",
    color = "Method"
  ) +
  scale_colour_manual(values = c("rule_of_16" = "black", "power.t.test" = "red")) +
  theme_minimal() +
  scale_x_log10(breaks = c(0.05, 0.1, 0.15, 0.2, 0.25, 0.33, 0.5, 0.66, 1, 1.5, 2)) +
  scale_y_log10(breaks = c(4, 5, 7, 8, 17, 37, 65, 150, 250, 400, 700, 1600, 6300)) +
  theme(
    legend.position = "bottom",
    panel.grid.minor = element_blank()
  )
p

# Save the plot
ggsave("results/ruleofthumb_vs_t_distribution_power_test.png", plot = p, width = 10, height = 8)

signal_to_noise_ratios_v2 <- c(0.05, 0.1, 0.15, 0.2, 0.25, 0.33, 0.5, 0.66, 1, 1.5, 2)
sapply(signal_to_noise_ratios_v2, function(ratio) t_distribution_power_test(d = ratio, sd = 1))

# sanity check
power.t.test(delta = 0.2, sd = 1, power = 0.8, sig.level = 0.05, type = "two.sample", alternative = "two.sided")
power.t.test(delta = 0.2, sd = 1, power = 0.8, sig.level = 0.05, type = "one.sample", alternative = "two.sided")
power.t.test(delta = 0.5, sd = 1, power = 0.8, sig.level = 0.05, type = "one.sample", alternative = "two.sided")


power.t.test(delta = d, sd = sd, power = 0.8, sig.level = 0.05, type = "two.sample", alternative = "two.sided")
power.t.test(delta = d, sd = sd, power = 0.8, sig.level = 0.05, type = "one.sample", alternative = "two.sided")

two_sample_t_test_power <- function(n, d, sd, sig.level = 0.05) {
  df <- 2 * n - 2
  c <- qt(1 - sig.level / 2, df)
  ncp <- sqrt(2 * n) * d / sd / 2
  1 - pt(c, df, ncp)
}

n <- 30
d <- 5
sd <- 10

two_sample_t_test_power(n, d, sd)

power.t.test(delta = d, sd = sd, n = n, sig.level = 0.05, type = "two.sample", alternative = "two.sided")

# Example usage
n <- 30
d <- 5
sd <- 10
two_sample_t_test_power(n, d, sd, sig.level = 0.05) # , alternative = "one.sided"
power.t.test(delta = d, sd = sd, n = n, sig.level = 0.05, type = "two.sample", alternative = "one.sided")



# Example usage
n <- 25
d <- 2
sd <- 4

power.t.test(delta = 2, sd = sqrt(4^2 + 4^2), sig.level = 0.05, power = 0.8, type = "paired", alternative = "one.sided")


two_sample_t_test_power(n, d, sd)
# paired_t_test_power(n, d, sd)
