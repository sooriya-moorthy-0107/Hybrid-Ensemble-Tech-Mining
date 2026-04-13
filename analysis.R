# ══════════════════════════════════════════════════════════════
#  TechPulse — analysis.R
#  Run this file top to bottom before building the Shiny app
# ══════════════════════════════════════════════════════════════

library(tidyverse)
library(lubridate)
library(forecast)
library(tseries)
library(strucchange)
library(corrplot)
library(patchwork)

# ── 1. LOAD & PARSE ───────────────────────────────────────────
df <- read.csv("MLTollsStackOverflow.csv", stringsAsFactors = FALSE)
df$month <- as.Date(paste0("01-", df$month), format = "%d-%y-%b")

cat("Rows:", nrow(df), "| Cols:", ncol(df), "\n")
cat("Date range:", format(min(df$month)), "→", format(max(df$month)), "\n")
cat("Missing values:", sum(is.na(df)), "\n")

# ── 2. EDA PLOT 1: Technology Race (top 6) ───────────────────
top6 <- c("python", "r", "tensorflow", "keras", "pytorch", "spark")

p1 <- df %>%
  select(month, all_of(top6)) %>%
  pivot_longer(-month, names_to = "technology", values_to = "count") %>%
  ggplot(aes(x = month, y = count, color = technology)) +
  geom_line(linewidth = 1.1) +
  geom_smooth(method = "loess", se = FALSE,
              linetype = "dashed", linewidth = 0.5, alpha = 0.6) +
  scale_color_brewer(palette = "Set1") +
  labs(
    title    = "Technology Adoption Race (2009–2019)",
    subtitle = "Monthly Stack Overflow question volume",
    x = NULL, y = "Monthly Questions",
    color = "Technology",
    caption  = "Source: MLToolsStackOverflow dataset"
  ) +
  theme_minimal(base_size = 13) +
  theme(legend.position = "bottom")

print(p1)
ggsave("plot_race.png", p1, width = 10, height = 5, dpi = 150)

# ── 3. EDA PLOT 2: Correlation Heatmap ───────────────────────
cor_mat <- cor(df %>% select(-month), use = "complete.obs")

png("plot_correlation.png", width = 1200, height = 1000, res = 120)
corrplot(cor_mat,
         method   = "color",
         type     = "upper",
         tl.cex   = 0.55,
         tl.col   = "black",
         col      = colorRampPalette(c("#2166AC","white","#D6604D"))(200),
         title    = "Inter-Technology Correlation (82 tags)",
         mar      = c(0, 0, 2, 0))
dev.off()
cat("Correlation heatmap saved.\n")

# ── 4. EDA PLOT 3: STL Decomposition (Python) ────────────────
ts_python <- ts(df$python, start = c(2009, 1), frequency = 12)

p3 <- autoplot(stl(ts_python, s.window = "periodic")) +
  labs(title = "STL Decomposition: Python",
       subtitle = "Separating trend, seasonality, and remainder") +
  theme_minimal(base_size = 13)

print(p3)
ggsave("plot_stl.png", p3, width = 10, height = 6, dpi = 150)

# ── 5. STATIONARITY TEST ──────────────────────────────────────
cat("\n── ADF Stationarity Test: TensorFlow ──\n")
ts_tf <- ts(df$tensorflow, start = c(2009, 1), frequency = 12)
adf_result <- adf.test(ts_tf)
print(adf_result)
# If p > 0.05 → non-stationary → auto.arima handles differencing

# ── 6. STRUCTURAL BREAKPOINT DETECTION ───────────────────────
cat("\n── Structural Breakpoints: TensorFlow ──\n")
bp <- breakpoints(ts_tf ~ 1)
summary(bp)
# This detects the exact month TensorFlow's growth accelerated

# ── 7. BACK-TEST: Train 2009–2014, Test 2015–2016 ────────────
train_tf  <- window(ts_tf, end   = c(2014, 12))
actual_tf <- window(ts_tf, start = c(2015,  1), end = c(2016, 12))

# ARIMA
cat("\nFitting ARIMA...\n")
model_arima  <- auto.arima(train_tf, stepwise = FALSE, approximation = FALSE)
fc_arima     <- forecast(model_arima, h = 24)

# NNETAR
cat("Fitting NNETAR (takes ~30 seconds)...\n")
model_nn     <- nnetar(train_tf, repeats = 20)
fc_nn        <- forecast(model_nn, h = 24)

# Hybrid ensemble
ensemble_fc  <- 0.6 * fc_arima$mean + 0.4 * fc_nn$mean

# MSE comparison
mse_arima   <- mean((actual_tf - fc_arima$mean)^2)
mse_nn      <- mean((actual_tf - fc_nn$mean)^2)
mse_hybrid  <- mean((actual_tf - ensemble_fc)^2)

results_table <- data.frame(
  Model       = c("ARIMA", "NNETAR", "Hybrid Ensemble"),
  MSE         = round(c(mse_arima, mse_nn, mse_hybrid), 2),
  Improvement = c(
    "—",
    paste0(round((1 - mse_nn / mse_arima) * 100, 1), "%"),
    paste0(round((1 - mse_hybrid / mse_arima) * 100, 1), "%")
  )
)
print(results_table)

# ── 8. BACK-TEST PLOT ─────────────────────────────────────────
bt_months <- seq(as.Date("2015-01-01"), by = "month", length.out = 24)

bt_df <- data.frame(
  month  = bt_months,
  Actual          = as.numeric(actual_tf),
  ARIMA           = as.numeric(fc_arima$mean),
  NNETAR          = as.numeric(fc_nn$mean),
  `Hybrid Ensemble` = as.numeric(ensemble_fc),
  check.names = FALSE
)

p_backtest <- bt_df %>%
  pivot_longer(-month, names_to = "Model", values_to = "value") %>%
  mutate(Model = factor(Model,
                        levels = c("Actual","ARIMA","NNETAR","Hybrid Ensemble"))) %>%
  ggplot(aes(x = month, y = value,
             color = Model, linewidth = Model, linetype = Model)) +
  geom_line() +
  scale_color_manual(values = c(
    "Actual"           = "black",
    "ARIMA"            = "#2166AC",
    "NNETAR"           = "#D6604D",
    "Hybrid Ensemble"  = "#1A7D3A"
  )) +
  scale_linewidth_manual(values = c(
    "Actual" = 1.4, "ARIMA" = 1, "NNETAR" = 1, "Hybrid Ensemble" = 1.3
  )) +
  scale_linetype_manual(values = c(
    "Actual" = "solid", "ARIMA" = "dashed",
    "NNETAR" = "dashed", "Hybrid Ensemble" = "solid"
  )) +
  labs(
    title    = "Back-Test Validation: TensorFlow Forecast vs Actual (2015–2016)",
    subtitle = "Model trained exclusively on 2009–2014 data — no future data leakage",
    x = NULL, y = "Monthly Stack Overflow Questions",
    caption  = paste0(
      "Hybrid MSE: ", round(mse_hybrid, 1),
      "  |  ARIMA MSE: ", round(mse_arima, 1),
      "  |  Improvement: ",
      round((1 - mse_hybrid / mse_arima) * 100, 1), "%"
    )
  ) +
  theme_minimal(base_size = 13) +
  theme(legend.position = "bottom",
        plot.caption    = element_text(face = "bold", size = 10))

print(p_backtest)
ggsave("plot_backtest.png", p_backtest, width = 11, height = 5.5, dpi = 150)

cat("\n✅ All analysis complete. Plots saved.\n")
cat("Now open app.R and run the Shiny app.\n")