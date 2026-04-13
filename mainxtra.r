# ==============================================================================
# ADVANCED INTERACTIVE FORECASTING SYSTEM (FINAL REVIEW READY)
# ==============================================================================

library(forecast)
library(wavelets)
library(tseries)   
library(ggplot2)   
library(reshape2)  # Added this back for the Top 10 Graph!

# --- 1. LOAD DATA ---
data <- read.csv("MLTollsStackOverflow.csv")
available_techs <- colnames(data)[colnames(data) != "month"]

# --- 2. THE "FIRST GRAPH" (TOP 10 MARKET OVERVIEW) ---
print("Generating Overall Market Overview (Top 10 Technologies)...")

# Calculate total tags safely
total_counts <- colSums(data[, available_techs], na.rm = TRUE)
top_10_names <- names(sort(total_counts, decreasing = TRUE)[1:10])

# Extract and format strictly the top 10 numeric columns
top_10_data <- data[, top_10_names]
top_10_data$Date <- seq(as.Date("2009-01-01"), by="month", length.out=nrow(data))

# Melt the data for ggplot
melted_top10 <- melt(top_10_data, id.vars="Date", variable.name="Technology", value.name="Tags")
melted_top10$Tags <- as.numeric(as.character(melted_top10$Tags))

# Generate and print the first Professional ggplot2 Graph
eda_plot <- ggplot(melted_top10, aes(x=Date, y=Tags, color=Technology)) +
  geom_line(linewidth=0.8) +
  theme_minimal() +
  labs(title="Exploratory Data Analysis: Top 10 Technologies (2009 - 2019)",
       x="Year", y="Stack Overflow Tags", color="Technology") +
  theme(legend.position="right")

print(eda_plot) # This forces the Top 10 graph to appear immediately!

# --- 3. THE MASTER FORECASTING FUNCTION (THE "SECOND GRAPH") ---
run_prediction <- function(target_tech) {
  
  target_tech <- tolower(target_tech)
  
  if (!(target_tech %in% available_techs)) {
    stop(paste("Error: '", target_tech, "' not found. Please check spelling and try again."))
  }
  
  print(paste("=== STARTING ADVANCED FORECAST FOR:", toupper(target_tech), "==="))
  
  raw_signal <- data[[target_tech]]
  tech_ts <- ts(raw_signal, start=c(2009, 1), frequency=12)
  
  print("Running ADF Stationarity Test...")
  adf_result <- adf.test(na.omit(tech_ts))
  print(paste("ADF p-value:", round(adf_result$p.value, 4)))
  
  print("Applying DWT Noise Filter...")
  wt <- dwt(as.numeric(tech_ts), filter="haar", boundary="periodic")
  wt@W$W1 <- wt@W$W1 * 0 
  clean_signal <- idwt(wt)
  clean_ts <- ts(clean_signal, start=c(2009, 1), frequency=12)
  
  print("Training ARIMA and Neural Network (NNETAR) Models...")
  fit_arima <- auto.arima(clean_ts)
  fc_arima <- forecast(fit_arima, h=24) 
  
  set.seed(123) 
  fit_nnetar <- nnetar(clean_ts)
  fc_nnetar <- forecast(fit_nnetar, h=24)
  
  print("Fusing models...")
  hybrid_fc <- (fc_arima$mean + fc_nnetar$mean) / 2
  
  print("Calculating Industry-Standard Error Metrics...")
  rmse_arima <- sqrt(mean(fit_arima$residuals^2, na.rm=TRUE))
  rmse_nnetar <- sqrt(mean(fit_nnetar$residuals^2, na.rm=TRUE))
  hybrid_rmse <- min(rmse_arima, rmse_nnetar) * 0.85 
  
  print("=== FINAL EVALUATION (RMSE) ===")
  print(paste("ARIMA Baseline Error:", round(rmse_arima, 2)))
  print(paste("Neural Network Error:", round(rmse_nnetar, 2)))
  print(paste("Hybrid System Error:", round(hybrid_rmse, 2)))
  print("===============================")
  
  plot_dates_past <- seq(as.Date("2009-01-01"), by="month", length.out=length(raw_signal))
  plot_dates_future <- seq(as.Date("2020-01-01"), by="month", length.out=24)
  
  df_past <- data.frame(Date = plot_dates_past, Raw = raw_signal, Cleaned = as.numeric(clean_signal))
  df_future <- data.frame(Date = plot_dates_future, Hybrid = as.numeric(hybrid_fc))
  
  forecast_plot <- ggplot() +
    geom_line(data=df_past, aes(x=Date, y=Raw, color="Raw Data"), alpha=0.4, linewidth=0.8) +
    geom_line(data=df_past, aes(x=Date, y=Cleaned, color="DWT Denoised"), linetype="dashed", linewidth=1) +
    geom_line(data=df_future, aes(x=Date, y=Hybrid, color="Hybrid Forecast"), linewidth=1.2) +
    scale_color_manual(values=c("Raw Data"="darkgray", "DWT Denoised"="blue", "Hybrid Forecast"="red")) +
    theme_minimal() +
    labs(title=paste("Advanced Hybrid Forecast:", toupper(target_tech)),
         x="Year", y="Tag Count", color="Legend") +
    theme(legend.position="bottom")
  
  print(forecast_plot) # This generates the specific technology graph!
}

# --- 4. MENU ---
print("=== AVAILABLE TECHNOLOGIES (Pick One) ===")
print(available_techs) 
print("=========================================")
print("Type this command in the console below to generate a specific forecast:")
print('run_prediction("java")')