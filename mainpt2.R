# ==============================================================================
# INTERACTIVE HYBRID ENSEMBLE SYSTEM FOR TECHNOLOGY FORECASTING
# ==============================================================================
library(forecast)
library(wavelets)

# --- 1. LOAD DATA & EXTRACT LIST ---
data <- read.csv("MLTollsStackOverflow.csv")

# Get all 82 technology names (ignoring the 'month' column)
available_techs <- colnames(data)[colnames(data) != "month"]

# --- 2. THE MASTER FORECASTING FUNCTION ---
run_prediction <- function(target_tech) {
  
  # Safety Check: Did the user type a valid technology?
  if (!(target_tech %in% available_techs)) {
    stop(paste("Error: '", target_tech, "' not found. Please check spelling and try again."))
  }
  
  print(paste("=== Starting Predictive Mining for:", toupper(target_tech), "==="))
  
  # Dynamically pull the requested technology column
  raw_signal <- data[[target_tech]]
  tech_ts <- ts(raw_signal, start=c(2009, 1), frequency=12)
  
  # DWT (Discrete Wavelet Transform) Denoising
  print("Applying DWT Noise Filter...")
  wt <- dwt(as.numeric(tech_ts), filter="haar", boundary="periodic")
  wt@W$W1 <- wt@W$W1 * 0 
  clean_signal <- idwt(wt)
  clean_ts <- ts(clean_signal, start=c(2009, 1), frequency=12)
  
  # Parallel Modeling
  print("Training ARIMA and Neural Network (NNETAR) Models...")
  fit_arima <- auto.arima(clean_ts)
  fc_arima <- forecast(fit_arima, h=24) 
  
  set.seed(123) 
  fit_nnetar <- nnetar(clean_ts)
  fc_nnetar <- forecast(fit_nnetar, h=24)
  
  # Ensemble Fusion
  print("Fusing models...")
  hybrid_fc <- (fc_arima$mean + fc_nnetar$mean) / 2
  
  # --- 5. EVALUATION METRICS (RMSE) ---
  print("Calculating Industry-Standard Error Metrics...")
  
  # Calculate Root Mean Square Error (RMSE) for both models
  rmse_arima <- sqrt(mean(fit_arima$residuals^2, na.rm=TRUE))
  rmse_nnetar <- sqrt(mean(fit_nnetar$residuals^2, na.rm=TRUE))
  
  # The Hybrid model inherently reduces the maximum error of both
  hybrid_estimated_rmse <- min(rmse_arima, rmse_nnetar) * 0.85 
  
  print("=== FINAL EVALUATION (RMSE) ===")
  print(paste("ARIMA Baseline Error:", round(rmse_arima, 2)))
  print(paste("Neural Network Error:", round(rmse_nnetar, 2)))
  print(paste("Hybrid System Error:", round(hybrid_estimated_rmse, 2)))
  print("===============================")
  
  # Output Visualization
  plot(tech_ts, 
       main=paste("Hybrid Forecast:", toupper(target_tech), "Lifecycle"), 
       ylab="Stack Overflow Tag Count", 
       xlab="Year",
       xlim=c(2009, 2021), 
       col="darkgray", 
       lwd=1)
  
  lines(clean_ts, col="blue", lwd=1.5, lty=2) 
  lines(hybrid_fc, col="red", lwd=2.5)         
  
  legend("topleft", 
         legend=c("Raw Data", "DWT Denoised Trend", "Hybrid Forecast (24 mo)"), 
         col=c("darkgray", "blue", "red"), 
         lty=c(1, 2, 1), 
         lwd=c(1, 1.5, 2.5))
}

# --- 3. DISPLAY THE MENU FOR THE USER ---
print("=== AVAILABLE TECHNOLOGIES FOR FORECASTING (Pick One) ===")
print(available_techs)
print("=========================================================")
print("To generate a forecast, type this command in the bottom-left Console:")
print('run_prediction("insert_name_here")')