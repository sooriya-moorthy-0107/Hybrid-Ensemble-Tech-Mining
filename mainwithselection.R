# ==============================================================================
# INTERACTIVE HYBRID ENSEMBLE SYSTEM FOR TECHNOLOGY FORECASTING
# ==============================================================================
library(forecast)
library(wavelets)

#Loading 
data <- read.csv("MLTollsStackOverflow.csv")

#This tells the 82 trends 
available_techs <- colnames(data)[colnames(data) != "month"]

#main line 
run_prediction <- function(target_tech) {
  
  
  if (!(target_tech %in% available_techs)) {
    stop(paste("Error: '", target_tech, "' not found. Please check spelling and try again."))
  }
  
  print(paste("=== Starting Predictive Mining for:", toupper(target_tech), "==="))
  
  
  raw_signal <- data[[target_tech]]
  tech_ts <- ts(raw_signal, start=c(2009, 1), frequency=12)
  
  # DWT (Deadweight Tonnage) Denoising
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
  
  # Metrics
  mse_val <- mean(fit_arima$residuals^2, na.rm=TRUE)
  max_signal <- max(clean_ts)
  psnr_val <- 10 * log10((max_signal^2) / mse_val)
  
  print("=== FORECAST COMPLETE ===")
  print(paste("MSE:", round(mse_val, 2)))
  print(paste("PSNR:", round(psnr_val, 2), "dB"))
  print("=========================")
  
  #Plotting area 
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

# Menu Driven part 
print("=== AVAILABLE TECHNOLOGIES FOR FORECASTING (Pick One) ===")
print(available_techs)
print("=========================================================")
print("To generate a forecast, type this command in the bottom-left Console:")
print('run_prediction("insert_name_here")')