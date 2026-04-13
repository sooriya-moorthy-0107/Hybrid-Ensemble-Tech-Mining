# ==============================================================================
# HYBRID ENSEMBLE SYSTEM FOR TECHNOLOGY LIFECYCLE FORECASTING
# ==============================================================================

library(forecast)
library(wavelets)

#loading

data <- read.csv("MLTollsStackOverflow.csv")
#Searchin for the Python
python_ts <- ts(data$python, start=c(2009, 1), frequency=12)

#Pre- Processing the signal
print("Applying Discrete Wavelet Transform (DWT) for denoising...")
# Decompose the signal
wt <- dwt(as.numeric(python_ts), filter="haar", boundary="periodic")
#Remove the noise
wt@W$W1 <- wt@W$W1 * 0 
# Reconstruct the clean signal
python_denoised <- idwt(wt)
python_clean_ts <- ts(python_denoised, start=c(2009, 1), frequency=12)

#PARALLEL MODELING (The Hybrid Engine)
print("Training ARIMA Model (Linear Path)...")
fit_arima <- auto.arima(python_clean_ts)
fc_arima <- forecast(fit_arima, h=24) # Forecast 24 months into the future

print("Training NNETAR Model (Non-Linear Neural Network Path)...")
# Set seed for reproducibility in neural networks
set.seed(123) 
fit_nnetar <- nnetar(python_clean_ts)
fc_nnetar <- forecast(fit_nnetar, h=24)

#ENSEMBLE FUSION (Alpha Blending Average)
print("Fusing models into Hybrid Forecast...")
hybrid_forecast <- (fc_arima$mean + fc_nnetar$mean) / 2

#EVALUATION METRICS (MSE & PSNR)
# Calculate Mean Squared Error based on ARIMA residuals
mse_val <- mean(fit_arima$residuals^2, na.rm=TRUE)

# Calculate Peak Signal-to-Noise Ratio (PSNR)
max_signal <- max(python_clean_ts)
psnr_val <- 10 * log10((max_signal^2) / mse_val)

print("=== FINAL MODEL EVALUATION ===")
print(paste("Mean Squared Error (MSE):", round(mse_val, 2)))
print(paste("Peak Signal-to-Noise Ratio (PSNR):", round(psnr_val, 2), "dB"))
print("==============================")

#OUTPUT
# Plotting the historical data vs. the Hybrid Forecast
plot(python_ts, 
     main="Hybrid Forecast: Python Technology Lifecycle", 
     ylab="Stack Overflow Tag Count", 
     xlab="Year",
     xlim=c(2009, 2021), 
     col="darkgray", 
     lwd=1)

# Overlay the smoothed trend and the future forecast
lines(python_clean_ts, col="blue", lwd=1.5, lty=2) # The DWT cleaned data
lines(hybrid_forecast, col="red", lwd=2.5)         # The Future Prediction

# Add a legend for clarity
legend("topleft", 
       legend=c("Raw Historical Data", "DWT Denoised Trend", "Hybrid Forecast (24 mo)"), 
       col=c("darkgray", "blue", "red"), 
       lty=c(1, 2, 1), 
       lwd=c(1, 1.5, 2.5))