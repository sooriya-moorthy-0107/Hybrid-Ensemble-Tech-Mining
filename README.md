# TechPulse: Hybrid Ensemble Tech Forecaster 📈

**TechPulse** is a predictive data mining system and interactive dashboard built in R. It is designed to forecast the lifecycle and market adoption of 82 distinct programming languages, frameworks, and libraries over a 24-month horizon. 

Relying on social media "hype" for technology adoption often leads to technical debt. To solve this, TechPulse analyzes 10 years of historical Stack Overflow engagement data and applies advanced signal processing to separate genuine market momentum from temporary stochastic noise.

## 🧠 Core Architecture
This project utilizes a **Multi-Stage Hybrid Ensemble** approach:
1. **Signal Denoising:** Uses **Discrete Wavelet Transform (DWT)** to filter out high-frequency noise and sudden social media spikes.
2. **Dimensionality Reduction:** Applies **SVD** to manage the high-dimensional landscape of 82 technologies.
3. **Dual-Path Forecasting:**
   * **ARIMA Path:** Captures steady, linear statistical growth.
   * **NNETAR Path:** Uses Neural Network Autoregression to learn complex, non-linear surges (S-curve adoption).
5. **Ensemble Fusion:** Integrates both models to achieve a **15.6% reduction in Root Mean Squared Error (RMSE)** compared to standalone forecasting methods.

---

## 📸 Dashboard Screenshots


### 1. Historical Explorer & DWT Denoising
*Visualizing the raw Stack Overflow counts versus the mathematically smoothed underlying trend.*
<img width="1919" height="1032" alt="image" src="https://github.com/user-attachments/assets/d3d87f73-3ea0-4b30-a8b3-3c83767f3c7a" />

### 2. The 24-Month Forecast Engine
*The Hybrid Ensemble projecting the future adoption lifecycle of the selected technology.*
<img width="1919" height="1036" alt="image" src="https://github.com/user-attachments/assets/39107bfa-10a6-4d09-8670-426056e8213b" />


### 3. Model Evaluation & RMSE Metrics
*Validating the accuracy of the Ensemble method against standalone ARIMA and NNETAR models.*
<img width="1919" height="1025" alt="image" src="https://github.com/user-attachments/assets/676c2181-8658-4640-b591-af88fb72a7f8" />

---

## 🚀 How to Run the Application Locally

To run this Shiny application on your local machine, ensure you have R and RStudio installed.

**1. Clone the repository:**
```bash
git clone [https://github.com/your-username/Hybrid-Ensemble-Tech-Mining.git](https://github.com/your-username/Hybrid-Ensemble-Tech-Mining.git)
```
**2. Install the required R packages:**

Open RStudio, run the following command in the console to install dependencies:
```bash
install.packages(c("shiny", "forecast", "wavelets", "tseries", "ggplot2", "dplyr", "bslib"))
```
**3. Run the App:**

Open the ```app.R``` file in RStudio and click the "Run App" button at the top of the script editor, or run:
```bash
shiny::runApp("app.R")
```
**4. Some Graphs**
<img width="1919" height="1035" alt="image" src="https://github.com/user-attachments/assets/45cfdf42-3851-4fde-9ddd-c30f610b0c0d" />
