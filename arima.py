import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA

# Example data (replace this with your actual time series data)
data = pd.Series([112, 118, 132, 129, 121, 135, 148, 148, 136, 119, 104, 118]) 

# Step 1: Fit ARIMA model (p=1, d=1, q=1)
model = ARIMA(data, order=(1, 1, 1))
model_fit = model.fit()

# Step 2: Summary of the model
print(model_fit.summary())

# Step 3: Forecast future values
forecast = model_fit.forecast(steps=5)  # Forecast the next 5 time steps
print(forecast)
