import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

df = pd.read_csv("raw_data.csv")

#############################
## Handle temperature data ##
#############################

temperature = df["temperature"].dropna()

# Create 3rd order lag model dataframe
lagged_temperature = pd.DataFrame({
    'y': temperature,
    'y_lag_1': temperature.shift(1),
    'y_lag_2': temperature.shift(2),
    'y_lag_3': temperature.shift(3)
}).dropna()

# Fit AR(3) model
X = lagged_temperature[['y_lag_1', 'y_lag_2', 'y_lag_3']]
y = lagged_temperature['y']
model = LinearRegression().fit(X, y)

# Generate de-noised predictions
y_hat = model.predict(X)

# Align y_hat with original series (accounting for dropped observations)
y_hat = pd.Series(index=temperature.index, dtype=float)
y_hat.iloc[3:] = model.predict(X)  # Start from index 3 since we dropped first 3 obs

# plot the original and de-noised temperature
plt.plot(temperature, label='Original')
plt.plot(y_hat, label='De-noised')
plt.legend()
plt.show()

#############################
## Handle humidity data ##
#############################

humidity = df["humidity"].dropna()

# Create 3rd order lag model dataframe
lagged_humidity = pd.DataFrame({
    'y': humidity,
    'y_lag_1': humidity.shift(1),
    'y_lag_2': humidity.shift(2),
    'y_lag_3': temperature.shift(3)
}).dropna()

# Fit AR(3) model
X = lagged_humidity[['y_lag_1', 'y_lag_2', 'y_lag_3']]
y = lagged_humidity['y']
model = LinearRegression().fit(X, y)

# Generate de-noised predictions
y_hat = model.predict(X)

# Align y_hat with original series (accounting for dropped observations)
y_hat = pd.Series(index=humidity.index, dtype=float)
y_hat.iloc[3:] = model.predict(X)  # Start from index 3 since we dropped first 3 obs

# plot the original and de-noised humidity
plt.plot(humidity, label='Original')
plt.plot(y_hat, label='De-noised')
plt.legend()
plt.show()

######################
## Handle pm25 data ##
######################

pm25 = df["pm25"].dropna()

# Create 3rd order lag model dataframe
lagged_pm25 = pd.DataFrame({
    'y': pm25,
    'y_lag_1': pm25.shift(1),
    'y_lag_2': pm25.shift(2),
    'y_lag_3': pm25.shift(3)
}).dropna()

# Fit AR(3) model
X = lagged_pm25[['y_lag_1', 'y_lag_2', 'y_lag_3']]
y = lagged_pm25['y']
model = LinearRegression().fit(X, y)

# Generate de-noised predictions
y_hat = model.predict(X)

# Align y_hat with original series (accounting for dropped observations)
y_hat = pd.Series(index=pm25.index, dtype=float)
y_hat.iloc[3:] = model.predict(X)  # Start from index 3 since we dropped first 3 obs

# plot the original and de-noised pm25
plt.plot(pm25, label='Original')
plt.plot(y_hat, label='De-noised')
plt.legend()
plt.show()