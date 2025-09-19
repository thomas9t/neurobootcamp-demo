import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

def main():
    df = pd.read_csv("raw_data.csv")
    denoise_signal(df, "temperature")
    denoise_signal(df, "humidity")
    denoise_signal(df, "pm25")


def denoise_signal(X, varname):
    signal = X[varname].dropna()

    # Create 3rd order lag model dataframe
    lagged_signal = pd.DataFrame({
        'y': signal,
        'y_lag_1': signal.shift(1),
        'y_lag_2': signal.shift(2),
        'y_lag_3': signal.shift(3)
    }).dropna()
    
    # Fit AR(3) model
    X = lagged_signal[['y_lag_1', 'y_lag_2', 'y_lag_3']]
    y = lagged_signal['y']
    model = LinearRegression().fit(X, y)

    # Generate de-noised predictions
    y_hat = model.predict(X)
    
    # Align y_hat with original series (accounting for dropped observations)
    y_hat = pd.Series(index=signal.index, dtype=float)
    y_hat.iloc[3:] = model.predict(X)  # Start from index 3 since we dropped first 3 obs

    # plot the original and de-noised signal
    plt.plot(signal, label='Original')
    plt.plot(y_hat, label='De-noised')
    plt.legend()
    plt.savefig(f'{varname}_denoised.png')
    plt.close()


if __name__ == "__main__":
    main()


    