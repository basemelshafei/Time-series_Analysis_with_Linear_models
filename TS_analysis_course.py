import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

pd.date_range('jan 01 2021', periods=12, freq='M')
date3 = pd.date_range('jan 01 2021', periods=8760, freq='H')
data_set = np.random.randint(1, 1000, (8760, 2))
data_set_df = pd.DataFrame(data_set)

data_set_df.set_index(date3, inplace=True)

data = pd.read_csv('Temp_Data.csv', index_col='DATE', parse_dates=True)

data.index.freq = 'D'

data.dropna(inplace=True)

data = pd.Dataframe(data['Temp'])

train = data.iloc[:510, 0]
test = data.iloc[510:, 0]

from statsmodels.tsa.seasonal import seasonal_decompose

decomp_results = seasonal_decompose(data)

decomp_results.plot()

decomp_results.seasonal.plot()

# finding parameters (p, d, q)

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

plot_acf(train, lags=50)
plot_pacf(train, lags=50)

from pmdarima import auto_arima

auto_arima(data, trace=True)

#  Developing ARIMA model

from statsmodels.tsa.arima_model import ARIMA

a_model = ARIMA(train, order=(1, 1, 2))
predictor = a_model.fit()
predictor.summary

predicted_results = predictor.predict(start=len(train), end=len(train)+len(test)-1, typ='levels')

plt.plot(test, color='red', label='Actual Temp')
plt.plot(predicted_results, color='blue', label='Predicted Temp')
plt.xlabel('Day')
plt.ylabel('Temp')
plt.legend()
plt.show()

import math
from sklearn.metrics import mean_squared_error
rmse = math.sqrt(mean_squared_error(test, predicted_results))


# SARIMAX (seasonal)

sns.heatmap(data.corr())

exo = data.iloc[:, 1:4]
exo_train = exo.iloc[:510]
exo_test = exo.iloc[510:]

decomp_results = seasonal_decompose(data['Temp'])

auto_arima(data, exogenous= exo, m=7, trace=True, D=1).summary()

from statsmodels.tsa.statespace.sarimax import SARIMAX

model = SARIMAX(train, exog=exo_train, order=(2, 0, 2), seasonal_order=(0, 1, 1, 7))
model_fit = model.fit()

prediction = model.predict(len(train), len(train)+len(test)-1, exog=exo_test, typ='levels')

plt.plot(test, color='red', label='Actual Temp')
plt.plot(prediction, color='blue', label='Predicted Temp')
plt.xlabel('Day')
plt.ylabel('Temp')
plt.legend()
plt.show()

import math
from sklearn.metrics import mean_squared_error
rmse = math.sqrt(mean_squared_error(test, predicted_results))









































