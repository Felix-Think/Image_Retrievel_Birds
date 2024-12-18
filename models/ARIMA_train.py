import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
#|%%--%%| <j1lXqUZbfh|L0TKw1XlHe>

#Load data
data = pd.read_csv("DATA_BIRDS.csv")
data

#|%%--%%| <L0TKw1XlHe|m8SHN5MIkf>

data.describe()

#|%%--%%| <m8SHN5MIkf|fAcn66mJHW>

# Drop Unnecessary Columns
# Season is not necessary because we have the Time data columns, from this we can get the season

data = data.drop(['Season'], axis=1)
#|%%--%%| <fAcn66mJHW|oIo2ApcgbM>

#Check for missing values
data.isnull().sum()

#|%%--%%| <oIo2ApcgbM|8I8oSrb1Hi>

data = data.dropna()
data.shape


#|%%--%%| <8I8oSrb1Hi|dRnvjTylou>

data['Time'] = pd.to_datetime(data['Time'])
data.dtypes
#|%%--%%| <dRnvjTylou|zVRm31alVN>

# Convert 'Time' column to datetime
data.set_index('Time', inplace = True)

#|%%--%%| <zVRm31alVN|nopO0UUNGE>

data.isnull().sum()
#|%%--%%| <nopO0UUNGE|tEpIMsfopS>
#Train size
train_size = int(len(data) * 0.8)
train, test = data[0:train_size], data[train_size:len(data)]
train.shape, test.shape

#|%%--%%| <tEpIMsfopS|Kaejdbu6TT>

#Check for stationarity
#In this section we will check if the data is stationary or not by using the ADF(Augmented Dickey-Fuller)
#test. The null hypothesis of the ADF test is that the time series is non-stationary. So, if the p-value
#is less than the significance level (0.05), we reject the null hypothesis and infer that the time series
#is indeed stationary.

adf_test = adfuller(train['Population'].dropna())
print('ADF Statistic: %f' % adf_test[0])
print('p-value: %f' % adf_test[1])

#|%%--%%| <Kaejdbu6TT|NVlFGypm8R>
r"""°°°
# We can see that the p-value is smaller than 0.05, the p_value here indicates that the time series is non-stationary. If the p-value of the test is geater than 0.05, then the null hypothesis is not rejected and the time series is stationary.
°°°"""
#|%%--%%| <NVlFGypm8R|nNhZwEYJIc>

#Differencing
#In this section we will difference the data to make it stationary. We will difference the data by one time period.
#|%%--%%| <nNhZwEYJIc|4U9avZfanE>

# Group by bird
grouped = data.groupby('Species Name')

for bird, group in grouped:
    print(f"Processing bird: {bird}")
    # Train size
    train_size = int(len(group) * 0.8)
    train, test = group[0:train_size], group[train_size:len(group)]
    print(train.shape, test.shape)
    # Check for stationarity
    adf_test = adfuller(train['Population'].dropna())
    print('ADF Statistic: %f' % adf_test[0])
    print('p-value: %f' % adf_test[1])
    # Differencing
    train['Population'] = train['Population'].diff().dropna()
    # Check for stationarity again after differencing
    adf_test_diff = adfuller(train['Population'].dropna())
    print('ADF Statistic after differencing: %f' % adf_test_diff[0])
    print('p-value after differencing: %f' % adf_test_diff[1])

#|%%--%%| <4U9avZfanE|MRV0dgxUZK>

# Train model for each bird
# In this section we will train an ARIMA model for each bird. We will use the ARIMA model to forecast the population of each bird

# Group by bird
grouped = data.groupby('Species Name')

for bird, gruop in grouped:
    print(f"Processing bird: {bird}")
    #Train size
    train_size = int(len(gruop) * 0.8)
    train, test = gruop[0:train_size], gruop[train_size:len(gruop)]
    print(train.shape, test.shape)
    #Call ARIMA model for each bird
    model = ARIMA(train['Population'], order=(5,1,0))
    model_fit = model.fit()
    #Predict
    forecast = model_fit.forecast(steps=len(test))
    #Calculate error
    MSE = mean_squared_error(test['Population'], forecast)
    print(f"Mean Squared Error: {MSE}")
    MAE = mean_absolute_error(test['Population'], forecast)
    print(f"Mean Absolute Error: {MAE}")
    #Plot
    plt.figure(figsize=(10,6))
    plt.plot(train['Population'], label='Training Data')
    plt.plot(test['Population'], label='Actual Data', color='green')
    plt.title(f"Actual for {bird}")
    plt.legend()
    plt.show()
    plt.figure(figsize=(10,6))
    plt.plot(train['Population'], label='Training Data')
    plt.plot(test.index, forecast, label='Forecasted Data', color='red')
    plt.title(f"Forecast for {bird}")
    plt.legend()
    plt.show()

