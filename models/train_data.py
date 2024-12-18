import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#|%%--%%| <4AlEEL598W|APEnSsDYs9>

# Read Data
data = pd.read_csv('DATA_BIRDS.csv')
data.shape
#|%%--%%| <APEnSsDYs9|dLsCnOjopf>

data.head()
#|%%--%%| <dLsCnOjopf|NQhW4sko1g>

data.describe()

#|%%--%%| <NQhW4sko1g|Hr3w93BxtQ>

data.dtypes

#|%%--%%| <Hr3w93BxtQ|fLALWulemv>
data.columns
#|%%--%%| <fLALWulemv|tOUrzydFoD>

# Drop Unnecessary Columns
# Season is not necessary because we have the Time data columns, from this we can get the season

data = data.drop(['Season'], axis=1)
#|%%--%%| <tOUrzydFoD|gEym2DQKjQ>

# Handle Missing Values
data.isna().sum()
#|%%--%%| <gEym2DQKjQ|7CklvVBSgp>

data = data.dropna()
data.shape

#|%%--%%| <7CklvVBSgp|OVnrlPhZ6y>

# Convert Time to Datetime

data['Time'] = pd.to_datetime(data['Time'])
data['Time']


#|%%--%%| <OVnrlPhZ6y|iAS1enHK3R>

# Extract Year, Month, Day, Hour, Minute from Time
data['Year'] = data['Time'].dt.year
data['Month'] = data['Time'].dt.month
data['Day'] = data['Time'].dt.day
data
#|%%--%%| <iAS1enHK3R|oJyUFMpBZK>

# Use Prophet for Time Series Forecasting

#Encode the Species Name to find number of unique species
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data['Species Name'] = le.fit_transform(data['Species Name'])

data['Species Name'].nunique()

# There are 7 unique species in the dataset
species_labels = le.classes_

# Print the labels of the species names
for i in range(7):
    print(f"Label {i}: {species_labels[i]}")


#|%%--%%| <oJyUFMpBZK|NLzTEKXZjz>

#!pip install prophet

#|%%--%%| <NLzTEKXZjz|DiLXVGTxh7>
#Create a new dataframe with the required columns
data_prophet = data[['Time', 'Species Name', 'Population']]

#Crete data for each Species Name
list_species = data_prophet['Species Name'].unique()

data_prophet_Chaomao = data_prophet[data_prophet['Species Name'] == 0]
data_prophet_Chichchoe = data_prophet[data_prophet['Species Name'] == 1]
data_prophet_CuGay = data_prophet[data_prophet['Species Name'] == 2]
data_prophet_Cong = data_prophet[data_prophet['Species Name'] == 3]
data_prophet_Cumeo = data_prophet[data_prophet['Species Name'] == 4]
data_prophet_Catnho = data_prophet[data_prophet['Species Name'] == 5]
data_prophet_Yenui = data_prophet[data_prophet['Species Name'] == 6]


data_prophet_Chaomao = data_prophet_Chaomao.rename(columns={'Time': 'ds', 'Population': 'y'})
data_prophet_Chichchoe = data_prophet_Chichchoe.rename(columns={'Time': 'ds', 'Population': 'y'})
data_prophet_CuGay = data_prophet_CuGay.rename(columns={'Time': 'ds', 'Population': 'y'})
data_prophet_Cong = data_prophet_Cong.rename(columns={'Time': 'ds', 'Population': 'y'})
data_prophet_Cumeo = data_prophet_Cumeo.rename(columns={'Time': 'ds', 'Population': 'y'})
data_prophet_Catnho = data_prophet_Catnho.rename(columns={'Time': 'ds', 'Population': 'y'})
data_prophet_Yenui = data_prophet_Yenui.rename(columns={'Time': 'ds', 'Population': 'y'})


data_prophet_Chaomao

#|%%--%%| <DiLXVGTxh7|GPHH2dpPHn>

#Drop the Species Name column to predict the population

data_prophet_Chaomao = data_prophet_Chaomao.drop(['Species Name'], axis=1)
data_prophet_Chichchoe = data_prophet_Chichchoe.drop(['Species Name'], axis=1)
data_prophet_CuGay = data_prophet_CuGay.drop(['Species Name'], axis=1)
data_prophet_Cong = data_prophet_Cong.drop(['Species Name'], axis=1)
data_prophet_Cumeo = data_prophet_Cumeo.drop(['Species Name'], axis=1)
data_prophet_Catnho = data_prophet_Catnho.drop(['Species Name'], axis=1)
data_prophet_Yenui = data_prophet_Yenui.drop(['Species Name'], axis=1)

data_prophet_Chaomao

#|%%--%%| <GPHH2dpPHn|dpAMjSzXFl>

from prophet import Prophet

# Create a model for each species_labels
model_Chaomao = Prophet()
model_Chichchoe = Prophet()
model_CuGay = Prophet()
model_Cong = Prophet()
model_Cumeo = Prophet()
model_Catnho = Prophet()
model_Yenui = Prophet()

#|%%--%%| <dpAMjSzXFl|MpLSCAAmQq>

# Fit the model for each species_labels

model_Chaomao.fit(data_prophet_Chaomao)

#|%%--%%| <MpLSCAAmQq|T2qgZgNIho>

prediction_Chaomao = model_Chaomao.make_future_dataframe(periods=730)
prediction_Chaomao

#|%%--%%| <T2qgZgNIho|qjYkk9W7Gr>

prediction_Chaomao = model_Chaomao.predict(prediction_Chaomao)
prediction_Chaomao[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

#|%%--%%| <qjYkk9W7Gr|NbCw6v8ude>
#Plot the prediction from 2024 to 2026
model_Chaomao.plot(prediction_Chaomao[500:])
plt.title('Chaomao Population Prediction')
plt.show()
#|%%--%%| <NbCw6v8ude|eJHAIpeU9Y>

model_Chichchoe.fit(data_prophet_Chichchoe)

#|%%--%%| <eJHAIpeU9Y|hkw75CTd4Z>

prediction_Chichchoe = model_Chichchoe.make_future_dataframe(periods=730)
prediction_Chichchoe = model_Chichchoe.predict(prediction_Chichchoe)
prediction_Chichchoe[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

#|%%--%%| <hkw75CTd4Z|0TtlLqjRfN>

#Plot the prediction from 2024 to 2026
model_Chichchoe.plot(prediction_Chichchoe[500:])
plt.title('Chichchoe Population Prediction')
plt.show()

#|%%--%%| <0TtlLqjRfN|7O7oZV8zuX>

model_CuGay.fit(data_prophet_CuGay)

#|%%--%%| <7O7oZV8zuX|xv74kRHm3I>

prediction_CuGay = model_CuGay.make_future_dataframe(periods=730)
prediction_CuGay = model_CuGay.predict(prediction_CuGay)
prediction_CuGay[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

#|%%--%%| <xv74kRHm3I|IzHlBbaHds>

#Plot the prediction from 2024 to 2026
model_CuGay.plot(prediction_CuGay)
plt.title('Cugay Population Prediction')
plt.show()

#|%%--%%| <IzHlBbaHds|MDdSRdQcS8>

model_Cong.fit(data_prophet_Cong)

#|%%--%%| <MDdSRdQcS8|UKcu6Lhm37>

prediction_Cong = model_Cong.make_future_dataframe(periods=730)
prediction_Cong = model_Cong.predict(prediction_Cong)
prediction_Cong[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

#|%%--%%| <UKcu6Lhm37|uMM3GWCJeZ>

#Plot the prediction from 2024 to 2026
model_Cong.plot(prediction_Cong)
plt.title('Cong Population Prediction')
plt.show()


#|%%--%%| <uMM3GWCJeZ|pVf3UTkvFA>


model_Cumeo.fit(data_prophet_Cumeo)

#|%%--%%| <pVf3UTkvFA|ZBS1lmRfaD>

prediction_Cumeo = model_Cumeo.make_future_dataframe(periods=730)
prediction_Cumeo = model_Cumeo.predict(prediction_Cumeo)
prediction_Cumeo[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

#|%%--%%| <ZBS1lmRfaD|jJUWw37h4u>

#Plot the prediction from 2024 to 2026
model_Cumeo.plot(prediction_Cumeo)
plt.title('Cumeo Population Prediction')
plt.show()

#|%%--%%| <jJUWw37h4u|G2W0j3BmkB>

model_Catnho.fit(data_prophet_Catnho)

#|%%--%%| <G2W0j3BmkB|DLDmLrkh7o>

prediction_Catnho = model_Catnho.make_future_dataframe(periods=730)
prediction_Catnho = model_Catnho.predict(prediction_Catnho)
prediction_Catnho[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

#|%%--%%| <DLDmLrkh7o|xx1RIG3ZtS>

#Plot the prediction from 2024 to 2026
model_Catnho.plot(prediction_Catnho)
plt.title('Catnho Population Prediction')
plt.show()

#|%%--%%| <xx1RIG3ZtS|Hmdng2avhH>

model_Yenui.fit(data_prophet_Yenui)

#|%%--%%| <Hmdng2avhH|CwTjoN3cvW>

prediction_Yenui = model_Yenui.make_future_dataframe(periods=730)
prediction_Yenui = model_Yenui.predict(prediction_Yenui)
prediction_Yenui[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

#|%%--%%| <CwTjoN3cvW|UXebYX1Hh7>

#Plot the prediction from 2024 to 2026
model_Yenui.plot(prediction_Yenui)
plt.title('Yenui Population Prediction')
plt.show()

#|%%--%%| <UXebYX1Hh7|BRb70VsK19>

#Save each model to a file
import pickle
pickle.dump(model_Chaomao, open('model_Chaomao.pkl', 'wb'))
pickle.dump(model_Chichchoe, open('model_Chichchoe.pkl', 'wb'))
pickle.dump(model_CuGay, open('model_CuGay.pkl', 'wb'))
pickle.dump(model_Cong, open('model_Cong.pkl', 'wb'))
pickle.dump(model_Cumeo, open('model_Cumeo.pkl', 'wb'))
pickle.dump(model_Catnho, open('model_Catnho.pkl', 'wb'))
pickle.dump(model_Yenui, open('model_Yenui.pkl', 'wb'))

