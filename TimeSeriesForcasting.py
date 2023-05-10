#!/usr/bin/env python
# coding: utf-8

# # Matthew Werner
# # Canisius College: DAT 512
# # Project 3: Times Series Forcasting<br>
# 
# ---
# 
# 
# 
# ---
# 
# 

# In[1]:


import pandas as pd
import numpy as np
import requests
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from prophet import Prophet
import matplotlib.pyplot as plt
 
get_ipython().run_line_magic('matplotlib', 'inline')
 
plt.rcParams['figure.figsize']=(20,10)
plt.style.use('ggplot')


# In[2]:


# Making sure the connection works to 311 dataset
# Checking to see if the data has less than 1000 rows
uri = 'https://data.buffalony.gov/resource/whkc-e5vr.json'
r = requests.get(uri)
print('Status code ',r.status_code)
print('Number of rows returned ',len(r.json()))
print('Endoced URI with params ',r.url)


# In[3]:


data311=pd.DataFrame(r.json())
print(data311.shape)
data311.head()


# In[4]:


# Pulling in 311 Service Requests API from Open Data Buffalo
# Setting limit parameter in order to extract all of the rows
# Date parameters set to limit data to end on 1/1/2023

params_dict = {
    '$where':'date_extract_y(open_date) >= 2018 and date_extract_y(open_date) < 2023',
    '$limit':1000000
}

uri = 'https://data.buffalony.gov/resource/whkc-e5vr.json'
r = requests.get(uri, params=params_dict)
print('Status code ',r.status_code)
print('Number of rows returned ',len(r.json()))
print('Endoced URI with params ',r.url)


# In[5]:


# Creating a Pandas dataframe for 311 data
data311=pd.DataFrame(r.json())
print(data311.shape)
data311.head()


# ## Data Cleaning and Preparation

# ### Filtering useful columns into new dataframe

# In[6]:


# Finding all column names
data311.columns


# In[7]:


# Selcting valuable columns
data311 = pd.DataFrame(data311[['case_reference', 'open_date', 'status', 'type', 'zip_code', 'reason']])
data311.head()


# In[8]:


# Dataframe shape
data311.shape


# In[9]:


# remove rows where zipcode = unknown
data311 = data311[data311['zip_code'] != 'UNKNOWN']
data311.head()


# In[10]:


# Show if there was a change in the number of rows in the filtered 311 dataset
# Number of rows decreased from original dataframe
data311.shape


# In[11]:


data311.info()


# In[12]:


# Converting to datetime and extracting only year, month, and day.
data311['open_date'] = pd.to_datetime(data311['open_date']).dt.strftime('%Y-%m-%d')


# In[13]:


# Checking to make sure it worked.
data311['open_date'].head()


# In[14]:


# Building a new dataframe with the days that a case was opened
# Counting the number of cases opened on this day
# Preparing for prophit; Rename count column as "y"
final311 = data311.groupby('open_date')['open_date'].count().reset_index(name="y") 


# In[15]:


# Rename date column as "ds"
final311 = final311.rename(columns={'open_date':'ds'})


# In[16]:


final311.head()


# In[17]:


final311.set_index('ds').y.plot();


# # Initial Forecast

# In[18]:


# Instantiate Model
model = Prophet()

# Fit Model
model.fit(final311)


# In[19]:


# Create future data frame
future = model.make_future_dataframe(periods= 365, freq = 'd')
future.tail()


# In[20]:


# To forecast this future data, we need to run it through Prophet's model.
# Add predictions to the forecast dataframe
forecast = model.predict(future)


# In[21]:


# The resulting forecast dataframe contains quite a bit of data, but we really only care about a few columns.
# First, let's look at the full dataframe:
forecast.tail().T


# In[22]:


# Selecting valuable columns.
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()


# In[23]:


# Plot the forecast
model.plot(forecast);


# ## Evaluating Initial Forecast

# In[24]:


#To do this, we have to get the y-hat and original y's from the data
metric_df = pd.concat([forecast[['ds','yhat']],final311['y']], axis=1)
metric_df.head()


# In[25]:


# The tail has NaN values, because they're predictions - there was no real Y. Let's drop those for model evaluation.
metric_df.dropna(inplace = True)


# In[26]:


# check the tail, because we added 12 months of forecast.
metric_df.tail()


# In[27]:


#Let's take a look at the numbers - from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
print("R-squared: ", r2_score(metric_df['y'], metric_df['yhat']))
print("Mean Squared Error: ", mean_squared_error(metric_df['y'], metric_df['yhat']))
print("RMSE: ", np.sqrt(mean_squared_error(metric_df['y'], metric_df['yhat'])))


# # Forcasting with holidays

# In[28]:


from datetime import date
import holidays


# In[29]:


us_holidays = holidays.UnitedStates(years = [2018,2019,2020,2021,2022])

holidays = pd.DataFrame({
  'holiday': us_holidays.values(),
  'ds': us_holidays.keys(),
  'lower_window': 0,
  'upper_window': 0,
})
holidays


# In[30]:


#Now let's set up prophet to model our data using holidays - Instantiate and fit the model
model = Prophet(holidays=holidays,
                weekly_seasonality=False)

# model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
model.fit(final311)


# In[31]:


#We've instantiated the model, so now we need to build our future dates to forecast into!
future = model.make_future_dataframe(periods=365, freq = 'd')
future.tail()

#... and then run our future data through prophet's model
forecast = model.predict(future)

forecast.head().T


# In[32]:


future['ds'] = future['ds'].to_numpy().astype('datetime64[M]')


# In[33]:


#while our new df contains a bit of data, we only care about a few features...
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()


# ## Visualizing with holidays

# In[34]:


# use Prophet's .plot() method to visualize your timeseries.
model.plot(forecast);


# In[35]:


metric_df = pd.concat([forecast[['ds','yhat']],final311['y']], axis=1)
metric_df.head()


# In[36]:


# The tail has NaN values, because they're predictions - there was no real Y. Let's drop those for model evaluation.
metric_df.dropna(inplace = True)


# In[37]:


print("R-squared: ", r2_score(metric_df['y'], metric_df['yhat']))
print("Mean Squared Error: ", mean_squared_error(metric_df['y'], metric_df['yhat']))
print("RMSE: ", np.sqrt(mean_squared_error(metric_df['y'], metric_df['yhat'])))


# In[38]:


# View the components
model.plot_components(forecast);


# # AMIRA Forcasting

# In[39]:


get_ipython().system('pip install pmdarima')


# In[40]:


get_ipython().system('pip install statsmodels')


# In[41]:


import statsmodels.api as sm
import statsmodels.formula.api as smf


# In[42]:


from statsmodels.tsa.stattools import adfuller


# In[43]:


def ad_test(dataset):
    dftest = adfuller(dataset, autolag = 'AIC')
    print('1. ADF : ', dftest[0])
    print('2. P-Value : ', dftest[1])
    print('3. Num Of Lags : ', dftest[2])
    print('4. Num Of Observations Used For ADF Regression and Critical Values Calculation : ', dftest[3])
    print('5. Critical Values : ')
    for key, val in dftest[4].items():
        print('\t', key, ':',val)


# In[44]:


ad_test(final311['y'])


# In[45]:


from pmdarima import auto_arima
# Ignore harmless warnings
import warnings
warnings.filterwarnings('ignore')


# In[46]:


stepwise_fit = auto_arima(final311['y'],
                          trace=True,
                          supress_warnings=True)


# In[47]:


stepwise_fit.summary() 


# In[48]:


final311 = final311.set_index('ds')


# In[49]:


final311.head()


# In[50]:


final311.index = pd.to_datetime(final311.index)


# In[51]:


final311.info()


# In[52]:


split_date = '2022-01-01'
final311_train = final311.loc[final311.index <= split_date].copy()
final311_test = final311.loc[final311.index > split_date].copy()


# In[53]:


# The storm in december of 2022 may skew the test data
fig, ax = plt.subplots(figsize=(20,10))
final311_train.plot(ax=ax, label='Training Data', title='Train/Test Data Split')
final311_test.plot(ax=ax, label='Testing Data')
ax.axvline(split_date, color='black', ls='--')
ax.legend(['Training Data', 'Test Data'])
plt.show()


# In[54]:


import statsmodels.api as sm
from statsmodels.tsa.arima_model import ARIMA


# In[55]:


model=sm.tsa.ARIMA(final311['y'], order=(5,0,5))
model=model.fit()
model.summary()


# In[56]:


start=len(final311_train)
end=len(final311_train)+len(final311_test)-1
pred=model.predict(start=start, end=end, typ='levels')
pred.index=final311.index[start:end+1]
print(pred)


# In[57]:


pred.plot(legend=True)
final311_test['y'].plot(legend=True)


# In[58]:


final311['y'].mean()


# In[59]:


from sklearn.metrics import mean_squared_error
from math import sqrt
rmse=sqrt(mean_squared_error(pred, final311_test['y']))
rmse


# In[60]:


#Fit model on all df5
model2 = sm.tsa.ARIMA(final311['y'], order=(5,0,5))
model2=model2.fit()
final311.tail() # predict into the future after training


# In[61]:


index_future_dates=pd.date_range(start='2023-01-01', end='2023-12-31')
pred=model2.predict(start=len(final311), end=len(final311)+(364), type='levels').rename('ARIMA Predictions')
pred.index=index_future_dates
print(pred)


# ## 2023 Arima Predictions

# In[62]:


pred.plot(figsize=(12,5),legend=True)

