"""
Ascend-Analytics DataScience Take-Home Assignment

SETUP
1. Create a python 3.11 environment with packages
  * pandas
  * scikit-learn
  * plotly
"""
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import numpy as np
import matplotlib.pyplot as plt

pd.options.plotting.backend = "plotly"


class Cols:
    timestamp = "timestamp"

    # LABEL
    # Value known up to midnight of current day
    day_ahead_price = "day_ahead_price"  # Hourly.

    # OTHER DATA
    # Values known up to runtime.
    real_time_price = "real_time_price"  # Spot price for current location. Updated every 15 minutes
    load_actual = "load_actual"  # Hourly total electricity load for all of California
    wind_actual = "wind_actual"  # Hourly total wind production for all of california

    # CAISO publishes predictions for the total load and total wind production for all of CA. We
    # fetch these once a day at ~6am and only save the predictions for the following day to our
    # raw data. As such, what you see in these columns are the "predictions for this timestamp
    # made yesterday at 6am". These columns do not need to be lagged.
    load_forecast = "load_forecast"  # Hourly
    wind_forecast = "wind_forecast"  # Hourly


def set_timestamp_index(pd_obj):
    pd_obj[Cols.timestamp] = pd.to_datetime(pd_obj[Cols.timestamp])
    pd_obj = pd_obj.set_index(Cols.timestamp)
    pd_obj.index = pd.DatetimeIndex(pd_obj.index)
    return pd_obj


raw_data = pd.read_csv("raw_data.csv")
raw_data = set_timestamp_index(raw_data)
y_true = pd.read_csv("y_true.csv")
y_true = set_timestamp_index(y_true)

runtime = pd.Timestamp("8/19/2023 07:00")
predict_timestamps = pd.date_range(pd.Timestamp("8/20/2023"), periods=24, freq="H")

# TODO: 1. Use raw_data to create a DataFrame of features. There should be 6 features: 1 label lag and 1 each for the
#  other 5 columns (lagged according to availability at runtime). DO NOT spend time feature engineering additional
#  features.

# Okay, here's my interpretation of the problem - 
# We want to predict day ahead price based on info in that
# row. My interpretation of "Day Ahead Price" is that 
# it was predicted in the past for that moment.
# For instance, we're trying to predict it for 8-20
# To dos:
# We need to do two types of lag (move back in time) 
# 1) the day ahead price prediction is a series dependent
# variable, so I need to lag it by one time interval so it
# can use its past value as a predictor
# 2) the wind actual needs to be moved forward in time by
# 24 hours because the wind from the previous day should be used
# to predict the 8-20 values, NOT SURE ABOUT THIS
# 3) load actual


# I want to look at the day ahead price
# looking only at the entries that are numbers (on the hour)
cutoff_timestamp = '2023-08-19 07:00:00'
filtered_data = raw_data[raw_data.index > cutoff_timestamp]
print(filtered_data['day_ahead_price'][~filtered_data['day_ahead_price'].isna()])
# ^ last entry is 23h on 8-19

# now there are other things known only up until runtime
cutoff_timestamp = '2023-08-19 00:00:00'
filtered_data = raw_data[raw_data.index > cutoff_timestamp]

# look also at the realtime price
# which is also known up until 11pm of the current day
print(filtered_data['real_time_price'][~filtered_data['real_time_price'].isna()])
# ^ last entry was 6:45am 8-19

# load actual
print(filtered_data['load_actual'][~filtered_data['load_actual'].isna()])
# ^ last entry was 6am 8-19

# wind actual
print(filtered_data['wind_actual'][~filtered_data['wind_actual'].isna()])
# ^ last entry was 6am 8-19

# wind forecast
print(filtered_data['wind_forecast'])
# ^ last entry is 23h on 8-20

# dropping all of the 15 minute forecast
# because most of the predictors are not on the 15 minute anyway
raw_data = raw_data[raw_data.index.minute == 0]



# lag the real_time_price, load_actual, and wind_actual columns by 24 hours
# I was reading this blog to learn about serial dependency:
# https://www.kaggle.com/code/ryanholbrook/time-series-as-features
lagged_df = pd.DataFrame({
    'day_ahead_price': raw_data['day_ahead_price'],
    'lagged_day_ahead_price': raw_data['day_ahead_price'].shift(24),
    'lagged_load_actual': raw_data['load_actual'].shift(18),
    'lagged_wind_actual': raw_data['wind_actual'].shift(18),
    'lagged_real_time_price': raw_data['real_time_price'].shift(18),
    'unlagged_load_forecast': raw_data['load_forecast'],
    'unlagged_wind_forecast': raw_data['wind_forecast']
})
print(lagged_df)
lagged_df = lagged_df.fillna(0)


# there are a lot of Nans here - looks like the 15 minute intervals
# aren't included in the load or wind or DAP, so I'll drop them
#lagged_df = lagged_df.dropna()




# overplot the lagged to make sure I did this right
plt.clf()
plt.figure(figsize=(10, 6))

plt.plot(lagged_df.index,
         lagged_df['lagged_day_ahead_price'],
         label="Lagged Day Ahead Price")
plt.plot(lagged_df.index,
         lagged_df['day_ahead_price'],
         label="Day Ahead Price")
plt.title("Day Ahead Price Over Time")
plt.xlabel("Date")
plt.ylabel("Day Ahead Price")
plt.legend()
plt.grid(True)
plt.show()




# TODO: 2. Fit a simple RandomForestRegressor model and use it to predict "day_ahead_price" for
#  all times in `predict_timestamps`

features = ['lagged_load_actual', 'lagged_wind_actual', 'lagged_real_time_price',
            'unlagged_load_forecast', 'unlagged_wind_forecast', 'lagged_day_ahead_price']
target = 'day_ahead_price'  # This is the day-ahead price for tomorrow

# Split data into features (X) and target (y)
# SHOULD BE DROPPING NANS
X = lagged_df[lagged_df.index < "2023-08-19 23:45:00"][features]
y = lagged_df[lagged_df.index < "2023-08-19 23:45:00"][target]

print('X train', X)
print('y train', y)

# Step 3: Initialize and train the RandomForestRegressor
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X, y)

print('X test', lagged_df[lagged_df.index > "2023-08-19 23:45:00"][features])
y_pred = rf.predict(lagged_df[lagged_df.index > "2023-08-19 23:45:00"][features])
print('predicted', y_pred)
print('y_test', y_true['day_ahead_price'])


# TODO: 3. Plot results and print MAE
print(np.shape(y_pred), np.shape(y_true))
plt.clf()
plt.scatter(y_pred, y_true['day_ahead_price'])
plt.xlabel('predicted')
plt.ylabel('true')
plt.show()

plt.clf()
plt.figure(figsize=(10, 6))

plt.plot(lagged_df.index,
         lagged_df['day_ahead_price'],
         label="Day Ahead Price")

plt.plot(y_true.index,
         y_true['day_ahead_price'],
         label="True Day Ahead Price")
plt.plot(y_true.index,
         y_pred,
         label="Predicted Day Ahead Price")

plt.title("Day Ahead Price Over Time")
plt.xlabel("Date")
plt.ylabel("Day Ahead Price")
plt.legend()
plt.grid(True)
plt.show()

mae = mean_absolute_error(y_true, y_pred)
print(mae)

# Things that could be improved:
# 1) CV split (Timeseriessplit from sklearn)
# 2) Getting the lags right / understanding what
# info should be available at prediction time
# 3) Additional lag features for cyclicity / seasons
# 4) Add 'isfuture' column to more easily separate out
# the future prediction columns


