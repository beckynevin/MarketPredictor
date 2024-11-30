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

# TODO: 2. Fit a simple RandomForestRegressor model and use it to predict "day_ahead_price" for
#  all times in `predict_timestamps`

# TODO: 3. Plot results and print MAE
