import pandas as pd
import numpy as np

## datetime
import calendar
from datetime import datetime

from datetime import datetime
from meteostat import Point, Hourly
import holidays

from data_pipeline import load_data, restructure_df 

def fetch_weather_data(start_time, end_time, data_path):
    # Define latitude and longitude for the location
    latitude = 38.9072
    longitude = -77.0379

    # Create a Point object for the location
    point = Point(latitude, longitude)

    # Fetch hourly weather data
    data = Hourly(point, start_time, end_time).fetch()

    weather_df = data.reset_index()

    # Rename columns
    weather_df.rename(columns={
        'time': 'datetime',
        'rhum': 'humidity',
        'wspd': 'windspeed',
    }, inplace=True)

    # Select relevant columns
    weather_df = weather_df[['datetime', 'temp', 'humidity', 'windspeed']]

    # Add new datetime columns
    weather_df['season'] = weather_df['datetime'].dt.month.apply(lambda x: (x % 12 + 3) // 3)
    weather_df['DayOfWeek'] = weather_df['datetime'].dt.dayofweek
    weather_df['month'] = weather_df['datetime'].dt.month
    weather_df['hour'] = weather_df['datetime'].dt.hour

    # Add 'holiday', 'weekday', and 'workingday' columns
    us_holidays = holidays.US(years=[start_time.year])
    weather_df['holiday'] = weather_df['datetime'].dt.date.isin(us_holidays).astype(int)  # Convert boolean to integer (1 or 0)
    weather_df['weekday'] = weather_df['datetime'].dt.weekday
    weather_df['workingday'] = weather_df.apply(lambda row: 1 if row['weekday'] < 5 and row['holiday'] == 0 else 0, axis=1)

    # Load and format base hourly data
    input_df = load_data(data_path)
    base_df = restructure_df(input_df)

    # Combine 'datetime' and 'hour' columns to create a new datetime column
    base_df['datetime'] = pd.to_datetime(base_df['datetime']) + pd.to_timedelta(base_df['hour'], unit='h')

    # Format the new datetime column as a string
    base_df['datetime'] = base_df['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')

    # Convert the 'datetime' column back to datetime objects
    base_df['datetime'] = pd.to_datetime(base_df['datetime'])

    # Add 1 year to the 'datetime' column
    base_df['datetime'] = base_df['datetime'] + pd.DateOffset(years=1)

    # Merge (inner join) the DataFrames based on 'datetime'
    X_test = pd.merge(weather_df, base_df[['datetime', 'weather']], on='datetime', how='inner')

    ## drop 'datetime' column 
    X_test.drop(columns = 'datetime', inplace = True)
    
    return X_test

