import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

## datetime
import calendar
from datetime import datetime


## Load historical training data
def load_data(data_path = r"data/hour.csv"):
    
    bike_sharing_data = pd.read_csv(data_path)
    return bike_sharing_data

def restructure_df(input_df):
    ## rename for better interpretability
    input_df.rename(columns={'dteday':'datetime',
                        'weathersit':'weather',
                        'mnth':'month',
                        'hr':'hour',
                        'yr':'year',
                        'hum':'humidity',
                        'cnt':'net_cnt'},inplace=True)

    input_df['datetime'] = pd.to_datetime(input_df['datetime'])

    ## Extract the weekdays and add 1 to convert them to the format 1-7
    input_df['DayOfWeek_num'] = input_df['datetime'].dt.dayofweek + 1

    ## Drop null values
    input_df.dropna(inplace=True)

    return input_df


def load_new_data(data_dir = r'data\new_data_logs'): 
    ## This function considers that all new data logs also have a timestamp at the end of their names
    
    ## Get a list of data files in the directory
    data_files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]

    if not data_files:
        print("No data files found in the directory.")
        return None

    ## find the latest data file based on the timestamp in the filename
    latest_data_filename = max(data_files)
    latest_data_path = os.path.join(data_dir, latest_data_filename)

    new_data = pd.read_csv(latest_data_path, index_col=0)

    return new_data

def append_all_data(base_df_dir = r"data/hour.csv" , new_df_dir = r'data\new_data_logs'):
    ## appends all historical data into single dataframe
    new_df = load_new_data(new_df_dir)
    base_df = load_data(base_df_dir)
    
    ## restructure data
    new_df = restructure_df(new_df)
    base_df = restructure_df(base_df)

    if not isinstance(new_df, pd.DataFrame):
        updated_df = base_df  ## no new data
    else:
        updated_df = pd.concat([base_df, new_df], ignore_index=True)

    return updated_df


## Data cleaning and feature engineering
def preprocess_data(bike_sharing_data):
    
    if not isinstance(bike_sharing_data, pd.DataFrame):
        print('No new data updation')
        return None, None
    
    hourly_cnt_df = bike_sharing_data
    
    hourly_cnt_df = restructure_df(hourly_cnt_df)

    X = hourly_cnt_df.drop(columns=['net_cnt', 'instant', 'datetime', 'holiday', 'atemp', 'casual', 'registered'])
    y = hourly_cnt_df['net_cnt']
    # print(X.columns)
    return X, y

def split_data(X, y, test_size = 0.2):
    ## Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, shuffle= False)
    return X_train, X_test, y_train, y_test
