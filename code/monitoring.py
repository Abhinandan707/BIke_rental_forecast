import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import logging
import os

import pickle


def load_latest_model(model_dir='models'):
    # Get a list of model files in the directory
    model_files = [f for f in os.listdir(model_dir) if f.endswith(".pkl")]

    if not model_files:
        print("No model files found in the directory.")
        return None

    ## Find the latest model based on the timestamp in the filename
    latest_model_filename = max(model_files)
    latest_model_path = os.path.join(model_dir, latest_model_filename)

    with open(latest_model_path, 'rb') as model_file:
        model = pickle.load(model_file)

    return model


def get_rmsle(y_true, y_pre):    
    log_y_true = np.log1p(y_true)
    log_y_pred = np.log1p(y_pre)
    squared_log_diff = (log_y_true - log_y_pred) ** 2
    mean_squared_log_error = np.mean(squared_log_diff)
    rmsle = np.sqrt(mean_squared_log_error)

    return rmsle

def monitor_model_performance(X_test, y_test, log_file, model= None, message = None):
    ## get appropriate model
    if model is None:
        model = load_latest_model()
    
    ## Make predictions on the test-data
    if model is not None:
        y_pred = model.predict(X_test)
    else:
        print('No model available')
        return None, None, None

    ## Calculate the RMSE (Root Mean Squared Error)
    rmse = mean_squared_error(y_test, y_pred, squared=False)

    ## Calculate RMLSE (Root Mean Log Squared Error)
    rmlse = get_rmsle(y_test, y_pred)

    ## Calculate the percentage error
    percentage_error = np.abs((y_test - y_pred) / y_test) * 100

    ## Calculate the mean percentage error
    mean_percentage_error = np.mean(percentage_error)

    ## Log performance metrics to a log file
    log_performance(log_file, 'rmse', rmse, message)
    log_performance(log_file, 'rmlse', rmlse, message)
    log_performance(log_file, 'mean_percentage_error', mean_percentage_error, message)
    
    return rmse, rmlse, mean_percentage_error 

def log_performance(log_file, param_name, param_value, message=None):
    # Configure logging
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')

    # Log the parameter name and value
    if param_value is not None:
        logging.info(f"{param_name}: {param_value}")

    # Log the message
    if message is not None:
        logging.info(message)

    # Log the NumPy array to a file
    if isinstance(param_value, np.ndarray):
        with open(f"{param_name}_array.txt", "w") as array_file:
            np.savetxt(array_file, param_value)
