import pandas as pd
import numpy as np
import os
import lightgbm as lgb
from datetime import datetime
from data_pipeline import preprocess_data, load_new_data
from monitoring import load_latest_model
import pickle
import os


def custom_asymmetric_train(y_true, y_pred):
    log_a = np.log(y_true + 1)
    log_b = np.log(y_pred + 1)
    calc = (log_a - log_b) ** 2
    residual = np.sqrt(np.mean(calc))

    ## 1st grad of the loss
    grad = -2 * (log_a - log_b) / (y_pred + 1)

    ## 2nd grad of the loss
    hess = 2 / (y_pred + 1) ** 2

    return grad, hess

def custom_rmsle_valid(y_true, y_pred):
    log_y_true = np.log1p(y_true)  # Logarithm of true values
    log_y_pred = np.log1p(y_pred)  # Logarithm of predicted values
    squared_log_diff = (log_y_true - log_y_pred) ** 2
    mean_squared_log_error = np.mean(squared_log_diff)
    rmsle = np.sqrt(mean_squared_log_error)

    return "custom_rmsle_eval", rmsle, False


def update_model_with_new_data(new_data):
    ## get latest model 
    if not isinstance(new_data, pd.DataFrame):
        return None
    
    model = load_latest_model()

    ## reuse preprocessing function from data_pipeline.py
    X_new, y_new = preprocess_data(new_data)
    
    ## Update the model with new data
    model.set_params(min_sum_in_hessian=0, min_data_in_leaf=0) ## LightGBM has predefined params to avoid overfitting on small data, seting them to zero to avoid failed checks
    model.fit(X_new, y_new)
    
    return model

def save_updated_model(model, model_dir='models'):
    # Create a unique filename with a timestamp
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    model_filename = f"bike_sharing_{timestamp}.pkl"
    model_path = os.path.join(model_dir, model_filename)

    with open(model_path, 'wb') as model_file:
        pickle.dump(model, model_file)
