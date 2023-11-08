from hyperopt import fmin, tpe, hp, Trials
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import train_test_split
import numpy as np
import lightgbm as lgb
import os

from data_pipeline import preprocess_data, split_data, append_all_data
from update_model import custom_asymmetric_train, custom_rmsle_valid


def read_optimal_params(output_directory, file_name):
    file_path = os.path.join(output_directory, file_name)
    hyperparameters = {}

    try:
        with open(file_path, 'r') as file:
            for line in file:
                key, value = line.strip().split(': ')
                if key in ["n_estimators", "num_leaves", "max_depth"]:
                    hyperparameters[key] = int(value)
                else:
                    hyperparameters[key] = float(value)

    except FileNotFoundError:
        print(f"File '{file_name}' not found in directory '{output_directory}'")
    
    return hyperparameters

## Define the objective function
def objective(params, X_train, y_train, X_test, y_test):
    
    ## Cast params to an integer
    params['n_estimators'] = int(params['n_estimators'])
    params['num_leaves'] = int(params['num_leaves'])
    params['max_depth'] = int(params['max_depth'])
    
    ## Create and configure the LightGBM model
    gbm = lgb.LGBMRegressor(
        n_estimators=params['n_estimators'],
        num_leaves=params['num_leaves'],
        learning_rate=params['learning_rate'],
        feature_fraction=params['feature_fraction'],
        max_depth=params['max_depth'],
        bagging_fraction=params['bagging_fraction'],
        lambda_l1=params['lambda_l1'],
        objective=custom_asymmetric_train,
        metrics=["rmse", "mae"],
        early_stopping_rounds=40,
        verbose=-1
    )

    gbm.set_params(**{'objective': custom_asymmetric_train}, metrics = ["rmse", 'mae'], early_stopping_rounds = 40, verbose =-1)
    
    ## Fit the model
    gbm.fit(X_train, y_train, eval_set=[(X_test, y_test)], eval_metric=custom_rmsle_valid)

    ## Make predictions on the validation data
    y_pred = gbm.predict(X_test)

    ## Calculate the custom RMSLE on the validation data
    log_y_true = np.log1p(y_test)
    log_y_pred = np.log1p(y_pred)
    squared_log_diff = (log_y_true - log_y_pred) ** 2
    mean_squared_log_error = np.mean(squared_log_diff)
    rmsle = np.sqrt(mean_squared_log_error)

    return rmsle


def hyperparameter_optimization(base_df_dir, new_df_dir, output_directory):
    input_data = append_all_data(base_df_dir, new_df_dir)
    X, y = preprocess_data(input_data)
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2)

    space = {
        'n_estimators': hp.quniform('n_estimators', 100, 500, 1),
        'num_leaves': hp.quniform('num_leaves', 7, 40, 1),
        'learning_rate': hp.uniform('learning_rate', 0.03, 0.1),
        'feature_fraction': hp.uniform('feature_fraction', 0.7, 1.0),
        'max_depth': hp.quniform('max_depth', 3, 15, 1),
        'bagging_fraction': hp.uniform('bagging_fraction', 0.7, 1.0),
        'lambda_l1': hp.uniform('lambda_l1', 0.0, 1.0)
    }

    trials = Trials()
    best = fmin(fn=lambda params: objective(params, X_train, y_train, X_test, y_test),
                space=space,
                algo=tpe.suggest,
                max_evals=1000,
                trials=trials)

    ## Get the best hyperparameters
    hyperopt_best_params = {
        'n_estimators': int(best['n_estimators']),
        'num_leaves': int(best['num_leaves']),
        'learning_rate': best['learning_rate'],
        'feature_fraction': best['feature_fraction'],
        'max_depth': int(best['max_depth']),
        'bagging_fraction': best['bagging_fraction'],
        'lambda_l1': best['lambda_l1']
    }

    ## Save the best hyperparameters as a text file in the output directory
    output_path = os.path.join(output_directory, 'best_hyperparameters.txt')
    with open(output_path, 'w') as file:
        for key, value in hyperopt_best_params.items():
            file.write(f'{key}: {value}\n')