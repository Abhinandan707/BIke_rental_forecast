import lightgbm as lgb
import numpy as np
import argparse
from datetime import datetime
from data_pipeline import preprocess_data, split_data, load_new_data, append_all_data
from update_model import update_model_with_new_data, save_updated_model, custom_asymmetric_train, custom_rmsle_valid
from monitoring import monitor_model_performance, load_latest_model, log_performance
from parameter_search import hyperparameter_optimization, read_optimal_params
from predict_future import fetch_weather_data


def parse_arguments():
    parser = argparse.ArgumentParser(description="Model Training and Monitoring")
    parser.add_argument("--base_data_path", type=str, default=r"data/hour.csv", help="Path to the initial data file")
    parser.add_argument("--new_data_path", type=str, default= r'data\new_data_logs' , help="Path to the new data folder")
    parser.add_argument("--model_dir", type=str, default="models", help="Directory where models are stored")
    parser.add_argument("--task_type", type=int, default=1, help="flag to decide- train the model with data split(0), just get inference on new data (1), train model on all data & save (2), update model on new_data_only(3) , get future rental count (4), update_optimal_params (5)")
    parser.add_argument("--dir_to_save_params", type=str, default= r"data\model_params", help="Place to save optimal LightGBM params")
    parser.add_argument("--dir_to_save_future_predn", type=str, default= r"data\future_data", help="Place to save future prediction")

    parser.add_argument("--start_time", type=lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S"), 
                        default="2013-01-01 00:00:00", help="Start time in format 'YYYY-MM-DD HH:MM:SS'")
    parser.add_argument("--end_time", type=lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S"), 
                        default="2013-01-31 23:00:00", help="End time in format 'YYYY-MM-DD HH:MM:SS'")
    
    # Add parameters for best_params
    parser.add_argument("--n_estimators", type=int, default=400, help="Number of iterations")
    parser.add_argument("--num_leaves", type=int, default=11, help="Number of leaves")
    parser.add_argument("--learning_rate", type=float, default=0.03, help="Learning rate")
    parser.add_argument("--feature_fraction", type=float, default=0.975924788696534, help="Feature fraction")
    parser.add_argument("--max_depth", type=int, default=11, help="Max depth")
    parser.add_argument("--min_child_weight", type=float, default=6.161164717805335, help="Min child weight")
    parser.add_argument("--bagging_fraction", type=float, default=0.9431662642717817, help="Bagging fraction")
    parser.add_argument("--bagging_freq", type=int, default=1, help="Bagging frequency")
    parser.add_argument("--lambda_l1", type=float, default=0.39565320258596776, help="L1 regularization lambda")

    return parser.parse_args()


def main():

    args = parse_arguments()
    best_params = read_optimal_params(args.dir_to_save_params, file_name = 'best_hyperparameters.txt')


    if args.task_type == 0:
        print('Training new model with data splits')
        
        ## Step 1: Load all data and latest model
        input_data = append_all_data(args.base_data_path, args.new_data_path)
        X, y = preprocess_data(input_data)
        X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2)

        current_model = lgb.LGBMRegressor(**best_params)
        current_model.set_params(**{'objective': custom_asymmetric_train}, metrics=["rmse", "mae"])
        current_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], eval_metric=custom_rmsle_valid)

        # Step 2: log model perfimance 
        monitor_model_performance(X_test, y_test, "logs/model_performance.log", current_model, message = 'Trained model with data splits')
    
    elif args.task_type == 1:
        print('Getting inference on the existing model')

        latest_data = load_new_data(args.new_data_path)
        X_latest, y_latest = preprocess_data(latest_data)
        latest_model = load_latest_model(model_dir=args.model_dir)
        monitor_model_performance(X_latest, y_latest, "logs/model_performance.log", latest_model, message = 'Only inference on latest model')

    elif args.task_type == 2:
        print('Training model on all available data & saving model')

        input_data = append_all_data(args.base_data_path, args.new_data_path)
        X, y = preprocess_data(input_data) 

        model2save = lgb.LGBMRegressor(**best_params)
        model2save.set_params(**{'objective': custom_asymmetric_train}, metrics=["rmse", "mae"])
        model2save.fit(X, y)

        save_updated_model(model2save, model_dir='models')

    elif args.task_type == 3:
        print('Updating model by just training on new data')

        latest_data = load_new_data(args.new_data_path)
        updated_model2save = update_model_with_new_data(latest_data)
        save_updated_model(updated_model2save, model_dir='models')

    elif args.task_type == 4:
        print('predicting future bike rentals')
        
        X_test_ = fetch_weather_data(args.start_time, args.end_time, args.base_data_path)
        latest_model = load_latest_model(model_dir=args.model_dir)
        y_pred_ = latest_model.predict(X_test_)
        current_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        full_path = args.dir_to_save_future_predn + '/' + 'predn_' + current_timestamp + 'npy' 
        np.save(full_path, y_pred_)

    elif args.task_type == 5:
        print('Only updating LightGBM parameters')

        hyperparameter_optimization(args.base_data_path, args.new_data_path, args.dir_to_save_params)

if __name__ == "__main__":
    main()