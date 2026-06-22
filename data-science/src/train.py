# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
Trains ML model using training dataset and evaluates using test dataset. Saves trained model.
"""

import argparse
from pathlib import Path
import os
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import mlflow
import mlflow.sklearn


def select_first_file(path):
    '''Selects the first file in a folder, assuming there's only one file.'''
    files = os.listdir(path)
    return os.path.join(path, files[0])


def parse_args():
    '''Parse input arguments'''

    parser = argparse.ArgumentParser("train")

    # -------- WRITE YOUR CODE HERE --------

    # Step 1: Define arguments for train data, test data, model output, and hyperparameters
    parser.add_argument("--train_data", type=str, help="Path to train data")
    parser.add_argument("--test_data", type=str, help="Path to test data")
    parser.add_argument("--model_output", type=str, help="Path of output model")
    parser.add_argument('--n_estimators', type=int, default=100,
                        help='The number of trees in the forest')
    parser.add_argument('--max_depth', type=int, default=None,
                        help='The maximum depth of the trees. If None, nodes are expanded '
                             'until all leaves are pure or contain less than '
                             'min_samples_split samples.')

    args = parser.parse_args()

    return args


def main(args):
    '''Read train and test datasets, train model, evaluate model, save trained model'''

    # -------- WRITE YOUR CODE HERE --------

    # Step 2: Read the train and test datasets from the provided paths
    train_df = pd.read_csv(select_first_file(args.train_data))
    test_df = pd.read_csv(select_first_file(args.test_data))

    # Step 3: Split into features (X) and target (y); 'price' is the target column
    y_train = train_df["price"].values
    X_train = train_df.drop("price", axis=1).values

    y_test = test_df["price"].values
    X_test = test_df.drop("price", axis=1).values

    # Step 4: Initialize and train the RandomForest Regressor
    rf_model = RandomForestRegressor(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=42
    )
    rf_model = rf_model.fit(X_train, y_train)

    # Step 5: Log hyperparameters for tracking in MLflow
    mlflow.log_param("n_estimators", args.n_estimators)
    mlflow.log_param("max_depth", args.max_depth)

    # Step 6: Predict on the test set and compute mean squared error
    rf_predictions = rf_model.predict(X_test)
    mse = mean_squared_error(y_test, rf_predictions)
    print(f'Mean Squared Error of Random Forest Regressor on test set: {mse:.4f}')

    # Step 7: Log the MSE metric and save the trained model
    mlflow.log_metric("MSE", float(mse))
    mlflow.sklearn.save_model(rf_model, args.model_output)


if __name__ == "__main__":
    
    mlflow.start_run()

    # Parse Arguments
    args = parse_args()

    lines = [
        f"Train dataset input path: {args.train_data}",
        f"Test dataset input path: {args.test_data}",
        f"Model output path: {args.model_output}",
        f"Number of Estimators: {args.n_estimators}",
        f"Max Depth: {args.max_depth}"
    ]

    for line in lines:
        print(line)

    main(args)

    mlflow.end_run()