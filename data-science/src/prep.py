# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
Prepares raw data and provides training and test datasets.
"""

import argparse
from pathlib import Path
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import mlflow

# Numeric columns to standardize
NUMERIC_COLS = ["Kilometers_Driven", "Mileage", "Engine", "Power", "Seats"]

def parse_args():
    '''Parse input arguments'''

    parser = argparse.ArgumentParser("prep")  # Create an ArgumentParser object
    parser.add_argument("--raw_data", type=str, help="Path to raw data")
    parser.add_argument("--train_data", type=str, help="Path to train dataset")
    parser.add_argument("--test_data", type=str, help="Path to test dataset")
    parser.add_argument("--test_train_ratio", type=float, default=0.2, help="Test-train ratio")
    args = parser.parse_args()

    return args

def main(args):
    '''Read, preprocess, split, and save datasets'''

    # Reading Data
    df = pd.read_csv(args.raw_data)

    # ------- WRITE YOUR CODE HERE -------

    # Step 1: Encode the categorical 'Segment' column using a mapping
    df['Segment'] = df['Segment'].map({'non-luxury segment': 0, 'luxury segment': 1})

    # Step 2: Split into train and test sets (split BEFORE scaling to avoid data leakage)
    train_df, test_df = train_test_split(
        df, test_size=args.test_train_ratio, random_state=42
    )

    # Standardize numeric feature columns (fit on train only, then transform test)
    scaler = StandardScaler()
    train_df[NUMERIC_COLS] = scaler.fit_transform(train_df[NUMERIC_COLS])
    test_df[NUMERIC_COLS] = scaler.transform(test_df[NUMERIC_COLS])

    # Step 3: Save train and test sets to their output directories
    os.makedirs(args.train_data, exist_ok=True)
    os.makedirs(args.test_data, exist_ok=True)
    train_df.to_csv(os.path.join(args.train_data, "data.csv"), index=False)
    test_df.to_csv(os.path.join(args.test_data, "data.csv"), index=False)

    # Step 4: Log row counts as metrics
    mlflow.log_metric("train_size", len(train_df))
    mlflow.log_metric("test_size", len(test_df))


if __name__ == "__main__":
    mlflow.start_run()

    # Parse Arguments
    args = parse_args()

    lines = [
        f"Raw data path: {args.raw_data}",
        f"Train dataset output path: {args.train_data}",
        f"Test dataset path: {args.test_data}",
        f"Test-train ratio: {args.test_train_ratio}",
    ]

    for line in lines:
        print(line)
    
    main(args)

    mlflow.end_run()