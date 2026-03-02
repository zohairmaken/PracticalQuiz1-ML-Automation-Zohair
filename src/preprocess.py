# Author: Zohaib | PracticalQuiz1 — Data Preprocessing Script

import pandas as pd
from sklearn.model_selection import train_test_split
import os

RAW_PATH   = "data/raw/dataset.csv"
PROC_DIR   = "data/processed"

def preprocess():
    # Load raw dataset
    df = pd.read_csv(RAW_PATH)
    print(f"[PREPROCESS] Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")

    # Fill missing values with column mean (numerical columns only)
    num_cols = df.select_dtypes(include="number").columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].mean())
    print(f"[PREPROCESS] Missing values handled (mean imputation on numerical columns)")

    # Separate features and target
    X = df.drop(columns=["passed"])
    y = df["passed"]

    # 80/20 train-test split with fixed seed for reproducibility
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    os.makedirs(PROC_DIR, exist_ok=True)

    X_train.to_csv(f"{PROC_DIR}/X_train.csv", index=False)
    X_test.to_csv(f"{PROC_DIR}/X_test.csv",  index=False)
    y_train.to_csv(f"{PROC_DIR}/y_train.csv", index=False)
    y_test.to_csv(f"{PROC_DIR}/y_test.csv",  index=False)

    print(f"[PREPROCESS] Train samples: {len(X_train)} | Test samples: {len(X_test)}")
    print("[PREPROCESS] Processed files saved to data/processed/")
    print("[PREPROCESS] DONE: Preprocessing complete.")

if __name__ == "__main__":
    preprocess()
