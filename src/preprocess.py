# Author: Zohaib | PracticalQuiz1 — Data Preprocessing Script

import pandas as pd
from sklearn.model_selection import train_test_split
import os

RAW_PATH   = "data/raw/student_scores.csv"
PROC_DIR   = "data/processed"

def preprocess():
    # Load raw dataset
    df = pd.read_csv(RAW_PATH)

    # Drop rows with missing values
    df.dropna(inplace=True)

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

    print("[PREPROCESS] Dataset loaded, cleaned, and split successfully.")
    print(f"[PREPROCESS] Train samples: {len(X_train)} | Test samples: {len(X_test)}")
    print("[PREPROCESS] Processed files saved to data/processed/")

if __name__ == "__main__":
    preprocess()
