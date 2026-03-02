# Author: Zohaib | PracticalQuiz1 — Model Training Script

import pandas as pd
import joblib
import os
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

PROC_DIR   = "data/processed"
MODEL_DIR  = "models"
MODEL_PATH = f"{MODEL_DIR}/logistic_model.pkl"

def train():
    # Load preprocessed data
    X_train = pd.read_csv(f"{PROC_DIR}/X_train.csv")
    X_test  = pd.read_csv(f"{PROC_DIR}/X_test.csv")
    y_train = pd.read_csv(f"{PROC_DIR}/y_train.csv").values.ravel()
    y_test  = pd.read_csv(f"{PROC_DIR}/y_test.csv").values.ravel()

    # Train Logistic Regression (feature-ml: tuned hyperparameters)
    model = LogisticRegression(max_iter=1000, random_state=42, C=1.0, solver='lbfgs')
    model.fit(X_train, y_train)

    # Evaluate
    y_pred   = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"[TRAIN] ✔ Logistic Regression Test Accuracy: {accuracy * 100:.2f}%")

    # Persist model
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"[TRAIN] ✔ Model artifact persisted → {MODEL_PATH}")
    print("[TRAIN] Training pipeline complete.")

if __name__ == "__main__":
    train()
