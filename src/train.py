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

    print(f"[TRAIN] Training samples: {len(X_train)} | Test samples: {len(X_test)}")

    # Train Logistic Regression
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    # Training accuracy
    train_accuracy = accuracy_score(y_train, model.predict(X_train))
    print(f"[TRAIN] Training Accuracy: {train_accuracy * 100:.2f}%")

    # Test accuracy (quick check during training)
    test_accuracy = accuracy_score(y_test, model.predict(X_test))
    print(f"[TRAIN] Test Accuracy:     {test_accuracy * 100:.2f}%")

    # Save model
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"[TRAIN] Model saved -> {MODEL_PATH}")
    print("[TRAIN] DONE: Training complete.")

if __name__ == "__main__":
    train()
