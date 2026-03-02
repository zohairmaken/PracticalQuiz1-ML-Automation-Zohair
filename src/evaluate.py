# Author: Zohaib | PracticalQuiz1 — Model Evaluation Script

import pandas as pd
import joblib
import os
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

PROC_DIR     = "data/processed"
MODEL_PATH   = "models/logistic_model.pkl"
RESULTS_DIR  = "results"
METRICS_PATH = f"{RESULTS_DIR}/metrics.txt"

def evaluate():
    # Load test data
    X_test = pd.read_csv(f"{PROC_DIR}/X_test.csv")
    y_test = pd.read_csv(f"{PROC_DIR}/y_test.csv").values.ravel()

    # Load trained model
    model = joblib.load(MODEL_PATH)
    print("[EVALUATE] Model loaded successfully.")

    # Predict
    y_pred   = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report   = classification_report(y_test, y_pred)
    cm       = confusion_matrix(y_test, y_pred)

    # Save metrics
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(METRICS_PATH, "w") as f:
        f.write("=== Model Evaluation Metrics ===\n\n")
        f.write(f"Accuracy: {accuracy * 100:.2f}%\n\n")
        f.write("Classification Report:\n")
        f.write(report + "\n")
        f.write("Confusion Matrix:\n")
        f.write(str(cm) + "\n")

    print(f"[EVALUATE] Accuracy:        {accuracy * 100:.2f}%")
    print(f"[EVALUATE] Metrics saved -> {METRICS_PATH}")
    print("[EVALUATE] DONE: Evaluation complete.")

if __name__ == "__main__":
    evaluate()
