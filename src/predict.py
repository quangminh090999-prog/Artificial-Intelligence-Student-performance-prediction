
"""Functions for predicting a student's exam score."""

import joblib
import pandas as pd

from src.config import MODEL_PATH


def predict_one(student_data):
    """Predict the exam score for one student."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model not found. Run python -m src.train first."
        )

    if isinstance(student_data, dict):
        student_data = pd.DataFrame([student_data])
    elif not isinstance(student_data, pd.DataFrame):
        raise TypeError("student_data must be a dictionary or DataFrame.")

    model = joblib.load(MODEL_PATH)
    prediction = model.predict(student_data)

    return float(prediction[0])
