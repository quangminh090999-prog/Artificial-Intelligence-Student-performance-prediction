"""Kiểm thử chuẩn dữ liệu và hàm dự đoán cho giao diện."""

import numpy as np
import pandas as pd
import pytest
import joblib
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR

from src.predict import (
    INPUT_COLUMNS,
    load_model,
    predict_batch,
    predict_batch_with_errors,
    prepare_input,
)
from src.preprocessing import build_preprocessor


class ConstantModel:
    """Mô hình nhỏ dùng để kiểm thử giao diện suy luận."""

    def predict(self, frame):
        return np.full(len(frame), 75.5)


def sample_input() -> dict[str, object]:
    return {
        "Hours_Studied": 20,
        "Attendance": 80,
        "Parental_Involvement": "Medium",
        "Access_to_Resources": "Medium",
        "Extracurricular_Activities": "Yes",
        "Sleep_Hours": 7,
        "Previous_Scores": 75,
        "Motivation_Level": "Medium",
        "Internet_Access": "Yes",
        "Tutoring_Sessions": 1,
        "Family_Income": "Low",
        "Teacher_Quality": "Medium",
        "School_Type": "Public",
        "Peer_Influence": "Positive",
        "Physical_Activity": 3,
        "Learning_Disabilities": "No",
        "Parental_Education_Level": "High School",
        "Distance_from_Home": "Near",
        "Gender": "Male",
    }


def test_prepare_input_keeps_training_column_order() -> None:
    frame = prepare_input(sample_input())

    assert frame.shape == (1, 19)
    assert frame.columns.tolist() == INPUT_COLUMNS


def test_prepare_input_rejects_missing_column() -> None:
    values = sample_input()
    values.pop("Attendance")

    with pytest.raises(ValueError, match="Attendance"):
        prepare_input(values)


def test_predict_batch_returns_prediction_for_every_row() -> None:
    frame = pd.DataFrame([sample_input(), sample_input()])

    result = predict_batch(ConstantModel(), frame)

    assert result["Predicted_Exam_Score"].tolist() == [75.5, 75.5]


def test_predict_batch_with_errors_only_predicts_valid_rows() -> None:
    invalid = sample_input()
    invalid["Gender"] = "Unknown"
    frame = pd.DataFrame([sample_input(), invalid])

    result = predict_batch_with_errors(ConstantModel(), frame)

    assert result.loc[0, "Predicted_Exam_Score"] == 75.5
    assert pd.isna(result.loc[1, "Predicted_Exam_Score"])
    assert "Gender" in result.loc[1, "Validation_Error"]


def test_saved_real_pipeline_supports_batch_prediction(tmp_path) -> None:
    rows = []
    for hours, attendance in [(10, 65), (20, 75), (30, 85), (40, 95)]:
        row = sample_input()
        row["Hours_Studied"] = hours
        row["Attendance"] = attendance
        rows.append(row)
    frame = pd.DataFrame(rows)
    target = [60.0, 68.0, 76.0, 84.0]
    pipeline = Pipeline(
        [
            ("preprocessor", build_preprocessor()),
            ("model", SVR(kernel="linear", C=1.0)),
        ]
    )
    pipeline.fit(frame, target)
    path = tmp_path / "pipeline.joblib"
    joblib.dump(pipeline, path)

    restored = load_model(path)
    result = predict_batch(restored, frame)

    assert len(result) == len(frame)
    assert np.isfinite(result["Predicted_Exam_Score"]).all()
