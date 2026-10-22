"""Kiểm thử chuẩn dữ liệu đầu vào cho giao diện dự đoán."""

import pytest

from src.predict import INPUT_COLUMNS, prepare_input


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
