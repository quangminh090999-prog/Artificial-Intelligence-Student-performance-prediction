"""Kiểm thử schema dữ liệu dự đoán một dòng và hàng loạt."""

import pandas as pd
import pytest

from src.input_schema import (
    INPUT_COLUMNS,
    InputValidationError,
    validate_input_frame,
    validate_input_rows,
)


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


def test_schema_normalizes_numeric_strings_and_whitespace() -> None:
    values = sample_input()
    values["Attendance"] = "80"
    values["Motivation_Level"] = " Medium "

    cleaned = validate_input_frame(pd.DataFrame([values]))

    assert cleaned.columns.tolist() == INPUT_COLUMNS
    assert cleaned.loc[0, "Attendance"] == 80
    assert cleaned.loc[0, "Motivation_Level"] == "Medium"


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("Attendance", "eighty", "phải là số"),
        ("Attendance", 101, "phải trong khoảng"),
        ("Gender", "Unknown", "giá trị không hợp lệ"),
        ("Teacher_Quality", "", "thiếu giá trị"),
    ],
)
def test_schema_rejects_invalid_values(column, value, message) -> None:
    values = sample_input()
    values[column] = value

    with pytest.raises(InputValidationError, match=message):
        validate_input_frame(pd.DataFrame([values]))


def test_schema_returns_errors_by_row_and_keeps_valid_rows() -> None:
    valid = sample_input()
    invalid = sample_input()
    invalid["Hours_Studied"] = 100
    frame = pd.DataFrame([valid, invalid])

    cleaned, errors = validate_input_rows(frame)

    assert cleaned.index.tolist() == [0]
    assert errors["row_number"].tolist() == [2]
    assert "Hours_Studied" in errors.loc[0, "errors"]


def test_schema_rejects_empty_frame() -> None:
    with pytest.raises(InputValidationError, match="không có dòng"):
        validate_input_frame(pd.DataFrame(columns=INPUT_COLUMNS))
