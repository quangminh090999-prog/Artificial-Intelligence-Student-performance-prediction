"""Kiểm tra schema cho dữ liệu dự đoán một dòng và hàng loạt."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd


NUMERIC_RANGES: dict[str, tuple[float, float]] = {
    "Hours_Studied": (0.0, 44.0),
    "Attendance": (0.0, 100.0),
    "Sleep_Hours": (0.0, 10.0),
    "Previous_Scores": (0.0, 100.0),
    "Tutoring_Sessions": (0.0, 8.0),
    "Physical_Activity": (0.0, 6.0),
}

CATEGORICAL_ALLOWED: dict[str, tuple[str, ...]] = {
    "Parental_Involvement": ("High", "Low", "Medium"),
    "Access_to_Resources": ("High", "Low", "Medium"),
    "Extracurricular_Activities": ("No", "Yes"),
    "Motivation_Level": ("High", "Low", "Medium"),
    "Internet_Access": ("No", "Yes"),
    "Family_Income": ("High", "Low", "Medium"),
    "Teacher_Quality": ("High", "Low", "Medium"),
    "School_Type": ("Private", "Public"),
    "Peer_Influence": ("Negative", "Neutral", "Positive"),
    "Learning_Disabilities": ("No", "Yes"),
    "Parental_Education_Level": ("College", "High School", "Postgraduate"),
    "Distance_from_Home": ("Far", "Moderate", "Near"),
    "Gender": ("Female", "Male"),
}

INPUT_COLUMNS = [
    "Hours_Studied",
    "Attendance",
    "Parental_Involvement",
    "Access_to_Resources",
    "Extracurricular_Activities",
    "Sleep_Hours",
    "Previous_Scores",
    "Motivation_Level",
    "Internet_Access",
    "Tutoring_Sessions",
    "Family_Income",
    "Teacher_Quality",
    "School_Type",
    "Peer_Influence",
    "Physical_Activity",
    "Learning_Disabilities",
    "Parental_Education_Level",
    "Distance_from_Home",
    "Gender",
]


class InputValidationError(ValueError):
    """Lỗi dữ liệu đầu vào kèm bảng chi tiết theo từng dòng."""

    def __init__(self, message: str, errors: pd.DataFrame | None = None):
        super().__init__(message)
        self.errors = errors if errors is not None else pd.DataFrame()


def _is_blank(value: Any) -> bool:
    return isinstance(value, str) and not value.strip()


def validate_input_rows(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chuẩn hóa dữ liệu, trả lại dòng hợp lệ và lỗi theo từng dòng."""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("Dữ liệu đầu vào phải là pandas DataFrame.")
    if frame.empty:
        raise InputValidationError("Dữ liệu đầu vào không có dòng nào.")
    if frame.columns.duplicated().any():
        duplicated = frame.columns[frame.columns.duplicated()].tolist()
        raise InputValidationError(
            "Tên cột bị trùng: " + ", ".join(map(str, duplicated))
        )

    missing_columns = [column for column in INPUT_COLUMNS if column not in frame]
    if missing_columns:
        raise InputValidationError(
            "Thiếu cột bắt buộc: " + ", ".join(missing_columns)
        )

    cleaned = frame.loc[:, INPUT_COLUMNS].copy().reset_index(drop=True)
    row_errors: dict[int, list[str]] = {
        position: [] for position in range(len(cleaned))
    }

    for column, (minimum, maximum) in NUMERIC_RANGES.items():
        original = cleaned[column]
        missing_mask = original.isna() | original.map(_is_blank)
        converted = pd.to_numeric(original, errors="coerce")
        invalid_type_mask = converted.isna() & ~missing_mask
        outside_mask = converted.notna() & ~converted.between(minimum, maximum)

        for position in cleaned.index[missing_mask]:
            row_errors[position].append(f"{column}: thiếu giá trị")
        for position in cleaned.index[invalid_type_mask]:
            row_errors[position].append(f"{column}: phải là số")
        for position in cleaned.index[outside_mask]:
            row_errors[position].append(
                f"{column}: phải trong khoảng [{minimum:g}, {maximum:g}]"
            )
        cleaned[column] = converted

    for column, allowed_values in CATEGORICAL_ALLOWED.items():
        original = cleaned[column]
        missing_mask = original.isna() | original.map(_is_blank)
        normalized = original.map(
            lambda value: value.strip() if isinstance(value, str) else value
        )
        invalid_mask = ~missing_mask & ~normalized.isin(allowed_values)

        for position in cleaned.index[missing_mask]:
            row_errors[position].append(f"{column}: thiếu giá trị")
        for position in cleaned.index[invalid_mask]:
            row_errors[position].append(
                f"{column}: giá trị không hợp lệ '{normalized.loc[position]}'"
            )
        cleaned[column] = normalized

    error_records = [
        {
            "row_number": position + 1,
            "errors": "; ".join(messages),
        }
        for position, messages in row_errors.items()
        if messages
    ]
    errors = pd.DataFrame(error_records, columns=["row_number", "errors"])
    invalid_positions = {record["row_number"] - 1 for record in error_records}
    valid_positions = [
        position for position in cleaned.index if position not in invalid_positions
    ]
    return cleaned.loc[valid_positions], errors


def validate_input_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Trả lại DataFrame đã chuẩn hóa hoặc báo toàn bộ lỗi tìm được."""
    valid, errors = validate_input_rows(frame)
    if not errors.empty:
        details = " | ".join(
            f"Dòng {row.row_number}: {row.errors}"
            for row in errors.itertuples(index=False)
        )
        raise InputValidationError("Dữ liệu đầu vào không hợp lệ. " + details, errors)
    return valid


def validate_input_mapping(input_data: Mapping[str, Any]) -> pd.DataFrame:
    """Kiểm tra một bản ghi dạng mapping và trả DataFrame một dòng."""
    if not isinstance(input_data, Mapping):
        raise TypeError("Một bản ghi đầu vào phải là mapping.")
    return validate_input_frame(pd.DataFrame([dict(input_data)]))
