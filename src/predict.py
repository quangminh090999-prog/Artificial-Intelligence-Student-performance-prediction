"""Nạp pipeline đã lưu và dự đoán điểm thi."""

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.config import MODEL_PATH


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


def load_model(path: Path = MODEL_PATH) -> Any:
    """Nạp pipeline học máy đã được huấn luyện từ ổ đĩa."""
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file mô hình tại {path}")
    return joblib.load(path)


def prepare_input(input_data: Mapping[str, Any]) -> pd.DataFrame:
    """Kiểm tra và chuyển một bản ghi đầu vào thành DataFrame đúng thứ tự cột."""
    missing_columns = [column for column in INPUT_COLUMNS if column not in input_data]
    if missing_columns:
        raise ValueError("Thiếu trường đầu vào: " + ", ".join(missing_columns))

    return pd.DataFrame(
        [{column: input_data[column] for column in INPUT_COLUMNS}],
        columns=INPUT_COLUMNS,
    )


def predict_one(model: Any, input_data: Mapping[str, Any]) -> float:
    """Dự đoán điểm thi cho một sinh viên bằng pipeline đã nạp."""
    frame = prepare_input(input_data)
    prediction = model.predict(frame)
    if len(prediction) != 1:
        raise ValueError("Mô hình không trả về đúng một kết quả dự đoán.")
    return float(prediction[0])
