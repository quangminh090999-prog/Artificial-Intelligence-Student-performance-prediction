"""Nạp pipeline đã lưu và dự đoán điểm thi."""

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.config import MODEL_PATH
from src.input_schema import (
    INPUT_COLUMNS,
    InputValidationError,
    validate_input_frame,
    validate_input_mapping,
    validate_input_rows,
)


def load_model(path: Path = MODEL_PATH) -> Any:
    """Nạp pipeline học máy đã được huấn luyện từ ổ đĩa."""
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file mô hình tại {path}")
    return joblib.load(path)


def prepare_input(input_data: Mapping[str, Any]) -> pd.DataFrame:
    """Kiểm tra và chuyển một bản ghi đầu vào thành DataFrame đúng thứ tự cột."""
    return validate_input_mapping(input_data)


def predict_one(model: Any, input_data: Mapping[str, Any]) -> float:
    """Dự đoán điểm thi cho một sinh viên bằng pipeline đã nạp."""
    frame = prepare_input(input_data)
    prediction = model.predict(frame)
    if len(prediction) != 1:
        raise ValueError("Mô hình không trả về đúng một kết quả dự đoán.")
    return float(prediction[0])


def predict_batch(model: Any, input_frame: pd.DataFrame) -> pd.DataFrame:
    """Dự đoán toàn bộ DataFrame; báo lỗi nếu có dòng không hợp lệ."""
    prepared = validate_input_frame(input_frame)
    predictions = model.predict(prepared)
    if len(predictions) != len(prepared):
        raise ValueError("Số dự đoán trả về không khớp số dòng đầu vào.")

    result = input_frame.copy().reset_index(drop=True)
    result["Predicted_Exam_Score"] = predictions
    return result


def predict_batch_with_errors(model: Any, input_frame: pd.DataFrame) -> pd.DataFrame:
    """Dự đoán dòng hợp lệ và ghi thông báo cho từng dòng lỗi."""
    valid, errors = validate_input_rows(input_frame)
    result = input_frame.copy().reset_index(drop=True)
    result["Predicted_Exam_Score"] = pd.Series(
        [pd.NA] * len(result), dtype="Float64"
    )
    result["Validation_Error"] = ""

    if not valid.empty:
        predictions = model.predict(valid)
        if len(predictions) != len(valid):
            raise ValueError("Số dự đoán trả về không khớp số dòng hợp lệ.")
        result.loc[valid.index, "Predicted_Exam_Score"] = predictions

    for row in errors.itertuples(index=False):
        result.loc[row.row_number - 1, "Validation_Error"] = row.errors

    return result


__all__ = [
    "INPUT_COLUMNS",
    "InputValidationError",
    "load_model",
    "prepare_input",
    "predict_one",
    "predict_batch",
    "predict_batch_with_errors",
]
