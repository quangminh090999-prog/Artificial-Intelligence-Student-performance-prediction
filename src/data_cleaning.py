"""Các bước làm sạch có thể tái sử dụng cho dữ liệu kết quả sinh viên."""

from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from src.config import CLEAN_DATA_PATH, DATA_PATH, TARGET_COLUMN


@dataclass(frozen=True)
class CleaningSummary:
    """Các số liệu dùng để kiểm tra và trình bày quá trình làm sạch."""

    rows_before: int
    rows_after: int
    duplicates_removed: int
    target_values_clipped: int
    missing_feature_values_preserved: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


def clean_student_data(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, CleaningSummary]:
    """Làm sạch dữ liệu mà không học tham số từ toàn bộ tập dữ liệu.

    Giá trị thiếu của các đặc trưng được giữ lại có chủ đích. Chúng sẽ được
    điền bằng pipeline sau khi chia train/test để tránh rò rỉ dữ liệu.
    """
    if data.empty:
        raise ValueError("Dữ liệu đầu vào không có dòng nào.")

    cleaned = data.copy(deep=True)
    cleaned.columns = cleaned.columns.str.strip()

    if cleaned.columns.duplicated().any():
        raise ValueError("Tên cột bị trùng sau khi chuẩn hóa khoảng trắng.")
    if TARGET_COLUMN not in cleaned.columns:
        raise ValueError(f"Dữ liệu thiếu cột mục tiêu: {TARGET_COLUMN}")

    text_columns = cleaned.select_dtypes(
        include=["object", "string"]
    ).columns
    for column in text_columns:
        normalized = cleaned[column].astype("string").str.strip()
        normalized = normalized.replace("", pd.NA)
        # SimpleImputer của scikit-learn nhận diện float NaN ổn định hơn
        # pandas.NA trong các cột object/string.
        cleaned[column] = normalized.astype(object).where(
            normalized.notna(), float("nan")
        )

    rows_before = len(cleaned)
    cleaned = cleaned.drop_duplicates().reset_index(drop=True)
    duplicates_removed = rows_before - len(cleaned)

    numeric_target = pd.to_numeric(cleaned[TARGET_COLUMN], errors="coerce")
    if numeric_target.isna().any():
        invalid_count = int(numeric_target.isna().sum())
        raise ValueError(
            f"Cột {TARGET_COLUMN} có {invalid_count} giá trị thiếu hoặc không phải số."
        )

    outside_target_range = ~numeric_target.between(0, 100)
    target_values_clipped = int(outside_target_range.sum())
    cleaned[TARGET_COLUMN] = numeric_target.clip(lower=0, upper=100)

    feature_columns = [
        column for column in cleaned.columns if column != TARGET_COLUMN
    ]
    missing_feature_values = int(
        cleaned[feature_columns].isna().sum().sum()
    )

    summary = CleaningSummary(
        rows_before=rows_before,
        rows_after=len(cleaned),
        duplicates_removed=duplicates_removed,
        target_values_clipped=target_values_clipped,
        missing_feature_values_preserved=missing_feature_values,
    )
    return cleaned, summary


def save_clean_data(
    data: pd.DataFrame,
    output_path: Path = CLEAN_DATA_PATH,
) -> Path:
    """Lưu dữ liệu sạch ở định dạng CSV UTF-8."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_path, index=False, encoding="utf-8-sig")
    return output_path


def clean_data_file(
    input_path: Path = DATA_PATH,
    output_path: Path = CLEAN_DATA_PATH,
) -> tuple[pd.DataFrame, CleaningSummary]:
    """Đọc, làm sạch và lưu toàn bộ tập dữ liệu."""
    if not input_path.exists():
        raise FileNotFoundError(f"Không tìm thấy dữ liệu tại {input_path}")

    raw_data = pd.read_csv(input_path)
    cleaned, summary = clean_student_data(raw_data)
    save_clean_data(cleaned, output_path)
    return cleaned, summary


if __name__ == "__main__":
    cleaned_frame, cleaning_summary = clean_data_file()
    print(f"Cleaned rows: {len(cleaned_frame):,}")
    print(f"Summary: {cleaning_summary.to_dict()}")
    print(f"Saved to: {CLEAN_DATA_PATH.relative_to(DATA_PATH.parents[2])}")
