"""Đọc dữ liệu gốc và thực hiện các kiểm tra tối thiểu."""

from pathlib import Path

import pandas as pd

from src.config import DATA_PATH, TARGET_COLUMN


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Đọc CSV và kiểm tra cột mục tiêu."""
    if not path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy dữ liệu tại {path}. "
            "Hãy đặt StudentPerformanceFactors.csv vào data/raw/."
        )

    data = pd.read_csv(path)
    if TARGET_COLUMN not in data.columns:
        raise ValueError(f"Dữ liệu thiếu cột mục tiêu: {TARGET_COLUMN}")
    if data.empty:
        raise ValueError("Tệp dữ liệu không có dòng nào.")
    return data


if __name__ == "__main__":
    frame = load_data()
    # Dung thong bao ASCII de tuong thich voi PowerShell dung bang ma cu.
    print(f"Loaded {len(frame):,} rows and {len(frame.columns)} columns.")
    print(f"Missing values: {int(frame.isna().sum().sum()):,}")
