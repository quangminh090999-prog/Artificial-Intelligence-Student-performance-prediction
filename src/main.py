"""Điều phối toàn bộ pipeline: load → clean → train → evaluate → export."""

import pandas as pd

from src.config import (
    MODEL_PATH,
    PROJECT_ROOT,
    RANDOM_STATE,
    RESULTS_PATH,
    TARGET_COLUMN,
    TEST_SIZE,
)
from src.data_cleaning import clean_student_data, save_clean_data
from src.data_loader import load_data
from src.evaluate import evaluate_model
from src.train import train_models
from src.train_decision_tree import train_decision_tree
from src.train_svr import train_svr


def run_pipeline(include_svr: bool = True, include_dt: bool = True) -> None:
    """Chạy toàn bộ pipeline từ đầu đến cuối và in tóm tắt kết quả.

    Parameters
    ----------
    include_svr:
        Có chạy tuning SVR không (mất vài phút).
    include_dt:
        Có chạy tuning Decision Tree không.
    """
    print("=" * 60)
    print("BƯỚC 1 — Đọc và làm sạch dữ liệu")
    print("=" * 60)
    raw = load_data()
    data, cleaning_summary = clean_student_data(raw)
    save_clean_data(data)
    print(f"  Dòng gốc      : {cleaning_summary.rows_before:,}")
    print(f"  Dòng sau làm sạch: {cleaning_summary.rows_after:,}")
    print(f"  Trùng lặp xóa : {cleaning_summary.duplicates_removed:,}")
    print(f"  Exam_Score clipped: {cleaning_summary.target_values_clipped:,}")

    print()
    print("=" * 60)
    print("BƯỚC 2 — Huấn luyện các mô hình cơ sở và Ridge")
    print("=" * 60)
    results_df = train_models()
    print()
    print(results_df.to_string(index=False))

    print()
    print("=" * 60)
    print("BƯỚC 3 — Đánh giá pipeline tốt nhất của nhóm")
    print("=" * 60)
    evaluate_model()

    if include_dt:
        print()
        print("=" * 60)
        print("BƯỚC 4 — Tuning Decision Tree (5-fold CV)")
        print("=" * 60)
        train_decision_tree()

    if include_svr:
        print()
        print("=" * 60)
        print("BƯỚC 5 — Tuning SVR (5-fold CV)")
        print("=" * 60)
        train_svr()

    print()
    print("=" * 60)
    print("BƯỚC 6 — Tổng hợp kết quả")
    print("=" * 60)
    _merge_and_print_summary(include_dt=include_dt, include_svr=include_svr)

    print()
    print("=" * 60)
    print("HOÀN THÀNH")
    print(f"  Mô hình tốt nhất (nhóm) : {MODEL_PATH.relative_to(PROJECT_ROOT)}")
    print(f"  Kết quả chi tiết        : {RESULTS_PATH.relative_to(PROJECT_ROOT)}")
    print("=" * 60)


def _metric_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Chuẩn hóa tên cột chỉ số từ các module huấn luyện khác nhau."""
    normalized = frame.rename(
        columns={
            "test_mae": "MAE",
            "test_rmse": "RMSE",
            "test_r2": "R2",
        }
    )
    required = ["model", "MAE", "RMSE", "R2"]
    missing = [column for column in required if column not in normalized.columns]
    if missing:
        raise ValueError(
            "File kết quả thiếu cột bắt buộc: " + ", ".join(missing)
        )
    return normalized[required]


def _merge_and_print_summary(
    include_dt: bool, include_svr: bool
) -> pd.DataFrame:
    """Đọc các file CSV kết quả, chuẩn hóa và lưu bảng so sánh chung."""
    frames = []

    comparison_path = RESULTS_PATH / "model_comparison.csv"
    if comparison_path.exists():
        frames.append(_metric_columns(pd.read_csv(comparison_path)))

    if include_dt:
        dt_path = RESULTS_PATH / "decision_tree_metrics.csv"
        if dt_path.exists():
            frames.append(_metric_columns(pd.read_csv(dt_path)))

    if include_svr:
        svr_path = RESULTS_PATH / "svr_metrics.csv"
        if svr_path.exists():
            svr_df = pd.read_csv(svr_path)
            if "model" not in svr_df.columns:
                svr_df.insert(0, "model", "SVR_Tuned")
            frames.append(_metric_columns(svr_df))

    if frames:
        combined = (
            pd.concat(frames, ignore_index=True)
            .drop_duplicates(subset="model")
            .sort_values("MAE")
        )
        combined.to_csv(RESULTS_PATH / "all_models_summary.csv", index=False)
        print(combined.to_string(index=False))
        return combined

    print("  Chưa có file kết quả nào để tổng hợp.")
    return pd.DataFrame(columns=["model", "MAE", "RMSE", "R2"])


if __name__ == "__main__":
    run_pipeline()
