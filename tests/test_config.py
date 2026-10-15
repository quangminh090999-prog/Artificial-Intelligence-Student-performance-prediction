"""Kiểm tra cấu hình cơ bản của dự án."""

from src.config import PROJECT_ROOT, TARGET_COLUMN


def test_project_root_exists() -> None:
    assert PROJECT_ROOT.exists()


def test_target_column_name() -> None:
    assert TARGET_COLUMN == "Exam_Score"
