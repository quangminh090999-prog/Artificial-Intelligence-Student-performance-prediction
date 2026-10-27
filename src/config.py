"""Cấu hình và đường dẫn dùng chung trong toàn dự án."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "StudentPerformanceFactors.csv"
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed"
CLEAN_DATA_PATH = PROCESSED_DATA_PATH / "StudentPerformanceFactors_cleaned.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "student_score_pipeline.joblib"
RESULTS_PATH = PROJECT_ROOT / "reports" / "results"
FIGURES_PATH = PROJECT_ROOT / "reports" / "figures"

TARGET_COLUMN = "Exam_Score"
TEST_SIZE = 0.2
RANDOM_STATE = 42
CV_FOLDS = 5
