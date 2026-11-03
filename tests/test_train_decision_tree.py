"""Kiểm thử cấu hình tuning Decision Tree mà không chạy GridSearch đầy đủ."""

import pandas as pd
import numpy as np
import pytest
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

from src.config import CV_FOLDS, RANDOM_STATE
from src.main import _merge_and_print_summary
from src.train_decision_tree import DT_PARAM_GRID, train_decision_tree
from src.preprocessing import build_preprocessor


def _build_dt_search(n_jobs: int = -1) -> GridSearchCV:
    """Tạo GridSearchCV cho Decision Tree (dùng trong test)."""
    pipeline = Pipeline(
        [
            ("preprocessor", build_preprocessor()),
            ("model", DecisionTreeRegressor(random_state=RANDOM_STATE)),
        ]
    )
    return GridSearchCV(
        pipeline,
        param_grid=DT_PARAM_GRID,
        scoring="neg_mean_absolute_error",
        cv=CV_FOLDS,
        n_jobs=n_jobs,
    )


def test_dt_search_uses_correct_estimator() -> None:
    search = _build_dt_search(n_jobs=1)

    assert isinstance(search, GridSearchCV)
    assert isinstance(search.estimator, Pipeline)
    assert isinstance(search.estimator.named_steps["model"], DecisionTreeRegressor)
    assert search.cv == CV_FOLDS
    assert search.scoring == "neg_mean_absolute_error"


def test_dt_grid_covers_required_parameters() -> None:
    keys = set(DT_PARAM_GRID.keys())

    assert {
        "model__max_depth",
        "model__min_samples_split",
        "model__min_samples_leaf",
    }.issubset(keys)


def test_dt_grid_contains_none_for_max_depth() -> None:
    """None trong max_depth cho phép cây phát triển không giới hạn."""
    assert None in DT_PARAM_GRID["model__max_depth"]


def test_train_decision_tree_returns_metrics_with_small_data(tmp_path, monkeypatch) -> None:
    """Chạy train_decision_tree với dữ liệu mẫu nhỏ và mock GridSearchCV để chạy nhanh."""
    import src.train_decision_tree as dt_module
    from sklearn.pipeline import Pipeline
    from sklearn.tree import DecisionTreeRegressor
    from src.preprocessing import build_preprocessor

    # Tạo dữ liệu mẫu tối thiểu
    np.random.seed(RANDOM_STATE)
    n = 60
    sample_df = pd.DataFrame(
        {
            "Hours_Studied": np.random.randint(1, 10, n).astype(float),
            "Attendance": np.random.randint(60, 100, n).astype(float),
            "Sleep_Hours": np.random.randint(4, 9, n).astype(float),
            "Previous_Scores": np.random.randint(50, 100, n).astype(float),
            "Tutoring_Sessions": np.random.randint(0, 5, n).astype(float),
            "Physical_Activity": np.random.randint(0, 5, n).astype(float),
            "Parental_Involvement": np.random.choice(["Low", "Medium", "High"], n),
            "Access_to_Resources": np.random.choice(["Low", "Medium", "High"], n),
            "Extracurricular_Activities": np.random.choice(["Yes", "No"], n),
            "Motivation_Level": np.random.choice(["Low", "Medium", "High"], n),
            "Internet_Access": np.random.choice(["Yes", "No"], n),
            "Family_Income": np.random.choice(["Low", "Medium", "High"], n),
            "Teacher_Quality": np.random.choice(["Low", "Medium", "High"], n),
            "School_Type": np.random.choice(["Public", "Private"], n),
            "Peer_Influence": np.random.choice(["Positive", "Neutral", "Negative"], n),
            "Learning_Disabilities": np.random.choice(["Yes", "No"], n),
            "Parental_Education_Level": np.random.choice(["High School", "College", "Postgraduate"], n),
            "Distance_from_Home": np.random.choice(["Near", "Moderate", "Far"], n),
            "Gender": np.random.choice(["Male", "Female"], n),
            "Exam_Score": np.random.randint(50, 100, n).astype(float),
        }
    )

    # Patch RESULTS_PATH sang tmp_path
    monkeypatch.setattr(dt_module, "RESULTS_PATH", tmp_path)

    # Mock load_data và clean_student_data
    from src.data_cleaning import CleaningSummary
    monkeypatch.setattr(dt_module, "load_data", lambda: sample_df)
    monkeypatch.setattr(
        dt_module,
        "clean_student_data",
        lambda df: (df, CleaningSummary(
            rows_before=n, rows_after=n,
            duplicates_removed=0, target_values_clipped=0,
            missing_feature_values_preserved=0,
        )),
    )

    # Mock GridSearchCV để bỏ qua tuning thật, chỉ fit pipeline đơn giản
    from unittest.mock import MagicMock
    from sklearn.model_selection import train_test_split as tts

    X = sample_df.drop(columns=["Exam_Score"])
    y = sample_df["Exam_Score"]
    X_train, _, y_train, _ = tts(X, y, test_size=0.2, random_state=RANDOM_STATE)
    real_pipeline = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("model", DecisionTreeRegressor(random_state=RANDOM_STATE, max_depth=3)),
    ])
    real_pipeline.fit(X_train, y_train)

    mock_search = MagicMock()
    mock_search.best_estimator_ = real_pipeline
    mock_search.best_params_ = {"model__max_depth": 3, "model__min_samples_split": 2, "model__min_samples_leaf": 1}
    mock_search.best_score_ = -1.5
    mock_search.cv_results_ = {
        "param_model__max_depth": [3],
        "param_model__min_samples_split": [2],
        "param_model__min_samples_leaf": [1],
        "mean_test_score": [-1.5],
        "std_test_score": [0.1],
        "rank_test_score": [1],
    }

    monkeypatch.setattr(dt_module, "GridSearchCV", lambda *a, **kw: mock_search)

    metrics = train_decision_tree()

    assert "MAE" in metrics
    assert "RMSE" in metrics
    assert "R2" in metrics
    assert metrics["CV_MAE"] == pytest.approx(1.5)
    assert metrics["MAE"] >= 0
    assert metrics["RMSE"] >= 0
    assert (tmp_path / "decision_tree_metrics.csv").exists()
    assert (tmp_path / "decision_tree_cv_results.csv").exists()

    cv_results = pd.read_csv(tmp_path / "decision_tree_cv_results.csv")
    assert "mean_cv_mae" in cv_results.columns
    assert cv_results.loc[0, "mean_cv_mae"] == pytest.approx(1.5)


def test_merge_summary_normalizes_svr_metric_names(tmp_path, monkeypatch) -> None:
    """Bảng chung phải đọc được cả cột test_mae của module SVR."""
    import src.main as main_module

    monkeypatch.setattr(main_module, "RESULTS_PATH", tmp_path)
    pd.DataFrame(
        [{"model": "Ridge", "MAE": 0.5, "RMSE": 1.8, "R2": 0.76}]
    ).to_csv(tmp_path / "model_comparison.csv", index=False)
    pd.DataFrame(
        [
            {
                "model": "SVR_Tuned",
                "test_mae": 0.4,
                "test_rmse": 1.7,
                "test_r2": 0.78,
            }
        ]
    ).to_csv(tmp_path / "svr_metrics.csv", index=False)

    combined = _merge_and_print_summary(include_dt=False, include_svr=True)

    assert combined["model"].tolist() == ["SVR_Tuned", "Ridge"]
    assert (tmp_path / "all_models_summary.csv").exists()
