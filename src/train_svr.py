"""Huấn luyện và tối ưu Support Vector Regression cho Người 5."""

from __future__ import annotations

import json
from time import perf_counter

import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR

from src.config import (
    CV_FOLDS,
    DATA_PATH,
    RANDOM_STATE,
    RESULTS_PATH,
    SVR_MODEL_PATH,
    TARGET_COLUMN,
    TEST_SIZE,
)
from src.data_cleaning import clean_student_data
from src.data_loader import load_data
from src.preprocessing import build_preprocessor


SVR_PARAM_GRID = [
    {
        "model__kernel": ["rbf"],
        "model__C": [1.0, 10.0, 100.0],
        "model__epsilon": [0.05, 0.1, 0.2],
        "model__gamma": ["scale", "auto"],
    },
    {
        "model__kernel": ["linear"],
        "model__C": [0.1, 1.0, 10.0],
        "model__epsilon": [0.05, 0.1, 0.2],
    },
]


def build_svr_search(n_jobs: int = -1) -> GridSearchCV:
    """Tạo GridSearchCV với tiền xử lý nằm trong từng fold."""
    pipeline = Pipeline(
        [
            ("preprocessor", build_preprocessor()),
            ("model", SVR()),
        ]
    )
    return GridSearchCV(
        pipeline,
        param_grid=SVR_PARAM_GRID,
        scoring="neg_mean_absolute_error",
        cv=CV_FOLDS,
        n_jobs=n_jobs,
        refit=True,
        return_train_score=True,
    )


def _stable_cv_table(search: GridSearchCV) -> pd.DataFrame:
    """Loại cột thời gian biến động để file kết quả có thể tái lập."""
    table = pd.DataFrame(search.cv_results_)
    timing_columns = [
        "mean_fit_time",
        "std_fit_time",
        "mean_score_time",
        "std_score_time",
    ]
    return table.drop(columns=timing_columns, errors="ignore")


def train_svr() -> pd.DataFrame:
    """Tuning SVR trên train, đánh giá một lần trên test và lưu artifact."""
    data, _ = clean_student_data(load_data(DATA_PATH))
    X = data.drop(columns=[TARGET_COLUMN])
    y = data[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    search = build_svr_search()
    started_at = perf_counter()
    search.fit(X_train, y_train)
    fit_seconds = perf_counter() - started_at

    predictions = search.best_estimator_.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions) ** 0.5
    r2 = r2_score(y_test, predictions)

    metrics = pd.DataFrame(
        [
            {
                "model": "SVR_Tuned",
                "cv_mae": -search.best_score_,
                "test_mae": mae,
                "test_rmse": rmse,
                "test_r2": r2,
                "fit_seconds": fit_seconds,
                "train_rows": len(X_train),
                "test_rows": len(X_test),
                "best_params": json.dumps(
                    search.best_params_,
                    ensure_ascii=False,
                    default=str,
                ),
            }
        ]
    )

    RESULTS_PATH.mkdir(parents=True, exist_ok=True)
    SVR_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    cv_path = RESULTS_PATH / "svr_cv_results.csv"
    metrics_path = RESULTS_PATH / "svr_metrics.csv"

    _stable_cv_table(search).to_csv(cv_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    joblib.dump(search.best_estimator_, SVR_MODEL_PATH)

    print("SVR tuning completed")
    print(f"Best parameters: {search.best_params_}")
    print(f"CV MAE: {-search.best_score_:.4f}")
    print(f"Test MAE: {mae:.4f}")
    print(f"Test RMSE: {rmse:.4f}")
    print(f"Test R2: {r2:.4f}")
    print(f"Saved CV results: {cv_path}")
    print(f"Saved metrics: {metrics_path}")
    print(f"Saved model: {SVR_MODEL_PATH}")
    return metrics


if __name__ == "__main__":
    train_svr()
