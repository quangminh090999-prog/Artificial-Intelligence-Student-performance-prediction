
"""Huấn luyện và đánh giá mô hình KNN."""

import json

import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline

from src.config import (
    DATA_PATH,
    RANDOM_STATE,
    RESULTS_PATH,
    TARGET_COLUMN,
    TEST_SIZE,
    CV_FOLDS,
)
from src.data_loader import load_data
from src.preprocessing import build_preprocessor


def run_knn():
    # Đọc dữ liệu
    df = load_data(DATA_PATH)
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN].clip(upper=100)

    # Chia train/test theo cấu hình chung của dự án
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    # Tham số chung cần thử
    common_params = {
        "model__n_neighbors": [3, 5, 7, 9, 11, 15],
        "model__weights": ["uniform", "distance"],
        "model__p": [1, 2],
    }

    # Mô hình 1: dùng toàn bộ đặc trưng
    full_pipeline = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("model", KNeighborsRegressor()),
    ])

    full_search = GridSearchCV(
        full_pipeline,
        param_grid=common_params,
        scoring="neg_mean_absolute_error",
        cv=CV_FOLDS,
        n_jobs=-1,
        refit=True,
    )
    full_search.fit(X_train, y_train)

    # Mô hình 2: chọn đặc trưng bên trong pipeline.
    # Việc này giúp mỗi fold chỉ chọn đặc trưng từ dữ liệu train của fold.
    selected_pipeline = Pipeline([
        ("preprocessor", build_preprocessor()),
        (
            "selector",
            SelectKBest(score_func=mutual_info_regression),
        ),
        ("model", KNeighborsRegressor()),
    ])

    selected_params = {
        **common_params,
        "selector__k": [5, 10, 15, 20],
    }

    selected_search = GridSearchCV(
        selected_pipeline,
        param_grid=selected_params,
        scoring="neg_mean_absolute_error",
        cv=CV_FOLDS,
        n_jobs=-1,
        refit=True,
    )
    selected_search.fit(X_train, y_train)

    # Đánh giá trên tập test, chỉ sau khi đã chọn được tham số tốt nhất
    searches = {
        "KNN_full_features": full_search,
        "KNN_selected_features": selected_search,
    }

    metrics = []
    cv_tables = []

    for name, search in searches.items():
        predictions = search.best_estimator_.predict(X_test)

        mae = mean_absolute_error(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)
        rmse = mse ** 0.5
        r2 = r2_score(y_test, predictions)

        metrics.append({
            "model": name,
            "cv_mae": -search.best_score_,
            "test_mae": mae,
            "test_rmse": rmse,
            "test_r2": r2,
            "best_params": json.dumps(
                search.best_params_,
                ensure_ascii=False,
                default=str,
            ),
        })

        table = pd.DataFrame(search.cv_results_)
        table.insert(0, "model", name)
        cv_tables.append(table)

        print(f"\n--- {name} ---")
        print(f"CV MAE tốt nhất: {-search.best_score_:.4f}")
        print(f"Test MAE: {mae:.4f}")
        print(f"Test RMSE: {rmse:.4f}")
        print(f"Test R²: {r2:.4f}")
        print(f"Tham số tốt nhất: {search.best_params_}")

    # Lưu ra file mới, không ghi đè báo cáo hiện có
    RESULTS_PATH.mkdir(parents=True, exist_ok=True)

    metrics_path = RESULTS_PATH / "knn_metrics.csv"
    pd.DataFrame(metrics).to_csv(metrics_path, index=False)

    cv_path = RESULTS_PATH / "knn_cv_results.csv"
    pd.concat(cv_tables, ignore_index=True).to_csv(
        cv_path,
        index=False,
    )

    print("\nĐã lưu kết quả:")
    print(metrics_path)
    print(cv_path)


if __name__ == "__main__":
    run_knn()
