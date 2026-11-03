"""Huấn luyện và tuning DecisionTreeRegressor cho dự án dự đoán điểm sinh viên."""

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

import src.config as _cfg
from src.data_cleaning import clean_student_data
from src.data_loader import load_data
from src.preprocessing import build_preprocessor

# Các hằng số này được khai báo riêng để test có thể monkeypatch dễ dàng.
CV_FOLDS = _cfg.CV_FOLDS
RANDOM_STATE = _cfg.RANDOM_STATE
RESULTS_PATH = _cfg.RESULTS_PATH
TARGET_COLUMN = _cfg.TARGET_COLUMN
TEST_SIZE = _cfg.TEST_SIZE

DT_PARAM_GRID = {
    "model__max_depth": [3, 5, 7, 10, None],
    "model__min_samples_split": [2, 5, 10],
    "model__min_samples_leaf": [1, 2, 4],
}


def train_decision_tree() -> dict:
    """Tuning DecisionTreeRegressor bằng GridSearchCV 5-fold trên tập train.

    Returns
    -------
    dict
        Các chỉ số đánh giá trên tập test: MAE, RMSE, R2 và tham số tốt nhất.
    """
    data, _ = clean_student_data(load_data())

    X = data.drop(columns=[TARGET_COLUMN])
    y = data[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    pipeline = Pipeline(
        [
            ("preprocessor", build_preprocessor()),
            ("model", DecisionTreeRegressor(random_state=RANDOM_STATE)),
        ]
    )

    search = GridSearchCV(
        pipeline,
        param_grid=DT_PARAM_GRID,
        scoring="neg_mean_absolute_error",
        cv=CV_FOLDS,
        n_jobs=-1,
        return_train_score=False,
    )
    search.fit(X_train, y_train)

    best_pipeline = search.best_estimator_
    predictions = best_pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions) ** 0.5
    r2 = r2_score(y_test, predictions)
    cv_mae = -search.best_score_
    best_params = {
        k.replace("model__", ""): v
        for k, v in search.best_params_.items()
    }

    # Lưu kết quả CV đầy đủ
    RESULTS_PATH.mkdir(parents=True, exist_ok=True)
    cv_df = pd.DataFrame(search.cv_results_)
    param_cols = [column for column in cv_df.columns if column.startswith("param_")]
    stable_cv = cv_df[
        param_cols + ["mean_test_score", "std_test_score", "rank_test_score"]
    ].copy()
    stable_cv["mean_cv_mae"] = -stable_cv.pop("mean_test_score")
    stable_cv["std_cv_mae"] = stable_cv.pop("std_test_score")
    stable_cv = stable_cv.rename(columns={"rank_test_score": "rank"})
    stable_cv.sort_values("rank").to_csv(
        RESULTS_PATH / "decision_tree_cv_results.csv", index=False
    )

    # Lưu chỉ số tập test
    metrics = {
        "model": "DecisionTree_Tuned",
        "CV_MAE": float(cv_mae),
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2": float(r2),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        **best_params,
    }
    pd.DataFrame([metrics]).to_csv(
        RESULTS_PATH / "decision_tree_metrics.csv", index=False
    )

    print(f"Best params : {best_params}")
    print(f"CV MAE      : {cv_mae:.4f}")
    print(f"Test MAE    : {mae:.4f}")
    print(f"Test RMSE   : {rmse:.4f}")
    print(f"Test R2     : {r2:.4f}")

    return metrics


if __name__ == "__main__":
    train_decision_tree()
