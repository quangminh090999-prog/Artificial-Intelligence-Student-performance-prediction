"""Huấn luyện, tuning, so sánh và lưu các mô hình hồi quy."""

import json

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR

from src.config import (
    CV_FOLDS,
    FIGURES_PATH,
    MODEL_PATH,
    PROJECT_ROOT,
    RANDOM_STATE,
    RESULTS_PATH,
    TARGET_COLUMN,
    TEST_SIZE,
)
from src.data_cleaning import clean_student_data
from src.data_loader import load_data
from src.preprocessing import build_preprocessor


RIDGE_ALPHAS = [0.01, 0.1, 1, 10, 100, 1000]


def calculate_metrics(y_true, y_pred) -> dict[str, float]:
    """Tính bộ chỉ số hồi quy dùng chung của dự án."""
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "R2": float(r2_score(y_true, y_pred)),
    }


def _build_pipeline(model) -> Pipeline:
    """Đặt toàn bộ tiền xử lý bên trong pipeline để tránh rò rỉ dữ liệu."""
    return Pipeline(
        [
            ("preprocessor", build_preprocessor()),
            ("model", model),
        ]
    )


def _save_ridge_cv_results(search: GridSearchCV) -> pd.DataFrame:
    """Lưu bảng CV Ridge với MAE dương, độ lệch chuẩn và thứ hạng."""
    cv_results = pd.DataFrame(search.cv_results_)
    table = cv_results[
        [
            "param_model__alpha",
            "mean_test_score",
            "std_test_score",
            "rank_test_score",
        ]
    ].copy()
    table["mean_CV_MAE"] = -table.pop("mean_test_score")
    table["std_CV_MAE"] = table.pop("std_test_score")
    table = table.rename(
        columns={
            "param_model__alpha": "alpha",
            "rank_test_score": "rank",
        }
    ).sort_values("rank")
    table.to_csv(RESULTS_PATH / "ridge_cv_results.csv", index=False)
    return table


def _save_ridge_coefficients(tuned_pipeline: Pipeline) -> pd.DataFrame:
    """Xuất hệ số Ridge và biểu đồ các hệ số có độ lớn cao nhất."""
    preprocessor = tuned_pipeline.named_steps["preprocessor"]
    ridge_model = tuned_pipeline.named_steps["model"]
    feature_names = preprocessor.get_feature_names_out()

    coefficient_df = pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": ridge_model.coef_,
        }
    )
    coefficient_df["absolute_coefficient"] = coefficient_df[
        "coefficient"
    ].abs()
    coefficient_df = coefficient_df.sort_values(
        "absolute_coefficient", ascending=False
    ).reset_index(drop=True)
    coefficient_df.to_csv(
        RESULTS_PATH / "ridge_coefficients.csv", index=False
    )

    plot_df = coefficient_df.head(20).sort_values("coefficient")
    colors = [
        "#1F77B4" if value >= 0 else "#D62728"
        for value in plot_df["coefficient"]
    ]
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.barh(plot_df["feature"], plot_df["coefficient"], color=colors)
    ax.axvline(0, color="#444444", linewidth=0.8)
    ax.set_title("Các hệ số Ridge có độ lớn cao nhất")
    ax.set_xlabel("Hệ số Ridge")
    ax.set_ylabel("Đặc trưng sau tiền xử lý")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(
        FIGURES_PATH / "09_ridge_coefficients.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)
    return coefficient_df


def train_models() -> pd.DataFrame:
    """Huấn luyện, đánh giá và lưu pipeline được chọn bằng CV MAE."""
    data, cleaning_summary = clean_student_data(load_data())
    X = data.drop(columns=[TARGET_COLUMN])
    y = data[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    models = {
        "DummyRegressor": DummyRegressor(strategy="mean"),
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(),
        "RandomForest": RandomForestRegressor(
            n_estimators=100,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "GradientBoosting": GradientBoostingRegressor(
            random_state=RANDOM_STATE,
        ),
        "SVR": SVR(),
    }

    RESULTS_PATH.mkdir(parents=True, exist_ok=True)
    FIGURES_PATH.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, float | str]] = []
    best_name = ""
    best_pipeline = None
    best_cv_mae = float("inf")
    best_test_metrics: dict[str, float] = {}

    for name, model in models.items():
        pipeline = _build_pipeline(model)
        cv_scores = cross_val_score(
            pipeline,
            X_train,
            y_train,
            scoring="neg_mean_absolute_error",
            cv=CV_FOLDS,
            n_jobs=-1,
        )
        cv_mae = float(-cv_scores.mean())

        pipeline.fit(X_train, y_train)
        metrics = calculate_metrics(y_test, pipeline.predict(X_test))
        results.append({"model": name, "CV_MAE": cv_mae, **metrics})

        if cv_mae < best_cv_mae:
            best_name = name
            best_pipeline = pipeline
            best_cv_mae = cv_mae
            best_test_metrics = metrics

    ridge_search = GridSearchCV(
        _build_pipeline(Ridge()),
        param_grid={"model__alpha": RIDGE_ALPHAS},
        scoring="neg_mean_absolute_error",
        cv=CV_FOLDS,
        n_jobs=-1,
        refit=True,
    )
    ridge_search.fit(X_train, y_train)
    tuned_pipeline = ridge_search.best_estimator_
    tuned_cv_mae = float(-ridge_search.best_score_)
    tuned_metrics = calculate_metrics(
        y_test, tuned_pipeline.predict(X_test)
    )
    results.append(
        {
            "model": "Ridge_Tuned",
            "CV_MAE": tuned_cv_mae,
            **tuned_metrics,
        }
    )

    if tuned_cv_mae < best_cv_mae:
        best_name = "Ridge_Tuned"
        best_pipeline = tuned_pipeline
        best_cv_mae = tuned_cv_mae
        best_test_metrics = tuned_metrics

    _save_ridge_cv_results(ridge_search)
    _save_ridge_coefficients(tuned_pipeline)

    results_df = pd.DataFrame(results).sort_values("CV_MAE")
    results_df.to_csv(RESULTS_PATH / "model_comparison.csv", index=False)

    if best_pipeline is None:
        raise RuntimeError("Không chọn được pipeline tốt nhất.")
    joblib.dump(best_pipeline, MODEL_PATH)

    summary = {
        "best_model": best_name,
        "best_cv_MAE": best_cv_mae,
        "test_MAE": best_test_metrics["MAE"],
        "test_RMSE": best_test_metrics["RMSE"],
        "test_R2": best_test_metrics["R2"],
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "best_ridge_alpha": float(
            ridge_search.best_params_["model__alpha"]
        ),
        "best_ridge_cv_MAE": tuned_cv_mae,
        "data_cleaning": cleaning_summary.to_dict(),
    }
    with open(
        RESULTS_PATH / "training_summary.json", "w", encoding="utf-8"
    ) as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    print(results_df.to_string(index=False))
    print(f"\nBest model by CV MAE: {best_name}")
    print(f"Best Ridge alpha: {ridge_search.best_params_['model__alpha']}")
    print(f"Saved model to: {MODEL_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Saved results to: {RESULTS_PATH.relative_to(PROJECT_ROOT)}")
    return results_df


if __name__ == "__main__":
    train_models()
