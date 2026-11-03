"""Tuning mô hình cây nâng cao và tổng hợp đánh giá chung của dự án."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import src.config as _cfg
from src.data_cleaning import clean_student_data
from src.data_loader import load_data
from src.preprocessing import build_preprocessor


CV_FOLDS = _cfg.CV_FOLDS
FIGURES_PATH = _cfg.FIGURES_PATH
MODELS_PATH = _cfg.MODEL_PATH.parent
RANDOM_STATE = _cfg.RANDOM_STATE
RESULTS_PATH = _cfg.RESULTS_PATH
TARGET_COLUMN = _cfg.TARGET_COLUMN
TEST_SIZE = _cfg.TEST_SIZE

RF_PARAM_DISTRIBUTIONS = {
    "model__n_estimators": [150, 250, 400],
    "model__max_depth": [None, 8, 12, 18],
    "model__min_samples_split": [2, 5, 10],
    "model__min_samples_leaf": [1, 2, 4],
    "model__max_features": ["sqrt", 0.7, 1.0],
}

GB_PARAM_DISTRIBUTIONS = {
    "model__n_estimators": [100, 150, 250],
    "model__learning_rate": [0.03, 0.05, 0.1],
    "model__max_depth": [2, 3, 4],
    "model__min_samples_leaf": [2, 4, 8],
    "model__subsample": [0.8, 1.0],
}


def _parameter_space_size(distributions: dict[str, list]) -> int:
    """Tính số cấu hình rời rạc có trong không gian tham số."""
    return math.prod(len(values) for values in distributions.values())


def _make_pipeline(model) -> Pipeline:
    return Pipeline(
        [
            ("preprocessor", build_preprocessor()),
            ("model", model),
        ]
    )


def _metrics(y_true, predictions) -> tuple[float, float, float]:
    return (
        float(mean_absolute_error(y_true, predictions)),
        float(mean_squared_error(y_true, predictions) ** 0.5),
        float(r2_score(y_true, predictions)),
    )


def _stable_cv_results(search: RandomizedSearchCV, model_name: str) -> pd.DataFrame:
    raw = pd.DataFrame(search.cv_results_)
    param_columns = [column for column in raw.columns if column.startswith("param_")]
    stable = raw[
        param_columns
        + [
            "mean_train_score",
            "std_train_score",
            "mean_test_score",
            "std_test_score",
            "mean_fit_time",
            "rank_test_score",
        ]
    ].copy()
    stable.insert(0, "model", model_name)
    stable["mean_train_mae"] = -stable.pop("mean_train_score")
    stable["std_train_mae"] = stable.pop("std_train_score")
    stable["mean_cv_mae"] = -stable.pop("mean_test_score")
    stable["std_cv_mae"] = stable.pop("std_test_score")
    stable = stable.rename(columns={"rank_test_score": "rank"})
    return stable.sort_values("rank")


def _save_diagnostic_figures(
    evaluation_rows: list[dict],
    all_models: pd.DataFrame,
) -> None:
    FIGURES_PATH.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for axis, row in zip(axes, evaluation_rows):
        axis.scatter(row["y_test"], row["test_predictions"], alpha=0.45, s=18)
        lower = min(float(np.min(row["y_test"])), float(np.min(row["test_predictions"])))
        upper = max(float(np.max(row["y_test"])), float(np.max(row["test_predictions"])))
        axis.plot([lower, upper], [lower, upper], "--", color="#c0392b", linewidth=1.5)
        axis.set_title(row["model"])
        axis.set_xlabel("Điểm thực tế")
        axis.set_ylabel("Điểm dự đoán")
        axis.grid(alpha=0.2)
    fig.suptitle("Dự đoán so với điểm thực tế")
    fig.tight_layout()
    fig.savefig(FIGURES_PATH / "10_advanced_predicted_vs_actual.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for axis, row in zip(axes, evaluation_rows):
        residuals = np.asarray(row["y_test"]) - np.asarray(row["test_predictions"])
        axis.hist(residuals, bins=28, color="#176b87", alpha=0.85, edgecolor="white")
        axis.axvline(0, linestyle="--", color="#c0392b", linewidth=1.5)
        axis.set_title(row["model"])
        axis.set_xlabel("Phần dư: thực tế - dự đoán")
        axis.set_ylabel("Số quan sát")
        axis.grid(axis="y", alpha=0.2)
    fig.suptitle("Phân phối phần dư của mô hình nâng cao")
    fig.tight_layout()
    fig.savefig(FIGURES_PATH / "11_advanced_residual_distribution.png", dpi=180)
    plt.close(fig)

    overfit = pd.DataFrame(
        {
            "model": [row["model"] for row in evaluation_rows],
            "Train MAE": [row["train_mae"] for row in evaluation_rows],
            "Test MAE": [row["test_mae"] for row in evaluation_rows],
        }
    ).set_index("model")
    axis = overfit.plot(kind="bar", figsize=(9, 5), color=["#5dade2", "#e67e22"])
    axis.set_title("So sánh Train MAE và Test MAE")
    axis.set_ylabel("MAE - thấp hơn là tốt hơn")
    axis.set_xlabel("")
    axis.tick_params(axis="x", rotation=0)
    axis.grid(axis="y", alpha=0.2)
    axis.figure.tight_layout()
    axis.figure.savefig(FIGURES_PATH / "12_advanced_overfitting_gap.png", dpi=180)
    plt.close(axis.figure)

    comparison = all_models.sort_values("MAE").head(10).copy()
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    colors = ["#176b87" if bool(value) else "#9fb3c8" for value in comparison["selected"]]
    for axis, metric in zip(axes, ["MAE", "RMSE", "R2"]):
        axis.barh(comparison["model"], comparison[metric], color=colors)
        axis.invert_yaxis()
        axis.set_title(metric)
        axis.grid(axis="x", alpha=0.2)
    fig.suptitle("So sánh các mô hình và lựa chọn mô hình triển khai")
    fig.tight_layout()
    fig.savefig(FIGURES_PATH / "13_all_models_comparison.png", dpi=180)
    plt.close(fig)


def choose_final_model(advanced_metrics: pd.DataFrame) -> pd.DataFrame:
    """Gộp kết quả toàn dự án và đánh dấu mô hình có Test MAE thấp nhất."""
    frames = []
    existing_path = RESULTS_PATH / "all_models_summary.csv"
    if existing_path.exists():
        existing = pd.read_csv(existing_path)
        required = ["model", "MAE", "RMSE", "R2"]
        if all(column in existing.columns for column in required):
            frames.append(existing[required])

    frames.append(advanced_metrics[["model", "MAE", "RMSE", "R2"]])
    combined = (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates(subset="model", keep="last")
        .sort_values(["MAE", "RMSE"], ascending=True)
        .reset_index(drop=True)
    )
    combined.insert(0, "rank", range(1, len(combined) + 1))
    combined["selected"] = False
    combined.loc[0, "selected"] = True
    combined["selection_reason"] = ""
    combined.loc[0, "selection_reason"] = (
        "Test MAE thấp nhất; kiểm tra thêm CV, overfitting và khả năng triển khai."
    )
    combined.to_csv(RESULTS_PATH / "final_model_selection.csv", index=False)
    return combined


def _write_report(metrics: pd.DataFrame, selection: pd.DataFrame) -> None:
    best = selection.iloc[0]
    lines = [
        "# Phần Người 4 - Mô hình nâng cao và đánh giá chung",
        "",
        "## Phương pháp",
        "",
        "- Tách train/test theo cùng `random_state=42` và tỷ lệ 80/20 của toàn nhóm.",
        "- Tiền xử lý nằm trong pipeline của từng fold để tránh rò rỉ dữ liệu.",
        "- Tuning Random Forest và Gradient Boosting bằng RandomizedSearchCV 5-fold, tối ưu MAE.",
        "- Đánh giá bằng CV MAE, Test MAE, RMSE, R2, thời gian và độ lệch train-test.",
        "",
        "## Kết quả mô hình nâng cao",
        "",
        "| Mô hình | CV MAE | Train MAE | Test MAE | RMSE | R2 | Chênh lệch MAE |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in metrics.itertuples(index=False):
        lines.append(
            f"| {row.model} | {row.CV_MAE:.4f} | {row.train_MAE:.4f} | "
            f"{row.MAE:.4f} | {row.RMSE:.4f} | {row.R2:.4f} | "
            f"{row.MAE_gap:.4f} |"
        )
    lines.extend(["", "## Tham số tốt nhất", ""])
    for row in metrics.itertuples(index=False):
        params = json.loads(row.best_params)
        readable = ", ".join(f"`{key}={value}`" for key, value in params.items())
        lines.append(f"- **{row.model}:** {readable}.")
    lines.extend(
        [
            "",
            "## Lựa chọn mô hình cuối",
            "",
            f"Mô hình được chọn là **{best['model']}** với Test MAE "
            f"{best['MAE']:.4f}, RMSE {best['RMSE']:.4f} và R2 {best['R2']:.4f}.",
            "Kết luận này dựa trên bảng so sánh chung; không chọn mô hình chỉ vì điểm train cao.",
            "",
            "## Phân tích overfitting",
            "",
            "Khoảng cách Train MAE và Test MAE càng lớn thì dấu hiệu overfitting càng rõ. "
            "Các biểu đồ phần dư và dự đoán-thực tế được dùng để phát hiện sai số có hệ thống.",
            "",
            "## Tệp minh chứng",
            "",
            "- `reports/results/advanced_model_cv_results.csv`",
            "- `reports/results/advanced_model_metrics.csv`",
            "- `reports/results/advanced_overfitting_analysis.csv`",
            "- `reports/results/final_model_selection.csv`",
            "- `reports/figures/10_advanced_predicted_vs_actual.png`",
            "- `reports/figures/11_advanced_residual_distribution.png`",
            "- `reports/figures/12_advanced_overfitting_gap.png`",
            "- `reports/figures/13_all_models_comparison.png`",
            "",
            "## Hạn chế",
            "",
            "R2 khoảng 0,77 là phù hợp với dữ liệu có nhiễu và các yếu tố con người chưa được quan sát. "
            "Kết quả dự đoán chỉ hỗ trợ tham khảo, không thay thế quyết định của giảng viên.",
        ]
    )
    report_path = _cfg.PROJECT_ROOT / "docs" / "phan_nguoi_4_mo_hinh_nang_cao.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def train_advanced_models(n_iter: int = 12) -> pd.DataFrame:
    """Tuning RF/GB, đánh giá overfitting và lưu toàn bộ sản phẩm Người 4."""
    data, _ = clean_student_data(load_data())
    X = data.drop(columns=[TARGET_COLUMN])
    y = data[TARGET_COLUMN]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    specifications = [
        (
            "RandomForest_Tuned",
            RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=1),
            RF_PARAM_DISTRIBUTIONS,
            MODELS_PATH / "random_forest_tuned.joblib",
        ),
        (
            "GradientBoosting_Tuned",
            GradientBoostingRegressor(random_state=RANDOM_STATE),
            GB_PARAM_DISTRIBUTIONS,
            MODELS_PATH / "gradient_boosting_tuned.joblib",
        ),
    ]

    metric_rows = []
    cv_frames = []
    evaluation_rows = []
    MODELS_PATH.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.mkdir(parents=True, exist_ok=True)

    for name, estimator, distributions, model_path in specifications:
        iterations = min(n_iter, _parameter_space_size(distributions))
        search = RandomizedSearchCV(
            _make_pipeline(estimator),
            param_distributions=distributions,
            n_iter=iterations,
            scoring="neg_mean_absolute_error",
            cv=CV_FOLDS,
            random_state=RANDOM_STATE,
            # Chạy tuần tự để ổn định trên Windows và môi trường có đường dẫn Unicode.
            # Hai model vẫn sử dụng đúng cùng fold và cùng không gian tìm kiếm.
            n_jobs=1,
            return_train_score=True,
        )
        started = time.perf_counter()
        search.fit(X_train, y_train)
        search_seconds = time.perf_counter() - started

        pipeline = search.best_estimator_
        train_predictions = pipeline.predict(X_train)
        predict_started = time.perf_counter()
        test_predictions = pipeline.predict(X_test)
        predict_seconds = time.perf_counter() - predict_started

        train_mae, train_rmse, train_r2 = _metrics(y_train, train_predictions)
        test_mae, test_rmse, test_r2 = _metrics(y_test, test_predictions)
        best_index = int(search.best_index_)
        best_params = {
            key.replace("model__", ""): value
            for key, value in search.best_params_.items()
        }
        metric_rows.append(
            {
                "model": name,
                "CV_MAE": float(-search.best_score_),
                "CV_MAE_std": float(search.cv_results_["std_test_score"][best_index]),
                "train_MAE": train_mae,
                "train_RMSE": train_rmse,
                "train_R2": train_r2,
                "MAE": test_mae,
                "RMSE": test_rmse,
                "R2": test_r2,
                "MAE_gap": test_mae - train_mae,
                "R2_gap": train_r2 - test_r2,
                "search_seconds": search_seconds,
                "predict_seconds": predict_seconds,
                "best_params": json.dumps(best_params, ensure_ascii=False),
            }
        )
        cv_frames.append(_stable_cv_results(search, name))
        evaluation_rows.append(
            {
                "model": name,
                "y_test": y_test,
                "test_predictions": test_predictions,
                "train_mae": train_mae,
                "test_mae": test_mae,
            }
        )
        joblib.dump(pipeline, model_path)

    metrics = pd.DataFrame(metric_rows).sort_values("MAE")
    metrics.to_csv(RESULTS_PATH / "advanced_model_metrics.csv", index=False)
    pd.concat(cv_frames, ignore_index=True).to_csv(
        RESULTS_PATH / "advanced_model_cv_results.csv", index=False
    )
    metrics[
        ["model", "train_MAE", "MAE", "MAE_gap", "train_R2", "R2", "R2_gap"]
    ].to_csv(RESULTS_PATH / "advanced_overfitting_analysis.csv", index=False)

    selection = choose_final_model(metrics)
    _save_diagnostic_figures(evaluation_rows, selection)
    _write_report(metrics, selection)

    print(metrics.to_string(index=False))
    # Dùng thông báo ASCII để không lỗi trên PowerShell dùng code page cp1252.
    print(f"\nSelected deployment model: {selection.iloc[0]['model']}")
    return metrics


if __name__ == "__main__":
    train_advanced_models()
