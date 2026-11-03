"""Tính permutation importance cho pipeline mô hình cuối."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

from src.config import (
    DATA_PATH,
    FIGURES_PATH,
    MODEL_PATH,
    RANDOM_STATE,
    RESULTS_PATH,
    TARGET_COLUMN,
    TEST_SIZE,
)
from src.data_cleaning import clean_student_data
from src.data_loader import load_data


def explain_model(
    model_path=MODEL_PATH,
    n_repeats: int = 10,
    output_stem: str = "permutation_importance",
) -> pd.DataFrame:
    """Đo mức tăng MAE khi xáo trộn từng cột trên tập test.

    Chỉ số thể hiện đóng góp dự đoán của đặc trưng đối với mô hình và dữ liệu
    hiện tại; không được diễn giải thành quan hệ nhân quả.
    """
    if n_repeats <= 0:
        raise ValueError("n_repeats phải là số nguyên dương")
    if not output_stem or any(char in output_stem for char in "\\/:"):
        raise ValueError("output_stem không hợp lệ")
    if not model_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy mô hình tại {model_path}. Hãy chạy python -m src.train."
        )

    data, _ = clean_student_data(load_data(DATA_PATH))
    X = data.drop(columns=[TARGET_COLUMN])
    y = data[TARGET_COLUMN]
    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    model = joblib.load(model_path)
    importance = permutation_importance(
        model,
        X_test,
        y_test,
        scoring="neg_mean_absolute_error",
        n_repeats=n_repeats,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    result = pd.DataFrame(
        {
            "feature": X_test.columns,
            "importance_mean": importance.importances_mean,
            "importance_std": importance.importances_std,
        }
    ).sort_values("importance_mean", ascending=False, ignore_index=True)

    RESULTS_PATH.mkdir(parents=True, exist_ok=True)
    FIGURES_PATH.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_PATH / f"{output_stem}.csv"
    figure_path = FIGURES_PATH / f"08_{output_stem}.png"
    result.to_csv(csv_path, index=False)

    plot_data = result.head(15).sort_values("importance_mean")
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(
        plot_data["feature"],
        plot_data["importance_mean"],
        xerr=plot_data["importance_std"],
        color="#176b87",
        alpha=0.9,
    )
    ax.set_title("Permutation importance theo đặc trưng")
    ax.set_xlabel("Mức tăng MAE khi xáo trộn đặc trưng")
    ax.set_ylabel("Đặc trưng")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(result.to_string(index=False))
    print(f"Saved table: {csv_path}")
    print(f"Saved figure: {figure_path}")
    return result


if __name__ == "__main__":
    explain_model()
