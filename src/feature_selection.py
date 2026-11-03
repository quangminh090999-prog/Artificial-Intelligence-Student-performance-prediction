
"""Feature selection bằng Mutual Information và SelectKBest."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_selection import (
    SelectKBest,
    mutual_info_regression,
)
from sklearn.model_selection import train_test_split

from src.config import (
    DATA_PATH,
    FIGURES_PATH,
    RANDOM_STATE,
    RESULTS_PATH,
    TARGET_COLUMN,
    TEST_SIZE,
)
from src.data_loader import load_data
from src.preprocessing import build_preprocessor


def mutual_info_with_random_state(X, y):
    """Tính Mutual Information với seed cố định để tái lập kết quả."""
    return mutual_info_regression(X, y, random_state=RANDOM_STATE)


def run_feature_selection(top_k: int = 10) -> pd.DataFrame:
    """Tính MI trên tập train, chọn đặc trưng và lưu bảng/biểu đồ.

    Parameters
    ----------
    top_k:
        Số đặc trưng sau tiền xử lý cần chọn. Nếu lớn hơn tổng số đặc
        trưng hiện có, hàm sẽ tự dùng toàn bộ đặc trưng.

    Returns
    -------
    pd.DataFrame
        Bảng điểm MI đã sắp xếp giảm dần.
    """
    if top_k <= 0:
        raise ValueError("top_k phải là số nguyên dương")

    # 1. Đọc dữ liệu và tách biến đầu vào, biến mục tiêu
    df = load_data(DATA_PATH)
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN].clip(upper=100)

    # 2. Chia train/test theo cấu hình chung
    X_train, X_test, y_train, _ = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    print(f"Số mẫu train: {len(X_train)}")
    print(f"Số mẫu test: {len(X_test)}")

    # 3. Chỉ fit bộ tiền xử lý trên train
    preprocessor = build_preprocessor()
    X_train_processed = preprocessor.fit_transform(X_train)

    # 4. Lấy tên đặc trưng sau One-Hot Encoding
    feature_names = preprocessor.get_feature_names_out()

    # 5. Tính MI và chọn đặc trưng chỉ từ tập train để tránh rò rỉ dữ liệu
    k = min(top_k, len(feature_names))
    selector = SelectKBest(
        score_func=mutual_info_with_random_state,
        k=k,
    )
    selector.fit(X_train_processed, y_train)

    # Tạo bảng theo đúng thứ tự đặc trưng sau tiền xử lý
    scores_df = pd.DataFrame(
        {
            "feature": feature_names,
            "mutual_information": selector.scores_,
            "selected": selector.get_support(),
        }
    )

    # Sắp xếp để dễ đọc và lấy danh sách đặc trưng đã chọn
    scores_df = scores_df.sort_values(
        "mutual_information",
        ascending=False,
    ).reset_index(drop=True)

    selected_features = scores_df.loc[
        scores_df["selected"], "feature"
    ].tolist()

    # 6. Lưu bảng điểm và danh sách đặc trưng được chọn
    RESULTS_PATH.mkdir(parents=True, exist_ok=True)
    FIGURES_PATH.mkdir(parents=True, exist_ok=True)

    scores_path = RESULTS_PATH / "feature_scores.csv"
    selected_path = RESULTS_PATH / "selected_features.csv"
    figure_path = FIGURES_PATH / "07_feature_importance_mi.png"

    scores_df.to_csv(scores_path, index=False)
    pd.DataFrame({"selected_feature": selected_features}).to_csv(
        selected_path,
        index=False,
    )

    # 7. Vẽ các đặc trưng có điểm MI cao nhất
    plot_df = scores_df.head(min(15, len(scores_df))).sort_values(
        "mutual_information"
    )
    colors = [
        "#1f77b4" if selected else "#b8c2cc"
        for selected in plot_df["selected"]
    ]

    fig, ax = plt.subplots(figsize=(11, 7))
    ax.barh(
        plot_df["feature"],
        plot_df["mutual_information"],
        color=colors,
    )
    ax.set_title("Mức độ quan trọng của đặc trưng theo Mutual Information")
    ax.set_xlabel("Mutual Information")
    ax.set_ylabel("Đặc trưng sau tiền xử lý")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Đã chọn {len(selected_features)} đặc trưng:")
    for feature in selected_features:
        print(f"- {feature}")
    print("\nĐã lưu kết quả:")
    print(scores_path)
    print(selected_path)
    print(figure_path)

    return scores_df


if __name__ == "__main__":
    run_feature_selection()
