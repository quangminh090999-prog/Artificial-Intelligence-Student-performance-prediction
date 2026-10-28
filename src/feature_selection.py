
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
from sklearn.pipeline import Pipeline

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


def run_feature_selection():
    """Tính MI, chọn đặc trưng và lưu kết quả."""

    # 1. Đọc dữ liệu và tách biến đầu vào, biến mục tiêu
    df = load_data(DATA_PATH)
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN].clip(upper=100)

    # 2. Chia train/test theo cấu hình chung
    X_train, X_test, y_train, y_test = train_test_split(
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

    # 5. Tính Mutual Information trên train
    mi_scores = mutual_info_regression(
        X_train_processed,
        y_train,
        random_state=RANDOM_STATE,
    )


    # Tạo bảng điểm MI theo đúng thứ tự đặc trưng ban đầu
    scores_df = pd.DataFrame({
        "feature": feature_names,
        "mutual_information": mi_scores,
    })

    # Chọn tối đa 10 đặc trưng có điểm MI cao nhất
    k = min(10, len(feature_names))
    selector = SelectKBest(
        score_func=mutual_info_regression,
        k=k,
    )
    selector.fit(X_train_processed, y_train)

    # Gán kết quả chọn trước khi sắp xếp bảng
    scores_df["selected"] = selector.get_support()

    # Sắp xếp để dễ đọc và lấy danh sách đặc trưng đã chọn
    scores_df = scores_df.sort_values(
        "mutual_information",
        ascending=False,
    )

    selected_features = scores_df.loc[
        scores_df["selected"], "feature"
    ].tolist()
