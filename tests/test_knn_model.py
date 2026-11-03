
"""Kiểm thử cơ bản cho mô hình KNN."""

import numpy as np
from sklearn.neighbors import KNeighborsRegressor


def test_knn_can_train_and_predict():
    """KNN phải huấn luyện và dự đoán được dữ liệu mẫu."""

    # Dữ liệu mẫu: 4 sinh viên, mỗi sinh viên có 2 đặc trưng
    X_train = np.array([
        [1, 80],
        [2, 85],
        [3, 90],
        [4, 95],
    ])

    # Điểm thi tương ứng
    y_train = np.array([60, 70, 80, 90])

    # Tạo mô hình KNN
    model = KNeighborsRegressor(
        n_neighbors=3,
        weights="distance",
        p=2,
    )

    # Huấn luyện mô hình
    model.fit(X_train, y_train)

    # Dự đoán điểm cho sinh viên mới
    X_test = np.array([[2, 86]])
    predictions = model.predict(X_test)

    # Kiểm tra kết quả dự đoán
    assert predictions.shape == (1,)
    assert np.isfinite(predictions).all()
    assert 60 <= predictions[0] <= 90
