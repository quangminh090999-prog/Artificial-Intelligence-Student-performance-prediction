# Phần Người 4 - Mô hình nâng cao và đánh giá chung

## Phương pháp

- Tách train/test theo cùng `random_state=42` và tỷ lệ 80/20 của toàn nhóm.
- Tiền xử lý nằm trong pipeline của từng fold để tránh rò rỉ dữ liệu.
- Tuning Random Forest và Gradient Boosting bằng RandomizedSearchCV 5-fold, tối ưu MAE.
- Đánh giá bằng CV MAE, Test MAE, RMSE, R2, thời gian và độ lệch train-test.

## Kết quả mô hình nâng cao

| Mô hình | CV MAE | Train MAE | Test MAE | RMSE | R2 | Chênh lệch MAE |
|---|---:|---:|---:|---:|---:|---:|
| GradientBoosting_Tuned | 0.7433 | 0.6007 | 0.6695 | 1.9057 | 0.7431 | 0.0688 |
| RandomForest_Tuned | 1.1443 | 0.7185 | 1.0677 | 2.1114 | 0.6846 | 0.3492 |

## Tham số tốt nhất

- **GradientBoosting_Tuned:** `subsample=1.0`, `n_estimators=250`, `min_samples_leaf=2`, `max_depth=3`, `learning_rate=0.1`.
- **RandomForest_Tuned:** `n_estimators=250`, `min_samples_split=10`, `min_samples_leaf=1`, `max_features=0.7`, `max_depth=12`.

## Lựa chọn mô hình cuối

Mô hình được chọn là **SVR_Tuned** với Test MAE 0.4063, RMSE 1.8003 và R2 0.7707.
Kết luận này dựa trên bảng so sánh chung; không chọn mô hình chỉ vì điểm train cao.

## Phân tích overfitting

Khoảng cách Train MAE và Test MAE càng lớn thì dấu hiệu overfitting càng rõ. Các biểu đồ phần dư và dự đoán-thực tế được dùng để phát hiện sai số có hệ thống.

## Tệp minh chứng

- `reports/results/advanced_model_cv_results.csv`
- `reports/results/advanced_model_metrics.csv`
- `reports/results/advanced_overfitting_analysis.csv`
- `reports/results/final_model_selection.csv`
- `reports/figures/10_advanced_predicted_vs_actual.png`
- `reports/figures/11_advanced_residual_distribution.png`
- `reports/figures/12_advanced_overfitting_gap.png`
- `reports/figures/13_all_models_comparison.png`

## Hạn chế

R2 khoảng 0,77 là phù hợp với dữ liệu có nhiễu và các yếu tố con người chưa được quan sát. Kết quả dự đoán chỉ hỗ trợ tham khảo, không thay thế quyết định của giảng viên.
