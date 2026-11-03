# BÁO CÁO SO SÁNH MÔ HÌNH KNN

## 1. Mục tiêu

Đánh giá và so sánh hiệu quả của mô hình K-Nearest Neighbors (KNN) khi sử dụng toàn bộ đặc trưng và khi sử dụng các đặc trưng được lựa chọn bằng phương pháp Mutual Information.

## 2. Kết quả thực nghiệm

| Chỉ số | KNN Full Features | KNN Selected Features |
|---|---:|---:|
| CV MAE | 1.4536 | 1.2915 |
| Test MAE | 1.3806 | 1.2833 |
| Test RMSE | 2.3464 | 2.2821 |
| Test R² | 0.6105 | 0.6315 |
| Số đặc trưng | Toàn bộ | 20 |

## 3. Tham số tối ưu

Cả hai mô hình đều sử dụng các tham số:

- Số láng giềng K: 15
- Trọng số: distance
- Phương pháp tính khoảng cách: Euclidean (p = 2)

Mô hình KNN Selected Features sử dụng thêm tham số selector__k = 20, tức là lựa chọn 20 đặc trưng đầu vào.

## 4. Nhận xét

Kết quả thực nghiệm cho thấy mô hình KNN Selected Features đạt hiệu quả tốt hơn mô hình KNN Full Features.

Test MAE giảm từ 1.3806 xuống 1.2833 và Test RMSE giảm từ 2.3464 xuống 2.2821. Đồng thời, hệ số R² tăng từ 0.6105 lên 0.6315.

Điều này cho thấy việc lựa chọn đặc trưng bằng Mutual Information giúp cải thiện khả năng dự đoán điểm thi của mô hình KNN trên tập dữ liệu thử nghiệm.

## 5. Kết luận

Trong hai mô hình được thử nghiệm, KNN Selected Features là mô hình có kết quả tốt hơn. Vì vậy, nhóm có thể ưu tiên mô hình này trong quá trình tích hợp và đánh giá hệ thống dự đoán kết quả học tập.