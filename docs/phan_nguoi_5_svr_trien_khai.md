# Phần Người 5 SVR suy luận AI và Streamlit

## 1 Mục tiêu và phạm vi

Phần này xây dựng Support Vector Regression, kiểm tra dữ liệu đầu vào và triển
khai dự đoán qua Streamlit. Toàn bộ tiền xử lý nằm trong Pipeline để imputation,
scaling và One Hot Encoding chỉ được fit từ dữ liệu train. Tập test chỉ được sử
dụng một lần sau khi GridSearchCV đã chọn xong tham số.

## 2 Tối ưu Support Vector Regression

Module `src/train_svr.py` sử dụng 5 fold cross validation với thước đo tối ưu là
MAE. Không gian tìm kiếm gồm:

- kernel: RBF và linear;
- C: 0.1, 1, 10 và 100 tùy kernel;
- epsilon: 0.05, 0.1 và 0.2;
- gamma: scale và auto cho kernel RBF.

Kết quả tốt nhất hiện tại:

| Chỉ số | Giá trị |
|---|---:|
| Kernel | linear |
| C | 10.0 |
| Epsilon | 0.2 |
| CV MAE | 0.4540 |
| Test MAE | 0.4063 |
| Test RMSE | 1.8003 |
| Test R2 | 0.7707 |

SVR đã tuning có Test MAE thấp hơn Ridge hiện tại, nhưng quyết định chọn mô
hình cuối phải dựa thêm vào độ ổn định, khả năng giải thích và kết quả tổng hợp
của Người 4. Mô hình SVR được lưu riêng, không ghi đè pipeline chính thức.

## 3 Kiểm tra dữ liệu và suy luận

Module `src/input_schema.py` kiểm tra đủ 19 cột, tên cột trùng, giá trị thiếu,
kiểu số, miền số và danh mục phân loại. Hàm dự đoán hỗ trợ hai chế độ:

- `predict_one`: dự đoán một sinh viên;
- `predict_batch`: chỉ trả kết quả khi toàn bộ dữ liệu hợp lệ;
- `predict_batch_with_errors`: dự đoán các dòng hợp lệ và ghi lỗi cho từng
  dòng không hợp lệ.

Cách làm này ngăn dữ liệu ngoài miền hoặc sai danh mục đi vào pipeline mà không
được phát hiện, đồng thời vẫn cho phép người dùng xử lý một tệp CSV có cả dòng
đúng và dòng sai.

## 4 Triển khai Streamlit

Ứng dụng gồm ba luồng chính:

1. Nhập 19 trường và dự đoán một sinh viên.
2. Tải CSV, xem trước, kiểm tra theo dòng, dự đoán hàng loạt và tải kết quả.
3. Xem MAE, RMSE, R2, bảng so sánh mô hình và kết quả SVR tuning.

Khi cả mô hình chính thức và SVR tuned đều có trên máy, người dùng có thể chọn
mô hình ở thanh bên. Nếu chưa có model, giao diện dự đoán đơn vẫn có chế độ minh
họa nhưng ghi rõ đây không phải kết quả AI.

## 5 Giải thích mô hình

Permutation importance đo mức tăng MAE khi xáo trộn từng đặc trưng trên tập
test. Với SVR tuned, hai đặc trưng nổi bật nhất là Attendance và Hours Studied.
Đây là mức đóng góp cho dự đoán của mô hình trên tập dữ liệu hiện tại, không
chứng minh quan hệ nhân quả.

## 6 Kiểm thử và khả năng tái lập

Bộ test kiểm tra thứ tự 19 cột, cột thiếu, sai kiểu, ngoài miền, danh mục mới,
lỗi theo từng dòng, dự đoán hàng loạt và cấu hình GridSearchCV. Các artifact gồm
CSV kết quả CV, CSV chỉ số test, pipeline SVR, bảng permutation importance và
biểu đồ tương ứng.

## 7 Hạn chế và hướng phát triển

- Dữ liệu có quy mô vừa và chỉ phản ánh một tập quan sát; cần kiểm tra trên dữ
  liệu của kỳ học hoặc trường khác.
- Batch prediction hiện xử lý CSV trong bộ nhớ; tệp rất lớn cần chia theo chunk.
- Schema đang dùng miền giá trị của dự án; khi nguồn dữ liệu thay đổi phải cập
  nhật quy tắc và giao diện đồng thời.
- Có thể bổ sung hiệu chỉnh khoảng dự đoán, theo dõi drift và triển khai trên
  dịch vụ web sau khi nhóm chốt mô hình cuối.
