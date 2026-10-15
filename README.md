# Dự đoán kết quả học tập của sinh viên

Dự án sử dụng các mô hình học máy để dự đoán `Exam_Score` từ những yếu tố như
thời gian học, chuyên cần, điểm trước đây, số buổi phụ đạo và điều kiện học tập.

## Mục tiêu

- Khảo sát và trực quan hóa dữ liệu học tập.
- Xây dựng baseline và nhiều mô hình hồi quy để so sánh.
- Đánh giá bằng MAE, RMSE và R² trên cùng một tập kiểm tra.
- Chọn mô hình tốt nhất và xây dựng giao diện Streamlit để dự đoán thử.
- Lưu đầy đủ hình ảnh, bảng kết quả và nội dung phục vụ báo cáo/thuyết trình.

## Cấu trúc thư mục

```text
.
|-- app.py                    # Giao diện Streamlit
|-- requirements.txt          # Thư viện Python
|-- data/
|   |-- raw/                  # Dữ liệu gốc (không đưa CSV lên Git)
|   `-- processed/            # Dữ liệu sau xử lý
|-- models/                   # Mô hình đã huấn luyện
|-- notebooks/                # Notebook EDA và thực nghiệm
|-- reports/
|   |-- figures/              # Biểu đồ dùng trong báo cáo
|   `-- results/              # Bảng chỉ số đánh giá
|-- src/
|   |-- config.py             # Đường dẫn và tham số dùng chung
|   |-- data_loader.py        # Đọc và kiểm tra dữ liệu
|   |-- preprocessing.py      # Tiền xử lý dữ liệu
|   |-- train.py              # Huấn luyện các mô hình
|   |-- evaluate.py           # Đánh giá và so sánh
|   `-- predict.py            # Nạp mô hình và dự đoán
`-- tests/                    # Kiểm thử tự động
```

## Chuẩn bị môi trường

Yêu cầu Python 3.10 trở lên.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Sao chép tệp `StudentPerformanceFactors.csv` vào `data/raw/`. Tệp CSV được loại
khỏi Git để repository nhẹ và tránh đăng dữ liệu khi chưa xác nhận giấy phép.

## Chạy dự án

```powershell
python -m src.data_loader
streamlit run app.py
pytest
```

Các lệnh huấn luyện và đánh giá sẽ được bổ sung khi từng thành viên hoàn thành
module được giao.

## Phân công 5 thành viên

1. Thành viên 1: kiểm tra dữ liệu, baseline, tích hợp và quản lý nhánh `main`.
2. Thành viên 2: EDA, biểu đồ và phần phân tích dữ liệu trong báo cáo.
3. Thành viên 3: tiền xử lý, Linear Regression và Ridge Regression.
4. Thành viên 4: Random Forest, Gradient Boosting và so sánh mô hình.
5. Thành viên 5: SVR, hàm dự đoán, giải thích mô hình và giao diện Streamlit.

## Quy trình Git của nhóm

```powershell
git switch main
git pull --ff-only origin main
git switch -c feature/ten-nhiem-vu
# sửa file, sau đó:
git add .
git commit -m "Mô tả ngắn gọn thay đổi"
git push -u origin feature/ten-nhiem-vu
```

Không đưa dữ liệu CSV, môi trường `.venv` hoặc file mô hình lớn lên GitHub.
