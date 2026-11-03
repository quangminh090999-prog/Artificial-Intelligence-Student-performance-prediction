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
|   |-- train_svr.py          # Tuning SVR riêng của Người 5
|   |-- evaluate.py           # Đánh giá và so sánh
|   |-- input_schema.py       # Kiểm tra 19 trường và lỗi theo dòng
|   |-- explain_model.py      # Permutation importance
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
python -m src.train
python -m src.evaluate
python -m src.feature_selection
python -m src.knn_model
python -m src.train_svr
streamlit run app.py
pytest
```

Sau khi chạy `python -m src.train`, ứng dụng dùng pipeline tốt nhất của nhóm tại
`models/student_score_pipeline.joblib`. Lệnh `python -m src.train_svr` tạo mô
hình SVR đã tuning tại `models/svr_pipeline.joblib`; có thể chọn mô hình này ở
thanh bên của ứng dụng.

Để tạo bảng và biểu đồ permutation importance cho SVR đã tuning:

```powershell
python -c "from src.config import SVR_MODEL_PATH; from src.explain_model import explain_model; explain_model(SVR_MODEL_PATH, output_stem='svr_permutation_importance')"
```

## Dự đoán CSV trên Streamlit

Tệp CSV phải có đủ 19 cột đầu vào với tên giống dữ liệu huấn luyện. Ứng dụng:

- kiểm tra cột thiếu, kiểu số, miền giá trị và danh mục;
- dự đoán những dòng hợp lệ;
- giữ lại dòng lỗi kèm cột `Validation_Error`;
- cho phép tải kết quả dưới dạng CSV.

Có thể thử ngay bằng tệp mẫu `data/sample_batch_input.csv`.

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
