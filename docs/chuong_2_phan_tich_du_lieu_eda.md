# CHƯƠNG 2: PHÂN TÍCH KHÁM PHÁ DỮ LIỆU (EDA) VÀ ĐỀ XUẤT QUY TẮC TIỀN XỬ LÝ

**Thực hiện:** Thành viên 2 – Phân tích dữ liệu và trực quan hóa (20%)  
**Đối tượng bàn giao:** Người 1 (Data Validation), Người 3 (Preprocessing), Người 4 (Model Comparison), Người 5 (Deployment).

---

## 2.1. Tổng quan Bộ dữ liệu Thực nghiệm

Bộ dữ liệu `StudentPerformanceFactors.csv` được sử dụng để xây dựng mô hình dự đoán kết quả thi học kỳ của sinh viên.
* **Quy mô tập dữ liệu:** $6,607$ dòng (bản ghi sinh viên) và $20$ cột thuộc tính.
* **Số dòng trùng lặp:** $0$ dòng (dữ liệu không có bản ghi trùng lặp hoàn toàn).
* **Cấu trúc thuộc tính:**
  * **Biến mục tiêu (Target Variable):** `Exam_Score` (biến liên tục dạng số nguyên, thang điểm học tập).
  * **Biến dự đoán (Features):** $19$ đặc trưng bao gồm:
    * **6 biến định lượng (Numeric Features):** `Hours_Studied`, `Attendance`, `Sleep_Hours`, `Previous_Scores`, `Tutoring_Sessions`, `Physical_Activity`.
    * **13 biến định tính (Categorical Features):** `Parental_Involvement`, `Access_to_Resources`, `Extracurricular_Activities`, `Motivation_Level`, `Internet_Access`, `Family_Income`, `Teacher_Quality`, `School_Type`, `Peer_Influence`, `Learning_Disabilities`, `Parental_Education_Level`, `Distance_from_Home`, `Gender`.

Toàn bộ thông tin mô tả chi tiết từng cột được tổng hợp và lưu trữ tại tệp `reports/data_summary.csv`.

---

## 2.2. Phân tích Biến Mục tiêu (`Exam_Score`) và Điểm Dị Biệt

### 2.2.1. Phân bố điểm số
* **Điểm trung bình (Mean):** $67.23$ điểm.
* **Điểm trung vị (Median):** $67.00$ điểm.
* **Độ lệch chuẩn (Std):** $3.93$ điểm.
* **Giá trị nhỏ nhất (Min):** $55$ điểm.
* **Giá trị lớn nhất (Max):** $101$ điểm.

Biểu đồ phân bố tần số (Histogram kết hợp đường ước lượng mật độ hạt nhân - KDE tại `reports/figures/01_exam_score_distribution.png`) cho thấy phân bố của điểm thi mang hình chuông cân đối (gần với phân bố chuẩn Gauss). Điểm tập trung dày đặc nhất ở khoảng 62 đến 72 điểm.

### 2.2.2. Phát hiện và xử lý điểm bất thường (Điểm 101)
* **Hiện tượng:** Bộ dữ liệu xuất hiện **01 bản ghi duy nhất có `Exam_Score = 101`**.
* **Đánh giá bản chất:** Thang điểm chuẩn của hầu hết hệ thống giáo dục tối đa là 100 điểm. Giá trị 101 có thể xuất phát từ:
  1. Sai sót nhập liệu (typo của người nhập dữ liệu);
  2. Lỗi làm tròn số thập phân (ví dụ từ 100.5 làm tròn lên);
  3. Cộng điểm thưởng vượt khung (bonus point).
* **Đề xuất xử lý (Bàn giao Mốc H1):**
  * Không nên giữ nguyên giá trị 101 vì sẽ gây sai lệch nhỏ cho hàm tối ưu bình phương tối thiểu (MSE/RMSE) của các mô hình tuyến tính và SVR.
  * Nhóm thống nhất đề xuất phương án **giới hạn trần (clip) về 100 điểm**: `df.loc[df['Exam_Score'] > 100, 'Exam_Score'] = 100`. Phương án này vừa giữ lại toàn bộ 19 thông tin đặc trưng giá trị của sinh viên đó, vừa đưa giá trị về đúng miền xác định chuẩn $[0, 100]$.

---

## 2.3. Phân tích Dữ liệu Thiếu (Missing Values)

Qua rà soát trên toàn bộ $6,607$ bản ghi, tập dữ liệu có chất lượng thu thập khá tốt, chỉ có **3 thuộc tính xuất hiện giá trị bị thiếu (NaN)** (minh họa tại `reports/figures/02_missing_values_summary.png`):

1. `Parental_Education_Level`: Thiếu $90$ dòng (tỷ lệ **$1.36\%$**).
2. `Teacher_Quality`: Thiếu $78$ dòng (tỷ lệ **$1.18\%$**).
3. `Distance_from_Home`: Thiếu $67$ dòng (tỷ lệ **$1.01\%$**).

### Nhận xét & Đề xuất kỹ thuật cho Người 3:
* Cả 3 cột thiếu đều là biến phân loại dạng chuỗi (`object`/`category`).
* Tỷ lệ thiếu rất thấp ($< 1.5\%$), nằm sâu dưới ngưỡng an toàn $5\%$.
* **Quy tắc tiền xử lý đề xuất:** Tuyệt đối không xóa bỏ các dòng dữ liệu này (vì sẽ làm mất gần 235 mẫu học tập quý giá). Nhóm khuyến nghị Người 3 sử dụng chiến lược **điền bằng giá trị xuất hiện nhiều nhất (Most Frequent / Mode Imputation)** trong Pipeline tiền xử lý (`SimpleImputer(strategy='most_frequent')`). Quá trình điền giá trị thiếu chỉ được fit trên tập Train để tránh rò rỉ dữ liệu (Data Leakage).

---

## 2.4. Phân tích Tương quan và Các Yếu tố Ảnh hưởng Trọng yếu

### 2.4.1. Ma trận tương quan tuyến tính Pearson ($r$)
Bảng ma trận tương quan giữa 6 biến số định lượng và điểm thi (minh họa tại `reports/figures/03_correlation_heatmap.png`) thể hiện:

| Thứ tự | Thuộc tính định lượng | Hệ số tương quan Pearson ($r$) | Đánh giá mức độ ảnh hưởng |
| :---: | :--- | :---: | :--- |
| **1** | `Attendance` (Chuyên cần) | **$+0.582$** | Tương quan dương mạnh nhất |
| **2** | `Hours_Studied` (Giờ tự học) | **$+0.447$** | Tương quan dương rõ rệt |
| **3** | `Previous_Scores` (Điểm trước đây) | **$+0.187$** | Tương quan thuận nhẹ |
| **4** | `Tutoring_Sessions` (Số buổi phụ đạo) | **$+0.165$** | Tương quan thuận nhẹ |
| **5** | `Physical_Activity` (Vận động) | $+0.024$ | Gần như không có tương quan tuyến tính |
| **6** | `Sleep_Hours` (Thời gian ngủ) | $-0.021$ | Gần như không có tương quan tuyến tính |

### 2.4.2. Lưu ý khoa học: Tương quan không đồng nghĩa với quan hệ nhân quả (Correlation $\neq$ Causation)
Hệ số tương quan Pearson $r = 0.582$ giữa `Attendance` và `Exam_Score` cho thấy sinh viên đi học đầy đủ có xu hướng đạt điểm cao hơn. Tuy nhiên, điều này không đồng nghĩa với việc chỉ cần có mặt tại lớp là sinh viên tự động đạt điểm giỏi. Chuyên cần là một biểu hiện đại diện (proxy) cho tính kỷ luật, mức độ tiếp thu bài giảng liên tục và sự nghiêm túc trong học tập.

### 2.4.3. Quan sát chi tiết qua biểu đồ tán xạ (Scatter & Regplot)
Biểu đồ hồi quy tại `reports/figures/05_hours_attendance_vs_exam_score.png` chỉ ra:
* Đường xu hướng (Trendline) giữa `Hours_Studied` và `Exam_Score` có độ dốc dương rất rõ nét. Sinh viên học từ 20 giờ/tuần trở lên hầu như đạt điểm trên 65.
* Tỷ lệ chuyên cần `Attendance` dưới $60\%$ có mật độ điểm thi dưới mức trung bình rất cao.

---

## 2.5. Khảo sát Ngoại lệ và Phân tích Nhóm Phân loại

### 2.5.1. Phân tích ngoại lệ (Boxplots)
Biểu đồ Boxplot tại `reports/figures/04_numeric_features_boxplots.png` áp dụng tiêu chuẩn khoảng tứ phân vị $IQR = Q_3 - Q_1$:
* Các biến `Hours_Studied` (từ 1 đến 44 giờ/tuần), `Attendance` (từ 60% đến 100%), `Sleep_Hours` (từ 4 đến 10 giờ/ngày) đều nằm trong giới hạn tự nhiên hợp lý của sinh viên, không xuất hiện các giá trị dị biệt âm hoặc vượt ngoài khả năng sinh học.

### 2.5.2. So sánh điểm số theo các nhóm phân loại (Categorical Comparisons)
Biểu đồ cột tại `reports/figures/06_categorical_factors_comparison.png` cho thấy:
* **Motivation Level (Động lực học tập):** Nhóm sinh viên có động lực "High" đạt điểm trung bình cao hơn rõ rệt ($68.5$ điểm) so với nhóm "Low" ($65.8$ điểm).
* **Access to Resources (Tiếp cận tài nguyên):** Sinh viên tiếp cận tài nguyên học tập ở mức "High" có điểm thi trung bình vượt trội ($68.7$ điểm) so với nhóm "Low" ($65.6$ điểm).
* **Teacher Quality (Chất lượng giảng viên):** Giảng viên được đánh giá chất lượng "High" giúp sinh viên đạt điểm trung bình cao hơn khoảng $2.5$ điểm so với nhóm "Low".
* **Parental Involvement (Sự quan tâm của gia đình):** Đồng hành của phụ huynh ở mức "High" tương ứng với mức điểm trung bình cao hơn.

---

## 2.6. Bảng Miền Giá Trị Hợp Lệ Đề Xuất (Bàn Giao Mốc H1)

Nhằm đảm bảo tính toàn vẹn dữ liệu cho toàn bộ hệ thống, Thành viên 2 đã tổng hợp bảng quy tắc kiểm định dữ liệu đầu vào (lưu trữ tại `reports/proposed_input_rules.json`):

### 1. Giới hạn các biến số định lượng
| Tên thuộc tính | Min thực tế | Max thực tế | Miền đề xuất kiểm định | Đơn vị |
| :--- | :---: | :---: | :---: | :--- |
| `Hours_Studied` | 1 | 44 | $[0, 50]$ | Giờ / tuần |
| `Attendance` | 60 | 100 | $[0, 100]$ | $\%$ |
| `Sleep_Hours` | 4 | 10 | $[0, 24]$ | Giờ / ngày |
| `Previous_Scores` | 40 | 100 | $[0, 100]$ | Điểm |
| `Tutoring_Sessions` | 0 | 8 | $[0, 20]$ | Buổi / tháng |
| `Physical_Activity` | 0 | 6 | $[0, 14]$ | Giờ / tuần |

### 2. Danh mục hợp lệ các biến định tính
* `Parental_Involvement`: `["High", "Low", "Medium"]`
* `Access_to_Resources`: `["High", "Low", "Medium"]`
* `Extracurricular_Activities`: `["No", "Yes"]`
* `Motivation_Level`: `["High", "Low", "Medium"]`
* `Internet_Access`: `["No", "Yes"]`
* `Family_Income`: `["High", "Low", "Medium"]`
* `Teacher_Quality`: `["High", "Low", "Medium"]`
* `School_Type`: `["Private", "Public"]`
* `Peer_Influence`: `["Negative", "Neutral", "Positive"]`
* `Learning_Disabilities`: `["No", "Yes"]`
* `Parental_Education_Level`: `["College", "High School", "Postgraduate"]`
* `Distance_from_Home`: `["Far", "Moderate", "Near"]`
* `Gender`: `["Female", "Male"]`

---

## 2.7. Kết luận và Khuyến nghị Kỹ thuật cho Các Thành viên

1. **Gửi Người 1 (Trưởng nhóm - Data Validation):** Tích hợp kiểm tra 19 trường theo đúng bảng quy tắc trên; thực hiện chuyển đổi `Exam_Score == 101` thành `100`.
2. **Gửi Người 3 (Tiền xử lý):** Tập trung điền dữ liệu khuyết thiếu bằng `SimpleImputer(strategy='most_frequent')` cho 3 cột phân loại; áp dụng `OneHotEncoder(handle_unknown='ignore')` để xử lý danh mục mới khi suy luận.
3. **Gửi Người 4 & 5 (Mô hình hóa & Streamlit):** Lưu ý rằng `Attendance` và `Hours_Studied` là hai đặc trưng có trọng số đóng góp dự đoán lớn nhất (có thể kiểm chứng lại bằng Feature Importance / Permutation Importance). Trên form Streamlit, Người 5 sử dụng các ngưỡng slider và selectbox khớp chuẩn xác với danh mục tại Mục 2.6.
