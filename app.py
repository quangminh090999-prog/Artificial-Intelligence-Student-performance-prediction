"""Giao diện Streamlit dự đoán kết quả học tập của sinh viên."""

import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from src.config import MODEL_PATH, RESULTS_PATH, SVR_MODEL_PATH
from src.predict import (
    InputValidationError,
    load_model,
    predict_batch_with_errors,
    predict_one,
)


st.set_page_config(
    page_title="Dự đoán kết quả sinh viên",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(180deg, #f4f8ff 0%, #ffffff 48%);
    }
    .main .block-container {
        max-width: 1180px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }
    .hero {
        padding: 1.6rem 1.8rem;
        margin-bottom: 1.2rem;
        border-radius: 18px;
        color: #ffffff;
        background: linear-gradient(120deg, #123b69 0%, #176b87 100%);
        box-shadow: 0 12px 30px rgba(18, 59, 105, 0.18);
    }
    .hero h1 {
        margin: 0 0 0.45rem 0;
        font-size: 2rem;
    }
    .hero p {
        margin: 0;
        color: #e6f4ff;
        font-size: 1rem;
    }
    .section-title {
        margin: 1.3rem 0 0.35rem 0;
        color: #123b69;
        font-size: 1.15rem;
        font-weight: 700;
    }
    .result-card {
        padding: 1.4rem;
        border: 1px solid #cfe2f3;
        border-radius: 16px;
        background: #ffffff;
        box-shadow: 0 8px 24px rgba(22, 107, 135, 0.10);
        text-align: center;
    }
    .result-label {
        color: #526579;
        font-size: 0.95rem;
    }
    .result-score {
        margin: 0.2rem 0;
        color: #123b69;
        font-size: 3rem;
        font-weight: 800;
    }
    .result-caption {
        color: #526579;
        font-size: 0.88rem;
    }
    div[data-testid="stForm"] {
        padding: 1.25rem;
        border: 1px solid #dce8f4;
        border-radius: 16px;
        background: rgba(255, 255, 255, 0.92);
    }
    div[data-testid="stForm"] label,
    div[data-testid="stForm"] label p {
        color: #18324a !important;
    }
    div[data-testid="stFormSubmitButton"] button {
        min-height: 3rem;
        border-radius: 10px;
        font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def get_model(model_path: str, modified_time: float):
    """Nạp model một lần và tự làm mới cache khi file model thay đổi."""
    del modified_time
    return load_model(Path(model_path))


def estimate_demo_score(values: dict[str, Any]) -> float:
    """Tạo kết quả minh họa để kiểm thử giao diện khi chưa có model thật."""
    score = (
        50
        + 0.20 * (values["Attendance"] - 60)
        + 0.18 * (values["Previous_Scores"] - 50)
        + 0.22 * (values["Hours_Studied"] - 1)
        + 0.45 * values["Tutoring_Sessions"]
    )
    score += {"Low": -1.0, "Medium": 0.0, "High": 1.0}[
        values["Motivation_Level"]
    ]
    score += {"Negative": -0.8, "Neutral": 0.0, "Positive": 0.8}[
        values["Peer_Influence"]
    ]
    return round(max(0.0, min(100.0, score)), 1)


def score_message(score: float) -> str:
    """Trả về mô tả ngắn theo mức điểm dự đoán."""
    if score >= 85:
        return "Kết quả dự đoán ở mức rất tốt."
    if score >= 70:
        return "Kết quả dự đoán ở mức tốt."
    if score >= 60:
        return "Kết quả dự đoán ở mức trung bình."
    return "Kết quả dự đoán còn thấp và cần được theo dõi thêm."


available_models = {}
if SVR_MODEL_PATH.exists():
    # SVR Tuned đang đứng đầu bảng so sánh chung nên là lựa chọn mặc định.
    available_models["SVR Tuned - mô hình triển khai"] = SVR_MODEL_PATH
if MODEL_PATH.exists():
    available_models["Mô hình đối chứng của nhóm"] = MODEL_PATH
if not available_models:
    available_models["SVR Tuned - mô hình triển khai"] = SVR_MODEL_PATH

with st.sidebar:
    selected_model_name = st.selectbox(
        "Mô hình sử dụng",
        options=list(available_models),
    )
selected_model_path = available_models[selected_model_name]

model = None
model_error = None
if selected_model_path.exists():
    try:
        model = get_model(
            str(selected_model_path),
            selected_model_path.stat().st_mtime,
        )
    except Exception as exc:  # Hiển thị lỗi thân thiện thay vì làm hỏng toàn bộ app.
        model_error = str(exc)

with st.sidebar:
    st.header("Thông tin hệ thống")
    if model is not None:
        st.success("Mô hình đã sẵn sàng")
        st.caption(f"Nguồn: `{selected_model_path.name}`")
    elif model_error:
        st.error("Không thể nạp mô hình")
        st.caption(model_error)
    else:
        st.warning("Đang dùng chế độ minh họa")
        st.caption(
            "Chưa có file mô hình. Giao diện sẽ tự nhận mô hình khi file "
            f"`{selected_model_path.name}` được đặt trong thư mục `models`."
        )

    st.divider()
    st.subheader("Cách sử dụng")
    st.write("1. Nhập đầy đủ thông tin sinh viên.")
    st.write("2. Nhấn **Dự đoán kết quả**.")
    st.write("3. Xem điểm dự đoán và dữ liệu đã nhập.")
    st.caption("Kết quả chỉ hỗ trợ tham khảo, không thay thế đánh giá của giảng viên.")

st.markdown(
    """
    <div class="hero">
        <h1>Dự đoán kết quả học tập của sinh viên</h1>
        <p>Nhập thông tin học tập, môi trường và hoàn cảnh cá nhân để ước lượng điểm thi.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.caption("Các trường có dấu * là thông tin bắt buộc.")

with st.form("student_prediction_form", clear_on_submit=False):
    st.markdown('<div class="section-title">1. Kết quả và thói quen học tập</div>', unsafe_allow_html=True)
    academic_left, academic_right = st.columns(2, gap="large")

    with academic_left:
        hours_studied = st.slider(
            "Số giờ học mỗi tuần *", 1, 44, 20, help="Hours_Studied trong dữ liệu gốc."
        )
        attendance = st.slider(
            "Tỷ lệ chuyên cần (%) *", 60, 100, 80, help="Attendance trong dữ liệu gốc."
        )
        previous_scores = st.slider(
            "Điểm học tập trước đây *", 50, 100, 75, help="Previous_Scores trong dữ liệu gốc."
        )

    with academic_right:
        sleep_hours = st.slider(
            "Số giờ ngủ mỗi ngày *", 4, 10, 7, help="Sleep_Hours trong dữ liệu gốc."
        )
        tutoring_sessions = st.slider(
            "Số buổi phụ đạo *", 0, 8, 1, help="Tutoring_Sessions trong dữ liệu gốc."
        )
        physical_activity = st.slider(
            "Số giờ hoạt động thể chất mỗi tuần *",
            0,
            6,
            3,
            help="Physical_Activity trong dữ liệu gốc.",
        )

    st.markdown('<div class="section-title">2. Động lực và môi trường học tập</div>', unsafe_allow_html=True)
    environment_left, environment_right = st.columns(2, gap="large")

    with environment_left:
        motivation_level = st.selectbox(
            "Mức độ động lực *", ["Low", "Medium", "High"], index=1
        )
        access_to_resources = st.selectbox(
            "Khả năng tiếp cận tài nguyên *", ["Low", "Medium", "High"], index=1
        )
        teacher_quality = st.selectbox(
            "Chất lượng giáo viên *", ["Low", "Medium", "High"], index=1
        )
        peer_influence = st.selectbox(
            "Ảnh hưởng từ bạn bè *", ["Negative", "Neutral", "Positive"], index=2
        )

    with environment_right:
        extracurricular_activities = st.selectbox(
            "Tham gia hoạt động ngoại khóa *", ["No", "Yes"], index=1
        )
        internet_access = st.selectbox(
            "Có truy cập Internet *", ["No", "Yes"], index=1
        )
        school_type = st.selectbox("Loại trường *", ["Public", "Private"], index=0)

    st.markdown('<div class="section-title">3. Gia đình và thông tin cá nhân</div>', unsafe_allow_html=True)
    personal_left, personal_right = st.columns(2, gap="large")

    with personal_left:
        parental_involvement = st.selectbox(
            "Mức độ tham gia của phụ huynh *", ["Low", "Medium", "High"], index=1
        )
        parental_education = st.selectbox(
            "Trình độ học vấn của phụ huynh *",
            ["High School", "College", "Postgraduate"],
            index=0,
        )
        family_income = st.selectbox(
            "Thu nhập gia đình *", ["Low", "Medium", "High"], index=0
        )

    with personal_right:
        distance_from_home = st.selectbox(
            "Khoảng cách từ nhà đến trường *", ["Near", "Moderate", "Far"], index=0
        )
        learning_disabilities = st.selectbox(
            "Có khó khăn trong học tập *", ["No", "Yes"], index=0
        )
        gender = st.selectbox("Giới tính *", ["Female", "Male"], index=1)

    submitted = st.form_submit_button(
        "Dự đoán kết quả", type="primary", use_container_width=True
    )

if submitted:
    input_data = {
        "Hours_Studied": hours_studied,
        "Attendance": attendance,
        "Parental_Involvement": parental_involvement,
        "Access_to_Resources": access_to_resources,
        "Extracurricular_Activities": extracurricular_activities,
        "Sleep_Hours": sleep_hours,
        "Previous_Scores": previous_scores,
        "Motivation_Level": motivation_level,
        "Internet_Access": internet_access,
        "Tutoring_Sessions": tutoring_sessions,
        "Family_Income": family_income,
        "Teacher_Quality": teacher_quality,
        "School_Type": school_type,
        "Peer_Influence": peer_influence,
        "Physical_Activity": physical_activity,
        "Learning_Disabilities": learning_disabilities,
        "Parental_Education_Level": parental_education,
        "Distance_from_Home": distance_from_home,
        "Gender": gender,
    }

    try:
        if model is not None:
            predicted_score = predict_one(model, input_data)
            result_caption = "Kết quả từ mô hình học máy của nhóm"
        else:
            predicted_score = estimate_demo_score(input_data)
            result_caption = "Kết quả minh họa giao diện, chưa phải dự đoán AI"

        st.markdown("### Kết quả dự đoán")
        result_column, detail_column = st.columns([1, 1.8], gap="large")
        with result_column:
            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-label">Điểm thi dự đoán</div>
                    <div class="result-score">{predicted_score:.1f}</div>
                    <div class="result-caption">{result_caption}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.info(score_message(predicted_score))

        with detail_column:
            if model is None:
                st.warning(
                    "Chưa có mô hình chính thức. Con số trên chỉ dùng để kiểm thử "
                    "bố cục và luồng thao tác."
                )
            with st.expander("Xem 19 trường gửi tới mô hình", expanded=True):
                st.json(input_data)
    except (ValueError, TypeError) as exc:
        st.error(f"Dữ liệu đầu vào chưa hợp lệ: {exc}")
    except Exception as exc:
        st.error("Không thể thực hiện dự đoán. Hãy kiểm tra lại file mô hình.")
        st.caption(str(exc))


st.divider()
st.header("Dự đoán hàng loạt từ CSV")
st.write(
    "Tải lên tệp CSV chứa 19 cột đầu vào. Hệ thống giữ lại mọi dòng, "
    "dự đoán các dòng hợp lệ và ghi lỗi cụ thể cho từng dòng không hợp lệ."
)

uploaded_file = st.file_uploader(
    "Chọn tệp CSV",
    type=["csv"],
    help="Tên cột phải giống dữ liệu huấn luyện của dự án.",
)

if uploaded_file is not None:
    try:
        batch_data = pd.read_csv(uploaded_file)
        st.subheader("Xem trước dữ liệu")
        st.caption(f"{len(batch_data)} dòng, {len(batch_data.columns)} cột")
        st.dataframe(batch_data.head(20), use_container_width=True)

        if model is None:
            st.warning(
                "Cần mô hình đã huấn luyện để dự đoán CSV. "
                "Hãy chạy `python -m src.train` trước."
            )
        elif st.button(
            "Kiểm tra và dự đoán CSV",
            type="primary",
            use_container_width=True,
        ):
            batch_result = predict_batch_with_errors(model, batch_data)
            valid_count = int(batch_result["Validation_Error"].eq("").sum())
            error_count = len(batch_result) - valid_count

            metric_left, metric_right = st.columns(2)
            metric_left.metric("Dòng dự đoán thành công", valid_count)
            metric_right.metric("Dòng có lỗi", error_count)
            if error_count:
                st.warning(
                    "Các dòng lỗi được giữ nguyên và có mô tả trong cột "
                    "Validation_Error."
                )
            else:
                st.success("Toàn bộ dữ liệu hợp lệ.")

            st.dataframe(batch_result, use_container_width=True)
            st.download_button(
                "Tải kết quả CSV",
                data=batch_result.to_csv(index=False).encode("utf-8-sig"),
                file_name="student_score_predictions.csv",
                mime="text/csv",
                use_container_width=True,
            )
    except InputValidationError as exc:
        st.error(str(exc))
        if not exc.errors.empty:
            st.dataframe(exc.errors, use_container_width=True)
    except (pd.errors.ParserError, UnicodeDecodeError) as exc:
        st.error("Không thể đọc tệp CSV. Hãy kiểm tra định dạng và bảng mã.")
        st.caption(str(exc))
    except Exception as exc:
        st.error("Không thể xử lý tệp CSV.")
        st.caption(str(exc))


st.divider()
st.header("Đánh giá mô hình")
comparison_path = RESULTS_PATH / "model_comparison.csv"
metrics_path = RESULTS_PATH / "evaluation_metrics.json"
svr_metrics_path = RESULTS_PATH / "svr_metrics.csv"

if metrics_path.exists():
    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        metric_columns = st.columns(3)
        metric_columns[0].metric("MAE", f"{metrics['MAE']:.4f}")
        metric_columns[1].metric("RMSE", f"{metrics['RMSE']:.4f}")
        metric_columns[2].metric("R²", f"{metrics['R2']:.4f}")
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        st.warning("Tệp chỉ số đánh giá không đúng định dạng.")
        st.caption(str(exc))
else:
    st.info("Chưa có chỉ số đánh giá. Hãy chạy `python -m src.evaluate`.")

if comparison_path.exists():
    try:
        comparison = pd.read_csv(comparison_path)
        st.subheader("So sánh các mô hình")
        st.dataframe(comparison, use_container_width=True, hide_index=True)
    except (pd.errors.ParserError, UnicodeDecodeError) as exc:
        st.warning("Không thể đọc bảng so sánh mô hình.")
        st.caption(str(exc))
else:
    st.info("Chưa có bảng so sánh. Hãy chạy `python -m src.train`.")

if svr_metrics_path.exists():
    try:
        svr_metrics = pd.read_csv(svr_metrics_path)
        st.subheader("Kết quả SVR sau tuning")
        st.dataframe(svr_metrics, use_container_width=True, hide_index=True)
    except (pd.errors.ParserError, UnicodeDecodeError) as exc:
        st.warning("Không thể đọc kết quả SVR tuning.")
        st.caption(str(exc))

st.caption(
    "Permutation importance và các chỉ số chỉ phản ánh khả năng dự đoán "
    "trên tập dữ liệu hiện tại, không chứng minh quan hệ nhân quả."
)
