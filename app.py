"""Giao diện Streamlit cho hệ thống dự đoán kết quả học tập sinh viên."""

import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from src.config import MODEL_PATH, RESULTS_PATH, SVR_MODEL_PATH
from src.predict import InputValidationError, load_model, predict_batch_with_errors, predict_one


st.set_page_config(
    page_title="StudentAI | Dự đoán kết quả học tập",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {--navy:#12345b;--blue:#176b87;--teal:#24a19c;--ink:#162b44;--muted:#64788f;--line:#d9e5f0}
    .stApp {color:var(--ink);background:radial-gradient(circle at 92% 4%,rgba(36,161,156,.11),transparent 24rem),linear-gradient(180deg,#f7faff 0%,#fff 52%)}
    .main .block-container {max-width:1220px;padding-top:2rem;padding-bottom:4rem}
    [data-testid="stSidebar"] {background:linear-gradient(180deg,#eef5fa 0%,#e7f0f7 100%);border-right:1px solid var(--line)}
    [data-testid="stSidebar"] .block-container {padding-top:1.6rem}
    [data-testid="stSidebar"] [role="radiogroup"] label {padding:.58rem .72rem;margin:.18rem 0;border-radius:10px}
    [data-testid="stSidebar"] [role="radiogroup"] label:hover {background:rgba(23,107,135,.08)}
    .brand {display:flex;align-items:center;gap:.72rem;margin-bottom:1.25rem}
    .brand-mark {display:grid;place-items:center;width:42px;height:42px;color:#fff;font-size:1.05rem;font-weight:800;border-radius:13px;background:linear-gradient(135deg,var(--navy),var(--blue));box-shadow:0 8px 18px rgba(18,52,91,.18)}
    .brand-name {color:var(--navy);font-size:1.2rem;font-weight:800;line-height:1.05}
    .brand-subtitle {color:var(--muted);font-size:.76rem;margin-top:.18rem}
    .hero {position:relative;overflow:hidden;padding:3.2rem 3.3rem;border-radius:28px;color:#fff;background:radial-gradient(circle at 86% 22%,rgba(79,209,197,.35),transparent 18rem),linear-gradient(120deg,#11345c 0%,#176b87 72%,#168897 100%);box-shadow:0 20px 48px rgba(18,52,91,.2)}
    .hero:after {content:"";position:absolute;right:-85px;bottom:-110px;width:320px;height:320px;border:1px solid rgba(255,255,255,.18);border-radius:50%;box-shadow:0 0 0 42px rgba(255,255,255,.05),0 0 0 88px rgba(255,255,255,.035)}
    .eyebrow {display:inline-flex;padding:.42rem .72rem;color:#dffbf7;font-size:.78rem;font-weight:700;letter-spacing:.08em;text-transform:uppercase;border:1px solid rgba(255,255,255,.24);border-radius:999px;background:rgba(255,255,255,.08)}
    .hero h1 {position:relative;z-index:1;max-width:780px;margin:1rem 0 .75rem;font-size:clamp(2.15rem,4vw,3.55rem);line-height:1.08;letter-spacing:-.035em}
    .hero p {position:relative;z-index:1;max-width:680px;margin:0;color:#e3f1f7;font-size:1.08rem;line-height:1.7}
    .trust-row {display:flex;flex-wrap:wrap;gap:.7rem;margin-top:1.55rem}.trust-chip {padding:.48rem .72rem;color:#e8f7fa;font-size:.82rem;border-radius:999px;background:rgba(255,255,255,.1)}
    .metric-grid,.feature-grid,.step-grid {display:grid;gap:1rem}.metric-grid {grid-template-columns:repeat(4,minmax(0,1fr));margin:1.25rem 0 2.4rem}.feature-grid,.step-grid {grid-template-columns:repeat(3,minmax(0,1fr))}
    .metric-card,.feature-card,.step-card {border:1px solid var(--line);border-radius:18px;background:rgba(255,255,255,.92);box-shadow:0 10px 28px rgba(18,52,91,.06)}
    .metric-card {padding:1.15rem 1.2rem}.metric-value {color:var(--navy);font-size:1.65rem;font-weight:800}.metric-label {margin-top:.18rem;color:var(--muted);font-size:.82rem}
    .feature-card,.step-card {padding:1.35rem}.feature-icon,.step-number {display:grid;place-items:center;width:40px;height:40px;margin-bottom:.9rem;border-radius:12px;color:var(--blue);font-weight:800;background:#e7f4f5}
    .feature-card h3,.step-card h3 {margin:0 0 .45rem;color:var(--navy);font-size:1rem}.feature-card p,.step-card p {margin:0;color:var(--muted);font-size:.9rem;line-height:1.55}
    .section-heading {margin:2.5rem 0 1rem}.section-heading h2 {margin:0;color:var(--navy);font-size:1.65rem}.section-heading p {margin:.35rem 0 0;color:var(--muted)}
    .page-header {padding:1.55rem 1.75rem;margin-bottom:1.2rem;border:1px solid var(--line);border-radius:20px;background:rgba(255,255,255,.9);box-shadow:0 10px 28px rgba(18,52,91,.06)}
    .page-header h1 {margin:0;color:var(--navy);font-size:2rem}.page-header p {margin:.45rem 0 0;color:var(--muted)}
    .result-card {padding:1.6rem;text-align:center;border:1px solid #cfe2f3;border-radius:18px;background:#fff;box-shadow:0 10px 28px rgba(22,107,135,.1)}
    .result-label {color:var(--muted);font-size:.9rem}.result-score {margin:.2rem 0;color:var(--navy);font-size:3rem;font-weight:850}.result-caption {color:var(--muted);font-size:.82rem}
    .status-box {padding:.9rem 1rem;margin:.8rem 0;color:#08784f;font-size:.88rem;border:1px solid #bfe8d6;border-radius:12px;background:#e8f7f0}.status-dot {display:inline-block;width:8px;height:8px;margin-right:.45rem;border-radius:50%;background:#0a9b67;box-shadow:0 0 0 4px rgba(10,155,103,.12)}
    div[data-testid="stForm"] {padding:1.4rem;border:1px solid var(--line);border-radius:20px;background:rgba(255,255,255,.92);box-shadow:0 10px 28px rgba(18,52,91,.05)}
    div[data-testid="stFormSubmitButton"] button,div[data-testid="stButton"] button {min-height:2.9rem;border-radius:11px;font-weight:750}
    [data-testid="stFileUploader"] {padding:.8rem;border:1px dashed #a9bed0;border-radius:16px;background:rgba(255,255,255,.8)}
    #MainMenu,footer {visibility:hidden}
    @media(max-width:900px){.hero{padding:2.2rem 1.5rem;border-radius:22px}.metric-grid{grid-template-columns:repeat(2,minmax(0,1fr))}.feature-grid,.step-grid{grid-template-columns:1fr}}
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
    """Tạo kết quả minh họa khi môi trường chưa có model."""
    score = 50 + .20 * (values["Attendance"] - 60) + .18 * (values["Previous_Scores"] - 50)
    score += .22 * (values["Hours_Studied"] - 1) + .45 * values["Tutoring_Sessions"]
    score += {"Low": -1.0, "Medium": 0.0, "High": 1.0}[values["Motivation_Level"]]
    score += {"Negative": -.8, "Neutral": 0.0, "Positive": .8}[values["Peer_Influence"]]
    return round(max(0.0, min(100.0, score)), 1)


def score_message(score: float) -> str:
    if score >= 85:
        return "Kết quả dự đoán ở mức rất tốt."
    if score >= 70:
        return "Kết quả dự đoán ở mức tốt."
    if score >= 60:
        return "Kết quả dự đoán ở mức trung bình."
    return "Kết quả dự đoán còn thấp và cần được theo dõi thêm."


def boundary_label(
    value: int,
    minimum: int,
    maximum: int,
    lower_bucket: bool = True,
    upper_bucket: bool = True,
) -> str:
    """Hiển thị giá trị biên dưới dạng ≤ hoặc ≥ nhưng vẫn trả số cho model."""
    if lower_bucket and value == minimum:
        return f"≤ {value}"
    if upper_bucket and value == maximum:
        return f"≥ {value}"
    return str(value)


def navigate(page: str) -> None:
    st.session_state.navigation = page


def page_header(title: str, description: str) -> None:
    st.markdown(f'<div class="page-header"><h1>{title}</h1><p>{description}</p></div>', unsafe_allow_html=True)


def load_svr_metrics() -> dict[str, float]:
    path = RESULTS_PATH / "svr_metrics.csv"
    if path.exists():
        try:
            row = pd.read_csv(path).iloc[0]
            return {"mae": float(row["test_mae"]), "rmse": float(row["test_rmse"]), "r2": float(row["test_r2"])}
        except (KeyError, IndexError, TypeError, ValueError, pd.errors.ParserError):
            pass
    return {"mae": .4063, "rmse": 1.8003, "r2": .7707}


available_models: dict[str, Path] = {}
if SVR_MODEL_PATH.exists():
    available_models["SVR Tuned – mô hình triển khai"] = SVR_MODEL_PATH
if MODEL_PATH.exists():
    available_models["Ridge Regression – mô hình đối chứng"] = MODEL_PATH
if not available_models:
    available_models["SVR Tuned – mô hình triển khai"] = SVR_MODEL_PATH

with st.sidebar:
    st.markdown('<div class="brand"><div class="brand-mark">AI</div><div><div class="brand-name">StudentAI</div><div class="brand-subtitle">Dự đoán kết quả học tập</div></div></div>', unsafe_allow_html=True)
    navigation = st.radio("Điều hướng", ["Trang chủ", "Dự đoán cá nhân", "Dự đoán CSV", "Kết quả mô hình"], key="navigation")
    st.divider()
    selected_model_name = st.selectbox("Mô hình sử dụng", options=list(available_models))

selected_model_path = available_models[selected_model_name]
model = None
model_error = None
if selected_model_path.exists():
    try:
        model = get_model(str(selected_model_path), selected_model_path.stat().st_mtime)
    except Exception as exc:
        model_error = str(exc)

with st.sidebar:
    if model is not None:
        st.markdown('<div class="status-box"><span class="status-dot"></span>Mô hình đã sẵn sàng</div>', unsafe_allow_html=True)
        st.caption(f"Nguồn: `{selected_model_path.name}`")
    elif model_error:
        st.error("Không thể nạp mô hình")
        st.caption(model_error)
    else:
        st.warning("Đang dùng chế độ minh họa")
        st.caption(f"Thiếu `{selected_model_path.name}` trong thư mục models.")
    st.divider()
    st.caption("Phiên bản học thuật · Kết quả chỉ mang tính tham khảo.")


def render_home() -> None:
    metrics = load_svr_metrics()
    st.markdown("""
        <div class="hero"><span class="eyebrow">Hệ thống hỗ trợ giáo dục bằng AI</span>
        <h1>Hiểu sớm kết quả học tập, hỗ trợ sinh viên đúng lúc</h1>
        <p>StudentAI sử dụng Support Vector Regression đã được tuning để ước lượng điểm thi từ thói quen học tập, mức độ chuyên cần và môi trường của sinh viên.</p>
        <div class="trust-row"><span class="trust-chip">✓ Pipeline chống rò rỉ dữ liệu</span><span class="trust-chip">✓ Kiểm định chéo 5 fold</span><span class="trust-chip">✓ Dự đoán đơn và CSV</span></div></div>
        """, unsafe_allow_html=True)
    primary, secondary, _ = st.columns([1, 1.35, .4])
    with primary:
        st.button("Bắt đầu dự đoán  →", type="primary", width="stretch", on_click=navigate, args=("Dự đoán cá nhân",))
    with secondary:
        st.button("Xem kết quả mô hình", width="stretch", on_click=navigate, args=("Kết quả mô hình",))
    st.markdown(f"""
        <div class="metric-grid">
        <div class="metric-card"><div class="metric-value">{metrics['r2']:.3f}</div><div class="metric-label">R² trên tập kiểm tra</div></div>
        <div class="metric-card"><div class="metric-value">{metrics['mae']:.3f}</div><div class="metric-label">Sai số tuyệt đối MAE</div></div>
        <div class="metric-card"><div class="metric-value">19</div><div class="metric-label">Yếu tố đầu vào</div></div>
        <div class="metric-card"><div class="metric-value">5-fold</div><div class="metric-label">Cross-validation</div></div></div>
        """, unsafe_allow_html=True)
    st.markdown('<div class="section-heading"><h2>Hệ thống phân tích những gì?</h2><p>Ba nhóm thông tin tạo nên góc nhìn toàn diện nhưng dễ nhập liệu.</p></div>', unsafe_allow_html=True)
    st.markdown("""
        <div class="feature-grid">
        <div class="feature-card"><div class="feature-icon">01</div><h3>Thói quen học tập</h3><p>Giờ học, chuyên cần, điểm trước đây, giấc ngủ, phụ đạo và hoạt động thể chất.</p></div>
        <div class="feature-card"><div class="feature-icon">02</div><h3>Môi trường học tập</h3><p>Động lực, tài nguyên, giáo viên, Internet, bạn bè và hoạt động ngoại khóa.</p></div>
        <div class="feature-card"><div class="feature-icon">03</div><h3>Gia đình & cá nhân</h3><p>Sự tham gia của phụ huynh, thu nhập, học vấn, khoảng cách và thông tin cá nhân.</p></div></div>
        """, unsafe_allow_html=True)
    st.markdown('<div class="section-heading"><h2>Quy trình sử dụng</h2><p>Từ dữ liệu đầu vào đến kết quả dự đoán trong ba bước.</p></div>', unsafe_allow_html=True)
    st.markdown("""
        <div class="step-grid">
        <div class="step-card"><div class="step-number">1</div><h3>Nhập thông tin</h3><p>Điền 19 trường hoặc tải tệp CSV theo đúng mẫu.</p></div>
        <div class="step-card"><div class="step-number">2</div><h3>AI xử lý</h3><p>Dữ liệu được kiểm tra, tiền xử lý và đưa qua pipeline SVR.</p></div>
        <div class="step-card"><div class="step-number">3</div><h3>Nhận kết quả</h3><p>Xem điểm dự đoán, lỗi theo dòng và tải kết quả khi cần.</p></div></div>
        """, unsafe_allow_html=True)
    st.info("StudentAI là sản phẩm học thuật. Kết quả hỗ trợ tham khảo, không thay thế đánh giá của giảng viên.")


def render_single_prediction() -> None:
    page_header("Dự đoán cho một sinh viên", "Điền đầy đủ 19 yếu tố. Các trường có dấu * là thông tin bắt buộc.")
    st.caption("Các mốc ≤ và ≥ biểu thị giá trị nằm ngoài hoặc bằng biên dữ liệu huấn luyện; hệ thống dùng chính giá trị biên để dự đoán.")
    with st.form("student_prediction_form", clear_on_submit=False):
        st.subheader("1. Kết quả và thói quen học tập")
        left, right = st.columns(2, gap="large")
        with left:
            hours_studied = st.select_slider(
                "Số giờ học mỗi tuần *", options=list(range(1, 45)), value=20,
                format_func=lambda value: boundary_label(value, 1, 44),
            )
            attendance = st.select_slider(
                "Tỷ lệ chuyên cần (%) *", options=list(range(60, 101)), value=80,
                format_func=lambda value: boundary_label(value, 60, 100, upper_bucket=False),
            )
            previous_scores = st.select_slider(
                "Điểm học tập trước đây *", options=list(range(50, 101)), value=75,
                format_func=lambda value: boundary_label(value, 50, 100, upper_bucket=False),
            )
        with right:
            sleep_hours = st.select_slider(
                "Số giờ ngủ mỗi ngày *", options=list(range(4, 11)), value=7,
                format_func=lambda value: boundary_label(value, 4, 10),
            )
            tutoring_sessions = st.select_slider(
                "Số buổi phụ đạo *", options=list(range(0, 9)), value=1,
                format_func=lambda value: boundary_label(value, 0, 8, lower_bucket=False),
            )
            physical_activity = st.select_slider(
                "Số giờ hoạt động thể chất mỗi tuần *", options=list(range(0, 7)), value=3,
                format_func=lambda value: boundary_label(value, 0, 6, lower_bucket=False),
            )
        st.subheader("2. Động lực và môi trường học tập")
        left, right = st.columns(2, gap="large")
        with left:
            motivation_level = st.selectbox("Mức độ động lực *", ["Low", "Medium", "High"], index=1)
            access_to_resources = st.selectbox("Khả năng tiếp cận tài nguyên *", ["Low", "Medium", "High"], index=1)
            teacher_quality = st.selectbox("Chất lượng giáo viên *", ["Low", "Medium", "High"], index=1)
            peer_influence = st.selectbox("Ảnh hưởng từ bạn bè *", ["Negative", "Neutral", "Positive"], index=2)
        with right:
            extracurricular_activities = st.selectbox("Tham gia hoạt động ngoại khóa *", ["No", "Yes"], index=1)
            internet_access = st.selectbox("Có truy cập Internet *", ["No", "Yes"], index=1)
            school_type = st.selectbox("Loại trường *", ["Public", "Private"])
        st.subheader("3. Gia đình và thông tin cá nhân")
        left, right = st.columns(2, gap="large")
        with left:
            parental_involvement = st.selectbox("Mức độ tham gia của phụ huynh *", ["Low", "Medium", "High"], index=1)
            parental_education = st.selectbox("Trình độ học vấn của phụ huynh *", ["High School", "College", "Postgraduate"])
            family_income = st.selectbox("Thu nhập gia đình *", ["Low", "Medium", "High"])
        with right:
            distance_from_home = st.selectbox("Khoảng cách từ nhà đến trường *", ["Near", "Moderate", "Far"])
            learning_disabilities = st.selectbox("Có khó khăn trong học tập *", ["No", "Yes"])
            gender = st.selectbox("Giới tính *", ["Female", "Male"], index=1)
        submitted = st.form_submit_button("Phân tích và dự đoán kết quả", type="primary", width="stretch")
    if not submitted:
        return
    input_data = {
        "Hours_Studied": hours_studied, "Attendance": attendance,
        "Parental_Involvement": parental_involvement, "Access_to_Resources": access_to_resources,
        "Extracurricular_Activities": extracurricular_activities, "Sleep_Hours": sleep_hours,
        "Previous_Scores": previous_scores, "Motivation_Level": motivation_level,
        "Internet_Access": internet_access, "Tutoring_Sessions": tutoring_sessions,
        "Family_Income": family_income, "Teacher_Quality": teacher_quality,
        "School_Type": school_type, "Peer_Influence": peer_influence,
        "Physical_Activity": physical_activity, "Learning_Disabilities": learning_disabilities,
        "Parental_Education_Level": parental_education, "Distance_from_Home": distance_from_home,
        "Gender": gender,
    }
    try:
        if model is not None:
            predicted_score = predict_one(model, input_data)
            caption = f"Kết quả từ {selected_model_name}"
        else:
            predicted_score, caption = estimate_demo_score(input_data), "Kết quả minh họa – chưa phải dự đoán AI"
        st.subheader("Kết quả phân tích")
        result_column, detail_column = st.columns([1, 1.8], gap="large")
        with result_column:
            st.markdown(f'<div class="result-card"><div class="result-label">Điểm thi dự đoán</div><div class="result-score">{predicted_score:.1f}</div><div class="result-caption">{caption}</div></div>', unsafe_allow_html=True)
            st.info(score_message(predicted_score))
        with detail_column:
            if model is None:
                st.warning("Thiếu model chính thức; kết quả chỉ dùng kiểm thử giao diện.")
            with st.expander("Xem dữ liệu gửi tới mô hình", expanded=True):
                st.json(input_data)
    except (ValueError, TypeError) as exc:
        st.error(f"Dữ liệu đầu vào chưa hợp lệ: {exc}")
    except Exception as exc:
        st.error("Không thể thực hiện dự đoán. Hãy kiểm tra lại file mô hình.")
        st.caption(str(exc))


def render_batch_prediction() -> None:
    page_header("Dự đoán hàng loạt từ CSV", "Tải tệp có đủ 19 cột. Hệ thống giữ mọi dòng và mô tả lỗi theo từng dòng.")
    st.markdown("**Quy trình:** Chọn tệp → xem trước → kiểm tra và dự đoán → tải kết quả.")
    uploaded_file = st.file_uploader("Chọn tệp CSV", type=["csv"], help="Tên cột phải giống dữ liệu huấn luyện.")
    if uploaded_file is None:
        st.info("Bạn có thể dùng `data/sample_batch_input.csv` để kiểm thử chức năng.")
        return
    try:
        batch_data = pd.read_csv(uploaded_file)
        st.subheader("Xem trước dữ liệu")
        st.caption(f"{len(batch_data)} dòng · {len(batch_data.columns)} cột")
        st.dataframe(batch_data.head(20), width="stretch")
        if model is None:
            st.warning("Cần model đã huấn luyện để dự đoán CSV.")
        elif st.button("Kiểm tra và dự đoán CSV", type="primary", width="stretch"):
            result = predict_batch_with_errors(model, batch_data)
            valid_count = int(result["Validation_Error"].eq("").sum())
            error_count = len(result) - valid_count
            c1, c2 = st.columns(2)
            c1.metric("Dòng dự đoán thành công", valid_count)
            c2.metric("Dòng có lỗi", error_count)
            st.success("Toàn bộ dữ liệu hợp lệ.") if not error_count else st.warning("Dòng lỗi được mô tả trong Validation_Error.")
            st.dataframe(result, width="stretch")
            st.download_button("Tải kết quả CSV", data=result.to_csv(index=False).encode("utf-8-sig"), file_name="student_score_predictions.csv", mime="text/csv", width="stretch")
    except InputValidationError as exc:
        st.error(str(exc))
        if not exc.errors.empty:
            st.dataframe(exc.errors, width="stretch")
    except (pd.errors.ParserError, UnicodeDecodeError) as exc:
        st.error("Không thể đọc tệp CSV. Hãy kiểm tra định dạng và bảng mã.")
        st.caption(str(exc))
    except Exception as exc:
        st.error("Không thể xử lý tệp CSV.")
        st.caption(str(exc))


def render_model_results() -> None:
    page_header("Kết quả và đánh giá mô hình", "Các mô hình được đánh giá trên cùng tập test; MAE và RMSE càng thấp càng tốt.")
    metrics = load_svr_metrics()
    columns = st.columns(3)
    columns[0].metric("SVR · MAE", f"{metrics['mae']:.4f}")
    columns[1].metric("SVR · RMSE", f"{metrics['rmse']:.4f}")
    columns[2].metric("SVR · R²", f"{metrics['r2']:.4f}")
    selection_path = RESULTS_PATH / "final_model_selection.csv"
    if selection_path.exists():
        try:
            selection = pd.read_csv(selection_path)
            st.subheader("Bảng xếp hạng mô hình")
            st.dataframe(selection[["rank", "model", "MAE", "RMSE", "R2", "selected"]], width="stretch", hide_index=True)
        except (KeyError, pd.errors.ParserError, UnicodeDecodeError) as exc:
            st.warning("Không thể đọc bảng lựa chọn mô hình.")
            st.caption(str(exc))
    comparison_path = RESULTS_PATH / "model_comparison.csv"
    if comparison_path.exists():
        try:
            with st.expander("Xem kết quả các mô hình cơ sở"):
                st.dataframe(pd.read_csv(comparison_path), width="stretch", hide_index=True)
        except (pd.errors.ParserError, UnicodeDecodeError) as exc:
            st.warning("Không thể đọc bảng so sánh mô hình cơ sở.")
            st.caption(str(exc))
    st.success("SVR Tuned được chọn triển khai vì đạt Test MAE thấp nhất và đã được kiểm thử cho dự đoán đơn lẫn CSV.")
    st.caption("Các chỉ số phản ánh khả năng dự đoán trên dữ liệu hiện tại, không chứng minh quan hệ nhân quả.")


if navigation == "Trang chủ":
    render_home()
elif navigation == "Dự đoán cá nhân":
    render_single_prediction()
elif navigation == "Dự đoán CSV":
    render_batch_prediction()
else:
    render_model_results()
