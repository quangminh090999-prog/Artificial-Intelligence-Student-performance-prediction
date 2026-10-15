"""Giao diện Streamlit của dự án.

Thành viên phụ trách giao diện sẽ thay phần minh họa bằng biểu mẫu nhập liệu
và gọi hàm dự đoán trong ``src.predict`` khi mô hình đã sẵn sàng.
"""

import streamlit as st


st.set_page_config(page_title="Dự đoán kết quả sinh viên", page_icon="🎓")
st.title("Dự đoán kết quả học tập của sinh viên")
st.info("Khung giao diện đã sẵn sàng. Mô hình dự đoán sẽ được tích hợp ở bước sau.")

st.subheader("Tiến độ dự án")
st.write(
    "Nhóm đang thực hiện phân tích dữ liệu, tiền xử lý, huấn luyện và đánh giá "
    "các mô hình hồi quy."
)
