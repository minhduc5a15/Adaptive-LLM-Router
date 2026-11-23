"""Demo offline: định tuyến prompt và xem kết quả thí nghiệm đã khóa."""

import json
from pathlib import Path
import re

import pandas as pd
import streamlit as st

from src.features import extract_structure_features
from src.preprocess_data import HAN_PATTERN
from src.router import DEFAULT_MODEL_PATH, load_router, route_prompts

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="Adaptive LLM Router", page_icon="🌳", layout="wide")
st.title("Adaptive LLM Router")
st.caption("Decision Tree + SVM · Chọn Mistral hoặc Mixtral từ nội dung câu hỏi")

if not DEFAULT_MODEL_PATH.exists():
    st.info("Chưa có mô hình. Chạy `.venv/bin/python src/train_models.py` trước khi mở demo.")
    st.stop()


@st.cache_resource
def get_bundle(path, modified_ns):
    return load_router(path)


bundle = get_bundle(str(DEFAULT_MODEL_PATH), DEFAULT_MODEL_PATH.stat().st_mtime_ns)
route_tab, results_tab, explanation_tab = st.tabs(["Thử định tuyến", "Kết quả test", "Cách hoạt động"])

with route_tab:
    st.write("Nhập một câu hỏi để xem Router chọn LLM nào. Demo chỉ định tuyến, chưa tạo câu trả lời từ LLM.")
    policies = {p["name"]: p for p in bundle["policies"]}
    names = list(policies)
    chosen_name = st.selectbox("Phương án", names, index=names.index(bundle["selected_policy"]["name"]))
    policy = policies[chosen_name]
    threshold = st.slider("Ngưỡng chọn Mixtral", 0.0, 1.0, float(policy["threshold"]), 0.01,
                          key=f"threshold_{chosen_name}")
    st.caption("Ngưỡng thấp: dễ chọn Mixtral hơn. Ngưỡng cao: dễ chọn Mistral hơn.")
    prompt = st.text_area("Câu hỏi", "Explain how binary search works and give a Python example.", height=180)
    if st.button("Định tuyến", type="primary"):
        if not prompt.strip():
            st.error("Hãy nhập một câu hỏi.")
        else:
            if re.search(HAN_PATTERN, prompt):
                st.warning("Prompt chứa chữ Hán, thuộc nhóm đã lọc khỏi dữ liệu huấn luyện.")
            result = route_prompts(bundle, [prompt], policy=policy, threshold=threshold)
            label = "Mixtral 8×7B" if result["route"][0] else "Mistral 7B"
            st.success(f"LLM được chọn: {label}")
            a, b, c = st.columns(3)
            a.metric("Điểm từ Decision Tree", f'{result["tree_probability"][0]:.1%}')
            b.metric("Điểm từ SVM", f'{result["svm_probability"][0]:.1%}')
            c.metric("Điểm dùng để định tuyến", f'{result["probability"][0]:.1%}')
            st.caption("Các điểm là xác suất ước lượng nhãn 1 (chọn Mixtral), không phải tỷ lệ LLM trả lời đúng.")
            with st.expander("Xem đặc trưng và phép kết hợp"):
                st.dataframe(extract_structure_features([prompt]), hide_index=True)
                st.write(f'Trọng số cây: {policy["tree_weight"]:.2f}; trọng số SVM: {1-policy["tree_weight"]:.2f}.')
                st.code("p = trọng_số_cây × p_tree + trọng_số_svm × p_svm\nchọn Mixtral nếu p ≥ ngưỡng")

with results_tab:
    metrics_path = ROOT / "reports" / "test_metrics.csv"
    summary_path = ROOT / "reports" / "summary.json"
    if metrics_path.exists() and summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        selected = summary["test_metrics"]
        st.subheader(f'Phương án chọn từ validation: {summary["selected_policy"]["name"]}')
        a, b, c = st.columns(3)
        a.metric("Câu được chấm đúng", f'{selected["answer_score"]:.2%}')
        b.metric("Tiết kiệm so với luôn Mixtral", f'{selected["cost_saving"]:.2%}')
        c.metric("Tỷ lệ gọi Mixtral", f'{selected["large_rate"]:.2%}')
        metrics = pd.read_csv(metrics_path)
        display = pd.DataFrame({
            "Phương án": metrics["name"],
            "Đúng nhãn Router": metrics["accuracy"].map(lambda p: f"{p:.2%}"),
            "Câu được chấm đúng": metrics["answer_score"].map(lambda p: f"{p:.2%}"),
            "Tổng chi phí benchmark (USD)": metrics["total_cost"].map(lambda p: f"{p:.6f}"),
            "Tiết kiệm": metrics["cost_saving"].map(lambda p: f"{p:.2%}"),
        })
        st.dataframe(display, hide_index=True)
        st.caption("Oracle dùng nhãn thật, chỉ là mốc tham chiếu. Chi phí lấy từ dataset, không phải tiền đã trả hoặc giá API hiện hành.")
        quality_ci = summary["bootstrap"]["answer_delta_95ci"]
        st.write(f'Khoảng bootstrap 95% cho thay đổi tỷ lệ đúng so với Mixtral: '
                 f'[{quality_ci[0]*100:+.2f}, {quality_ci[1]*100:+.2f}] điểm phần trăm.')
        chart = ROOT / "reports" / "figures" / "quality_cost.png"
        if chart.exists():
            st.image(str(chart))
        st.caption("Các kết quả trên dùng cấu hình đã khóa. Thay ngưỡng trong tab demo không thay đổi báo cáo test.")
    else:
        st.info("Chưa có báo cáo test; hãy chạy hết src/train_models.py.")

with explanation_tab:
    st.markdown("""
1. **Decision Tree** đọc số ký tự, số phần tách theo khoảng trắng và số dòng.
2. **SVM tuyến tính** đọc TF-IDF của câu hỏi.
3. Hai mô hình được hiệu chỉnh xác suất bằng các fold trong train.
4. **Ensemble** tính trung bình có trọng số của hai dự đoán.
5. Điểm được so sánh với ngưỡng để chọn Mistral hoặc Mixtral.

Nhãn `1` nghĩa là Mistral được chấm sai và Mixtral được chấm đúng trong benchmark.
Nhãn `0` gồm cả trường hợp hai LLM cùng sai. Vì vậy đúng nhãn Router và đúng câu trả lời là hai chỉ số khác nhau.

Cấu hình được chọn trên validation; test dùng để báo cáo cuối. Kết quả hiện tại áp dụng cho cặp LLM và phân bố RouterBench đã khảo sát.
""")
