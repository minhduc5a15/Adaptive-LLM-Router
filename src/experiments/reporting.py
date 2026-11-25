"""Báo cáo so sánh theo protocol đã khóa; không chọn lại bằng test."""

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/adaptive_llm_router_matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def markdown_table(data):
    lines = ["| Bản | Đúng | Số câu đúng | Tiết kiệm | Gọi Mixtral |",
             "| --- | ---: | ---: | ---: | ---: |"]
    for row in data.itertuples():
        lines.append(f"| {row.experiment} | {row.answer_score:.2%} | {row.correct_count} | "
                     f"{row.cost_saving:.2%} | {row.large_rate:.2%} |")
    return "\n".join(lines)


def write_report(reports, design, selection, metrics, summary):
    winner = selection["selected_experiment"]
    validation = metrics[(metrics.split == "validation") & (metrics.objective == "fixed_budget")]
    test = metrics[(metrics.split == "test") & (metrics.objective.isin(["fixed_budget", "baseline"]))]
    secondary = metrics[(metrics.split == "test") & (metrics.objective == "quality_constraint")]
    comparison = summary["selected_vs_A"]
    low, high = comparison["answer_delta_95ci"]
    if winner == "A":
        interpretation = "Validation chọn giữ A. So sánh A với chính A bằng 0 theo định nghĩa, không phải bằng chứng thống kê về các bản nâng cấp. Xem các so sánh thăm dò bên dưới."
    elif low <= 0 <= high:
        interpretation = "Khoảng cho chênh lệch tỷ lệ đúng chứa 0; chưa có bằng chứng rõ về tăng tỷ lệ đúng so với A."
    elif low > 0:
        interpretation = "Khoảng mô tả cho chênh lệch tỷ lệ đúng nằm trên 0; vẫn cần xem chi phí thực đạt được và giới hạn test đã xem."
    else:
        interpretation = "Khoảng mô tả cho chênh lệch tỷ lệ đúng nằm dưới 0: bản chọn có kết quả thấp hơn A trên test này."
    exploratory_lines = ["| Bản so với A | Chênh lệch đúng (điểm %) | Khoảng 95% | Chênh lệch tiết kiệm (điểm %) |",
                         "| --- | ---: | --- | ---: |"]
    for name, item in summary["exploratory_comparisons_vs_A"].items():
        ci = item["answer_delta_95ci"]
        exploratory_lines.append(f"| {name} | {item['answer_delta']*100:+.2f} | "
                                 f"[{ci[0]*100:+.2f}, {ci[1]*100:+.2f}] | {item['saving_delta']*100:+.2f} |")
    exploratory_table = "\n".join(exploratory_lines)
    text = f"""# Thí nghiệm Router v2 — {design['run_name']}

## Quyết định được chốt trước test

- Chọn **{winner}** bằng validation. Không xếp hạng rồi chọn lại bằng test.
- Chính: tối đa tỷ lệ câu được chấm đúng, với tiết kiệm validation ít nhất **{design['target_saving']:.0%}** so với always Mixtral.
- Phụ: chi phí thấp nhất, với mức giảm tỷ lệ đúng validation không quá **{design['max_quality_drop']*100:.1f} điểm phần trăm**.
- Ngưỡng và trọng số là cố định sau lựa chọn. Ngân sách/giới hạn đúng trên validation không bảo đảm giữ nguyên trên test.
- Mô hình chọn theo log loss validation cho từng mục tiêu; sigmoid calibration theo 3 fold nhóm prompt trong train. Không fit TF-IDF trên validation/test.
- Thời gian huấn luyện và lựa chọn: {selection['training_seconds']:.1f} giây. Artifact v1 được giữ nguyên.

## Thiết kế

| Bản | Đặc trưng | Mục tiêu |
| --- | --- | --- |
| A | Cây 3 đặc trưng; TF-IDF từ đơn 5000 | Nhãn định tuyến cũ |
| B_word | Cây mở rộng; word 1–2 gram | Nhãn cũ |
| B | Như B_word + char_wb 3–5 gram | Nhãn cũ |
| C | Như A | Dự đoán đúng của từng LLM |
| D_word | Như B_word | Dự đoán đúng của từng LLM |
| D | Như B | Dự đoán đúng của từng LLM |

C/D có bốn bộ phân loại: Tree và SVM cho mỗi LLM. Kết hợp xác suất bằng cùng alpha,
sau đó gain = P(Mixtral đúng) − P(Mistral đúng). Gain có thể âm, không phải xác suất.
Nhãn cũ vẫn hợp lệ cho bài toán phân loại lựa chọn; đổi mục tiêu là giả thuyết cần kiểm chứng.

## Validation — tiêu chí chính

{markdown_table(validation)}

## Test cũ — tiêu chí chính, ngưỡng đã khóa

{markdown_table(test)}

Oracle dùng đáp án chấm sẵn để chọn; không triển khai được. Các điểm đo đúng theo benchmark,
không phải accuracy dự đoán nhãn Router. Chi phí là ước tính lưu trong dataset.
A được chọn lại ngưỡng theo ngân sách chung; không so trực tiếp với điểm v1 54,41% vốn dùng tiêu chí phụ.

## Test cũ — tiêu chí phụ để đối chiếu v1

{markdown_table(secondary)}

## So sánh ghép cặp {winner} với A (tiêu chí chính)

- Chênh lệch tỷ lệ đúng: **{comparison['answer_delta']*100:+.2f} điểm phần trăm**.
- Khoảng bootstrap 95%: **[{low*100:+.2f}, {high*100:+.2f}] điểm phần trăm**.
- Thêm {comparison['gained_correct']} câu đúng và mất {comparison['lost_correct']} câu đúng so với A.
- Chênh lệch tiết kiệm: {comparison['saving_delta']*100:+.2f} điểm phần trăm;
  khoảng 95% [{comparison['saving_delta_95ci'][0]*100:+.2f}, {comparison['saving_delta_95ci'][1]*100:+.2f}].

{interpretation}

Bootstrap 1000 lần lấy mẫu theo nhóm prompt, dùng cùng mẫu cho cả hai router; giữ mô hình/ngưỡng cố định.
Không tính độ bất định do training/model selection; không điều chỉnh multiple comparisons.
Không chọn người thắng từ các khoảng test.

### Các so sánh thăm dò với A

{exploratory_table}

Các khoảng trên chưa điều chỉnh cho nhiều phép so sánh, chỉ giúp mô tả kết quả.
Chi phí test của các bản không bằng nhau: phải đọc cả cột chênh lệch tiết kiệm.
Không dùng bảng này để thay bản được chọn bằng validation.

## Giới hạn và cách diễn giải

- Test này đã được xem ở v1 và dùng định hướng v2. Đây là phân tích tiếp theo, không phải holdout độc lập mới.
- Không có cam kết v2 tốt hơn; cần đọc cả tỷ lệ đúng và chi phí tại các chính sách đã khóa.
- A/B/C/D giúp tách đổi mục tiêu và đổi đặc trưng; *_word giúp tách tác dụng character n-gram.
  B so với A vẫn đổi đồng thời đặc trưng cây và word n-gram, chưa tách riêng hai yếu tố đó.
- Tăng số đặc trưng có thể làm học các dấu hiệu đặc thù benchmark; chưa có thí nghiệm giữ riêng toàn bộ benchmark mới.
- Dữ liệu khác thực sự chưa dùng cần được giữ kín để đánh giá xác nhận. Chia lại dữ liệu cũ không tạo tính độc lập mới.
- Ngưỡng cố định được chọn có xét tổng chi phí validation, không dùng chi phí đầu ra thực của prompt mới.
- Cây vẫn học dấu hiệu hình thức; hai xác suất đúng ước lượng có sai số và hiệu của chúng cũng có sai số.
- Pipeline cũ và demo Streamlit vẫn dùng v1. Dùng CLI experiments.predict để thử artifact thí nghiệm.

## File để kiểm tra

- `protocol.json`: thiết kế, grid, seed, phiên bản, checksum nguồn/dữ liệu trước training.
- `train_folds.csv`: fold hiệu chỉnh của từng sample_id, chung mọi target.
- `validation_candidates.csv`: log loss/Brier từng cấu hình; `validation_sweeps.csv`: mọi ngưỡng/trọng số.
- `validation_policies.csv`: Tree/SVM/Ensemble riêng cho cả hai mục tiêu.
- `selection.json`: chính sách đã khóa, bản được chọn, đường dẫn và SHA256 artifact.
- `metrics.csv`: validation/test với chính sách khóa, cùng baseline; `probability_metrics.csv`: xác suất theo từng target.
- `test_predictions.csv`: dự đoán từng dòng cho tiêu chí chính; `test_by_benchmark.csv`: phân tích từng nhóm.
- `summary.json`: đối chiếu ghép cặp, checksum test và mã nguồn lúc đánh giá.
  `protocol.json` giữ nguyên checksum mã nguồn lúc huấn luyện; sửa phần trình bày báo cáo không huấn luyện/chọn lại mô hình.

Đọc mã theo thứ tự: `config.py` → `features.py` → `models.py` → `inference.py` → `policy.py` → `run.py` → `evaluate.py`.
"""
    (reports / "RESULTS.md").write_text(text, encoding="utf-8")
    # Đồ thị chỉ dùng chính sách khóa, không quét ngưỡng trên test.
    figure_dir = Path(__file__).resolve().parents[2] / "reports" / "figures" / design["run_name"]
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, split in zip(axes, ["validation", "test"]):
        points = metrics[(metrics.split == split) & (metrics.objective == "fixed_budget")]
        for row in points.itertuples():
            ax.scatter(row.cost_saving*100, row.answer_score*100, s=70,
                       marker="*" if row.experiment == winner else "o")
            ax.annotate(row.experiment, (row.cost_saving*100, row.answer_score*100),
                        xytext=(5, 5), textcoords="offset points")
        ax.axvline(design["target_saving"]*100, linestyle="--", color="gray", label="Validation budget target")
        ax.set(xlabel="Cost saving vs always Mixtral (%)", ylabel="Correct answers (%)",
               title=f"{split}: frozen policies")
        ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(figure_dir / "comparison.png", dpi=160)
    plt.close(fig)
