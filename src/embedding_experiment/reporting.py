"""Trình bày rõ hiệu quả cây riêng, trọng số cố định, và chọn lại trên validation."""

import json
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/adaptive_llm_router_matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.embedding_experiment.common import ROOT


def table(data):
    lines = ["| Đầu vào cây | Đúng | Số câu đúng | Tiết kiệm | Gọi Mixtral | FN | FP |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in data.itertuples():
        lines.append(f"| {r.variant} | {r.answer_score:.2%} | {r.correct_count} | {r.cost_saving:.2%} | "
                     f"{r.large_rate:.2%} | {r.fn} | {r.fp} |")
    return "\n".join(lines)


def write_report(reports, design, selection, metrics, groups, summary):
    winner = selection["selected_variant"]
    test = metrics[metrics.split == "test"]
    large = test[test.variant == "Always Large"].iloc[0]
    old = test[(test.variant == "structure") & (test["mode"] == "fixed_ensemble")].iloc[0]
    chosen = test[(test.variant == winner) & (test["mode"] == "tuned_router")].iloc[0]
    paired = summary["exploratory_paired_comparisons"].get(winner, {}).get("tuned_router")
    evidence = ""
    if paired:
        ci = paired["answer_delta_95ci"]
        evidence = (f"So với structure/tuned_router, bản được chọn cứu được {paired['gained_correct']} câu "
                    f"nhưng mất {paired['lost_correct']} câu đúng. Khoảng bootstrap 95% cho chênh lệch "
                    f"tỷ lệ đúng: [{ci[0]*100:+.2f}, {ci[1]*100:+.2f}] điểm phần trăm.")
        if ci[0] <= 0 <= ci[1]:
            evidence += " Chưa có bằng chứng rõ về tăng tỷ lệ đúng; chưa nên thay baseline chỉ từ kết quả này."
    policy_lines = ["| Đầu vào cây | Depth | Min leaf | Trọng số cây | Ngưỡng |",
                    "| --- | ---: | ---: | ---: | ---: |"]
    for variant, config in selection["best_configs"].items():
        policy = selection["policies"][variant]["tuned_router"]
        policy_lines.append(f"| {variant} | {config['max_depth']} | {config['min_samples_leaf']} | "
                            f"{policy['tree_weight']:.1f} | {policy['threshold']:.2f} |")
    sections = []
    for mode, explanation in [
        ("tree_only", "Cây riêng: loại ảnh hưởng của SVM; trọng số cây = 1."),
        ("fixed_ensemble", "10% cây + 90% SVM như baseline; chỉ chọn lại ngưỡng trên validation."),
        ("tuned_router", "Chọn lại trọng số và ngưỡng trên validation, cho phép cả Tree/SVM riêng.")]:
        sections.append(f"## {mode}\n\n{explanation}\n\n### Validation\n\n" +
                        table(metrics[(metrics.split == "validation") & (metrics["mode"] == mode)]) +
                        "\n\n### Test cũ, chính sách đã khóa\n\n" +
                        table(metrics[(metrics.split == "test") & (metrics["mode"] == mode)]))
    error_lines = ["| Tập | Nhóm | Đầu vào cây | Số mẫu | Thật sự cần Mixtral | FN | FP | FP làm mất đúng |",
                   "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    focused = groups[(groups["mode"] == "fixed_ensemble") & groups["group"].isin(["all", "hellaswag", "word_count_ge_200"])]
    for r in focused.itertuples():
        error_lines.append(f"| {r.split} | {r.group} | {r.variant} | {r.n_samples} | {r.label1_count} | "
                           f"{r.fn} | {r.fp} | {r.fp_harmful} |")
    comparison_lines = ["| Đầu vào cây | Chế độ | Δ đúng (điểm %) | Khoảng 95% | Δ tiết kiệm (điểm %) |",
                        "| --- | --- | ---: | --- | ---: |"]
    for variant, modes in summary["exploratory_paired_comparisons"].items():
        for mode, result in modes.items():
            ci = result["answer_delta_95ci"]
            comparison_lines.append(f"| {variant} | {mode} | {result['answer_delta']*100:+.2f} | "
                                    f"[{ci[0]*100:+.2f}, {ci[1]*100:+.2f}] | {result['saving_delta']*100:+.2f} |")
    enc = json.loads((reports / "encoding_train_validation.json").read_text())
    enc.update(json.loads((reports / "encoding_evaluation.json").read_text()))
    encoding_lines = ["| Tập | Prompt | Chunk | Prompt cần nhiều chunk | Thời gian encode (giây) |",
                      "| --- | ---: | ---: | ---: | ---: |"]
    for split in ["train", "validation", "test"]:
        stats = enc[split]["stats"]
        encoding_lines.append(f"| {split} | {stats['prompts']} | {stats['chunks']} | {stats['multi_chunk_prompts']} | {stats['seconds']:.1f} |")
    verdict = ("Validation chọn giữ ba đặc trưng cấu trúc ở tiêu chí chính. Chưa có lý do từ lựa chọn này để thay baseline bằng embedding."
               if winner == "structure" else
               f"Validation chọn **{winner}** theo tiêu chí chi phí với giới hạn giảm tỷ lệ đúng. Đọc cả chi phí và tỷ lệ đúng test; lựa chọn này không phải bảo đảm cải thiện ngoài dữ liệu đã xem.")
    text = f"""# Decision Tree với embedding — {design['run_name']}

## Kết luận lựa chọn

{verdict}

- Tiêu chí chính khóa trước fitting: chi phí validation thấp nhất với tỷ lệ đúng ≥ always Mixtral − 2 điểm phần trăm.
- Chọn cấu hình cây bằng log loss validation; seed 42, cùng 3 fold nhóm prompt, sigmoid calibration.
- Giữ nguyên SVM đã fit của v1, không chọn lại C hoặc TF-IDF. Nhãn vẫn là nhãn định tuyến cũ.
- Ba đầu vào: structure = 3 đặc trưng; embedding = 384 chiều; combined = 3 + 384 = 387 chiều.
- Đã thử 24 cấu hình cây (8 cho mỗi đầu vào). Không fit encoder, không dùng score/cost/eval_name làm feature.
- Thời gian tạo embedding train/validation và chọn mô hình: {selection['training_and_encoding_seconds']:.1f} giây.
- Test đã được xem và phân tích lỗi trước thí nghiệm. Đây là đánh giá tiếp theo, không phải holdout độc lập mới.

### Đối chiếu với kết quả v1 trên cùng test

| Phương án | Câu trả lời đúng | Số câu đúng | Tiết kiệm chi phí LLM |
| --- | ---: | ---: | ---: |
| Luôn Mixtral | {large.answer_score:.2%} | {large.correct_count} | {large.cost_saving:.2%} |
| Router v1 (structure, fixed_ensemble) | {old.answer_score:.2%} | {old.correct_count} | {old.cost_saving:.2%} |
| Variant validation chọn ({winner}, tuned_router) | {chosen.answer_score:.2%} | {chosen.correct_count} | {chosen.cost_saving:.2%} |

Bản được chọn thay đổi {(chosen.answer_score-old.answer_score)*100:+.2f} điểm phần trăm tỷ lệ đúng
và {(chosen.cost_saving-old.cost_saving)*100:+.2f} điểm phần trăm tiết kiệm so với v1.
Mức giảm đúng so với luôn Mixtral trên test là {(large.answer_score-chosen.answer_score)*100:.2f} điểm phần trăm;
ràng buộc 2 điểm chỉ được áp dụng khi chọn trên validation.

{evidence}

### Cấu hình khóa — tuned_router

{chr(10).join(policy_lines)}

Trọng số cây bằng 0 nghĩa là chính sách được chọn thực tế chỉ dùng SVM,
không được diễn giải thành lợi ích của embedding trong cây. Tương tự, chính sách
tree_only với ngưỡng 0 và tỷ lệ gọi Mixtral 100% chỉ tái tạo always Mixtral.

## Encoder và prompt dài

Dùng `{design['encoder']['model_id']}`, revision `{design['encoder']['revision']}`, ONNX float32 trên CPU.
Mỗi chunk có tối đa 254 content tokens + CLS/SEP. Chia không chồng lặp, giữ toàn bộ token.
Mean pooling theo attention mask trong từng chunk, gồm CLS/SEP; gộp vector thô theo số content tokens rồi L2-normalize.
Đây là phép gộp đoạn tự định nghĩa, không phải `encode()` cắt 256 token mặc định của Sentence Transformers.
Phép gộp không mô hình hóa tương tác giữa các chunk; embedding ngữ nghĩa không đảm bảo dự đoán được lợi ích LLM.
Encoder đóng băng nên có thể tạo vector trước các fold mà không fit vào validation/test.

{chr(10).join(encoding_lines)}

Thời gian trên là throughput theo batch, không phải độ trễ một request. `inference_timings.csv` ghi riêng thời gian SVM/cây với embedding đã có.
**Chi phí benchmark trong các bảng chỉ là chi phí LLM đã lưu, chưa cộng chi phí tạo embedding/CPU.**
Không gọi API trả lời, không phát sinh hóa đơn suy luận Mistral/Mixtral.

{chr(10).join(sections)}

## Đọc lỗi nhãn đúng cách

- FN: nhãn 1 nhưng chọn Mistral; Mistral sai, Mixtral đúng.
- FP: nhãn 0 nhưng chọn Mixtral. Có thể cả hai cùng sai, cùng đúng, hoặc chỉ Mistral đúng.
- FP làm mất đúng: Mistral đúng, Mixtral sai. FP không đồng nghĩa câu trả lời sai.
- FN/FP là lỗi nhãn; tỷ lệ câu trả lời đúng lấy điểm LLM được chọn.

### Nhóm đã định trước — giữ trọng số 10% cây / 90% SVM

{chr(10).join(error_lines)}

Nhóm hellaswag và prompt ≥200 phần tách theo khoảng trắng được chọn từ phân tích test cũ; đây là kiểm tra thăm dò.
`group_errors.csv` có thêm kết quả từng benchmark và các chế độ khác, gồm mẫu số để tính tỷ lệ FN/FP.
Không sửa ngưỡng theo từng benchmark sau khi xem các bảng này.

## So sánh ghép cặp với structure trong cùng chế độ — test cũ

{chr(10).join(comparison_lines)}

Δ dương nghĩa là tăng tỷ lệ đúng/tiết kiệm. Bootstrap 1000 lần theo nhóm prompt, cùng mẫu cho hai router,
giữ mô hình và chính sách cố định. Các khoảng chưa điều chỉnh nhiều phép so sánh, không bao gồm độ bất định
huấn luyện/chọn tham số, không khắc phục được việc test đã xem. So sánh tại chính sách khóa, không coi chi phí các bản bằng nhau.
Khi khoảng Δ đúng chứa 0, chưa có bằng chứng rõ về tăng tỷ lệ đúng từ phép so sánh đó.

## Giới hạn

- Encoder tiếng Anh, giới hạn ngữ cảnh từng chunk; gộp trung bình có thể làm mờ chi tiết quyết định giữa các đáp án.
- Cây tách từng chiều embedding tại một nút; vector tốt cho ngữ nghĩa không nhất thiết phù hợp cây đơn lẻ.
- Pretraining encoder dùng dữ liệu ngoài (model card có WikiHow và nhiều nguồn khác); chưa loại trừ trùng lặp với nguồn benchmark.
- Tối ưu log loss không đồng nhất tối ưu chất lượng/chi phí; giữ cùng quy trình cho ba bản để so sánh.
- Giới hạn giảm đúng trên validation không đảm bảo trên test. Không chọn lại bản thắng bằng test.
- Muốn kết luận tổng quát cần dữ liệu thật sự chưa dùng; thay seed chia lại dữ liệu cũ không tạo kiểm chứng độc lập.

## Tái lập và đọc mã

Xem `docs/EMBEDDING_EXPERIMENT.md`. Thứ tự đọc:
`common.py` → `encoder.py` → `inference.py` → `train.py` → `evaluate.py`.

`protocol.json` lưu thiết kế và SHA256 nguồn lúc training; `selection.json` khóa cấu hình/artifact trước test;
`summary.json` ghi SHA256 nguồn lúc đánh giá. `validation_error_examples.csv` chứa ví dụ cải thiện/mất đúng trên validation.
`predictions.csv` ghi dự đoán mọi variant/mode trên validation/test; cache `.npy` ở data/processed được Git bỏ qua.

Nguồn encoder và pooling: [model card MiniLM](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2).
"""
    (reports / "RESULTS.md").write_text(text, encoding="utf-8")
    figure_dir = ROOT / "reports" / "figures" / design["run_name"]
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, split in zip(axes, ["validation", "test"]):
        points = metrics[(metrics.split == split) & (metrics["mode"] == "tuned_router")]
        for r in points.itertuples():
            ax.scatter(r.cost_saving * 100, r.answer_score * 100, marker="*" if r.variant == winner else "o", s=90)
            ax.annotate(r.variant, (r.cost_saving * 100, r.answer_score * 100), xytext=(5, 6), textcoords="offset points")
        ax.margins(x=.25, y=.25)
        ax.set(title=f"{split}: locked tuned policies", xlabel="LLM cost saving (%)", ylabel="Correct answers (%)")
        ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(figure_dir / "comparison.png", dpi=160)
    plt.close(fig)
