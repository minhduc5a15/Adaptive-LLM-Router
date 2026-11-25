"""Báo cáo kết quả từ chính sách khóa, tách đúng nhãn khỏi đúng câu trả lời."""


def table(rows):
    lines = ["| Đầu vào SVM | Số câu đúng | Tỷ lệ đúng | Tiết kiệm LLM | Trọng số cây | Ngưỡng | FN | FP |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in rows.itertuples():
        lines.append(f"| {r.variant} | {r.correct_count} | {r.answer_score:.2%} | {r.cost_saving:.2%} | "
                     f"{r.tree_weight:.1f} | {r.threshold:.2f} | {r.fn} | {r.fp} |")
    return "\n".join(lines)


def write_report(reports, selection, metrics, groups, summary):
    winner = selection['selected_variant']
    test = metrics[metrics.split == 'test']
    old = test[(test.variant == 'tfidf') & (test['mode'] == 'fixed_ensemble')].iloc[0]
    chosen = test[(test.variant == winner) & (test['mode'] == 'tuned_router')].iloc[0]
    large = test[test.variant == 'Always Large'].iloc[0]
    sections = []
    for mode, description in [('svm_only', 'SVM riêng, không có cây.'),
                              ('fixed_ensemble', '10% cây cũ + 90% SVM, chọn lại ngưỡng trên validation.'),
                              ('tuned_router', 'Chọn trọng số cây từ 0 đến 1 và ngưỡng trên validation.')]:
        sections.append(f"## {mode}\n\n{description}\n\n### Validation\n\n" +
                        table(metrics[(metrics.split == 'validation') & (metrics['mode'] == mode)]) +
                        "\n\n### Test cũ\n\n" + table(test[test['mode'] == mode]))
    paired = ['| So với TF-IDF cùng chế độ | Chế độ | Δ đúng (điểm %) | Khoảng 95% | Cứu / mất đúng | Δ tiết kiệm (điểm %) |',
              '| --- | --- | ---: | --- | ---: | ---: |']
    for variant, modes in summary['paired_comparisons'].items():
        for mode, r in modes.items():
            ci = r['answer_delta_95ci']
            paired.append(f"| {variant} | {mode} | {r['answer_delta']*100:+.2f} | [{ci[0]*100:+.2f}, {ci[1]*100:+.2f}] | "
                          f"{r['gained_correct']} / {r['lost_correct']} | {r['saving_delta']*100:+.2f} |")
    errors = ['| Đầu vào SVM | Nhóm | Số mẫu | Nhãn 1 | FN | FP | FP làm mất đúng |',
              '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    focused = groups[(groups.split == 'test') & (groups['mode'] == 'fixed_ensemble') &
                     groups['group'].isin(['all', 'hellaswag', 'word_count_ge_200'])]
    for r in focused.itertuples():
        errors.append(f"| {r.variant} | {r.group} | {r.n_samples} | {r.label1_count} | {r.fn} | {r.fp} | {r.fp_harmful} |")
    configs = ', '.join(f"{v}: C={c['C']:g}" for v, c in selection['best_configs'].items())
    conclusion = ('Theo tiêu chí đã khóa, giữ router baseline TF-IDF. Các bản embedding có thể tăng số câu đúng '
                  'nhưng cần đối chiếu mức chi phí tăng thêm; chưa đủ để tuyên bố thay thế baseline.'
                  if winner == 'tfidf' else
                  'Variant này được chọn từ validation; kết quả test không được dùng để đổi lựa chọn.')
    text = f"""# Embedding cho SVM

## Lựa chọn và đối chiếu baseline

Validation chọn **{winner}**, khóa trước khi đọc test trong lần chạy này.
Tiêu chí: chi phí thấp nhất với tỷ lệ đúng validation ≥ always Mixtral − 2 điểm phần trăm.
Cấu hình SVM theo log loss validation: {configs}.

{conclusion}

- Router v1: {old.correct_count}/2825 câu đúng (**{old.answer_score:.2%}**), tiết kiệm **{old.cost_saving:.2%}**.
- Router được chọn: {chosen.correct_count}/2825 câu đúng (**{chosen.answer_score:.2%}**), tiết kiệm **{chosen.cost_saving:.2%}**.
- Thay đổi so với v1: {(chosen.answer_score-old.answer_score)*100:+.2f} điểm % đúng và {(chosen.cost_saving-old.cost_saving)*100:+.2f} điểm % tiết kiệm.
- Luôn Mixtral: {large.answer_score:.2%} đúng. Router được chọn giảm {(large.answer_score-chosen.answer_score)*100:.2f} điểm % trên test.
- Thời gian fit/chọn cấu hình và kiểm tra cache: {selection['training_seconds']:.1f} giây; không bao gồm thời gian tạo embedding ở thí nghiệm trước.

Cần đọc đồng thời tỷ lệ đúng và chi phí. Tăng tỷ lệ đúng khi gọi Mixtral nhiều hơn chưa chứng minh router tốt hơn tại cùng ngân sách.
Trọng số cây 0 nghĩa là chỉ dùng SVM; trọng số 1 nghĩa là SVM không đóng góp.

## Thiết kế

- Giữ nguyên cây v1 với 3 đặc trưng, nhãn, train/validation/test và 3 fold theo nhóm prompt.
- Ba đầu vào SVM: TF-IDF 5000 mục, MiniLM 384 chiều, hoặc ghép hai block rồi chuẩn hóa L2.
- C = 0.1 / 1 / 10: tổng 9 cấu hình. LinearSVC max_iter=10000; cảnh báo không hội tụ khiến run dừng.
- Hiệu chỉnh sigmoid, ensemble=False. TF-IDF nằm trong pipeline và chỉ fit phần train của từng fold.
- Encoder đóng băng, xử lý toàn bộ token bằng cách chia đoạn giống thí nghiệm cây; tái sử dụng cache có checksum.
- Bản TF-IDF phải tái tạo xác suất SVM v1 trên cả validation và test đến sai số 1e-12.
- Chính sách được chọn chỉ bằng validation. Không chọn lại mô hình theo kết quả test.

{chr(10).join(chr(10) + s + chr(10) for s in sections)}

## So sánh ghép cặp trên test cũ

{chr(10).join(paired)}

Bootstrap 1000 lần theo nhóm prompt, giữ mô hình/chính sách cố định; không điều chỉnh nhiều phép so sánh.
Khoảng Δ đúng chứa 0 nghĩa là chưa có bằng chứng rõ về cải thiện tỷ lệ đúng từ so sánh đó.
Các chính sách không nhất thiết có cùng chi phí. Khoảng bất định cho Δ tiết kiệm có trong summary.json.

## Nhóm lỗi — fixed_ensemble

{chr(10).join(errors)}

FN: Mistral sai/Mixtral đúng nhưng chọn Mistral. FP: chọn Mixtral khi nhãn 0;
chỉ nhóm FP làm mất đúng là Mistral đúng/Mixtral sai. Các FP còn lại có thể cả hai cùng đúng hoặc cùng sai.
Nhóm prompt dài dùng ít nhất 200 phần tách theo khoảng trắng. Đây là phân tích thăm dò từ lỗi đã xem.

## Giới hạn và tái lập

Test đã được xem ở các thí nghiệm trước; đây không phải holdout độc lập mới.
Giới hạn giảm đúng trên validation không bảo đảm trên test. Không suy rộng kết quả của một encoder sang mọi embedding.
Chi phí chỉ gồm chi phí LLM lưu trong RouterBench, chưa cộng CPU tạo embedding; không gọi API Mistral/Mixtral.
Embedding có thể bỏ sót chi tiết đáp án khi gộp đoạn. Chưa kiểm tra trùng nguồn pretraining encoder với benchmark.

Xem `docs/SVM_EMBEDDING_EXPERIMENT.md`. Protocol, checksum nguồn/dữ liệu, fold, bảng tham số,
chính sách, dự đoán và bootstrap đều được lưu cạnh báo cáo. Artifact và cache được Git bỏ qua.
"""
    (reports / 'RESULTS.md').write_text(text, encoding='utf-8')
