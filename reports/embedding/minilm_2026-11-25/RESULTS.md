# Decision Tree với embedding — minilm_2026-11-25

## Kết luận lựa chọn

Validation chọn **embedding** theo tiêu chí chi phí với giới hạn giảm tỷ lệ đúng. Đọc cả chi phí và tỷ lệ đúng test; lựa chọn này không phải bảo đảm cải thiện ngoài dữ liệu đã xem.

- Tiêu chí chính khóa trước fitting: chi phí validation thấp nhất với tỷ lệ đúng ≥ always Mixtral − 2 điểm phần trăm.
- Chọn cấu hình cây bằng log loss validation; seed 42, cùng 3 fold nhóm prompt, sigmoid calibration.
- Giữ nguyên SVM đã fit của v1, không chọn lại C hoặc TF-IDF. Nhãn vẫn là nhãn định tuyến cũ.
- Ba đầu vào: structure = 3 đặc trưng; embedding = 384 chiều; combined = 3 + 384 = 387 chiều.
- Đã thử 24 cấu hình cây (8 cho mỗi đầu vào). Không fit encoder, không dùng score/cost/eval_name làm feature.
- Thời gian tạo embedding train/validation và chọn mô hình: 738.3 giây.
- Test đã được xem và phân tích lỗi trước thí nghiệm. Đây là đánh giá tiếp theo, không phải holdout độc lập mới.

### Đối chiếu với kết quả v1 trên cùng test

| Phương án | Câu trả lời đúng | Số câu đúng | Tiết kiệm chi phí LLM |
| --- | ---: | ---: | ---: |
| Luôn Mixtral | 56.21% | 1588 | 0.00% |
| Router v1 (structure, fixed_ensemble) | 54.41% | 1537 | 20.54% |
| Variant validation chọn (embedding, tuned_router) | 54.27% | 1533 | 20.79% |

Bản được chọn thay đổi -0.14 điểm phần trăm tỷ lệ đúng
và +0.25 điểm phần trăm tiết kiệm so với v1.
Mức giảm đúng so với luôn Mixtral trên test là 1.95 điểm phần trăm;
ràng buộc 2 điểm chỉ được áp dụng khi chọn trên validation.

So với structure/tuned_router, bản được chọn cứu được 6 câu nhưng mất 10 câu đúng. Khoảng bootstrap 95% cho chênh lệch tỷ lệ đúng: [-0.42, +0.14] điểm phần trăm. Chưa có bằng chứng rõ về tăng tỷ lệ đúng; chưa nên thay baseline chỉ từ kết quả này.

### Cấu hình khóa — tuned_router

| Đầu vào cây | Depth | Min leaf | Trọng số cây | Ngưỡng |
| --- | ---: | ---: | ---: | ---: |
| structure | 5 | 20 | 0.1 | 0.21 |
| embedding | 5 | 20 | 0.1 | 0.22 |
| combined | 5 | 50 | 0.0 | 0.21 |

Trọng số cây bằng 0 nghĩa là chính sách được chọn thực tế chỉ dùng SVM,
không được diễn giải thành lợi ích của embedding trong cây. Tương tự, chính sách
tree_only với ngưỡng 0 và tỷ lệ gọi Mixtral 100% chỉ tái tạo always Mixtral.

## Encoder và prompt dài

Dùng `sentence-transformers/all-MiniLM-L6-v2`, revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`, ONNX float32 trên CPU.
Mỗi chunk có tối đa 254 content tokens + CLS/SEP. Chia không chồng lặp, giữ toàn bộ token.
Mean pooling theo attention mask trong từng chunk, gồm CLS/SEP; gộp vector thô theo số content tokens rồi L2-normalize.
Đây là phép gộp đoạn tự định nghĩa, không phải `encode()` cắt 256 token mặc định của Sentence Transformers.
Phép gộp không mô hình hóa tương tác giữa các chunk; embedding ngữ nghĩa không đảm bảo dự đoán được lợi ích LLM.
Encoder đóng băng nên có thể tạo vector trước các fold mà không fit vào validation/test.

| Tập | Prompt | Chunk | Prompt cần nhiều chunk | Thời gian encode (giây) |
| --- | ---: | ---: | ---: | ---: |
| train | 22605 | 27562 | 4848 | 450.9 |
| validation | 2826 | 3446 | 614 | 57.2 |
| test | 2825 | 3499 | 657 | 57.5 |

Thời gian trên là throughput theo batch, không phải độ trễ một request. `inference_timings.csv` ghi riêng thời gian SVM/cây với embedding đã có.
**Chi phí benchmark trong các bảng chỉ là chi phí LLM đã lưu, chưa cộng chi phí tạo embedding/CPU.**
Không gọi API trả lời, không phát sinh hóa đơn suy luận Mistral/Mixtral.

## tree_only

Cây riêng: loại ảnh hưởng của SVM; trọng số cây = 1.

### Validation

| Đầu vào cây | Đúng | Số câu đúng | Tiết kiệm | Gọi Mixtral | FN | FP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| structure | 56.09% | 1585 | 2.36% | 94.41% | 16 | 1746 |
| embedding | 56.33% | 1592 | 0.00% | 100.00% | 0 | 1888 |
| combined | 54.42% | 1538 | 17.63% | 80.04% | 73 | 1397 |

### Test cũ, chính sách đã khóa

| Đầu vào cây | Đúng | Số câu đúng | Tiết kiệm | Gọi Mixtral | FN | FP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| structure | 55.82% | 1577 | 2.34% | 94.37% | 21 | 1706 |
| embedding | 56.21% | 1588 | 0.00% | 100.00% | 0 | 1844 |
| combined | 54.41% | 1537 | 16.64% | 80.78% | 82 | 1383 |
## fixed_ensemble

10% cây + 90% SVM như baseline; chỉ chọn lại ngưỡng trên validation.

### Validation

| Đầu vào cây | Đúng | Số câu đúng | Tiết kiệm | Gọi Mixtral | FN | FP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| structure | 54.49% | 1540 | 21.82% | 76.54% | 81 | 1306 |
| embedding | 54.49% | 1540 | 22.18% | 76.19% | 83 | 1298 |
| combined | 54.42% | 1538 | 21.39% | 77.00% | 81 | 1319 |

### Test cũ, chính sách đã khóa

| Đầu vào cây | Đúng | Số câu đúng | Tiết kiệm | Gọi Mixtral | FN | FP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| structure | 54.41% | 1537 | 20.54% | 77.56% | 96 | 1306 |
| embedding | 54.27% | 1533 | 20.79% | 77.13% | 101 | 1299 |
| combined | 54.41% | 1537 | 19.77% | 78.27% | 94 | 1324 |
## tuned_router

Chọn lại trọng số và ngưỡng trên validation, cho phép cả Tree/SVM riêng.

### Validation

| Đầu vào cây | Đúng | Số câu đúng | Tiết kiệm | Gọi Mixtral | FN | FP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| structure | 54.49% | 1540 | 21.82% | 76.54% | 81 | 1306 |
| embedding | 54.49% | 1540 | 22.18% | 76.19% | 83 | 1298 |
| combined | 54.42% | 1538 | 21.50% | 76.68% | 83 | 1312 |

### Test cũ, chính sách đã khóa

| Đầu vào cây | Đúng | Số câu đúng | Tiết kiệm | Gọi Mixtral | FN | FP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| structure | 54.41% | 1537 | 20.54% | 77.56% | 96 | 1306 |
| embedding | 54.27% | 1533 | 20.79% | 77.13% | 101 | 1299 |
| combined | 54.44% | 1538 | 20.06% | 77.63% | 96 | 1308 |

## Đọc lỗi nhãn đúng cách

- FN: nhãn 1 nhưng chọn Mistral; Mistral sai, Mixtral đúng.
- FP: nhãn 0 nhưng chọn Mixtral. Có thể cả hai cùng sai, cùng đúng, hoặc chỉ Mistral đúng.
- FP làm mất đúng: Mistral đúng, Mixtral sai. FP không đồng nghĩa câu trả lời sai.
- FN/FP là lỗi nhãn; tỷ lệ câu trả lời đúng lấy điểm LLM được chọn.

### Nhóm đã định trước — giữ trọng số 10% cây / 90% SVM

| Tập | Nhóm | Đầu vào cây | Số mẫu | Thật sự cần Mixtral | FN | FP | FP làm mất đúng |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| validation | all | structure | 2826 | 938 | 81 | 1306 | 105 |
| validation | hellaswag | structure | 1015 | 215 | 68 | 362 | 13 |
| validation | word_count_ge_200 | structure | 453 | 111 | 26 | 155 | 13 |
| validation | all | embedding | 2826 | 938 | 83 | 1298 | 103 |
| validation | hellaswag | embedding | 1015 | 215 | 70 | 355 | 13 |
| validation | word_count_ge_200 | embedding | 453 | 111 | 26 | 156 | 13 |
| validation | all | combined | 2826 | 938 | 81 | 1319 | 107 |
| validation | hellaswag | combined | 1015 | 215 | 68 | 370 | 14 |
| validation | word_count_ge_200 | combined | 453 | 111 | 25 | 159 | 14 |
| test | all | structure | 2825 | 981 | 96 | 1306 | 115 |
| test | hellaswag | structure | 1000 | 208 | 79 | 374 | 22 |
| test | word_count_ge_200 | structure | 496 | 124 | 44 | 173 | 11 |
| test | all | embedding | 2825 | 981 | 101 | 1299 | 114 |
| test | hellaswag | embedding | 1000 | 208 | 81 | 376 | 23 |
| test | word_count_ge_200 | embedding | 496 | 124 | 44 | 176 | 11 |
| test | all | combined | 2825 | 981 | 94 | 1324 | 117 |
| test | hellaswag | combined | 1000 | 208 | 76 | 388 | 24 |
| test | word_count_ge_200 | combined | 496 | 124 | 40 | 177 | 12 |

Nhóm hellaswag và prompt ≥200 phần tách theo khoảng trắng được chọn từ phân tích test cũ; đây là kiểm tra thăm dò.
`group_errors.csv` có thêm kết quả từng benchmark và các chế độ khác, gồm mẫu số để tính tỷ lệ FN/FP.
Không sửa ngưỡng theo từng benchmark sau khi xem các bảng này.

## So sánh ghép cặp với structure trong cùng chế độ — test cũ

| Đầu vào cây | Chế độ | Δ đúng (điểm %) | Khoảng 95% | Δ tiết kiệm (điểm %) |
| --- | --- | ---: | --- | ---: |
| embedding | tree_only | +0.39 | [+0.00, +0.81] | -2.34 |
| embedding | fixed_ensemble | -0.14 | [-0.42, +0.14] | +0.25 |
| embedding | tuned_router | -0.14 | [-0.42, +0.14] | +0.25 |
| combined | tree_only | -1.42 | [-2.05, -0.74] | +14.30 |
| combined | fixed_ensemble | +0.00 | [-0.25, +0.25] | -0.77 |
| combined | tuned_router | +0.04 | [-0.25, +0.32] | -0.48 |

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
