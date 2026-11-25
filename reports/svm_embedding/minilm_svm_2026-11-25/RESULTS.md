# Embedding cho SVM

## Lựa chọn và đối chiếu baseline

Validation chọn **tfidf**, khóa trước khi đọc test trong lần chạy này.
Tiêu chí: chi phí thấp nhất với tỷ lệ đúng validation ≥ always Mixtral − 2 điểm phần trăm.
Cấu hình SVM theo log loss validation: tfidf: C=0.1, embedding: C=0.1, combined: C=0.1.

Theo tiêu chí đã khóa, giữ router baseline TF-IDF. Các bản embedding có thể tăng số câu đúng nhưng cần đối chiếu mức chi phí tăng thêm; chưa đủ để tuyên bố thay thế baseline.

- Router v1: 1537/2825 câu đúng (**54.41%**), tiết kiệm **20.54%**.
- Router được chọn: 1537/2825 câu đúng (**54.41%**), tiết kiệm **20.54%**.
- Thay đổi so với v1: +0.00 điểm % đúng và +0.00 điểm % tiết kiệm.
- Luôn Mixtral: 56.21% đúng. Router được chọn giảm 1.81 điểm % trên test.
- Thời gian fit/chọn cấu hình và kiểm tra cache: 113.7 giây; không bao gồm thời gian tạo embedding ở thí nghiệm trước.

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


## svm_only

SVM riêng, không có cây.

### Validation

| Đầu vào SVM | Số câu đúng | Tỷ lệ đúng | Tiết kiệm LLM | Trọng số cây | Ngưỡng | FN | FP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| tfidf | 1538 | 54.42% | 21.50% | 0.0 | 0.21 | 83 | 1312 |
| embedding | 1541 | 54.53% | 17.90% | 0.0 | 0.19 | 68 | 1432 |
| combined | 1542 | 54.56% | 19.17% | 0.0 | 0.19 | 71 | 1368 |

### Test cũ

| Đầu vào SVM | Số câu đúng | Tỷ lệ đúng | Tiết kiệm LLM | Trọng số cây | Ngưỡng | FN | FP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| tfidf | 1538 | 54.44% | 20.06% | 0.0 | 0.21 | 96 | 1308 |
| embedding | 1544 | 54.65% | 16.76% | 0.0 | 0.19 | 75 | 1413 |
| combined | 1559 | 55.19% | 17.36% | 0.0 | 0.19 | 68 | 1366 |


## fixed_ensemble

10% cây cũ + 90% SVM, chọn lại ngưỡng trên validation.

### Validation

| Đầu vào SVM | Số câu đúng | Tỷ lệ đúng | Tiết kiệm LLM | Trọng số cây | Ngưỡng | FN | FP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| tfidf | 1540 | 54.49% | 21.82% | 0.1 | 0.21 | 81 | 1306 |
| embedding | 1541 | 54.53% | 18.46% | 0.1 | 0.19 | 68 | 1421 |
| combined | 1540 | 54.49% | 19.47% | 0.1 | 0.19 | 74 | 1369 |

### Test cũ

| Đầu vào SVM | Số câu đúng | Tỷ lệ đúng | Tiết kiệm LLM | Trọng số cây | Ngưỡng | FN | FP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| tfidf | 1537 | 54.41% | 20.54% | 0.1 | 0.21 | 96 | 1306 |
| embedding | 1549 | 54.83% | 16.95% | 0.1 | 0.19 | 71 | 1408 |
| combined | 1556 | 55.08% | 17.38% | 0.1 | 0.19 | 70 | 1367 |


## tuned_router

Chọn trọng số cây từ 0 đến 1 và ngưỡng trên validation.

### Validation

| Đầu vào SVM | Số câu đúng | Tỷ lệ đúng | Tiết kiệm LLM | Trọng số cây | Ngưỡng | FN | FP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| tfidf | 1540 | 54.49% | 21.82% | 0.1 | 0.21 | 81 | 1306 |
| embedding | 1538 | 54.42% | 18.93% | 0.2 | 0.19 | 71 | 1407 |
| combined | 1539 | 54.46% | 20.56% | 0.3 | 0.19 | 77 | 1343 |

### Test cũ

| Đầu vào SVM | Số câu đúng | Tỷ lệ đúng | Tiết kiệm LLM | Trọng số cây | Ngưỡng | FN | FP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| tfidf | 1537 | 54.41% | 20.54% | 0.1 | 0.21 | 96 | 1306 |
| embedding | 1549 | 54.83% | 17.23% | 0.2 | 0.19 | 71 | 1395 |
| combined | 1543 | 54.62% | 18.85% | 0.3 | 0.19 | 82 | 1340 |


## So sánh ghép cặp trên test cũ

| So với TF-IDF cùng chế độ | Chế độ | Δ đúng (điểm %) | Khoảng 95% | Cứu / mất đúng | Δ tiết kiệm (điểm %) |
| --- | --- | ---: | --- | ---: | ---: |
| embedding | svm_only | +0.21 | [-0.50, +0.89] | 52 / 46 | -3.30 |
| embedding | fixed_ensemble | +0.42 | [-0.21, +1.06] | 50 / 38 | -3.59 |
| embedding | tuned_router | +0.42 | [-0.18, +1.06] | 48 / 36 | -3.31 |
| combined | svm_only | +0.74 | [+0.25, +1.27] | 39 / 18 | -2.70 |
| combined | fixed_ensemble | +0.67 | [+0.18, +1.17] | 36 / 17 | -3.16 |
| combined | tuned_router | +0.21 | [-0.28, +0.67] | 29 / 23 | -1.69 |

Bootstrap 1000 lần theo nhóm prompt, giữ mô hình/chính sách cố định; không điều chỉnh nhiều phép so sánh.
Khoảng Δ đúng chứa 0 nghĩa là chưa có bằng chứng rõ về cải thiện tỷ lệ đúng từ so sánh đó.
Các chính sách không nhất thiết có cùng chi phí. Khoảng bất định cho Δ tiết kiệm có trong summary.json.

## Nhóm lỗi — fixed_ensemble

| Đầu vào SVM | Nhóm | Số mẫu | Nhãn 1 | FN | FP | FP làm mất đúng |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| tfidf | all | 2825 | 981 | 96 | 1306 | 115 |
| tfidf | hellaswag | 1000 | 208 | 79 | 374 | 22 |
| tfidf | word_count_ge_200 | 496 | 124 | 44 | 173 | 11 |
| embedding | all | 2825 | 981 | 71 | 1408 | 128 |
| embedding | hellaswag | 1000 | 208 | 59 | 445 | 31 |
| embedding | word_count_ge_200 | 496 | 124 | 31 | 209 | 15 |
| combined | all | 2825 | 981 | 70 | 1367 | 122 |
| combined | hellaswag | 1000 | 208 | 56 | 435 | 27 |
| combined | word_count_ge_200 | 496 | 124 | 32 | 196 | 13 |

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
