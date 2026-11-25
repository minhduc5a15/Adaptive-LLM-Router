# Thí nghiệm Router v2 — v2_2026-11-25

## Quyết định được chốt trước test

- Chọn **A** bằng validation. Không xếp hạng rồi chọn lại bằng test.
- Chính: tối đa tỷ lệ câu được chấm đúng, với tiết kiệm validation ít nhất **20%** so với always Mixtral.
- Phụ: chi phí thấp nhất, với mức giảm tỷ lệ đúng validation không quá **2.0 điểm phần trăm**.
- Ngưỡng và trọng số là cố định sau lựa chọn. Ngân sách/giới hạn đúng trên validation không bảo đảm giữ nguyên trên test.
- Mô hình chọn theo log loss validation cho từng mục tiêu; sigmoid calibration theo 3 fold nhóm prompt trong train. Không fit TF-IDF trên validation/test.
- Thời gian huấn luyện và lựa chọn: 1273.1 giây. Artifact v1 được giữ nguyên.

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

| Bản | Đúng | Số câu đúng | Tiết kiệm | Gọi Mixtral |
| --- | ---: | ---: | ---: | ---: |
| A | 54.67% | 1545 | 20.25% | 78.66% |
| B_word | 54.49% | 1540 | 20.30% | 78.06% |
| B | 54.35% | 1536 | 20.24% | 77.78% |
| C | 54.28% | 1534 | 20.83% | 77.71% |
| D_word | 54.42% | 1538 | 20.48% | 76.79% |
| D | 54.42% | 1538 | 21.02% | 75.62% |

## Test cũ — tiêu chí chính, ngưỡng đã khóa

| Bản | Đúng | Số câu đúng | Tiết kiệm | Gọi Mixtral |
| --- | ---: | ---: | ---: | ---: |
| Always Small | 27.15% | 767 | 66.40% | 0.00% |
| Always Large | 56.21% | 1588 | 0.00% | 100.00% |
| Oracle | 61.88% | 1748 | 46.22% | 34.73% |
| A | 54.65% | 1544 | 18.70% | 79.86% |
| B_word | 54.62% | 1543 | 18.11% | 79.79% |
| B | 54.48% | 1539 | 18.95% | 78.73% |
| C | 54.48% | 1539 | 20.49% | 77.52% |
| D_word | 54.62% | 1543 | 18.41% | 78.69% |
| D | 54.23% | 1532 | 19.45% | 77.27% |

Oracle dùng đáp án chấm sẵn để chọn; không triển khai được. Các điểm đo đúng theo benchmark,
không phải accuracy dự đoán nhãn Router. Chi phí là ước tính lưu trong dataset.
A được chọn lại ngưỡng theo ngân sách chung; không so trực tiếp với điểm v1 54,41% vốn dùng tiêu chí phụ.

## Test cũ — tiêu chí phụ để đối chiếu v1

| Bản | Đúng | Số câu đúng | Tiết kiệm | Gọi Mixtral |
| --- | ---: | ---: | ---: | ---: |
| A | 54.41% | 1537 | 20.54% | 77.56% |
| B_word | 54.37% | 1536 | 19.79% | 78.05% |
| B | 54.48% | 1539 | 18.95% | 78.73% |
| C | 54.58% | 1542 | 18.72% | 79.33% |
| D_word | 54.34% | 1535 | 19.96% | 77.49% |
| D | 54.23% | 1532 | 19.45% | 77.27% |

## So sánh ghép cặp A với A (tiêu chí chính)

- Chênh lệch tỷ lệ đúng: **+0.00 điểm phần trăm**.
- Khoảng bootstrap 95%: **[+0.00, +0.00] điểm phần trăm**.
- Thêm 0 câu đúng và mất 0 câu đúng so với A.
- Chênh lệch tiết kiệm: +0.00 điểm phần trăm;
  khoảng 95% [+0.00, +0.00].

Validation chọn giữ A. So sánh A với chính A bằng 0 theo định nghĩa, không phải bằng chứng thống kê về các bản nâng cấp. Xem các so sánh thăm dò bên dưới.

Bootstrap 1000 lần lấy mẫu theo nhóm prompt, dùng cùng mẫu cho cả hai router; giữ mô hình/ngưỡng cố định.
Không tính độ bất định do training/model selection; không điều chỉnh multiple comparisons.
Không chọn người thắng từ các khoảng test.

### Các so sánh thăm dò với A

| Bản so với A | Chênh lệch đúng (điểm %) | Khoảng 95% | Chênh lệch tiết kiệm (điểm %) |
| --- | ---: | --- | ---: |
| B_word | -0.04 | [-0.39, +0.35] | -0.59 |
| B | -0.18 | [-0.64, +0.28] | +0.25 |
| C | -0.18 | [-0.67, +0.32] | +1.79 |
| D_word | -0.04 | [-0.71, +0.60] | -0.28 |
| D | -0.42 | [-1.17, +0.28] | +0.76 |

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
