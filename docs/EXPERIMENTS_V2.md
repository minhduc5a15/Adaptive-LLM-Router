# Hướng dẫn thí nghiệm Router v2

Lần chạy hoàn chỉnh ngày 25/11/2026: [báo cáo kết quả](../reports/experiments/v2_2026-11-25/RESULTS.md). Đã thử 75 cấu hình, validation chọn A; chưa có bằng chứng rõ về cải thiện của các bản mới. Mô hình và chính sách v1 được giữ nguyên.

## 1. Câu hỏi nghiên cứu

**Có thể tăng tỷ lệ câu trả lời được chấm đúng, trong khi vẫn tiết kiệm ít nhất 20% chi phí trên validation, bằng cách cải tiến đặc trưng hoặc mục tiêu học không?**

Đây là giả thuyết, chưa phải kết luận. Quy trình này giữ nguyên dữ liệu Mistral/Mixtral, ba tập và seed của v1. Không huấn luyện hoặc gọi hai LLM. Script v1, `models/router.joblib`, các báo cáo v1 và demo Streamlit được giữ nguyên.

Tiêu chí chính: tối đa tỷ lệ câu trả lời đúng với tiết kiệm validation ≥ 20%. Tiêu chí phụ: tối thiểu chi phí với tỷ lệ đúng validation giảm không quá 2 **điểm phần trăm** so với always Mixtral, để đối chiếu v1. Hai tiêu chí có thể chọn ngưỡng/trọng số khác nhau. Bản cuối được chọn theo tiêu chí chính.

Ngân sách được kiểm tra trên validation; tỷ lệ tiết kiệm thực tế trên test có thể cao hoặc thấp hơn 20%. Không thay ngưỡng sau khi xem test để làm đẹp kết quả.

## 2. Các thí nghiệm

| Tên | Đặc trưng Tree | Đặc trưng SVM | Mục tiêu học |
| --- | --- | --- | --- |
| A | 3 đặc trưng v1 | Từ đơn, tối đa 5000 | Nhãn định tuyến cũ |
| B_word | 9 đặc trưng | Word n-gram 1–2, tối đa 15000 | Nhãn cũ |
| B | 9 đặc trưng | Như B_word + char_wb 3–5, tối đa 20000 | Nhãn cũ |
| C | Như A | Như A | Đúng/sai của từng LLM |
| D_word | Như B_word | Như B_word | Đúng/sai của từng LLM |
| D | Như B | Như B | Đúng/sai của từng LLM |

- A → C: ảnh hưởng của đổi mục tiêu học khi giữ biểu diễn đầu vào.
- A → B: ảnh hưởng của gói đặc trưng mới khi giữ mục tiêu học.
- B_word → B và D_word → D: đóng góp character n-gram.
- B → D: đổi mục tiêu trên cùng biểu diễn mới.
- Các cặp A/B và C/D thay cả đặc trưng cây lẫn từ vựng SVM; không đủ để kết luận từng thay đổi riêng lẻ tốt hơn.

Cả 6 bản có cùng grid cho độ sâu, kích thước lá và C. Các mô hình chung `(thuật toán, profile, target)` được fit một lần rồi tái sử dụng; không coi các mô hình trùng nhau là các lần thử độc lập.

## 3. Nhãn và phép tính của C/D

A/B vẫn học `label = 1` khi Mistral sai và Mixtral đúng. Nhãn này hợp lệ cho việc dự đoán lựa chọn, nhưng gộp các mức lợi ích khác nhau ở nhãn 0.

C/D có bốn bộ phân loại nhị phân:

| Bộ phân loại | X | y |
| --- | --- | --- |
| Tree cho Mistral | Đặc trưng từ prompt | `small_score` |
| Tree cho Mixtral | Đặc trưng từ prompt | `large_score` |
| SVM cho Mistral | TF-IDF từ prompt | `small_score` |
| SVM cho Mixtral | TF-IDF từ prompt | `large_score` |

Đây vẫn là **hai thuật toán**, với bốn lần áp dụng. `small_score` và `large_score` chỉ là mục tiêu học và dữ liệu đánh giá, không phải đầu vào dự đoán. Vì CSV hiện tại đã có hai cột này, không cần sửa tiền xử lý hoặc chia lại tập.

Với trọng số cây alpha:

```text
p_small = alpha × p_tree_small + (1 − alpha) × p_svm_small
p_large = alpha × p_tree_large + (1 − alpha) × p_svm_large
gain = p_large − p_small
chọn Mixtral nếu gain ≥ threshold, ngược lại chọn Mistral
```

`gain` nằm trong [-1, 1]; không phải xác suất. Hiệu hai xác suất bằng lợi ích kỳ vọng với điểm nhị phân, không đòi hỏi hai LLM độc lập:

```text
E[large_score − small_score | prompt]
  = P(Mistral sai, Mixtral đúng | prompt)
  − P(Mistral đúng, Mixtral sai | prompt)
```

Hai xác suất đều được ước lượng, nên hiệu có thể sai. Không mặc định cách học này luôn tốt hơn nhãn cũ.

## 4. Đặc trưng: đếm cụ thể gì?

| Cột | Định nghĩa |
| --- | --- |
| `char_count` | Số ký tự, gồm khoảng trắng |
| `word_count` | Số phần sau `split()` theo khoảng trắng |
| `line_count` | Số ký tự xuống dòng + 1, gồm dòng trống |
| `code_fence_count` | Số dấu ba backtick; một khối hoàn chỉnh thường có 2 dấu |
| `math_symbol_count` | Số ký tự trong tập `+ - * / = < > ^ % √ ∑ ∫ ≤ ≥ ≠` |
| `question_mark_count` | Số dấu `?`, không đồng nghĩa số yêu cầu trong prompt |
| `digit_ratio` | Số ký tự số / số ký tự; prompt rỗng trả về 0 |
| `avg_word_length` | Tổng độ dài các phần tách theo khoảng trắng / số phần |
| `sentence_end_count` | Số cụm dấu `. ! ?`, không phải bộ tách câu ngữ nghĩa |

Đây là dấu hiệu hình thức, không phải điểm độ khó. Ví dụ `-` cũng có thể là dấu đầu dòng.

TF-IDF mở rộng dùng `min_df=3`. `char_wb` lấy n-gram bên trong ranh giới từ với khoảng trắng đệm; không đọc toàn bộ cú pháp code. Hai nhánh word/char chuẩn hóa L2 riêng rồi ghép bằng `FeatureUnion`; ma trận giữ sparse. Giới hạn 15000/20000 là số đặc trưng tối đa, không bảo đảm học đủ số đó. C của SVM được chọn lại cho mỗi profile.

## 5. Ngăn rò rỉ và lưu dấu vết

1. Đọc và kiểm tra train/validation, giữ nguyên nhóm prompt như v1.
2. Ghi `protocol.json` trước fitting: seed, grid, mục tiêu, phiên bản thư viện, SHA256 dữ liệu và mã nguồn.
3. Dùng 3 fold `StratifiedGroupKFold`, stratify theo nhãn định tuyến cũ, dùng cùng fold cho mọi target; kiểm tra mỗi target có cả hai lớp ở mỗi fold.
4. TF-IDF nằm **trong** pipeline của `CalibratedClassifierCV`. Mỗi fold fit từ vựng trên phần train của fold đó.
5. Với `ensemble=False`, sigmoid học từ dự đoán ngoài fold; mô hình cơ sở cuối fit toàn train. Validation không tham gia fitting.
6. Chọn cấu hình theo log loss validation cho từng thuật toán/profile/target. Chọn trọng số và ngưỡng bằng validation theo tiêu chí đã định.
7. Lưu artifact cho mọi bản và `selection.json` trước khi kết thúc lệnh train. Script train không nạp test.
8. Lệnh evaluate kiểm tra checksum, nạp artifact rồi đánh giá test với ngưỡng cố định. Không fit/chọn lại mô hình.

Cả lựa chọn cấu hình và ngưỡng dùng cùng validation, nên điểm validation có thể lạc quan. Giữ lưới thử nghiệm hữu hạn, công khai mọi cấu hình. Không coi validation là ước lượng tổng quát hóa độc lập.

## 6. Chi phí được dùng ở đâu?

Chỉ khi tính chỉ số và chọn ngưỡng trên validation. Khi nhận prompt mới, router chỉ dùng prompt và chính sách cố định, không biết trước số token đầu ra hoặc chi phí thực tế của hai LLM.

Vì vậy đây là chính sách ngưỡng cố định được chọn **có xét chi phí**, chưa phải router ước lượng chi phí riêng cho từng prompt. Giá trong dataset là chi phí benchmark đã lưu, không phải giá API hiện hành.

## 7. Lệnh chạy

Từ thư mục gốc dự án, dùng Python trong `.venv`:

```bash
# Tạo một run mới. Tên đã tồn tại sẽ bị từ chối để giữ kết quả cũ.
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m src.experiments.run --run-name v2_2026-11-25

# Chỉ chạy sau khi run hoàn thành và selection.json đã được ghi.
.venv/bin/python -m src.experiments.evaluate --run-name v2_2026-11-25

# Mặc định chọn bản được chọn trên validation, không mặc định D.
.venv/bin/python -m src.experiments.predict --run-name v2_2026-11-25 --prompt "Explain binary search."

# Xem riêng bản D, kể cả khi D không phải bản được chọn.
.venv/bin/python -m src.experiments.predict --run-name v2_2026-11-25 --experiment D --prompt "Explain binary search."

.venv/bin/python -m pytest -q
```

Để lặp lại thí nghiệm, dùng tên run mới. Hai biến môi trường trên giới hạn luồng tính toán, không đổi thuật toán. Không cần Modal hoặc GPU cho quy trình này.

Mô hình nằm ở `models/experiments/<run>/` và được Git bỏ qua. Báo cáo nằm ở `reports/experiments/<run>/`; biểu đồ ở `reports/figures/<run>/comparison.png`. Pipeline v1 và demo cũ tiếp tục dùng `models/router.joblib`.

## 8. Đọc kết quả thế nào?

Bắt đầu từ `RESULTS.md`. Xem bảng validation để biết vì sao chọn bản cuối, sau đó xem **cả tỷ lệ đúng lẫn tiết kiệm** trên test. Đừng chỉ lấy bản có test tốt nhất rồi đổi lựa chọn.

A ở tiêu chí chính được chọn lại ngưỡng theo ngân sách ≥20%; không nhất thiết có điểm test 54,41% như v1. Dùng bảng tiêu chí phụ để đối chiếu quy tắc v1.

Báo cáo bootstrap ghép cặp theo nhóm prompt cho bản đã chọn so với A và always Large. Khoảng 95% chỉ mô tả dao động lấy mẫu của tập này với mô hình cố định; không bao gồm toàn bộ độ bất định huấn luyện/chọn mô hình, không sửa được việc đã xem test. Xem cả khoảng của chênh lệch tỷ lệ đúng lẫn chênh lệch tiết kiệm.

Nếu validation chọn A, so sánh A với chính A bằng 0 theo định nghĩa. Báo cáo cũng có các so sánh thăm dò B/C/D với A; các khoảng này chưa điều chỉnh cho nhiều phép so sánh, không dùng để chọn lại mô hình. `protocol.json` lưu checksum mã lúc training; `summary.json` lưu riêng checksum mã lúc đánh giá, để truy vết cả những sửa đổi chỉ liên quan báo cáo sau training.

**Test đã được phân tích trong v1 và ảnh hưởng định hướng v2.** Các kết quả tiếp theo phải gọi là đánh giá tiếp theo trên test cũ. Một holdout mới cần dữ liệu thực sự chưa dùng; chỉ đổi seed chia lại dữ liệu cũ không tạo được kiểm chứng độc lập. Giữ riêng toàn bộ benchmark mới là một nghiên cứu tổng quát hóa khác, chưa thực hiện ở đây.

## 9. Bản đồ mã nguồn

| File trong `src/experiments/` | Trách nhiệm |
| --- | --- |
| `config.py` | Các bản A/B/C/D, grid và protocol |
| `features.py` | Đặc trưng cây và biểu diễn TF-IDF |
| `data.py` | Nạp/kiểm tra dữ liệu, SHA256, JSON |
| `models.py` | Fold, candidate, calibration, chọn bằng log loss |
| `inference.py` | Xác suất, lợi ích và định tuyến chỉ từ prompt |
| `policy.py` | Chấm kết quả, quét và chọn ngưỡng |
| `run.py` | Điều phối train/validation và khóa artifact |
| `evaluate.py` | Kiểm tra artifact, đánh giá, bootstrap ghép cặp |
| `reporting.py` | Bảng giải thích và biểu đồ |
| `predict.py` | CLI dùng artifact với prompt mới |

Tham khảo phương pháp: [scikit-learn calibration](https://scikit-learn.org/stable/modules/calibration.html), [CARROT](https://arxiv.org/abs/2502.03261). Đây là thí nghiệm Tree/SVM đơn giản, không phải bản tái hiện đầy đủ CARROT.
