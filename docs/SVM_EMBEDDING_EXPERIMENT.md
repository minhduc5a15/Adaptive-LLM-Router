# Thí nghiệm đưa embedding vào SVM

## Giả thuyết và yếu tố được giữ cố định

SVM có thể khai thác tổ hợp các chiều embedding tốt hơn cây tách từng chiều.
Thí nghiệm này kiểm tra giả thuyết đó, không giả định kết quả sẽ tốt hơn.

- Giữ cây v1 ba đặc trưng, đã fit và hiệu chỉnh, nguyên trạng.
- Giữ nhãn `small_score == 0 AND large_score == 1`, dữ liệu và seed 42.
- Dùng cùng 3 fold StratifiedGroupKFold theo prompt cho mọi SVM.
- Giữ encoder và cách chia đoạn/gộp vector của thí nghiệm cây; không fine-tune.
- Tái sử dụng cache theo checksum nội dung/thứ tự, encoder và mã encode.

| Variant | Đầu vào SVM |
| --- | --- |
| tfidf | TF-IDF từ đơn, max_features=5000 như v1 |
| embedding | MiniLM 384 chiều đã L2-normalize |
| combined | Ghép TF-IDF và embedding, trọng số hai block bằng nhau, L2-normalize vector tổng |

`combined` giữ ma trận sparse. Không thêm PCA, đặc trưng thủ công mới, class
weights hoặc thay nhãn để tránh đổi nhiều yếu tố cùng lúc. Nếu TF-IDF của
prompt không có từ trong vocabulary, block TF-IDF bằng 0 và vector tổng
được chuẩn hóa từ block embedding còn lại.

## Huấn luyện và lựa chọn

Mỗi variant thử LinearSVC với C = 0.1, 1, 10, max_iter=10000, random_state=42.
Cảnh báo không hội tụ khiến run dừng để không âm thầm dùng mô hình chưa hội tụ.
Hiệu chỉnh sigmoid qua CalibratedClassifierCV với ensemble=False. Transformer
TF-IDF nằm trong pipeline: từ vựng và IDF chỉ được fit trên phần train của
mỗi fold. Sau dự đoán ngoài fold, mô hình cơ sở fit lại toàn train.

Chọn C có log loss validation thấp nhất cho mỗi variant. TF-IDF phải tái tạo
xác suất của SVM v1 đến sai số 1e-12; đó là kiểm tra tính tương đương baseline.

Ba chế độ:

1. `svm_only`: trọng số cây 0.
2. `fixed_ensemble`: trọng số cây 0.1 như router v1.
3. `tuned_router`: trọng số cây từ 0 đến 1, bước 0.1.

Mọi chế độ chọn ngưỡng 0 đến 1 bước 0.01, cộng 1.01 (luôn Mistral), bằng
validation. Tiêu chí chính giống thí nghiệm cây: tối thiểu hóa chi phí với
tỷ lệ trả lời đúng ≥ always Mixtral − 2 điểm phần trăm. Variant chính được
chọn từ chế độ tuned_router; hòa chi phí thì ưu tiên tỷ lệ đúng cao hơn,
rồi thứ tự tfidf/embedding/combined. Lưu selection và artifact trước test.

## Đọc mã

Mã mới nằm trong `src/svm_embedding_experiment/`:

| File | Vai trò |
| --- | --- |
| common.py | Hằng số, đường dẫn và checksum mã nguồn |
| features.py | Dựng DataFrame đầu vào, fit/transform ba biểu diễn |
| train.py | Thử 9 cấu hình, chọn chính sách, lưu artifact |
| evaluate.py | Nạp mô hình khóa, đo kết quả và phân tích nhóm lỗi |
| reporting.py | Sinh báo cáo từ bảng kết quả |
| predict.py | Định tuyến prompt mới bằng artifact |

Dùng lại encoder, kiểm tra dữ liệu, phép quét ngưỡng và bootstrap từ
`src/embedding_experiment/`; không thay nội dung các file của thí nghiệm cũ.

## Chạy lại

Cần `.venv`, `requirements-embeddings.txt`, dữ liệu, artifact baseline và
encoder đã chuẩn bị như `docs/EMBEDDING_EXPERIMENT.md`.

```bash
# Dùng tên mới khi chạy lại; không ghi đè run cũ.
TOKENIZERS_PARALLELISM=false OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python -m src.svm_embedding_experiment.train --run-name my_svm_run

.venv/bin/python -m src.svm_embedding_experiment.evaluate --run-name my_svm_run

.venv/bin/python -m src.svm_embedding_experiment.predict \
  --run-name my_svm_run --variant combined --prompt "Explain binary search."

.venv/bin/python -m pytest -q
```

Lần chạy mặc định: `minilm_svm_2026-11-25`.
Cache dùng chung: `data/processed/embeddings/minilm_2026-11-25/`; nếu chưa có
thì tự tạo bằng encoder đã tải. Test cache chỉ được mở trong evaluate.
Báo cáo ở `reports/svm_embedding/<run>/`, mô hình ở `models/svm_embedding/<run>/`.
Mô hình và cache được Git bỏ qua; không có API LLM nào được gọi.

## Đọc kết quả

`RESULTS.md` có cả validation và test, SVM riêng và router kết hợp.
`validation_candidates.csv` ghi log loss/Brier/thời gian từng cấu hình;
`validation_sweeps.csv` ghi mọi chính sách thử trên validation;
`selection.json` khóa lựa chọn; `predictions.csv` và `group_errors.csv` cho
phép kiểm tra lỗi từng mẫu/nhóm. Protocol và summary lưu checksum nguồn,
dữ liệu và phiên bản. `stats.seconds` trong metadata cache là thời gian
encode của lần đầu, không được cộng thành thời gian encode của run mới.

Đúng nhãn router khác đúng câu trả lời. FN là bỏ lỡ Mixtral khi chỉ Mixtral
đúng. FP có thể cả hai cùng đúng, cùng sai hoặc chỉ Mistral đúng. Báo cáo tách
FP làm mất đúng và đo chi phí cùng tỷ lệ trả lời đúng.

Test đã dùng phân tích lỗi và gợi ý thí nghiệm; đây là kiểm tra thăm dò,
không phải holdout mới. Bootstrap 1.000 lần theo nhóm prompt giữ mô hình và
chính sách cố định, không điều chỉnh nhiều so sánh hoặc bất định lựa chọn
tham số. Không chọn lại variant từ test. Chi phí là chi phí LLM của benchmark,
chưa bao gồm CPU tạo embedding. Tăng tỷ lệ đúng bằng cách gọi Mixtral nhiều
hơn không đủ chứng minh chất lượng định tuyến được cải thiện tại cùng ngân sách.

## Nhật ký sửa lỗi trước khi đánh giá

Lần đầu dừng ở kiểm tra baseline validation: xác suất lệch tối đa 3.36e-7.
Nguyên nhân là transformer mới gọi `fit` rồi `transform` riêng, làm đổi thứ
tự phần tử trong CSR so với `TfidfVectorizer.fit_transform` của v1. Đã gọi
trực tiếp `fit_transform` gốc và thêm kiểm thử thứ tự CSR. Lưới tham số và
tiêu chí chọn giữ nguyên; chưa đọc test khi sửa. Protocol của lần dừng và
lý do được giữ tại `reports/svm_embedding/failed_attempt_01/`.
