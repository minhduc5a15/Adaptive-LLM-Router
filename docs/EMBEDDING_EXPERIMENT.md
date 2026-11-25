# Thí nghiệm embedding cho Decision Tree

## Câu hỏi cần trả lời

Thay ba đặc trưng độ dài/cấu trúc bằng vector ngữ nghĩa có cải thiện việc
định tuyến Mistral–Mixtral không? Thử cả phương án ghép hai nhóm đặc trưng.
Đây là thí nghiệm tiếp nối baseline, không thay quy trình tiền xử lý hay nhãn.

| Variant | Đầu vào cây | Số chiều |
| --- | --- | ---: |
| `structure` | char_count, word_count, line_count | 3 |
| `embedding` | MiniLM đóng băng | 384 |
| `combined` | Ba cột cấu trúc rồi đến embedding | 387 |

Giữ nguyên SVM đã lưu trong `models/router.joblib`. Chương trình kiểm tra
checksum train/validation và phiên bản sklearn trước khi tái sử dụng SVM.
Cây `structure` phải tái tạo xác suất validation của cây v1 đến sai số 1e-12.

## Thiết kế so sánh

1. Giữ nguyên train/validation/test, nhãn và seed 42.
2. Ghi `protocol.json` trước khi mã hóa và fit cây.
3. Mỗi variant thử cùng 8 cấu hình: depth 3/5/8/12 × min_samples_leaf 20/50.
4. Hiệu chỉnh sigmoid bằng 3 fold chia theo nhóm prompt trong train,
   `ensemble=False`. Chọn cây có log loss validation thấp nhất.
5. Chọn ngưỡng/chính sách bằng validation, tối thiểu hóa chi phí với tỷ lệ
   trả lời đúng không thấp hơn always Mixtral quá 2 điểm phần trăm.
6. Lưu mô hình và `selection.json`; chỉ sau đó lệnh evaluate mới đọc test.

Ba chế độ giúp phân biệt ảnh hưởng của biểu diễn và của việc kết hợp mô hình:

| Chế độ | Trọng số cây | Cách chọn ngưỡng |
| --- | --- | --- |
| `tree_only` | 1 | Validation |
| `fixed_ensemble` | 0,1; SVM 0,9 như v1 | Validation |
| `tuned_router` | 0 đến 1, bước 0,1 | Validation |

Ngưỡng từ 0 đến 1, bước 0,01; thêm 1,01 để biểu diễn luôn chọn Mistral.
Variant chính được chọn theo chi phí validation ở chế độ `tuned_router`,
hòa thì ưu tiên tỷ lệ đúng cao hơn, rồi thứ tự structure/embedding/combined.
Cho phép trọng số 0 hoặc 1 để kiểm tra liệu kết hợp có thực sự cần thiết.
Vì lưới này rộng hơn lưới ensemble v1, dùng `fixed_ensemble/structure` để
đối chiếu trực tiếp với kết quả router v1 đã công bố.

## Tạo embedding như thế nào?

Encoder: `sentence-transformers/all-MiniLM-L6-v2`, revision
`1110a243fdf4706b3f48f1d95db1a4f5529b4d41`, ONNX float32, CPU 4 threads.
Không fine-tune encoder và không gọi API LLM. Mô hình và tokenizer được tải
một lần; dữ liệu prompt được mã hóa tại máy.

Prompt dài được xử lý như sau:

1. Tokenize toàn bộ, không cắt bớt token.
2. Chia các đoạn không chồng lặp, tối đa 254 content tokens mỗi đoạn.
3. Thêm CLS/SEP để có tối đa 256 tokens; padding có attention mask.
4. Lấy trung bình vector token có mask bằng 1, gồm CLS/SEP, trong mỗi đoạn.
5. Lấy trung bình các vector đoạn theo trọng số số content tokens của đoạn.
6. Chuẩn hóa L2 một lần để nhận vector 384 chiều của prompt.

Đây là cách gộp đoạn của thí nghiệm, khác cách cắt 256 token mặc định trong
model card. Không mất token nhưng không giữ đầy đủ quan hệ giữa các đoạn.
Encoder đóng băng, không học từ các tập nên việc cache vector trước chia
fold không đưa thông tin học từ validation vào train.

Cache lưu thứ tự sample_id/prompt, SHA256 dữ liệu, encoder, mã encode và
mảng `.npy`. Cache sai thứ tự hoặc khác checksum bị từ chối.

## Chạy lại

Cần dữ liệu đã tiền xử lý và artifact v1 từ quy trình ở README.

```bash
.venv/bin/python -m pip install -r requirements-embeddings.txt

# Chỉ bước này cần mạng để tải encoder chính thức.
HF_HUB_DISABLE_XET=1 .venv/bin/python -m src.embedding_experiment.prepare

# Chọn tên run mới; train từ chối ghi đè run đã tồn tại.
TOKENIZERS_PARALLELISM=false OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python -m src.embedding_experiment.train --run-name my_minilm_run

.venv/bin/python -m src.embedding_experiment.evaluate --run-name my_minilm_run

.venv/bin/python -m src.embedding_experiment.predict \
  --run-name my_minilm_run --variant embedding --mode fixed_ensemble \
  --prompt "Explain binary search."

.venv/bin/python -m pytest -q
```

Không chỉ định `--variant` thì predict dùng variant được validation chọn.
Predict chỉ chọn LLM, không tạo câu trả lời hay biết trước prompt mới đúng/sai.

## Đọc mã và kết quả

| File | Vai trò |
| --- | --- |
| `common.py` | Đường dẫn, hằng số, kiểm tra dữ liệu và checksum |
| `prepare.py` | Tải encoder theo revision, lưu manifest |
| `encoder.py` | Tokenize, chia đoạn, pooling và cache |
| `inference.py` | Ghép đặc trưng, lấy xác suất, áp dụng chính sách |
| `train.py` | Fit cây và khóa lựa chọn từ validation |
| `evaluate.py` | Nạp artifact, đo kết quả, phân tích lỗi |
| `reporting.py` | Tạo báo cáo và biểu đồ |
| `predict.py` | Định tuyến prompt mới |

Mã nằm trong `src/embedding_experiment/`. Kết quả nằm riêng tại
`reports/embedding/<run>/`; artifact tại `models/embedding/<run>/`; cache
tại `data/processed/embeddings/<run>/`. Baseline v1 được giữ nguyên.

- `RESULTS.md`: bảng kết quả và diễn giải.
- `validation_candidates.csv`: 24 cấu hình cây và log loss.
- `validation_sweeps.csv`: mọi chính sách đã thử trên validation.
- `selection.json`: cấu hình, chính sách và variant khóa trước test.
- `metrics.csv`: điểm trả lời, chi phí và điểm phân loại theo từng chế độ.
- `group_errors.csv`: lỗi theo benchmark và nhóm prompt dài.
- `predictions.csv`: dự đoán từng mẫu để kiểm tra lại.
- `validation_error_examples.csv`: ví dụ được/mất câu đúng trên validation.
- `protocol.json`, `summary.json`: thiết kế, checksum, phiên bản, bootstrap.

## Diễn giải khoa học

Nhãn 1 nghĩa là Mistral sai và Mixtral đúng. FN bỏ lỡ trường hợp này.
FP chỉ có nghĩa chọn Mixtral khi nhãn bằng 0: có thể cả hai cùng đúng,
cùng sai, hoặc Mistral đúng/Mixtral sai. Vì vậy cần xem cả số câu trả lời
đúng và chi phí, không kết luận từ accuracy nhãn hay FN đơn lẻ.

Test cũ đã được dùng phân tích lỗi và gợi ý thí nghiệm này. Các nhóm HellaSwag
và prompt ít nhất 200 phần tách theo khoảng trắng là phân tích thăm dò.
Bootstrap ghép cặp theo nhóm prompt, 1.000 lần, giữ mô hình/chính sách cố định;
không bao gồm bất định do chọn cấu hình hay hiệu chỉnh nhiều phép so sánh.
Không được coi đây là holdout độc lập mới hoặc bảo đảm tổng quát hóa.

Chi phí trong bảng là chi phí LLM lưu trong RouterBench, chưa cộng CPU tạo
embedding. Thời gian encode theo batch không phải độ trễ của request đơn.
Encoder học từ dữ liệu ngoài; chưa kiểm tra trùng nguồn với benchmark.

Nguồn: [MiniLM model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2).
