# Trọng số embedding và đường đánh đổi chất lượng–chi phí

## Giả thuyết

Biểu diễn ghép bằng trọng số 1 chưa chắc cân bằng tốt thông tin từ vựng và
ngữ nghĩa. Thử `X = normalize([X_tfidf, λ * X_MiniLM])` với
λ = 0.25 / 0.5 / 1 / 2. Nhân λ **trước** chuẩn hóa L2 toàn vector; không
chuẩn hóa lại từng block sau nhân. Khi cả hai block có norm 1, norm của
block MiniLM sau ghép bằng `λ / sqrt(1 + λ²)` nên λ vẫn ảnh hưởng kết quả.
Nếu prompt không chứa từ trong vocabulary, TF-IDF bằng 0; chuẩn hóa vector
tổng khi đó chỉ giữ hướng MiniLM, điều này là hệ quả của công thức.

## Thiết kế trước khi xem test

- Giữ encoder, cache, chia đoạn, nhãn, tập dữ liệu và cây v1 nguyên trạng.
- Chế độ chính: 10% cây + 90% SVM. Giữ trọng số kết hợp này cố định để tách
  tác động λ. SVM riêng được báo cáo như phân tích bổ sung với cùng C đã chọn.
- Mỗi λ thử C = 0.1 / 1 / 10. Hai mốc TF-IDF riêng và embedding riêng cũng
  thử cùng ba C: tổng 18 cấu hình, không ưu ái số lần tuning cho các bản ghép.
- TF-IDF fit trong từng fold của pipeline. CalibratedClassifierCV sigmoid,
  ensemble=False, 3 StratifiedGroupKFold theo prompt, seed 42.
- LinearSVC max_iter=10000; cảnh báo không hội tụ làm run dừng.
- Các cấu hình tham chiếu ở C cũ phải tái tạo xác suất trước đến sai số 1e-12,
  đặc biệt λ=1 phải khớp bản `combined` trước.

## Chọn λ từ toàn bộ đường

Với score định tuyến s, ngưỡng t chọn Mixtral khi s ≥ t. Xét mọi score khác
nhau trên validation, cộng 0 (luôn Mixtral) và 1.01 (luôn Mistral).
Vì chỉ quyết định định tuyến có tác động đến outcome, các ngưỡng nằm giữa
hai score liên tiếp không cần được quét riêng. Các score bằng nhau phải
được chuyển cùng lúc. Dùng tổng tích lũy để tính toàn bộ đường trong O(n log n).

Tại yêu cầu tiết kiệm tối thiểu b, đặt:

`Q*(b) = max AnswerQuality(t), với CostSaving(t) >= b`.

Điểm chọn C và λ là **diện tích chuẩn hóa**:

`Area = integral_0^Ssmall Q*(b) db / Ssmall`.

`Ssmall` là mức tiết kiệm của always Mistral trên tập đang xét. Nó tạo miền
khả thi chung cho mọi biểu diễn, gồm toàn khoảng từ 0 đến always Mistral.
Tích phân chính xác hàm bậc thang, không chỉ trung bình vài ngưỡng, không
nối tuyến tính để giả lập trộn ngẫu nhiên các chính sách. Area có đơn vị
chất lượng trung bình trên các yêu cầu budget, không phải ROC-AUC.

Chọn C tốt nhất cho mỗi biểu diễn, sau đó chọn λ tốt nhất trong bốn λ;
đồng thời ghi bản tốt nhất trong cả sáu biểu diễn. Hòa diện tích thì ưu tiên
log loss thấp hơn, C nhỏ hơn, thứ tự variant đã khai báo. Toàn bộ lựa chọn
dựa trên `fixed_ensemble` validation. SVM riêng không được chọn lại C.

Trọng số budget là đều theo độ rộng saving. Chọn tiêu chí này có thể cho
λ khác với việc chỉ tối ưu quanh 20% saving. Muốn đổi trọng số budget cần
một thí nghiệm được định nghĩa trước, không sửa tiêu chí sau khi xem test.

## Đánh giá test mà không chọn ngưỡng từ test

1. Lưu toàn bộ tập ngưỡng validation và artifact trong selection trước test.
2. Áp dụng từng ngưỡng đó lên test để có đường đánh đổi đầy đủ của các
   chính sách đã định. Không sinh ngưỡng mới từ nhãn test.
3. Đường Pareto và diện tích test chỉ **mô tả** khả năng đạt được trong
   tập chính sách đã khóa; việc lấy đường biên dùng nhãn test nên không
   được coi nó là chính sách triển khai đã chọn trước.
4. Riêng đánh giá chuyển chính sách: trên validation chọn ngưỡng tốt nhất
   cho 101 mức budget cách đều và thêm 10/20/30/40/50/60%. Giữ ngưỡng đó trên
   test, xuất cả budget yêu cầu lẫn saving thực tế, không chọn lại để vừa budget test.
5. So sánh ghép cặp tại budget validation 20%: λ đã chọn với TF-IDF và λ=1;
   bootstrap theo nhóm prompt 1.000 lần, giữ mô hình và chính sách cố định.

Các đường đều chia sẻ cùng miền budget trong một tập. Miền validation và
test có thể hơi khác do chi phí always Mistral khác; không dùng chênh lệch
diện tích giữa hai tập như một phép so sánh cùng dữ liệu. Saving test ở cùng
budget validation có thể khác nhau; cần đọc cả quality và saving.

## Mã và chạy lại

`src/svm_lambda_experiment/`:

- `features.py`: subclass transformer cũ, thêm trọng số MiniLM; không sửa
  class đã lưu trong các artifact trước.
- `curves.py`: quét ngưỡng, Pareto, chọn policy theo budget, tích phân chính xác.
- `common.py`: lưới, chế độ và đường dẫn riêng.
- `train.py`: 18 cấu hình, lựa chọn validation, khóa artifact/ngưỡng.
- `evaluate.py`: áp dụng lên test, tính bảng và bootstrap.
- `reporting.py`: báo cáo và hình PNG/SVG xuất được.
- `predict.py`: thử prompt mới bằng ngưỡng budget khóa trên validation.

```bash
# Cần dữ liệu, encoder và artifact của thí nghiệm SVM trước.
TOKENIZERS_PARALLELISM=false OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python -m src.svm_lambda_experiment.train --run-name my_lambda_run

.venv/bin/python -m src.svm_lambda_experiment.evaluate --run-name my_lambda_run

.venv/bin/python -m src.svm_lambda_experiment.predict \
  --run-name my_lambda_run --budget 0.2 --prompt "Explain binary search."

.venv/bin/python -m pytest -q
```

Tên mặc định: `lambda_2026-11-26`. Không ghi đè run đã tồn tại.
Báo cáo/hình ở `reports/svm_lambda/<run>/`, model ở `models/svm_lambda/<run>/`.
Dùng lại cache `data/processed/embeddings/minilm_2026-11-25/`. `protocol.json`
lưu mã nguồn lúc training; summary lưu mã nguồn lúc evaluate để ghi nhận
các phần đánh giá/báo cáo được hoàn thiện sau khi khởi động training.

Các bảng quan trọng: `validation_candidates.csv`, `validation_candidate_curves.csv`,
`selection.json`, `threshold_curves.csv`, `pareto_frontiers.csv`, `curve_areas.csv`,
`locked_budget_policies.csv`, `budget_evaluation.csv`, `predictions.csv`.

## Giới hạn

Test đã được dùng phân tích nhiều lần; đây là nghiên cứu thăm dò, chưa có
holdout độc lập mới. Diện tích validation cũng có thể bị overfit qua chọn
siêu tham số. Bootstrap không bao gồm độ bất định training/lựa chọn, không
điều chỉnh nhiều so sánh. Chi phí chỉ là chi phí LLM lưu trong RouterBench,
chưa tính CPU embedding; không gọi API LLM. Tăng quality cùng tăng chi phí
không đồng nghĩa cải thiện tại cùng ngân sách.
