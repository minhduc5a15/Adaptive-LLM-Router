# Adaptive LLM Router

Đồ án Học máy cơ bản: định tuyến prompt giữa **Mistral 7B Chat** và **Mixtral 8×7B Chat** bằng **Decision Tree**, **SVM tuyến tính** và kết hợp dự đoán của hai mô hình. Đã hoàn thành thí nghiệm offline, lưu mô hình, báo cáo và demo Streamlit.

[Báo cáo kết quả](reports/RESULTS.md) · [Chuẩn bị dữ liệu](DATA_PREPARATION.md) · [Đề cương](notes/Adaptive_LLM_Router.md)

## Kết quả lần chạy 23/11/2026

Đánh giá trên **2.825 prompt test**; cấu hình và ngưỡng được chọn trước trên validation.

| Phương án | Câu được chấm đúng | Tiết kiệm chi phí so với luôn Mixtral |
| --- | ---: | ---: |
| Luôn Mistral | 27,15% | 66,40% |
| Luôn Mixtral | 56,21% | 0% |
| Decision Tree | 55,82% | 2,34% |
| SVM | 54,44% | 20,06% |
| **Ensemble được chọn** | **54,41%** | **20,54%** |

Ensemble dùng **10% điểm của cây + 90% điểm của SVM**, chọn Mixtral khi điểm kết hợp **≥ 0,21**. Tiết kiệm 20,54% chi phí benchmark với mức giảm tỷ lệ đúng 1,81 điểm phần trăm. Khoảng bootstrap 95% cho mức giảm là 0,99–2,62 điểm phần trăm: kết quả khả quan trên dữ liệu này, chưa bảo đảm mức giảm luôn dưới 2 điểm trên dữ liệu mới. Ensemble chỉ khác SVM một chút; chưa đủ để khẳng định kết hợp tốt hơn nói chung.

Chi phí là ước tính đã lưu trong RouterBench, không phải tiền đã trả hay giá API hiện hành. Đúng nhãn Router và đúng câu trả lời là hai chỉ số khác nhau; xem bảng đầy đủ trong báo cáo.

## Thí nghiệm embedding cho Decision Tree

Thí nghiệm riêng so sánh ba đầu vào của cây: **3 đặc trưng cấu trúc**,
**384 chiều MiniLM**, và **387 chiều kết hợp**. SVM v1 được giữ nguyên;
chọn cấu hình và chính sách bằng validation trước khi đánh giá test.

[Thiết kế và hướng dẫn chạy](docs/EMBEDDING_EXPERIMENT.md) ·
[Kết quả thí nghiệm](reports/embedding/minilm_2026-11-25/RESULTS.md)

Mã nằm trong `src/embedding_experiment/`; cài thêm `requirements-embeddings.txt`.
Encoder chạy trên CPU, không cần GPU hoặc gọi Mistral/Mixtral. Prompt dài được
chia đoạn để giữ toàn bộ token. Kết quả và mô hình được lưu riêng với baseline.

Lần chạy `minilm_2026-11-25`: validation chọn 10% cây embedding + 90% SVM,
ngưỡng 0,22. Trên 2.825 mẫu test cũ, router đúng **1.533 câu (54,27%)**,
tiết kiệm **20,79%** chi phí LLM; baseline đúng 1.537 câu, tiết kiệm 20,54%.
Chưa cải thiện tỷ lệ đúng; chưa có bằng chứng đủ để thay baseline. Chi phí
embedding chưa được cộng vào mức tiết kiệm trên. Đây là phân tích tiếp nối
trên test đã xem, không phải kiểm chứng bằng holdout độc lập mới.

## Thí nghiệm embedding cho SVM

Giữ cây v1 nguyên trạng và so sánh SVM dùng **TF-IDF**, **MiniLM**, hoặc
**TF-IDF + MiniLM**. Mã nằm trong `src/svm_embedding_experiment/`, dùng lại
encoder/cache đã có, không cần cài thêm thư viện.

[Thiết kế và hướng dẫn chạy](docs/SVM_EMBEDDING_EXPERIMENT.md) ·
[Kết quả thí nghiệm](reports/svm_embedding/minilm_svm_2026-11-25/RESULTS.md)

Lần chạy `minilm_svm_2026-11-25` thử 9 cấu hình. Với 10% cây + 90% SVM,
TF-IDF + embedding đúng **1.556/2.825 câu (55,08%)**, so với baseline
1.537 câu (54,41%); tiết kiệm chi phí LLM giảm từ **20,54% xuống 17,38%**.
Như vậy tăng tỷ lệ đúng đi kèm tăng chi phí. Theo tiêu chí chi phí validation
thấp nhất với giới hạn giảm đúng 2 điểm phần trăm, **validation vẫn chọn
baseline TF-IDF**. Không chọn lại phương án từ test đã xem; chi phí CPU
embedding chưa được tính vào mức tiết kiệm.

## Thí nghiệm trọng số MiniLM λ

Thử `normalize([TF-IDF, λ × MiniLM])` với λ = 0.25 / 0.5 / 1 / 2,
chọn λ và C bằng diện tích đường Answer Quality–Cost Saving trên validation.
Giữ 10% cây + 90% SVM; hai mốc TF-IDF/MiniLM riêng được tuning cùng lưới C.
Đánh giá mọi ngưỡng đã khóa từ validation và nhiều chính sách budget trên test.

[Thiết kế và lệnh chạy](docs/SVM_LAMBDA_EXPERIMENT.md) ·
[Báo cáo và biểu đồ](reports/svm_lambda/lambda_2026-11-26/RESULTS.md)

Lần chạy `lambda_2026-11-26`: validation chọn **λ=2, C=0.1**. Diện tích
chuẩn hóa đường test đạt **48,775%**, so với **48,269%** của TF-IDF và
**48,679%** của λ=1. Đây là chất lượng trung bình trên các yêu cầu tiết kiệm,
không phải chất lượng tại một ngưỡng cụ thể. Với ngưỡng chọn cho budget
validation 20%, λ=2 đạt **54,87% đúng**, tiết kiệm thực tế **18,61%** trên test.
Mức tăng còn nhỏ, các đường có giao nhau; test đã xem và chi phí CPU embedding
chưa được tính. Không chọn lại λ hoặc ngưỡng từ đường biên test.

## Chạy dự án

Đã kiểm tra với Python **3.12.3** và các phiên bản trong `requirements.txt`.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt

# Tạo train/validation/test từ file 0-shot đã tải
.venv/bin/python src/preprocess_data.py

# Huấn luyện, chọn cấu hình, lưu mô hình và đánh giá cuối
.venv/bin/python src/train_models.py

# Tạo lại báo cáo từ mô hình đã lưu, không chọn lại cấu hình
.venv/bin/python src/evaluate_models.py

# Định tuyến một câu hỏi mới
.venv/bin/python -m src.predict --prompt "Explain binary search."

# Mở demo
.venv/bin/streamlit run app.py

# Kiểm tra tính toán và khả năng nạp lại mô hình
.venv/bin/python -m pytest -q
```

`train_models.py` mặc định chọn chính sách có chi phí validation thấp nhất với mức giảm tỷ lệ đúng tối đa **2 điểm phần trăm** so với luôn Mixtral. `--max-quality-drop 0.02` là tham số cho quy tắc này. Nếu nghiên cứu tiêu chí khác, phải chọn trước bằng validation; không điều chỉnh theo kết quả test đã xem rồi tuyên bố đó là kiểm thử độc lập mới.

Dữ liệu thô đã nằm trong `data/raw/` ở workspace này. Trên máy mới, cần lấy lại file `routerbench_0shot.pkl` đúng nguồn, revision và SHA-256 trong `DATA_PREPARATION.md`. Các dataset, mô hình và ảnh sinh ra được Git bỏ qua. Chỉ nạp pickle/joblib do nguồn tin cậy cung cấp.

## Quy trình học máy

1. Giữ cặp điểm 0/1, chuyển prompt thành văn bản, loại prompt chứa chữ Hán theo quy tắc đã ghi trong tài liệu.
2. Chia theo nhóm prompt, khoảng 80/10/10, tránh cùng prompt xuất hiện ở nhiều tập.
3. **Decision Tree:** ba cột `char_count`, `word_count`, `line_count`. Thử độ sâu 3/5/8/12 và số mẫu tối thiểu ở lá 20/50.
4. **SVM:** `TfidfVectorizer(max_features=5000)` + `LinearSVC`, thử C = 0,1 / 1 / 10.
5. Hiệu chỉnh sigmoid bằng dự đoán ngoài fold trong train; 3 fold được chia theo prompt. TF-IDF nằm trong pipeline nên được fit riêng từng fold. Sau đó mô hình cơ sở được fit toàn train.
6. Chọn cấu hình mỗi họ theo log loss validation. Thử 9 trọng số kết hợp và 101 ngưỡng; chọn theo tiêu chí điểm trả lời–chi phí.
7. Lưu `selection.json` và `router.joblib` trước khi đọc test. Đánh giá các baseline, mô hình riêng và ensemble với chính sách đã khóa.
8. Xuất bảng số liệu, biểu đồ, dự đoán từng mẫu và ví dụ sai. Test không được dùng để chọn lại tham số.

## Đọc mã theo thứ tự

| File | Vai trò |
| --- | --- |
| `src/preprocess_data.py` | Lọc, tạo nhãn và chia dữ liệu |
| `src/features.py` | Ba đặc trưng cấu trúc |
| `src/train_models.py` | Huấn luyện, hiệu chỉnh và chọn chính sách |
| `src/router.py` | Nạp mô hình, kết hợp xác suất và định tuyến |
| `src/evaluation.py` | Tính điểm nhãn, điểm trả lời, chi phí, khoảng bootstrap |
| `src/reporting.py` | Bảng kết quả, biểu đồ và phân tích sai số |
| `src/evaluate_models.py` | Đánh giá lại mô hình đã lưu |
| `src/predict.py` | Nhập prompt từ dòng lệnh |
| `app.py` | Demo trực quan và bảng test |
| `tests/` | Kiểm thử các phép tính và artifact |

## Artifact

- `models/router.joblib`: hai mô hình đã hiệu chỉnh và các chính sách đã chọn.
- `reports/RESULTS.md`: kết luận, phương pháp và giới hạn.
- `reports/selection.json`: tham số, phiên bản thư viện, checksum train/validation.
- `reports/summary.json`: kết quả chính, khoảng bất định, checksum test.
- `reports/*metrics.csv`: kết quả validation/test; `classification_at_050.csv` báo cáo riêng ngưỡng 0,5.
- `reports/test_predictions.csv`, `test_by_benchmark.csv`, `error_examples.csv`: phân tích chi tiết.
- `reports/figures/`: đường điểm trả lời–chi phí, confusion matrix, calibration và các tầng đầu của cây.

## Phạm vi

Thí nghiệm sử dụng kết quả LLM đã có trong RouterBench, **không cần chạy hoặc huấn luyện Mistral/Mixtral**. Demo với prompt mới chỉ chọn LLM, không tạo câu trả lời. Chưa thực hiện đánh giá trên một nhóm benchmark hoàn toàn chưa thấy lúc train; kết quả hiện tại thuộc cùng hỗn hợp benchmark. Cặp LLM khác, ngôn ngữ khác hoặc dữ liệu thực tế cần đánh giá riêng.

Nguồn: [RouterBench](https://huggingface.co/datasets/withmartian/routerbench), revision `784021482c3f320c6619ed4b3bb3b41a21424fcb`. Phương pháp hiệu chỉnh: [scikit-learn](https://scikit-learn.org/stable/modules/calibration.html).
