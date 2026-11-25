# Adaptive LLM Router

Đồ án Học máy cơ bản: định tuyến prompt giữa **Mistral 7B Chat** và **Mixtral 8×7B Chat** bằng **Decision Tree**, **SVM tuyến tính** và kết hợp dự đoán của hai mô hình. Đã hoàn thành thí nghiệm offline, lưu mô hình, báo cáo và demo Streamlit.

[Báo cáo kết quả](reports/RESULTS.md) · [Chuẩn bị dữ liệu](DATA_PREPARATION.md) · [Đề cương](notes/Adaptive_LLM_Router.md)

## Thí nghiệm nâng cấp v2

[Hướng dẫn và thiết kế A/B/C/D](docs/EXPERIMENTS_V2.md) mô tả các thí nghiệm bổ sung đặc trưng và dự đoán xác suất đúng của từng LLM. Mã nằm riêng trong `src/experiments/`; dữ liệu, mô hình và demo v1 giữ nguyên.

```bash
# Dùng tên run mới nếu thư mục này đã tồn tại.
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m src.experiments.run --run-name v2_2026-11-25
.venv/bin/python -m src.experiments.evaluate --run-name v2_2026-11-25
.venv/bin/python -m src.experiments.predict --run-name v2_2026-11-25 --prompt "Explain binary search."
```

Tiêu chí chính: tỷ lệ đúng validation cao nhất với tiết kiệm ít nhất 20%. Script train chỉ đọc train/validation và khóa lựa chọn trước khi lệnh evaluate đọc test. Test cũ đã được xem; kết quả v2 là phân tích tiếp theo, không phải đánh giá trên holdout độc lập mới.

**Kết quả ngày 25/11/2026:** đã chạy 75 cấu hình cho 6 bản trong khoảng 21 phút. Validation chọn giữ **A**, chưa có bằng chứng rõ để thay bằng bản nâng cấp. A theo chính sách mới đạt 54,65% đúng / 18,70% tiết kiệm trên test; D đạt 54,23% / 19,45%. Ràng buộc tiết kiệm 20% trên validation không được bảo đảm trên test. Xem [báo cáo v2 đầy đủ](reports/experiments/v2_2026-11-25/RESULTS.md). Theo tiêu chí phụ cũ, A tái tạo đúng kết quả v1: 54,41% / 20,54%.

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
