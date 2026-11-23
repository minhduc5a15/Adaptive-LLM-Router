# Kết quả Adaptive LLM Router

## Kết luận

Khả quan trong phạm vi thí nghiệm: tiết kiệm chi phí và đạt mức giảm điểm cho phép theo ước lượng điểm trên test.

- Phương án chọn từ validation: **Ensemble**, trọng số cây **0.10**, ngưỡng **0.21**.
- Test: **54.41%** câu được chấm đúng; tiết kiệm **20.54%** so với luôn chọn Mixtral.
- Thay đổi tỷ lệ đúng so với Mixtral: **-1.81 điểm phần trăm**.
- Khoảng bootstrap 95% cho thay đổi điểm: [-2.62, -0.99] điểm phần trăm.
- Khoảng bootstrap 95% cho tiết kiệm chi phí: [19.03%, 22.02%].

Các khoảng trên lấy mẫu lại các nhóm prompt test (1.000 lần), giữ cố định mô hình. Chúng không bao gồm bất định do huấn luyện lại hay thay đổi phân bố tác vụ.

Khoảng bất định của thay đổi điểm vượt mức giảm 2.0 điểm phần trăm ở phía bất lợi; việc đạt tiêu chí theo ước lượng điểm không phải cam kết thống kê cho dữ liệu mới.

## Dữ liệu và quy trình

- Train / validation / test: **22,605 / 2,826 / 2,825** mẫu.
- RouterBench 0-shot, Mistral 7B Chat → Mixtral 8×7B Chat; lọc điểm nhị phân và prompt chứa chữ Hán theo DATA_PREPARATION.md.
- Nhãn 1: Small sai, Large đúng. Nhãn 0 gồm cả hai mô hình cùng sai; dự đoán đúng nhãn không đồng nghĩa câu trả lời đúng.
- Các tập không có prompt chung. Hiệu chỉnh sigmoid bằng dự đoán ngoài fold trên train, 3 fold StratifiedGroupKFold theo prompt.
- Pipeline SVM fit TF-IDF riêng từng fold; mô hình cơ sở cuối cùng fit toàn train. Validation không dùng để fit tham số nội bộ của cây/SVM hoặc hàm hiệu chỉnh; dùng để chọn cấu hình, trọng số kết hợp và ngưỡng.
- Decision Tree dùng đúng ba cột: char_count, word_count (tách theo khoảng trắng), line_count.
- Cấu hình cây chọn bằng log loss validation: `{"max_depth": 5, "min_samples_leaf": 20}`. Cây có độ sâu 5, 31 lá.
- SVM tuyến tính: `{"C": 0.1, "max_features": 5000}`; TF-IDF tối đa 5.000 mục.
- Thử 8 cấu hình cây và 3 giá trị C cho SVM. Mỗi họ chọn log loss validation nhỏ nhất.
- Ensemble thử 9 trọng số cây từ 0,1 đến 0,9. Mỗi phương án thử 101 ngưỡng từ 0 đến 1.
- Quy tắc chọn chính sách đặt trước khi xem test: chi phí validation thấp nhất, tỷ lệ đúng giảm tối đa **2.0 điểm phần trăm** so với Always Large.
- Lưu selection.json và router.joblib trước khi nạp test. Các bảng test không được dùng để đổi mô hình, trọng số hoặc ngưỡng.
- Huấn luyện và lựa chọn: 30.3 giây. Dự đoán cả hai mô hình trên batch test: 0.166 giây (không phải độ trễ một yêu cầu tương tác).

## Validation

| Phương án | Đúng nhãn Router | F1 nhãn 1 | Câu được chấm đúng | Tổng chi phí (USD benchmark) | Tiết kiệm | Gọi Mixtral |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Always Small | 66.81% | 0.0000 | 27.88% | 0.093411 | 67.02% | 0.00% |
| Always Large | 33.19% | 0.4984 | 56.33% | 0.283233 | 0.00% | 100.00% |
| Random 50% | 50.42% | 0.4061 | 42.25% | 0.188119 | 33.58% | 50.28% |
| Random matched | 41.26% | 0.4747 | 50.46% | 0.240940 | 14.93% | 78.63% |
| Decision Tree | 37.65% | 0.5114 | 56.09% | 0.276542 | 2.36% | 94.41% |
| SVM | 50.64% | 0.5507 | 54.42% | 0.222340 | 21.50% | 76.68% |
| Ensemble | 50.92% | 0.5527 | 54.49% | 0.221423 | 21.82% | 76.54% |
| Oracle (labels) | 100.00% | 1.0000 | 61.08% | 0.149272 | 47.30% | 33.19% |

## Test

| Phương án | Đúng nhãn Router | F1 nhãn 1 | Câu được chấm đúng | Tổng chi phí (USD benchmark) | Tiết kiệm | Gọi Mixtral |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Always Small | 65.27% | 0.0000 | 27.15% | 0.096879 | 66.40% | 0.00% |
| Always Large | 34.73% | 0.5155 | 56.21% | 0.288353 | 0.00% | 100.00% |
| Random 50% | 50.58% | 0.4188 | 42.16% | 0.191896 | 33.45% | 50.30% |
| Random matched | 41.35% | 0.4827 | 50.02% | 0.247019 | 14.33% | 78.65% |
| Decision Tree | 38.87% | 0.5265 | 55.82% | 0.281602 | 2.34% | 94.37% |
| SVM | 50.30% | 0.5577 | 54.44% | 0.230505 | 20.06% | 77.63% |
| Ensemble | 50.37% | 0.5580 | 54.41% | 0.229127 | 20.54% | 77.56% |
| Oracle (labels) | 100.00% | 1.0000 | 61.88% | 0.155070 | 46.22% | 34.73% |

Random matched dùng xác suất gọi Large bằng tỷ lệ của chính sách được chọn trên validation; các lần rút ngẫu nhiên có seed 42. Oracle dùng nhãn thật nên chỉ là mốc tham chiếu không triển khai được.

Mọi giá chi phí là ước tính được lưu trong RouterBench, không phải hóa đơn đã trả hoặc giá API hiện tại. Precision, recall và F1 dùng nhãn 1 làm lớp dương.

## Diễn giải kết quả

- Ensemble so với SVM: tiết kiệm thêm +0.48 điểm phần trăm chi phí tham chiếu, thay đổi tỷ lệ đúng -0.04 điểm phần trăm. Chênh lệch nhỏ này chưa chứng minh kết hợp tốt hơn SVM nói chung.
- AUC của phương án chọn: 0.691. Khả năng phân biệt nhãn ở mức vừa phải.
- Tại ngưỡng 0,5, accuracy của phương án chọn là 67.26%, so với baseline luôn Small 65.27%.
- Ngưỡng 0.21 được chọn để ưu tiên giữ điểm câu trả lời, làm recall nhãn 1 đạt 90.21% nhưng accuracy Router còn 50.37%. Đây là các mục tiêu khác nhau; không nên dùng accuracy riêng để kết luận về chi phí và điểm trả lời.

## Phân tích sai số của phương án được chọn

- TN=538, FP=1306, FN=96, TP=885.
- FN: bỏ lỡ trường hợp Mixtral có điểm 1 còn Mistral có điểm 0.
- FP: gọi Mixtral khi nhãn là 0; có thể tăng chi phí mà không tăng điểm, hoặc làm giảm điểm khi Small đúng và Large sai.
- [Kết quả theo benchmark](test_by_benchmark.csv); các nhóm ít mẫu không đủ để kết luận chắc chắn.
- [Ví dụ sai](error_examples.csv): tối đa 10 trường hợp mỗi loại, chọn theo thứ tự dataset, không phải mẫu đại diện ngẫu nhiên.
- [Chi tiết từng mẫu test](test_predictions.csv).
- [Độ đúng phân loại tại ngưỡng 0,5](classification_at_050.csv) cho train/validation/test giúp quan sát chênh lệch khả năng khái quát.

## Biểu đồ

![Điểm trả lời và chi phí](figures/quality_cost.png)

Ngôi sao là cấu hình đã khóa từ validation. Đường test ở các ngưỡng khác chỉ để mô tả đánh đổi, không dùng để lựa chọn lại.

![Ma trận nhầm lẫn](figures/confusion_matrices.png)

![Hiệu chỉnh xác suất](figures/calibration.png)

![Các tầng đầu của cây](figures/decision_tree.png)

## Giới hạn

- Đây là kiểm thử trên các prompt chưa thấy trong cùng hỗn hợp benchmark. Chưa thực hiện thí nghiệm giữ riêng toàn bộ nhóm benchmark chưa thấy khi train.
- Ba đặc trưng cấu trúc chỉ là phiên bản ban đầu. TF-IDF chủ yếu ghi nhận từ xuất hiện, chưa mô hình hóa đầy đủ ngữ nghĩa.
- Lọc chữ Hán không đảm bảo dữ liệu chỉ có tiếng Anh; word_count vẫn là phép đếm theo khoảng trắng.
- Ngưỡng đạt ràng buộc trên validation không đảm bảo giữ được ràng buộc trên mọi tập dữ liệu mới. Đọc cả khoảng bất định, không chỉ ước lượng điểm.
- Các xác suất đã qua hiệu chỉnh vẫn là ước lượng; biểu đồ calibration thể hiện sai lệch còn lại.
- Đổi cặp LLM, ngôn ngữ hoặc tác vụ cần đánh giá lại. Demo prompt mới chỉ định tuyến, không gọi LLM tạo câu trả lời.

## Tái lập và đọc mã

```bash
.venv/bin/python src/train_models.py
.venv/bin/python -m src.predict --prompt "Explain binary search."
.venv/bin/streamlit run app.py
```

- Mô hình: `models/router.joblib`; seed, phiên bản thư viện và checksum dữ liệu: [selection.json](selection.json), [summary.json](summary.json).
- [Các cấu hình đã thử](validation_candidates.csv); [toàn bộ lựa chọn ngưỡng validation](validation_thresholds.csv).
- Thứ tự đọc mã: features.py → train_models.py → router.py → evaluation.py → reporting.py → app.py.

## Tài liệu phương pháp

- [scikit-learn: Probability calibration](https://scikit-learn.org/stable/modules/calibration.html).
- [scikit-learn: Classification metrics](https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics).
