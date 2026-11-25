# Ghi chú chuẩn bị dữ liệu RouterBench

**Cập nhật v2 (25/11/2026):** thí nghiệm mới dùng lại đúng các CSV và nhóm prompt hiện tại. `small_score` và `large_score` là hai mục tiêu học cho các bản C/D, không đưa vào đặc trưng. Không cần tạo lại split; xem [thiết kế v2](docs/EXPERIMENTS_V2.md).

Khảo sát dữ liệu ghi nhận ngày 15/11/2026; tiền xử lý cập nhật ngày 21/11/2026. Đã hoàn thành huấn luyện Decision Tree, SVM, kết hợp và đánh giá test ngày 23/11/2026. Phần dưới ghi dữ liệu và các mốc validation; kết quả cuối nằm trong [reports/RESULTS.md](reports/RESULTS.md).

## Nguồn và tính toàn vẹn

- Nguồn: [withmartian/routerbench trên Hugging Face](https://huggingface.co/datasets/withmartian/routerbench).
- Revision cố định: `784021482c3f320c6619ed4b3bb3b41a21424fcb`.
- Ba file nằm trong `data/raw/`, đã đối chiếu kích thước và SHA-256 với thông tin từ nguồn. Thư mục này được Git bỏ qua.

| File | Kích thước (byte) | SHA-256 |
| --- | ---: | --- |
| `routerbench_0shot.pkl` | 99.567.659 | `ba4f77f19517610a707c374e99322d7750c30fc4ae7ff5527888595a1e65d36d` |
| `routerbench_5shot.pkl` | 171.091.324 | `fbd7d3d16fba2759a18fa0ad44409d3e0e92ba80d03d90827eed1a6d084c9ffe` |
| `routerbench_raw.pkl` | 1.198.134.681 | `18e919267500a1d49302e0efe25cf7d9aa2f5b74aa209f5230442c2cfbee7971` |

Hai file 0-shot và 5-shot đã được khảo sát; file `raw` mới được xác minh checksum, chưa nạp vào bộ nhớ. **Pickle có thể thực thi mã khi nạp**; chỉ nạp file từ nguồn đã xác minh.

## Cấu trúc quan sát được

| Thuộc tính | 0-shot | 5-shot |
| --- | ---: | ---: |
| Số dòng | 36.497 | 36.511 |
| Số cột | 37 | 37 |
| `sample_id` khác nhau | 36.497 | 36.497 |
| `prompt` khác nhau | 36.481 | 36.497 |
| Nhóm `eval_name` | 86 | 86 |

- Mỗi dòng có `sample_id`, `prompt`, `eval_name`; 11 cột điểm chất lượng của LLM, 11 cột `|model_response`, 11 cột `|total_cost` và `oracle_model_to_route_to`.
- `prompt` là **chuỗi biểu diễn một danh sách các đoạn văn bản**, không phải văn bản phẳng: trong 0-shot có 27.420 danh sách một phần tử và 9.077 danh sách hai phần tử. Tất cả 36.497 chuỗi này đọc được bằng `ast.literal_eval`. Khi tự viết bước tiền xử lý, cần chọn cách ghép các phần tử và áp dụng giống nhau cho mọi tập. Không dùng `eval`.
- 0-shot không thiếu prompt, điểm hay chi phí ở 11 mô hình. Có 16 prompt trùng nguyên văn dù `sample_id` khác nhau.
- 5-shot có 14 dòng thuộc `arc-challenge` thiếu điểm của các mô hình và trùng `sample_id` với dòng khác. Toàn bộ 36.497 `sample_id` của 0-shot cũng xuất hiện trong 5-shot. Nếu kết hợp hai phiên bản, cần chia tập theo cùng ID/prompt để tránh rò rỉ.
- Điểm nằm trong khoảng 0–1 nhưng **không phải tất cả đều là 0 hoặc 1**. Các dòng có điểm lẻ tập trung ở `grade-school-math`, `consensus_summary` và một số nhóm nhỏ. Vì vậy, quy tắc nhãn “Small sai, Large đúng” chỉ áp dụng trực tiếp cho dòng mà **cả hai điểm đều là 0 hoặc 1**. Nếu muốn giữ điểm lẻ, người thực hiện phải định nghĩa rõ ngưỡng và giải thích ý nghĩa của nhãn mới.
- Chi phí ở các cột `|total_cost` là ước tính trong benchmark, không nên xem là giá API hiện hành.

## Hai cặp LLM có thể dùng cho bài toán nhị phân

Các số khảo sát dưới đây tính trên **0-shot**, chỉ giữ dòng có điểm của cả hai LLM bằng 0 hoặc 1, **trước khi lọc prompt chứa chữ Hán**. Nhãn 1 nghĩa là Small đạt 0 và Large đạt 1. Tỷ lệ chi phí là trung bình chi phí Large chia trung bình chi phí Small trong cùng tập lọc.

| Small → Large | Dòng hợp lệ | Nhãn 1 | Tỷ lệ nhãn 1 | Điểm đúng Small / Large | Tỷ lệ chi phí |
| --- | ---: | ---: | ---: | ---: | ---: |
| Mistral 7B Chat → Mixtral 8×7B Chat | 28.617 | 9.540 | 33,34% | 27,41% / 55,27% | 3,01× |
| GPT-3.5 Turbo 1106 → GPT-4 1106 Preview | 28.664 | 7.056 | 24,62% | 62,31% / 81,55% | 11,23× |

Cặp Mistral/Mixtral diễn giải rõ sự khác biệt về quy mô mô hình và có đủ hai lớp để thử nghiệm; cặp GPT-3.5/GPT-4 thể hiện rõ hơn bài toán chi phí. Dự án hiện đã chọn cặp **Mistral/Mixtral**. Trước khi lọc chữ Hán, 28.617 dòng hợp lệ của cặp này chỉ có 28.605 prompt khác nhau, nên vẫn phải xử lý trùng lặp khi chia tập. Chi phí Mixtral lớn hơn Mistral ở 99,92% dòng của tập lọc điểm; khi đánh giá cần dùng chi phí thật của từng dòng thay vì luôn giả định Large đắt hơn.

## Yêu cầu phần cứng và giới hạn của bản offline

**Không cần chạy Mistral 7B hoặc Mixtral 8×7B trên laptop** để làm thí nghiệm offline. RouterBench đã lưu prompt, điểm chất lượng, chi phí và phản hồi của từng LLM. Phần cần huấn luyện là Router dùng Decision Tree và SVM; chất lượng và chi phí được tính bằng cách tra kết quả LLM có sẵn trên tập test. Cách làm này không cần VRAM để suy luận hai LLM.

Demo offline có thể nhập một prompt và hiển thị **LLM mà Router chọn**. Với prompt đã có trong benchmark, có thể hiển thị phản hồi đã lưu để minh họa. Với prompt mới ngoài benchmark, Router vẫn chọn được LLM, nhưng **không thể tạo câu trả lời mới** nếu không gọi API hoặc một dịch vụ suy luận bên ngoài. Chỉ bổ sung phần gọi LLM thật nếu yêu cầu của học phần đòi hỏi demo trả lời trực tiếp; khi đó cần chốt phương án tài nguyên riêng.

## Tiền xử lý hiện tại (cập nhật 21/11/2026)

Chạy `.venv/bin/python src/preprocess_data.py` để tạo lại ba file CSV trong `data/processed/`. Script dùng file **0-shot**, cặp **Mistral 7B Chat → Mixtral 8×7B Chat** và hạt giống ngẫu nhiên `42`. Các file kết quả được Git bỏ qua.

1. Chỉ giữ dòng mà cả hai điểm là `0` hoặc `1`: giữ **28.617** trong **36.497** dòng, loại **7.880** dòng có ít nhất một điểm lẻ.
2. Dùng `ast.literal_eval` đọc danh sách trong `prompt`, bỏ khoảng trắng ở đầu/cuối từng phần rồi ghép các phần bằng hai dấu xuống dòng thành `prompt_text`.
3. Loại prompt chứa chữ Hán bằng biểu thức `r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]"`, kể cả prompt trộn ngôn ngữ: loại thêm **361** dòng, còn **28.256** dòng.
4. Tạo `label = 1` khi Small được `0` và Large được `1`; ba cặp điểm còn lại có `label = 0`. Tổng cộng **9.525 nhãn 1** và **18.731 nhãn 0**.
5. Chia theo `prompt_text` bằng `GroupShuffleSplit`, tỷ lệ xấp xỉ 80/10/10. Có **10 dòng trùng prompt** sau chuẩn hóa; tất cả nằm cùng tập, không có prompt chung giữa các tập.

Bộ lọc chữ Hán là quy tắc đơn giản nhằm loại các prompt tiếng Trung trong phạm vi thực nghiệm, chưa phải bộ nhận diện ngôn ngữ hoàn chỉnh: chữ Hán cũng có thể xuất hiện trong tiếng Nhật và bộ lọc không bao phủ mọi ký tự Hán hiếm. Dữ liệu còn lại không được coi là hoàn toàn tiếng Anh. Bộ lọc được áp dụng **trước khi chia tập**; các tập được tạo lại từ đầu với cùng seed, nên số dòng không bằng việc chỉ xóa mẫu khỏi từng tập cũ.

| File | Số dòng | Nhãn 1 |
| --- | ---: | ---: |
| `routerbench_0shot_train.csv` | 22.605 | 7.606 |
| `routerbench_0shot_validation.csv` | 2.826 | 938 |
| `routerbench_0shot_test.csv` | 2.825 | 981 |

Đã đọc lại ba CSV để kiểm tra: đủ 8 cột, không thiếu dữ liệu hoặc có prompt rỗng, không còn prompt khớp bộ lọc chữ Hán, nhãn đúng quy tắc, không có ID hoặc prompt chung giữa các tập. Tổng hợp ba tập chứa đúng toàn bộ mẫu hợp lệ từ dữ liệu gốc: **28.256 dòng**, tương ứng **28.246 prompt khác nhau**.

Mỗi file có 8 cột: `sample_id`, `eval_name`, `prompt_text`, `label`, `small_score`, `large_score`, `small_cost`, `large_cost`. **Chỉ `prompt_text` được dùng làm đặc trưng đầu vào**; `label` là mục tiêu dự đoán, còn điểm/chi phí dùng để đánh giá Router. `sample_id` và `eval_name` chỉ là thông tin để truy vết và phân tích. Phản hồi LLM không được sao chép vào file đã xử lý; vẫn có thể tra lại file gốc bằng `sample_id`.

TF-IDF cho SVM nằm trong pipeline ở `src/train_models.py`: dùng `TfidfVectorizer(max_features=5000)`. Khi hiệu chỉnh xác suất bằng 3 fold trong train, mỗi fold học bộ từ vựng riêng; mô hình cơ sở cuối cùng học toàn train. Validation và test chỉ được biến đổi bằng bộ đã học. Đầu vào Decision Tree gồm ba đặc trưng cấu trúc trong `src/features.py`: số ký tự, số phần tách theo khoảng trắng và số dòng.

## Hai mốc so sánh trên validation (cập nhật 21/11/2026)

Chạy `.venv/bin/python src/evaluate_baselines.py`. Mỗi cách chọn được áp dụng lên cùng **2.826 prompt** của validation sau khi lọc chữ Hán và chia lại dữ liệu; script lấy điểm và chi phí của mô hình được chọn ở từng dòng.

| Cách chọn | Tỷ lệ câu được chấm đúng | Chi phí TB / câu | Tổng chi phí | Tỷ lệ gọi Mixtral | Accuracy so với nhãn Router |
| --- | ---: | ---: | ---: | ---: | ---: |
| Luôn chọn Mistral | 27,88% | 0,00003305 | 0,093411 | 0% | 66,81% |
| Luôn chọn Mixtral | 56,33% | 0,00010022 | 0,283233 | 100% | 33,19% |

Tỷ lệ câu được chấm đúng là trung bình điểm `0/1` của LLM được chọn. Đây là **kết quả trả lời**, còn tỷ lệ chọn mô hình nằm ở cột "Tỷ lệ gọi Mixtral". Chi phí là **ước tính của benchmark**, không phải giá API hiện hành. Accuracy so với nhãn Router chỉ đo mức khớp với quy tắc nhãn `0/1`: dù Always Small có accuracy cao hơn, tỷ lệ câu được chấm đúng thấp hơn nhiều. Vì vậy cần báo cáo cả tỷ lệ đúng, chi phí và tỷ lệ chọn Mixtral khi so sánh Router.

## Huấn luyện và đánh giá đã hoàn thành (23/11/2026)

1. Huấn luyện và hiệu chỉnh Decision Tree/SVM bằng các fold train không trùng prompt. Chọn cấu hình bằng log loss validation.
2. Chọn trọng số/ngưỡng trên validation để giảm chi phí với mức giảm điểm tối đa 2 điểm phần trăm so với Always Large; lưu cấu hình trước khi nạp test.
3. Đánh giá test và lưu mô hình, báo cáo, biểu đồ. Ensemble được chọn dùng 10% điểm cây + 90% điểm SVM, ngưỡng 0,21; tiết kiệm 20,54%, giảm tỷ lệ đúng 1,81 điểm phần trăm trên test. Xem khoảng bất định và các giới hạn trong [báo cáo](reports/RESULTS.md).

Các kết quả test đã được xem; nghiên cứu thêm tham số dựa vào chính các kết quả này phải được ghi rõ là phân tích tiếp theo, không gọi là một lần kiểm thử độc lập mới.

Trong buổi vấn đáp, nên giải thích được vì sao nhãn 0 bao gồm cả trường hợp hai LLM cùng sai, vì sao phải tránh trùng prompt giữa các tập và vì sao accuracy của Router không thay thế được phép đo tỷ lệ đúng–chi phí.
