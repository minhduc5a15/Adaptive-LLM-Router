# Ghi chú chuẩn bị dữ liệu RouterBench

Ghi nhận ngày 15/11/2026. Đây là kết quả **khảo sát dữ liệu**, chưa có pipeline xử lý, mô hình huấn luyện hoặc kết quả kiểm thử. Các phép khảo sát được chạy tạm thời và không lưu thành mã nguồn dự án.

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

Các số dưới đây tính trên **0-shot**, chỉ giữ dòng có điểm của cả hai LLM bằng 0 hoặc 1. Nhãn 1 nghĩa là Small đạt 0 và Large đạt 1. Tỷ lệ chi phí là trung bình chi phí Large chia trung bình chi phí Small trong cùng tập lọc.

| Small → Large | Dòng hợp lệ | Nhãn 1 | Tỷ lệ nhãn 1 | Điểm đúng Small / Large | Tỷ lệ chi phí |
| --- | ---: | ---: | ---: | ---: | ---: |
| Mistral 7B Chat → Mixtral 8×7B Chat | 28.617 | 9.540 | 33,34% | 27,41% / 55,27% | 3,01× |
| GPT-3.5 Turbo 1106 → GPT-4 1106 Preview | 28.664 | 7.056 | 24,62% | 62,31% / 81,55% | 11,23× |

Cặp Mistral/Mixtral diễn giải rõ sự khác biệt về quy mô mô hình và có đủ hai lớp để thử nghiệm; cặp GPT-3.5/GPT-4 thể hiện rõ hơn bài toán chi phí. Đây là hai **ứng viên**, chưa phải quyết định cuối cùng. Với Mistral/Mixtral, 28.617 dòng hợp lệ chỉ có 28.605 prompt khác nhau, nên vẫn phải xử lý trùng lặp khi chia tập. Chi phí Mixtral lớn hơn Mistral ở 99,92% dòng của tập lọc; khi đánh giá cần dùng chi phí thật của từng dòng thay vì luôn giả định Large đắt hơn.

## Quyết định cần chốt trước khi tự viết mã

1. Chọn **một phiên bản dữ liệu** để triển khai đầu tiên; 0-shot là lựa chọn đơn giản hơn vì không có dòng thiếu điểm.
2. Chọn **một cặp Small/Large**, ghi lý do dựa trên điểm và chi phí quan sát được.
3. Quyết định chỉ dùng điểm nhị phân hay đặt quy tắc riêng cho điểm lẻ. Ghi lại số dòng bị loại và phân bố nhãn sau lọc.
4. Xác định cách biến danh sách trong `prompt` thành văn bản đầu vào, rồi kiểm tra prompt trùng **sau** bước chuẩn hóa này.
5. Chia train/validation/test theo prompt hoặc nhóm prompt. Chỉ fit TF-IDF trên train; không đưa điểm, chi phí, phản hồi LLM hoặc `oracle_model_to_route_to` vào đặc trưng Router.
6. Chốt baseline và chỉ số đánh giá trước khi xem test: Always Small, Always Large, Decision Tree, SVM; báo cáo cả chất lượng, tổng chi phí và tỷ lệ gọi Large.

Trong buổi vấn đáp, nên giải thích được vì sao nhãn 0 bao gồm cả trường hợp hai LLM cùng sai, vì sao phải tránh trùng prompt giữa các tập và vì sao accuracy của Router không thay thế được phép đo chất lượng–chi phí.
