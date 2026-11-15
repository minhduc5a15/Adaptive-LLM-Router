# Adaptive LLM Router

Khung chuẩn bị cho đồ án Học máy cơ bản: định tuyến prompt giữa Small LLM và Large LLM bằng **Decision Tree** và **SVM**. [Báo cáo đề xuất](notes/Adaptive_LLM_Router.md) mô tả bài toán và kế hoạch thực nghiệm. **Chưa có mã nguồn mô hình hoặc kết quả huấn luyện**; người thực hiện đồ án sẽ tự viết phần mã.

## Thiết lập

Yêu cầu Python 3.10 trở lên. Trên Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Ba file RouterBench (khoảng 1,47 GB) đã được tải vào `data/raw/` và xác minh SHA-256. [Ghi chú chuẩn bị dữ liệu](DATA_PREPARATION.md) lưu nguồn, phiên bản và checksum. Các file dataset không được đưa vào Git.

> RouterBench được phân phối dưới dạng Python pickle. Chỉ giải tuần tự hóa file từ nguồn đã xác minh; pickle có thể chạy mã khi nạp.

## Cấu trúc

```text
notes/Adaptive_LLM_Router.md     Báo cáo đề xuất (file cục bộ, không theo dõi bằng Git)
DATA_PREPARATION.md              Kết quả khảo sát dữ liệu và các quyết định cần chốt
data/raw/                        File RouterBench gốc
data/processed/                  Dữ liệu sau ghép cặp LLM và chia tập
src/                            Chỗ để bạn tự viết mã nguồn Python
models/                          Pipeline và mô hình đã huấn luyện
reports/figures/                 Biểu đồ thực nghiệm
tests/                          Chỗ để bạn tự viết kiểm thử
```

## Các bước triển khai tiếp theo

1. Đọc ghi chú khảo sát dữ liệu, chọn cặp Small/Large và quy tắc xử lý điểm lẻ.
2. Tạo nhãn `1` khi Small sai và Large đúng, còn lại `0`; chia train/validation/test theo prompt trước khi fit TF-IDF.
3. Huấn luyện Decision Tree trên đặc trưng cấu trúc và SVM trên TF-IDF.
4. Chọn ngưỡng định tuyến trên validation; đánh giá trên test cùng Always Small, Always Large và Random Router.
5. Lưu mô hình, biểu đồ chất lượng–chi phí và dựng demo Streamlit.

Nguồn: [RouterBench trên Hugging Face](https://huggingface.co/datasets/withmartian/routerbench), revision `784021482c3f320c6619ed4b3bb3b41a21424fcb`.
