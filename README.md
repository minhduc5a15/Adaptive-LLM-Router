# Adaptive LLM Router

Khung dự án cho đồ án Học máy cơ bản: định tuyến prompt giữa Small LLM và Large LLM bằng **Decision Tree** và **SVM**. [Báo cáo đề xuất](notes/Adaptive_LLM_Router.md) mô tả bài toán và kế hoạch thực nghiệm. Dự án hiện ở giai đoạn chuẩn bị dữ liệu; chưa có mô hình đã huấn luyện hay kết quả đánh giá.

## Thiết lập

Yêu cầu Python 3.10 trở lên. Trên Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

Tải toàn bộ RouterBench (khoảng 1,47 GB):

```bash
routerbench-download
```

Lệnh tải cố định phiên bản dataset và xác minh SHA-256 của từng file. Chạy lại lệnh sẽ bỏ qua các file đã đúng checksum và tiếp tục file tải dở. Dữ liệu nằm trong `data/raw/` và không được đưa vào Git.

> RouterBench được phân phối dưới dạng Python pickle. Chỉ giải tuần tự hóa các file khi đã xác minh nguồn và checksum; pickle có thể chạy mã khi nạp. Script tải chỉ tải và kiểm tra checksum, không mở pickle.

## Cấu trúc

```text
notes/Adaptive_LLM_Router.md     Báo cáo đề xuất (file cục bộ, không theo dõi bằng Git)
data/raw/                        File RouterBench gốc
data/processed/                  Dữ liệu sau ghép cặp LLM và chia tập
src/adaptive_llm_router/         Mã nguồn Python
  download.py                    Tải và xác minh dataset
  features.py                    Đặc trưng cấu trúc từ prompt
  models.py                      Cấu hình Decision Tree và SVM
  evaluation.py                  Chỉ số chất lượng và chi phí
models/                          Pipeline và mô hình đã huấn luyện
reports/figures/                 Biểu đồ thực nghiệm
tests/                           Kiểm thử cho pipeline khi triển khai
```

## Các bước triển khai tiếp theo

1. Khảo sát schema của RouterBench, chọn cặp Small/Large có đủ prompt chung và điểm chất lượng/chi phí.
2. Tạo nhãn `1` khi Small sai và Large đúng, còn lại `0`; chia train/validation/test theo prompt trước khi fit TF-IDF.
3. Huấn luyện Decision Tree trên đặc trưng cấu trúc và SVM trên TF-IDF.
4. Chọn ngưỡng định tuyến trên validation; đánh giá trên test cùng Always Small, Always Large và Random Router.
5. Lưu mô hình, biểu đồ chất lượng–chi phí và dựng demo Streamlit.

Nguồn: [RouterBench trên Hugging Face](https://huggingface.co/datasets/withmartian/routerbench), revision `784021482c3f320c6619ed4b3bb3b41a21424fcb`.
