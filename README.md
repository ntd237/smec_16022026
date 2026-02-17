# SMEC: Sequential Matryoshka Embedding Compression

Triển khai thuật toán **SMEC (Sequential Matryoshka Embedding Compression)** dựa trên báo cáo nghiên cứu và bài báo cùng tên. Dự án này cung cấp một khung huấn luyện để nén vector nhúng từ các mô hình ngôn ngữ lớn (LLMs) cho mục đích tìm kiếm thông tin (Information Retrieval).

## Tính năng chính

- **Sequential Matryoshka Representation Learning (SMRL)**: Huấn luyện tuần tự các kích thước vector giảm dần để ổn định gradient.
- **Adaptive Dimension Selection (ADS)**: Sử dụng Gumbel-Softmax để học cách chọn các chiều quan trọng nhất thay vì cắt tỉa tĩnh.
- **Selective Cross-Batch Memory (S-XBM)**: Hàng đợi bộ nhớ FIFO để lưu trữ và truy xuất các hard negatives từ các batch trước đó.

## Cài đặt

Yêu cầu Python 3.10+ và các thư viện trong `requirements.txt`.

```bash
pip install -r requirements.txt
```

## Cấu trúc Dự án

```
.
├── main.py                 # Entry point cho huấn luyện và đánh giá
├── requirements.txt        # Các thư viện phụ thuộc
├── src/
│   ├── data/
│   │   └── loader.py       # Data loader cho MTEB/BEIR
│   ├── models/
│   │   ├── ads.py          # Module Adaptive Dimension Selection
│   │   ├── memory.py       # Module S-XBM Memory
│   │   └── smec_wrapper.py # Wrapper kết hợp Backbone + ADS
│   ├── loss.py             # Contrastive Loss + S-XBM support
│   ├── trainer.py          # Vòng lặp huấn luyện SMRL tuần tự
│   └── evaluate.py         # Script đánh giá dùng MTEB
└── docs/                   # Tài liệu dự án
```

## Hướng dẫn Sử dụng

### 1. Huấn luyện (Training)

Chạy lệnh sau để bắt đầu huấn luyện mô hình theo chiến lược SMRL:

```bash
python main.py --mode train --model_name bert-base-uncased --dataset_name quora --epochs 3
```

Các tham số tùy chọn:
- `--model_name`: Tên backbone model (mặc định: `bert-base-uncased`).
- `--output_dir`: Thư mục lưu checkpoint.
- `--batch_size`: Kích thước batch.
- `--lr`: Learning rate.

### 2. Đánh giá (Evaluation)

Chạy lệnh sau để đánh giá mô hình trên các tác vụ MTEB (ví dụ: QuoraRetrieval):

```bash
python main.py --mode eval --model_name bert-base-uncased --output_dir ./checkpoints
```

## Kết quả (Dự kiến)

Mô hình SMEC được kỳ vọng sẽ đạt hiệu suất nén tốt hơn so với các phương pháp MRL truyền thống, đặc biệt ở các mức nén cao (ví dụ: 12x, 14x).

## Tham khảo

- **SMEC: Rethinking Matryoshka Representation Learning for Retrieval Embedding Compression**
- Dataset: MTEB/BEIR Benchmark
