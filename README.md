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
│   │   └── loader.py       # Data loader cho MTEB (Hỗ trợ Quora-duplicates)
│   ├── models/
│   │   ├── ads.py          # Module Adaptive Dimension Selection (Gumbel-Softmax)
│   │   ├── memory.py       # Module Selective Cross-Batch Memory (S-XBM)
│   │   └── smec_wrapper.py # Wrapper kết hợp Backbone + ADS
│   ├── loss.py             # Contrastive Loss (InfoNCE) + S-XBM support
│   ├── trainer.py          # Vòng lặp huấn luyện SMRL (Hỗ trợ FP16 Mixed Precision)
│   └── evaluate.py         # Script đánh giá dùng MTEB (Kế thừa EncoderProtocol)
└── docs/
    ├── architecture.md     # Tài liệu chi tiết kiến trúc hệ thống
    └── evaluation_report.md # Báo cáo kết quả và so sánh nén embedding
```

## Hướng dẫn Sử dụng

### 1. Huấn luyện (Training)

Chạy huấn luyện tuần tự qua các Dimension (768 -> 384 -> 192):

```bash
python main.py --mode train --model_name bert-base-uncased --dataset_name quora --epochs 3 --batch_size 32 --max_length 128
```

**Các tham số chính:**
- `--model_name`: Backbone model từ HuggingFace (mặc định: `bert-base-uncased`).
- `--dataset_name`: Dataset cho huấn luyện (mặc định: `quora`).
- `--epochs`: Số lượng epoch huấn luyện cho **mỗi** chiều không gian.
- `--batch_size`: Kích thước batch (khuyên dùng 32 cho VRAM 8GB).
- `--max_length`: Độ dài tối đa của sequence (mặc định: 128).
- `--lr`: Learning rate (mặc định: 2e-5).

### 2. Đánh giá (Evaluation)

Dự án hỗ trợ đánh giá tự động trên bộ tiêu chuẩn **MTEB (STSBenchmark)**:

**Tự động đánh giá tất cả checkpoint trong thư mục:**
```bash
python main.py --mode eval --output_dir ./checkpoints
```

**Đánh giá một file checkpoint cụ thể:**
```bash
python main.py --mode eval --checkpoint ./checkpoints/checkpoint_dim_192
```

*Kết quả sẽ được lưu vào thư mục `results/` dưới dạng các file JSON chi tiết.*

## Hiệu năng & Tối ưu hóa

- **GPU Acceleration**: Tự động sử dụng CUDA nếu khả dụng.
- **Mixed Precision**: Sử dụng `torch.cuda.amp` để tối ưu hóa bộ nhớ và tốc độ trên các dòng card RTX.
- **Auto-Dimension**: Lớp ADS tự động điều chỉnh Mask weights dựa trên checkpoint được load.

## Tham khảo

- **SMEC Paper**: Rethinking Matryoshka Representation Learning for Retrieval Embedding Compression.
- **Framework**: PyTorch & MTEB.
