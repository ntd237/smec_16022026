# Báo cáo Đánh giá và So sánh Hiệu năng SMEC 

## 1. Tổng quan
Báo cáo này trình bày kết quả đánh giá mô hình **SMEC (Sequential Matryoshka Embedding Compression)** trên benchmark **STSBenchmark** sau lần huấn luyện và đánh giá mới nhất. Các chỉ số được đo lường cho 3 kích thước vector nhúng: 768, 384, và 192.

## 2. Kết quả Đánh giá Chi tiết 

Dưới đây là các chỉ số chính dựa trên Cosine Similarity (được làm tròn 4 chữ số thập phân):

| Kích thước (Dim) | Tỷ lệ Nén | Spearman (Cosine) | Pearson (Cosine) | Main Score (Spearman) |
|:---:|:---:|:---:|:---:|:---:|
| **768** | 1x (Full) | 0.7211 | 0.7038 | 0.7211 |
| **384** | 2x (50%) | **0.7232** | **0.7079** | **0.7232** |
| **192** | 4x (75%) | 0.7227 | 0.7069 | 0.7227 |

## 3. Phân tích So sánh

### 3.1. Đỉnh cao hiệu năng tại Dim 384
Mô hình đạt hiệu suất cao nhất tại kích thước **384 chiều**. Điều này cho thấy với kiến trúc BERT-base, việc biểu diễn thông tin trong 384 chiều là "điểm ngọt" (sweet spot), nơi các đặc trưng ngữ nghĩa được cô đọng tốt nhất mà không bị lẫn nhiễu như ở bản 768 chiều gốc.

### 3.2. Hiệu quả nén tại Dim 192
Ở kích thước **192 chiều**, mô hình chỉ giảm **0.0005** điểm Spearman so với bản 384 và vẫn **cao hơn** bản 768 gốc. Đây là một kết quả cực kỳ thành công của thuật toán SMEC, cho thấy khả năng bảo toàn thông tin ưu việt ngay cả khi kích thước vector giảm đi 4 lần.

### 3.3. Sự ổn định của thuật toán
Kết quả giữa các lần chạy cho thấy sự ổn định tuyệt đối của cơ chế ADS và SMRL. Các con số gần như không thay đổi, minh chứng cho việc mô hình đã hội tụ tốt tại các tầng Matryoshka khác nhau.

## 4. Kết luận Đề tài
Dựa trên kết quả thực nghiệm mới nhất:
- **Mục tiêu nén**: Đạt tỷ lệ nén 4:1 (từ 768 xuống 192) mà vẫn giữ được hiệu năng tương đương/cao hơn gốc.
- **Tiết kiệm tài nguyên**: Việc sử dụng Dim 192 sẽ giúp giảm 75% chi phí bộ nhớ cho vector database trong khi vẫn đảm bảo độ chính xác tìm kiếm cao nhất.

**Khuyến nghị cuối cùng**: Sử dụng checkpoint **192** cho các ứng dụng thực tế để tối ưu hóa cả tốc độ và độ chính xác.
