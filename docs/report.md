# SMEC: Tái định nghĩa Học biểu diễn Matryoshka để Nén Vector Nhúng trong Tìm kiếm Thông tin  
*(Sequential Matryoshka Embedding Compression – SMEC)*

---

## Tóm tắt Điều hành

Tài liệu này trình bày các phát hiện và phương pháp luận từ nghiên cứu **Sequential Matryoshka Embedding Compression (SMEC)**, một khung huấn luyện mới nhằm giải quyết các thách thức về **chi phí lưu trữ (storage cost)** và **chi phí tính toán (computational cost)** do các vector nhúng (embedding) cao chiều từ các mô hình ngôn ngữ lớn (Large Language Models – LLMs) gây ra.

SMEC mở rộng và tái định nghĩa **Matryoshka Representation Learning (MRL)** bằng cách giới thiệu ba thành phần cốt lõi:

1. **Sequential Matryoshka Representation Learning (SMRL)**
2. **Adaptive Dimension Selection (ADS)**
3. **Selective Cross-Batch Memory (S-XBM)**

Kết quả thực nghiệm cho thấy SMEC đạt **tỷ lệ nén không tổn hao (lossless compression)** lên tới **14×** đối với mô hình LLM2Vec-7B, đồng thời vượt trội so với các phương pháp hiện đại như *Matryoshka-Adaptor* và *Search-Adaptor* trên nhiều phương thức dữ liệu (text, image, multimodal).

---

## Bối cảnh và Thách thức hiện tại

Các mô hình ngôn ngữ lớn như **:contentReference[oaicite:0]{index=0}** hay **:contentReference[oaicite:1]{index=1}** thường sinh ra các vector nhúng có độ phân giải rất cao (từ 1.024 đến 4.096 chiều), cho phép mã hóa cấu trúc ngữ nghĩa tinh vi. Tuy nhiên, trong triển khai thực tế, các embedding này tạo ra nhiều rào cản nghiêm trọng.

### Các thách thức chính

- **Chi phí tính toán và lưu trữ**  
  Chi phí bộ nhớ tăng tuyến tính theo số lượng tài liệu và số chiều embedding, đặc biệt trong các hệ thống truy vấn động hoặc xử lý văn bản dài.

- **Lời nguyền đa chiều (Curse of Dimensionality)**  
  Hiệu suất của các thuật toán tìm kiếm hàng xóm gần nhất như **:contentReference[oaicite:2]{index=2}** hoặc **:contentReference[oaicite:3]{index=3}** suy giảm đáng kể khi số chiều tăng, dẫn đến độ trễ truy vấn tăng nhanh.

- **Hạn chế của MRL truyền thống**
  - **Biến động gradient**: Tối ưu hóa đồng thời nhiều không gian chiều gây mất cân bằng cường độ gradient.
  - **Suy giảm thông tin**: Cắt tỉa tĩnh (ví dụ `D → D/2`) không phản ánh mức độ quan trọng khác nhau của từng chiều.
  - **Lấy mẫu hạn chế**: Huấn luyện theo batch nhỏ, thiếu đa dạng và phụ thuộc vào dữ liệu gán nhãn.

---

## Khung giải pháp Sequential Matryoshka Embedding Compression (SMEC)

SMEC khắc phục các hạn chế trên thông qua ba đổi mới kỹ thuật chính.

---

## 1. Sequential Matryoshka Representation Learning (SMRL)

Khác với MRL tối ưu **song song nhiều kích thước**, SMRL áp dụng **chiến lược nén tuần tự**:

\[
D \rightarrow \frac{D}{2} \rightarrow \frac{D}{4} \rightarrow \dots
\]

### Đặc điểm chính

- **Giảm biến động gradient**  
  Mỗi giai đoạn chỉ tối ưu *một bước giảm chiều*, giúp gradient ổn định hơn.

- **Đóng băng tham số (parameter freezing)**  
  Sau khi hội tụ ở một mức chiều, các tham số được cố định để tránh suy giảm hiệu suất ở các bước sau.

- **Khả năng đào tạo tiếp tục (continual compression)**  
  Cho phép nén sâu hơn mà không cần huấn luyện lại từ đầu.

---

## 2. Adaptive Dimension Selection (ADS)

ADS thay thế chiến lược cắt tỉa tĩnh bằng **lựa chọn thứ nguyên động và có thể học được**.

### Cơ chế

- Sử dụng **tham số hóa khả học** kết hợp với **Gumbel-Softmax**
- Biến bài toán chọn chiều rời rạc thành bài toán **có thể vi phân (differentiable)**

### Hiệu quả

- Ở mức nén **6×**, ADS vẫn giữ được **>80% các chiều quan trọng**
- Trong khi MRL chỉ cho thấy **tương quan tuyến tính yếu** giữa độ quan trọng và tỷ lệ nén

---

## 3. Selective Cross-Batch Memory (S-XBM)

S-XBM tăng cường học tương phản (contrastive learning) giữa embedding gốc và embedding nén.

### Cơ chế hoạt động

- Duy trì **FIFO memory queue** chứa các đặc trưng lịch sử
- Chỉ truy xuất **top-k mẫu tương tự nhất** để tạo batch mới

### Lợi ích

- Tập trung vào **hard samples** giàu thông tin
- Lưu trữ embedding từ **backbone đóng băng** để tránh *feature drift*

---

## Kết quả Thực nghiệm và Phân tích Hiệu suất

SMEC được đánh giá trên nhiều bộ dữ liệu:

- Văn bản: **:contentReference[oaicite:4]{index=4}**
- Hình ảnh: Products-10K
- Đa phương thức: Fashion-200K

---

## Hiệu quả Nén không Tổn hao

Trên BEIR (Quora):

- **LLM2Vec-7B (3584D)**: ~**14× lossless compression**
- **LLM2Vec-1B (1536D)**: ~**12× lossless compression**

---

## So sánh với các phương pháp khác (NDCG@10)

| Mô hình gốc | Kích thước nén | Search-Adaptor | MRL-Adaptor | **SMEC (Ours)** |
|------------|---------------|----------------|-------------|-----------------|
| OpenAI text-embedding-3-large | 128 | ~51.5% | ~54.5% | **~56.5%** |
| LLM2Vec (Qwen2-7B) | 128 | ~51.0% | ~53.8% | **~55.8%** |
| LLM2Vec (Qwen2-7B) | 256 | ~56.5% | ~58.0% | **~59.2%** |

---

## Phân tích Tầm quan trọng của Thứ nguyên (WARE)

Sử dụng **Weighted Average Reconstruction Error (WARE)**:

- Ở **256 chiều**:
  - **ADS**: 83.6% chiều quan trọng được chọn
  - **MRL**: 17.4%

→ ADS vượt trội rõ rệt trong việc bảo toàn thông tin ngữ nghĩa.

---

## Phân tích Gradient và Hội tụ

- **Biến động gradient**  
  MRL có phương sai gradient cao hơn SMRL trên cả:
  - Rank loss  
  - MSE loss  
  - Cross-Entropy loss

- **Hội tụ**  
  - SMRL bắt đầu hội tụ ~epoch 15  
  - MRL tiếp tục dao động đến sau epoch 20

- **Ablation study**  
  Đóng góp hiệu suất:  
  **SMRL > ADS > S-XBM**

---

## Kết luận và Hướng phát triển

**SMEC** đại diện cho một bước tiến quan trọng trong **nén vector nhúng cho tìm kiếm thông tin quy mô lớn**, cho phép:

- Giảm mạnh chi phí hạ tầng
- Duy trì hoặc cải thiện độ chính xác
- Tương thích với nhiều phương thức dữ liệu và backbone hiện đại

### Hạn chế và hướng tương lai

- Phụ thuộc vào dữ liệu gán nhãn theo miền
- Hạn chế khả năng tổng quát hóa xuyên miền

**Hướng nghiên cứu tiếp theo**:
- Huấn luyện *end-to-end* toàn bộ mô hình biểu diễn
- Sinh trực tiếp embedding đa độ phân giải
- Đánh giá trên các tập dữ liệu đa dạng và OOD lớn hơn

---