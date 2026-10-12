# BÁO CÁO TIẾN ĐỘ THỰC HIỆN ĐỀ TÀI GR2

**Đề tài:** Xây dựng trợ lý AI sử dụng kiến trúc AI Agent hỗ trợ khách hàng ngân hàng  
*(AI Banking Assistant)*

* **Ngày báo cáo:** 12/10/2026 (Tuần từ 05/10/2026 đến 12/10/2026)
* **Giai đoạn:** Giai đoạn 1 — Xây dựng cơ sở tri thức và hệ thống RAG
* **Phiên bản:** 1.0


### 1. Kết quả đạt được trong tuần này (05/10/2026 – 12/10/2026)

Trong tuần làm việc vừa qua, tôi đã hoàn thành hai nhiệm vụ kỹ thuật cốt lõi:
* **Nhiệm vụ 1:** Crawl được trọn bộ cơ sở dữ liệu tri thức trực tiếp từ ngân hàng Vietcombank phục vụ cho RAG.
* **Nhiệm vụ 2:** Xây dựng hoàn chỉnh một hệ thống RAG bản v1.0 và đã chạy thử nghiệm thành công End-to-End.

#### 1.1. Thu thập bộ dữ liệu từ ngân hàng Vietcombank
Dữ liệu được thu thập trực tiếp từ cổng thông tin chính thức của Vietcombank (vietcombank.com.vn), bao gồm các quy định, biểu phí, cẩm nang sản phẩm và quy trình nghiệp vụ khách hàng cá nhân. Bộ dữ liệu sau khi crawl đã được làm sạch và phân loại chuẩn hóa:
* **62 tệp Markdown (.md)** chuẩn hóa theo 11 nhóm nghiệp vụ bán lẻ cốt lõi (tài khoản, thẻ, tiết kiệm, vay vốn, ngân hàng số, bảo hiểm, bảo mật...).
* **19 tệp PDF gốc** lưu trữ các điều khoản mở tài khoản trực tuyến, quy chế phát hành thẻ quốc tế và biểu phí.
* **7 tệp Word (.docx)** chuyên sâu về biểu lãi suất, tỷ giá, cẩm nang thẻ và phòng chống gian lận.
* **Bộ dataset benchmark tổng hợp gồm 110 tệp** (50 PDF, 50 MD, 10 Excel biểu phí/lãi suất) dùng để đánh giá định lượng thuật toán.

#### 1.2. Xây dựng và chạy End-to-End hệ thống RAG bản v1.0
Tôi đã hoàn thành xây dựng hệ thống Modular RAG Pipeline v1.0 với quy trình xử lý khép kín, từ lúc nhận câu hỏi của người dùng cho đến khi sinh câu trả lời có kèm căn cứ trích dẫn nguồn:
* **Pha Ingestion & Dual-Indexing:** Tài liệu được phân đoạn bảo toàn breadcrumbs Chương/Điều (Hierarchy Metadata) và cặp hỏi đáp (FAQ Pair). Lập chỉ mục kép song song: Sparse BM25 (Okapi BM25) và Dense Vector Store (Cosine HNSW 1024 chiều).
* **Bước 1 (Query Parser):** Chuẩn hóa các từ viết tắt chuyên ngành (eKYC, TKTT, Smart OTP, VCB Digibank) và trích xuất từ khóa.
* **Bước 2 (Hybrid Retrieval):** Truy vấn đồng thời trên Dense Vector và BM25, sau đó hợp nhất danh sách bằng thuật toán Reciprocal Rank Fusion (RRF k=60) để lấy ra Top-5 ứng viên.
* **Bước 3 (Reranking):** So khớp câu hỏi với các câu hỏi chuẩn (Golden Questions) và độ trùng lặp từ vựng để tinh lọc ra Top-3 chunks chính xác nhất.
* **Bước 4 (Context Synthesizer):** Đóng gói ngữ cảnh Markdown, gắn thẻ nguồn tham chiếu pháp lý `[Điều X, Khoản Y]` và kiểm soát giới hạn token.
* **Bước 5 (LLM Serving):** Truyền ngữ cảnh kèm System Guardrails ngân hàng vào LLM (DeepSeek / Gemini) để sinh câu trả lời chính xác, tránh bịa đặt số liệu.

#### 1.3. Sơ đồ kiến trúc RAG v1.0 (Draw.io Flow Diagram)
Toàn bộ luồng dữ liệu và các phân hệ của bản v1.0 đã được thiết kế chi tiết trên Draw.io, lưu tại thư mục `docs/flow diagram/rag pipeline/v1/rag_modular_pipeline_v1.drawio`:

![Sơ đồ luồng xử lý Modular RAG Pipeline v1.0](../flow%20diagram/rag%20pipeline/v1/rag_modular_pipeline_v1.png)

*Hình 1: Sơ đồ luồng xử lý hệ thống Modular RAG Pipeline v1.0 (Bản vẽ Draw.io)*

#### 1.4. Kết quả kiểm thử End-to-End và độ trễ thực tế
Thử nghiệm thực tế với câu hỏi nghiệp vụ: *"Hạn mức mở tài khoản eKYC tối đa bao nhiêu 1 ngày?"*

* **Câu trả lời sinh ra:**  
  "Theo quy định mở tài khoản thanh toán trực tuyến (eKYC) của Vietcombank:  
  (1) Hạn mức thông thường: Giao dịch tối đa 100.000.000 VNĐ/ngày (và 100.000.000 VNĐ/tháng).  
  (2) Hạn mức nâng cao: Sau khi hoàn tất xác thực sinh trắc học CCCD gắn chip qua NFC theo Quyết định 2345/QĐ-NHNN, hạn mức được nâng lên tối đa 400.000.000 VNĐ/ngày (và 1.000.000.000 VNĐ/tháng).  
  Giao dịch từ 10.000.000 VNĐ/lần trở lên bắt buộc phải xác thực khuôn mặt."
* **Dẫn chứng nguồn (Citations):**  
  `[Quy chế eKYC Vietcombank - Điều 12, Khoản 2; Quyết định 2345/QĐ-NHNN]`

**Bảng 1: Đo lường thời gian trễ chi tiết từng module (Latency breakdown)**

| Phân hệ xử lý | Thời gian (ms) | Tỷ trọng (%) | Đánh giá hiệu năng |
|---|:---:|:---:|---|
| 1. Query Parsing | 1.2 ms | 0.14% | Chuẩn hóa từ viết tắt & trích keyword |
| 2. Hybrid Retrieval (RRF) | 8.5 ms | 1.02% | Truy xuất song song Dense + Sparse |
| 3. Reranking | 3.1 ms | 0.37% | Golden Question Match & lọc nhiễu |
| 4. Context Synthesizer | 0.8 ms | 0.10% | Đóng gói Markdown & gắn citations |
| 5. LLM Inference | 820.0 ms | 98.37% | Sinh câu trả lời qua mô hình ngôn ngữ |
| **TỔNG CỘNG TOÀN TRÌNH** | **833.6 ms** | **100.0%** | **Đạt tiêu chuẩn thời gian thực (< 1.5s)** |

**Bảng 2: Kết quả so sánh 4 cấu hình RAG trên tập câu hỏi kiểm thử**

| Cấu hình Pipeline | Chiến lược Ingestion & Retrieval | Hit Rate@3 | MRR@3 |
|---|---|:---:|:---:|
| Pipeline 1 | FixedWindow + Dense Only (Cosine 1024d) | 72.0% | 0.65 |
| Pipeline 2 | FAQPair + Classic Okapi BM25 | 84.5% | 0.78 |
| Pipeline 3 | HierarchyMetadata + Linear Hybrid (α=0.5) | 91.2% | 0.86 |
| **Pipeline 4 (Bản v1.0)** | **Multi-Strategy + Advanced RRF (k=60) + Rerank** | **98.5%** | **0.96** |


### 2. Định hướng tuần tới: Xây dựng hệ thống chunking DLA (Document Layout Awareness)
Định hướng trọng tâm của tuần tới là xây dựng hệ thống chunking theo hướng Document Layout Awareness (DLA). Mục tiêu là có thể tiếp nhận và xử lý các định dạng dữ liệu phức tạp ngoài Markdown như PDF và Excel, bảo toàn cấu trúc bảng biểu, hình ảnh và tích hợp OCR trích xuất nội dung.

* **Phân tích bố cục tài liệu (DLA Engine):** Nhận diện không gian hình học của trang tài liệu: phân biệt rõ vùng Tiêu đề (Headings), Đoạn văn (Paragraphs), Bảng biểu (Tables) và Hình ảnh (Figures) trước khi chia chunk.
* **Xử lý bảng biểu Excel & PDF:** Thuật toán làm phẳng bảng (*Table Flattening*) cho Excel và PDF: tự động xử lý ô gộp, kế thừa tiêu đề đa cấp và chuyển đổi từng dòng dữ liệu thành một biểu diễn ngữ nghĩa hoàn chỉnh (*Row-level Contextual Chunks*).
* **Tích hợp OCR:** Tích hợp thư viện OCR (như RapidOCR chạy offline trên CPU) để trích xuất chữ tiếng Việt từ các tài liệu PDF scan và hình ảnh quy trình.
