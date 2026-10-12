# BÁO CÁO NGHIÊN CỨU & ĐỐI SOÁT HỌC THUẬT: PHÂN HỆ RAG & BENCHMARK CHO ĐỒ ÁN AI BANKING ASSISTANT (GR2)

> **Phương pháp thực hiện**: Tuân thủ quy định **Source-First & Deep Research Directive**.  
> **Nguồn tri thức**: Khởi chạy Deep Research nạp **39 tài liệu học thuật và kỹ thuật chuyên sâu** vào sổ tay **`AI Agent banking`** (`1938151d-f91d-47df-b395-4b27aac0a211`) kết hợp đối soát sổ tay **`RAG`** (`880c1277-4ac1-4b2b-9565-54913bda81df`).  
> **Các tài liệu nguồn tiêu biểu**:
> - *FinanceBench: A New Benchmark for Financial Question Answering* (Patronus AI, arXiv / GitHub)
> - *Late Chunking: Contextual Chunk Embeddings Using Long-Context Embedding Models* (Jina AI & Weaviate, arXiv)
> - *Contextual Retrieval in AI Systems* (Anthropic / Claude Cookbook)
> - *Stop chunking tables: How we built an agentic GraphRAG for financial disclosures with Docling* (Red Hat Developer)
> - *RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval* (Stanford University, ICLR)
> - *Why Off-the-Shelf LLMs Fail in Fintech — And Take Months to Get Right* (SEC 10-K Experiment)
> - *The Hidden Cost of 98% Accuracy: A Practical Guide to RAG Architecture Selection*

---

## I. TẠI SAO NAIVE RAG THẤT BẠI TRONG NGÂN HÀNG & TÀI CHÍNH

Theo kết quả công bố từ benchmark chuẩn hóa **FinanceBench (Patronus AI)** trên 10,231 câu hỏi tài chính thực tế:
* Các cấu hình RAG truyền thống (Naive RAG: Vector Store + LLM như GPT-4-Turbo) **trả lời sai hoặc từ chối trả lời lên tới 81%** (độ chính xác chỉ đạt khoảng 19% - 35%).
* Nguyên nhân thất bại cốt lõi đến từ 4 đặc trưng bệnh lý cấu trúc (*Structural Pathology*) của tài liệu ngân hàng:

1. **Bẫy phá vỡ cấu trúc bảng biểu (*Table Flattening*):**
   * Báo cáo ngân hàng, biểu phí và hồ sơ tín dụng chứa dày đặc bảng đa chiều. Khi Naive RAG cắt đoạn tĩnh (fixed-size chunking), các bảng biểu bị làm dẹt thành văn bản thô (*word soup*), cắt đứt mối liên kết giữa con số với tiêu đề cột/hàng.
2. **Đứt gãy chú thích chân trang (*Severed Footnotes*):**
   * Các điều kiện ngoại lệ, quy tắc tính lãi suất và biên độ phí quan trọng thường nằm ở phần Footnotes cuối trang. Naive RAG làm đứt mối liên kết giữa dữ liệu trong bảng và các ghi chú giải thích này.
3. **Bẫy đồng chỉ & Mất ngữ cảnh (*Co-reference Traps & Context Loss*):**
   * Các đoạn trích chứa các đại từ thay thế (*"nó"*, *"khoản vay này"*, *"sản phẩm trên"*, *"so với quý trước"*) khi bị bóc tách độc lập vào một chunk sẽ hoàn toàn mất đối tượng tham chiếu. Một con số lãi suất hay biểu phí không gắn liền với tên gói dịch vụ và kỳ hạn cụ thể sẽ trở nên vô giá trị hoặc gây hiểu nhầm nghiêm trọng.
4. **Bất lực trước các truy vấn tổng hợp toàn cục (*Global Aggregation Failure*):**
   * Tìm kiếm vector chỉ so khớp cục bộ theo ngữ nghĩa tương đồng (thường trả về các điều khoản rủi ro chung chung - *boilerplate text*) nhưng hoàn toàn bất lực trước các câu hỏi đòi hỏi tính tổng hợp hoặc xếp hạng trên toàn bộ kho tài liệu (ví dụ: *"Tổng hợp 5 gói vay mua nhà có lãi suất ưu đãi thấp nhất"*).

---

## II. ĐỐI SOÁT & SO SÁNH CHUYÊN SÂU CÁC KỸ THUẬT RAG NÂNG CAO

| Kỹ Thuật RAG | Cơ Chế Kỹ Thuật Cốt Lõi | Ưu Điểm Đột Phá Cho Ngân Hàng | Đánh Đổi Kỹ Thuật (Trade-offs) |
| :--- | :--- | :--- | :--- |
| **Late Chunking** *(Jina AI)* | Nạp toàn bộ tài liệu dài qua mô hình embedding ngữ cảnh dài (8192 tokens) để tính toán ma trận Self-Attention toàn cục trước, sau đó mới chia ranh giới chunk và Mean-Pooling. | Khắc phục triệt để **Bẫy đồng chỉ**; chunk vector mang đầy đủ ngữ cảnh của văn bản gốc; **không tốn thêm chi phí và độ trễ gọi LLM**. | Giới hạn ở độ dài ngữ cảnh tối đa của mô hình embedding (cần sliding window nếu tài liệu >8k tokens). |
| **Contextual Retrieval** *(Anthropic)* | Dùng LLM (kết hợp Prompt Caching) tạo một đoạn mô tả ngữ cảnh ngắn (50–100 từ) gắn vào đầu mỗi chunk trước khi index vector và BM25. | **Giảm tới 49% lỗi truy xuất** (giảm 67% khi kết hợp Reranker); đưa tỷ lệ Pass@10 đạt **95.26%**. | Tốn chi phí tiền xử lý ban đầu khi nạp tài liệu (ingestion cost). |
| **Table Extraction** *(Docling / IBM)* | Bóc tách layout phân cấp, giữ nguyên hình học bảng biểu (cells, rows, columns, headers) và xuất ra cấu trúc JSON/Markdown hoặc ánh xạ Typed Graph. | Không làm nát bảng biểu thành *word soup*; bảo toàn 100% quan hệ giữa các con số tài chính, điều kiện và footnotes. | Cần bổ sung pipeline parser chuyên biệt trong hạ tầng ETL. |
| **RAPTOR Tree Retrieval** *(Stanford / ICLR)* | Gom cụm các đoạn văn bản bằng GMM, dùng LLM tóm tắt đệ quy từ dưới lên (Bottom-up) để dựng cây phân cấp từ chi tiết đến vĩ mô. | Cho phép trả lời các câu hỏi suy luận đa bước (*Multi-step Reasoning*) và câu hỏi tổng quan toàn bộ tài liệu quy chế. | Tốn tài nguyên tính toán và chi phí xây dựng cây tóm tắt đệ quy. |
| **Cross-Encoder Reranker** | Tìm kiếm diện rộng Top 50–100 ứng viên (Hybrid Search), sau đó dùng mô hình Cross-Encoder chấm điểm tương quan ma trận $(Query, Chunk)$. | **Giảm 47% lỗi truy xuất** so với RAG cơ bản; loại bỏ triệt để các đoạn văn bản nhiễu trước khi đưa vào LLM sinh câu trả lời. | Tăng độ trễ truy vấn khoảng 100–200ms trên mỗi câu hỏi. |

---

## III. BỘ CHỈ SỐ BENCHMARK CHUẨN HÓA CHO AI BANKING ASSISTANT

Theo chuẩn đo lường từ **FinanceBench** và khung **RAGAS**, bài toán đánh giá RAG cho ngân hàng cần được chia tách độc lập giữa **Bộ truy xuất (Retriever)** và **Bộ sinh (Generator)**:

### 1. Chỉ số Đánh giá Bộ truy xuất (Retriever Quality)
* **Recall@K ($K \in \{3, 5, 10\}$)**: Tỷ lệ các đoạn văn bản chuẩn (Ground Truth Contexts) được tìm thấy trong Top-K kết quả trả về.
* **MRR (Mean Reciprocal Rank)**: Điểm trung bình của vị trí nghịch đảo của tài liệu chuẩn đầu tiên. Đo khả năng đẩy tài liệu chính xác nhất lên vị trí số 1.
* **Hit Rate@K**: Tỷ lệ câu hỏi mà hệ thống tìm được ít nhất 1 tài liệu chuẩn trong Top-K.
* **Context Precision (RAGAS)**: Mức độ ưu tiên xếp hạng các chunk chứa thông tin hữu ích ở các vị trí đầu danh sách (chống hiện tượng *Lost in the Middle*).
* **Context Recall (RAGAS)**: Tỷ lệ thông tin cần thiết để trả lời câu hỏi được thu hồi đầy đủ từ kho văn bản.

### 2. Chỉ số Đánh giá Bộ sinh (Generator Quality & An Toàn Ngân Hàng)
* **Faithfulness (Độ trung thực - Chỉ số sống còn)**:
  $$\text{Faithfulness} = \frac{\text{Số lượng phát biểu trong câu trả lời có bằng chứng từ Context}}{\text{Tổng số phát biểu sinh ra}}$$
  *Bắt buộc đạt $\ge 0.90$ để ngăn chặn tuyệt đối rủi ro ảo giác về lãi suất, phí phạt và điều khoản pháp lý.*
* **Answer Correctness (Độ chính xác thực tế)**: Đo lường mức độ đồng thuận ngữ nghĩa và dữ kiện giữa câu trả lời sinh ra và đáp án chuẩn (Ground Truth Answer).
* **Answer Relevance (Độ liên quan)**: Đánh giá câu trả lời có tập trung giải quyết đúng câu hỏi hay bị lan man sang các điều khoản không liên quan.

### 3. Đánh Đổi Vận Hành: Precision-Recall, Độ Trễ & Chi Phí (Trade-off Analysis)
* **Quy luật Precision-Recall Trade-off trong Tài chính**:
  * Khi tăng $K$ từ 5 lên 20: Information Recall tăng nhẹ (từ 0.064 lên 0.086), nhưng **Precision tụt dốc**, làm loãng ngữ cảnh nạp vào LLM, dẫn đến tăng nguy cơ ảo giác và tăng độ trễ.
  * **Giải pháp tối ưu**: Duy trì $K = 5 \div 10$ kết hợp với Cross-Encoder Reranker để tối đa hóa cả Recall lẫn Precision.
* **Cơ chế Page-level Retrieval**:
  * Thay vì chia nhỏ thành 36,000 chunks, chuyển sang đánh chỉ mục và truy xuất theo trang tài liệu kèm tóm tắt ngữ cảnh giúp giảm không gian tìm kiếm xuống còn 12,000 đơn vị, tăng tính ổn định và định vị thông tin chính xác hơn 30%.
* **Độ trễ & Chi phí (Latency & Token Cost)**:
  * Vector RAG thuần túy: Nhanh (~3.4s) nhưng độ chính xác rất thấp (~14% - 19%).
  * Agentic / Hybrid RAG + Reranking: Độ trễ khoảng ~5s - 8s nhưng độ chính xác tăng vọt lên **76% - 88%**.

---

## IV. BẢNG DỮ LIỆU ĐỐI CHUẨN THỰC NGHIỆM ĐÃ ĐƯỢC CHỨNG MINH (BENCHMARK EVIDENCE)

| Kiến Trúc / Cấu Hình RAG | Độ Chính Xác (Accuracy / Correctness) | Tỷ Lệ Lỗi / Từ Chối Trả Lời | Tỷ Lệ Pass@10 | Ghi Chú Thực Nghiệm Nguồn |
| :--- | :---: | :---: | :---: | :--- |
| **Naive Vector RAG (Baseline)** | 19.0% – 35.0% | **81.0%** (Sai hoặc từ chối) | 87.15% | Thử nghiệm FinanceBench trên GPT-4-Turbo / Llama-2. |
| **+ Contextual Embeddings (Anthropic)** | Cải thiện +5% đến +7% | Giảm 49% lỗi truy xuất | 92.34% | Claude Cookbook Experiment. |
| **+ Hybrid Search (Vector + BM25)** | Cải thiện độ phủ từ khóa | Tăng thu hồi mã sản phẩm | 93.21% | Kết hợp Sparse & Dense Retrieval. |
| **+ Cross-Encoder Reranking** | Cải thiện độ chính xác vượt bậc | Giảm 47% lỗi truy xuất | **95.26%** | Đưa tài liệu đúng vào Top-10 cho 95% câu hỏi. |
| **Specialized Financial RAG Stack** | **76.0% – 88.0%** | Giảm thiểu ảo giác tối đa | **> 96.0%** | Kết hợp Docling table parser, Page-level retrieval và Gemini/GPT-4. |

---

## V. KIẾN TRÚC RAG KHUYẾN NGHỊ TRIỂN KHAI CHO AI BANKING ASSISTANT (GR2)

```
[Văn Bản Ngân Hàng: PDF / Biểu Phí / Quy Định]
                       │
                       ▼
         [Tầng 1: Parser Chuyên Dụng (Docling)]
       (Bóc tách Layout, Bảng Biểu & Footnotes)
                       │
         ┌─────────────┴─────────────┐
         ▼                           ▼
[Late Chunking (Jina AI)]    [Contextual Retrieval (Anthropic)]
(Bảo toàn Attention 8k)      (Gắn ngữ cảnh cấp văn bản vào Chunk)
         │                           │
         └─────────────┬─────────────┘
                       ▼
    [Tầng 2: PostgreSQL pgvector + BM25]
     (Chỉ mục HNSW + Reciprocal Rank Fusion RRF k=60)
                       │
                       ▼
       [Tầng 3: Cross-Encoder Reranker]
   (Chấm điểm tương quan, lọc lấy Top 5 Chunks)
                       │
                       ▼
    [Tầng 4: LangGraph Critic & Verification]
 (Kiểm tra Faithfulness >= 0.90 + Sinh Citations)
                       │
                       ▼
        [Câu Trả Lời Chuẩn Xác Cho Khách Hàng]
```

### Kế Hoạch 5 Bước Thực Nghiệm Benchmark Cho GR2:
1. **Bước 1 (Golden Test Set):** Xây dựng bộ test 100 câu hỏi theo format của FinanceBench đại diện cho 4 nhóm nghiệp vụ ngân hàng.
2. **Bước 2 (Ablation Benchmark):** Chạy thực nghiệm 5 phiên bản: Baseline EXP-000 (LLM thuần) $\rightarrow$ EXP-001 (Vector RAG) $\rightarrow$ EXP-002 (Hybrid RRF) $\rightarrow$ EXP-003 (Reranker) $\rightarrow$ EXP-004 (Late Chunking + Docling).
3. **Bước 3 (Đo lường Retriever):** Thu thập các chỉ số `Recall@3`, `Recall@5`, `MRR`, `Context Precision`.
4. **Bước 4 (Đo lường Generator):** Dùng `RAGAS` đo `Faithfulness`, `Answer Relevance`, `Answer Correctness`.
5. **Bước 5 (Phân tích Đánh đổi):** Lập biểu đồ tương quan giữa độ chính xác và độ trễ / chi phí token để đưa ra kết luận khoa học cho đồ án.
