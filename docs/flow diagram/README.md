# Thư Mục Sơ Đồ & Luồng Nghiệp Vụ (Flow Diagrams)

Thư mục này là **nơi lưu trữ tập trung bắt buộc** cho toàn bộ các sơ đồ luồng (flow diagrams), kiến trúc hệ thống, sequence diagrams, quy trình nghiệp vụ và các bản vẽ liên quan của dự án **AI Banking Assistant**.

---

## 1. Quy định Lưu trữ
* **Định dạng file nguồn**: `.drawio` (hoặc `.xml`).
* **Định dạng kết xuất (Preview/Export)**: `.png` hoặc `.svg`.
* **Vị trí**: Toàn bộ file sơ đồ và hình ảnh render tương ứng bắt buộc nằm trong thư mục này:
  `docs/flow diagram/`

## 2. Công cụ Sử dụng
Hệ thống sử dụng kỹ năng `drawio-diagram-architect` kèm công cụ CLI `drawio-tool`:
* Khởi tạo file mới: `drawio-tool new "docs/flow diagram/<ten_luong>.drawio" --title "<Tên Sơ Đồ>"`
* Thêm node / cập nhật: `drawio-tool add-node "docs/flow diagram/<ten_luong>.drawio" ...`
* Kết xuất ảnh preview: `drawio-tool export "docs/flow diagram/<ten_luong>.drawio" --page 1 --output "docs/flow diagram/<ten_luong>.png"`

## 3. Danh mục Luồng Dự Án
* **Hệ Thống RAG Banking Toàn Trình (Vietcombank Assistant)** (`docs/flow diagram/rag_banking_pipeline_flow.drawio` & `docs/flow diagram/rag_banking_pipeline_flow.png`):
  * **Phản ánh chính xác 100% mã nguồn thực tế của thư mục `rag_banking`**:
  * **4 Giai đoạn cốt lõi**:
    1. *Giai đoạn 1 (Multi-Format Document Layout-Aware Chunking)*: Hỗ trợ 4 định dạng thực tế (.md, .xlsx, Digital .pdf, Scanned .pdf qua RapidOCR CPU 100% offline). Trích xuất Header Breadcrumbs, làm phẳng bảng Excel/Markdown, phủ 100% breadcrumbs trên 1.789 chunks không có chunk rỗng.
    2. *Giai đoạn 2 (Dual-Indexing Pipeline)*: Lập chỉ mục song song Dense Vector HNSW 384d (ONNX all-MiniLM-L6-v2, batch size 128 đạt throughput 38.5 chunks/s) trên Weaviate v4 (kèm Auto-Fallback In-Memory) và Sparse Lexical Okapi BM25 ($k_1=1.5, b=0.75$).
    3. *Giai đoạn 3 (Online Query & Hybrid Retrieval RRF)*: Phân phối truy vấn song song, hợp nhất thứ hạng Reciprocal Rank Fusion ($k=60, w_{\text{dense}}=0.4, w_{\text{bm25}}=0.6$, candidate $K=15$) lọc ra Top-5 chunks tinh tuyển trong ~215.6 ms.
    4. *Giai đoạn 4 (LLM Serving, Observability & Benchmark Auto-Tuning)*: RAG Prompt đóng gói 5 nguyên tắc ngân hàng, gọi DeepSeek-V4-Flash qua FPT Cloud API (kèm offline fallback), Bird Tracer 🦜 (@traceable LangSmith) đo đạc vi mô từng span và lưu `bird_traces.jsonl`, bảng Auto-Tuning cải thiện Hit Rate@1 từ 65.0% lên 80.0% (+15%).
* **Modular RAG Pipeline v1.0 (Bản Tối Giản)** (`docs/flow diagram/rag pipeline/v1/rag_modular_pipeline_v1.drawio` & `v1/rag_modular_pipeline_v1.png`):
  * **Sơ đồ luồng phong cách tối giản chuẩn kỹ thuật**: Đen trắng, viền sắc nét, nhãn ngắn gọn trên mũi tên (documents, split, 1024d, terms, query, parsed, top 20, top 5, top 3, prompt, answer).
  * **Toàn bộ chuỗi 5 phân hệ tuần tự**: BaseChunker ➔ Dual Indexing (Dense 1024d & BM25) ➔ Query Parser ➔ Hybrid Retriever (RRF k=60) ➔ Cross-match Reranker ➔ Context Synthesizer ➔ Banking LLM ➔ RAGResponse.
* **Modular RAG Pipeline v2.0** (`docs/flow diagram/rag pipeline/rag_modular_pipeline_flow.drawio` & `v2/rag_modular_pipeline_v2.drawio`):
  * **Thay thế hoàn toàn bản v1.0**: Phản ánh chính xác 100% mã nguồn thực tế của thư viện `rag_modular`.
  * **3 Giai đoạn cốt lõi**:
    1. *Giai đoạn 1 (Ingestion & Dual-Indexing)*: 6 chiến lược Chunking (`FixedWindow`, `SentenceRegex`, `FAQPair`, `HierarchyMetadata`, `ParagraphSplit`, `SemanticSliding`), Dense Embedding nơ-ron 1024d (`CloudEmbedding`, `Vietnamese_Embedding`, `Multilingual-E5`), Vector Databases (`ChromaVectorDB`, `QdrantVectorDB`, `WeaviateVectorDB`), và Sparse BM25 (`BM25Okapi`, `BM25Plus`, `TFIDF`).
    2. *Giai đoạn 2 (Online Query & Multi-Stage Refinement)*: 5 phân hệ tuần tự gồm `BaseQueryParser` (chuẩn hóa từ viết tắt ngân hàng, trích từ khóa, HyDE, Multi-Query), `BaseRetriever` (truy xuất song song & Reciprocal Rank Fusion RRF $k=60$), `BaseReranker` (FAQ Golden Question Cross Match $60/40$), `BaseContextSynthesizer` (Markdown ngân hàng & Token Budget Truncation), và `BaseLLM` (`CloudLLM` DeepSeek-V3/R1 kèm Auto-Fallback sang `MockBankLLM`).
    3. *Giai đoạn 3 (Output & Profiling)*: `RAGResponse` (Answer + Citations), `ModularRAGPipeline` Profiler (`rag_modular.rag_pipeline`, phân hệ điều phối đổi tên từ `core`), đo đạc `latency_breakdown_ms` vi mô từng module, và Ma trận 4 Pipelines đối chuẩn Benchmark (Hit@3 đạt 98.5% | MRR 0.96).
* `langgraph_agent_orchestration_flow.drawio`: Luồng điều phối StateGraph của Multi-Agent Banking Assistant.
* `fraud_risk_mitigation_flow.drawio`: Luồng kiểm soát an toàn giao dịch, phát hiện lừa đảo & xác thực OTP/Sinh trắc học.
* `late_chunking_document_flow.drawio`: Luồng xử lý và phân đoạn tài liệu ngân hàng chuẩn hóa.
