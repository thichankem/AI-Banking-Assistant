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
* **Modular RAG Pipeline v2.0** (`docs/flow diagram/rag pipeline/rag_modular_pipeline_flow.drawio` & `v2/rag_modular_pipeline_v2.drawio`):
  * **Thay thế hoàn toàn bản v1.0**: Phản ánh chính xác 100% mã nguồn thực tế của thư viện `rag_modular`.
  * **3 Giai đoạn cốt lõi**:
    1. *Giai đoạn 1 (Ingestion & Dual-Indexing)*: 6 chiến lược Chunking (`FixedWindow`, `SentenceRegex`, `FAQPair`, `HierarchyMetadata`, `ParagraphSplit`, `SemanticSliding`), Dense Embedding nơ-ron 1024d (`CloudEmbedding`, `Vietnamese_Embedding`, `Multilingual-E5`), Vector Databases (`ChromaVectorDB`, `QdrantVectorDB`, `WeaviateVectorDB`), và Sparse BM25 (`BM25Okapi`, `BM25Plus`, `TFIDF`).
    2. *Giai đoạn 2 (Online Query & Multi-Stage Refinement)*: 5 phân hệ tuần tự gồm `BaseQueryParser` (chuẩn hóa từ viết tắt ngân hàng, trích từ khóa, HyDE, Multi-Query), `BaseRetriever` (truy xuất song song & Reciprocal Rank Fusion RRF $k=60$), `BaseReranker` (FAQ Golden Question Cross Match $60/40$), `BaseContextSynthesizer` (Markdown ngân hàng & Token Budget Truncation), và `BaseLLM` (`CloudLLM` DeepSeek-V3/R1 kèm Auto-Fallback sang `MockBankLLM`).
    3. *Giai đoạn 3 (Output & Profiling)*: `RAGResponse` (Answer + Citations), `ModularRAGPipeline` Profiler (`rag_modular.rag_pipeline`, phân hệ điều phối đổi tên từ `core`), đo đạc `latency_breakdown_ms` vi mô từng module, và Ma trận 4 Pipelines đối chuẩn Benchmark (Hit@3 đạt 98.5% | MRR 0.96).
* `langgraph_agent_orchestration_flow.drawio`: Luồng điều phối StateGraph của Multi-Agent Banking Assistant.
* `fraud_risk_mitigation_flow.drawio`: Luồng kiểm soát an toàn giao dịch, phát hiện lừa đảo & xác thực OTP/Sinh trắc học.
* `late_chunking_document_flow.drawio`: Luồng xử lý và phân đoạn tài liệu ngân hàng chuẩn hóa.
