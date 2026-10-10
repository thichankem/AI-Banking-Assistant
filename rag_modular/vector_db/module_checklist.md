# Vector Database & Indexing Ecosystem Checklist

## 1. Phân hệ Embedded Vector Databases (Zero-Infrastructure / Cục bộ / Đơn giản nhất)
- [x] **ChromaDB** (`ChromaVectorDB` - **Đã triển khai mặc định**: In-memory & Persistent Disk, HNSW Cosine Index, dễ dùng nhất)
- [x] **Qdrant Embedded** (`QdrantVectorDB` - **Đã triển khai**: Rust-based engine, chạy trên `:memory:` hoặc file đĩa, HNSW cực nhanh)
- [ ] **LanceDB** (Serverless, dựa trên Apache Arrow Columnar, hỗ trợ DiskANN lưu trữ chi phí thấp)
- [ ] **SQLite-vec** (Extension C thuần cho SQLite, tích hợp tìm kiếm vector trực tiếp vào CSDL SQLite quen thuộc)
- [ ] **FAISS (Facebook AI Similarity Search)** (Thư viện C++ chuẩn mực: Flat, IVF, HNSW, Product Quantization)

## 2. Phân hệ Dedicated Client-Server & Cloud Vector Databases (Quy mô Doanh nghiệp lớn)
- [x] **Weaviate** (`WeaviateVectorDB` - **Đã triển khai connector**: Go-based, GraphQL & REST API, Hybrid Search BM25 + Vector)
- [ ] **Milvus / Zilliz** (CSDL Vector phân tán Cloud-Native, quản lý hàng tỷ vector, hỗ trợ tăng tốc GPU)
- [ ] **Pinecone** (CSDL Vector Serverless hàng đầu trên đám mây, tự động scale, quản lý 100% qua API)
- [ ] **Qdrant Server / Cluster** (Triển khai cụm Docker / Kubernetes cho tải RPS cực lớn)
- [ ] **Vespa** (Hệ thống tìm kiếm lớn từ Yahoo, xử lý multi-stage ranking và vector quy mô petabyte)
- [ ] **Deep Lake** (Tối ưu hóa đặc biệt cho Deep Learning & Multi-modal: hình ảnh, âm thanh, video)

## 3. Phân hệ Mở rộng Vector từ CSDL Truyền thống (Relational & Search Extensions)
- [ ] **PostgreSQL + pgvector / pgvectorscale** (ACID Transaction, HNSW/IVFFlat index, StreamingDiskANN)
- [ ] **Elasticsearch / OpenSearch** (k-NN search qua Apache Lucene HNSW / Faiss engine)
- [ ] **Redis Vector Search** (In-memory, độ trễ cực thấp sub-millisecond, RedisVL)
- [ ] **MongoDB Atlas Vector Search** (CSDL Document NoSQL kết hợp Vector Search)
- [ ] **ClickHouse Vector Search** (CSDL OLAP phân tích tốc độ cao)

## 4. Các Giải thuật Lập chỉ mục Vector (Indexing Algorithms) để nghiên cứu sâu
- [x] **HNSW (Hierarchical Navigable Small World)** (Chuẩn công nghiệp, Recall > 98%, độ trễ tìm kiếm logarit)
- [ ] **IVF-PQ (Inverted File + Product Quantization)** (Lượng tử hóa vector, nén giảm 70-90% dung lượng RAM)
- [ ] **DiskANN / Vamana Graph** (Chỉ mục đồ thị đọc trực tiếp từ NVMe SSD, không bị nghẽn RAM)
- [ ] **SCaNN (Google Anisotropic Quantization)** (Tối ưu hóa phép đo góc cho bài toán MIPS)
- [ ] **Scalar Quantization (SQ8/SQ4)** (Chuyển đổi float32 sang int8/int4 để tăng tốc tính toán)

## 5. Trạng thái Deprecated & Xóa bỏ
- [~] ~~**Numpy Exact Flat Scan (Cosine / Euclidean / Dot Product)**~~ (ĐÃ XÓA BỎ: Quét mảng thô sơ đã được thay thế 100% bằng ChromaDB)
- [~] ~~**Toy Clustered IVF (K-Means Python thuần)**~~ (ĐÃ XÓA BỎ: Thay thế bằng các engine HNSW chuyên nghiệp)
