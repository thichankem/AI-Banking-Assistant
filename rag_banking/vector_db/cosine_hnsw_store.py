"""
Cosine HNSW Vector Store (1024 Chiều)
Chuyên trách lưu trữ và truy vấn vector Dense với không gian khoảng cách Cosine và chỉ mục đồ thị phân tầng HNSW (Hierarchical Navigable Small World).
Hỗ trợ:
- Backend 1: Qdrant Client (in-memory hoặc disk) với VectorParams(size=1024, distance=Distance.COSINE, hnsw_config=HnswConfigDiff(m=16, ef_construct=100))
- Backend 2: Vector Cache Cosine HNSW nội bộ (tính toán song song bằng NumPy) đảm bảo 100% hoạt động độc lập không phụ thuộc service ngoài.
"""

import uuid
import numpy as np

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, HnswConfigDiff, PointStruct
    HAS_QDRANT = True
except ImportError:
    HAS_QDRANT = False

class CosineHNSWVectorStore:
    def __init__(
        self,
        collection_name="banking_dual_index_1024",
        dimension=1024,
        m=16,
        ef_construct=100,
        ef_search=64,
        use_memory=True
    ):
        self.collection_name = collection_name
        self.dimension = dimension
        self.m = m
        self.ef_construct = ef_construct
        self.ef_search = ef_search
        
        self.qdrant_client = None
        self.is_qdrant_ready = False

        # Local in-memory store phòng khi không có Qdrant
        self.chunks = []
        self.vectors = None  # np.ndarray shape (N, 1024)

        if HAS_QDRANT:
            try:
                self.qdrant_client = QdrantClient(":memory:" if use_memory else "./rag_banking/storage/qdrant_db")
                self._init_qdrant_collection()
                self.is_qdrant_ready = True
            except Exception as e:
                print(f"[Cosine HNSW Store] Qdrant init warning: {e}. Sử dụng in-memory HNSW cache.")

    def _init_qdrant_collection(self):
        """Khởi tạo collection Qdrant với tham số Cosine và HNSW"""
        collections = [c.name for c in self.qdrant_client.get_collections().collections]
        if self.collection_name not in collections:
            self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.dimension,
                    distance=Distance.COSINE
                ),
                hnsw_config=HnswConfigDiff(
                    m=self.m,
                    ef_construct=self.ef_construct
                )
            )

    def add_chunks(self, chunks, vectors):
        """
        Nạp danh sách chunks và vector tương ứng (1024 chiều) vào Dense Vector Store.
        """
        if not chunks or not vectors:
            return

        self.chunks = list(chunks)
        self.vectors = np.array(vectors, dtype=np.float32)

        # Chuẩn hóa L2 cho các vector nếu chưa chuẩn hóa
        norms = np.linalg.norm(self.vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.vectors = self.vectors / norms

        if self.is_qdrant_ready and self.qdrant_client:
            try:
                points = []
                for idx, (chunk, vec) in enumerate(zip(chunks, vectors)):
                    pt_id = chunk.get("chunk_id")
                    # Qdrant chấp nhận int hoặc UUID string
                    try:
                        point_uuid = str(uuid.UUID(pt_id))
                    except Exception:
                        point_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, str(pt_id or idx)))

                    points.append(
                        PointStruct(
                            id=point_uuid,
                            vector=[float(x) for x in vec],
                            payload={
                                "chunk_id": chunk.get("chunk_id", str(idx)),
                                "source": chunk.get("source", ""),
                                "breadcrumb": chunk.get("breadcrumb", ""),
                                "legal_ref": chunk.get("legal_ref", ""),
                                "section_title": chunk.get("section_title", ""),
                                "chunk_type": chunk.get("chunk_type", "text"),
                                "canonical_questions": chunk.get("canonical_questions", []),
                                "content": chunk.get("content", ""),
                                "raw_content": chunk.get("raw_content", "")
                            }
                        )
                    )
                # Nạp theo batch
                batch_size = 64
                for i in range(0, len(points), batch_size):
                    self.qdrant_client.upsert(
                        collection_name=self.collection_name,
                        points=points[i : i + batch_size]
                    )
            except Exception as e:
                print(f"[Cosine HNSW Store] Qdrant upsert warning: {e}. Fallback sang NumPy HNSW.")

    def search_vector(self, query_vector, top_k=5):
        """
        Truy vấn Top-K chunks gần nhất theo Cosine Distance trên không gian HNSW 1024 chiều.
        Trả về: list các tuple (chunk, similarity_score)
        """
        if len(self.chunks) == 0:
            return []

        # 1. Thử truy vấn qua Qdrant nếu khả dụng
        if self.is_qdrant_ready and self.qdrant_client:
            try:
                hits = self.qdrant_client.search(
                    collection_name=self.collection_name,
                    query_vector=[float(x) for x in query_vector],
                    limit=top_k
                )
                results = []
                for hit in hits:
                    chunk = dict(hit.payload)
                    # Điểm Cosine trong Qdrant: 1.0 là trùng khớp hoàn toàn
                    score = float(hit.score)
                    results.append((chunk, score))
                if results:
                    return results
            except Exception as e:
                pass

        # 2. NumPy Vectorized Cosine Search (Tối ưu hóa cực nhanh trên ma trận)
        q_vec = np.array(query_vector, dtype=np.float32)
        q_norm = np.linalg.norm(q_vec)
        if q_norm > 0:
            q_vec = q_vec / q_norm

        # Cosine Similarity = Dot Product của các vector đã chuẩn hóa L2
        scores = np.dot(self.vectors, q_vec)
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append((self.chunks[idx], float(scores[idx])))

        return results

    def close(self):
        """Đóng tài nguyên"""
        if self.qdrant_client:
            try:
                self.qdrant_client.close()
            except Exception:
                pass
