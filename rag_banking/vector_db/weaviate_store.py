"""
Weaviate Vector Database Store
Quản lý lưu trữ và truy xuất vector ngân hàng bằng Weaviate (phiên bản weaviate-client v4).
Hỗ trợ:
1. Kết nối Weaviate Server cục bộ (Docker localhost:8080) hoặc Weaviate Cloud (WCD).
2. Tự động dự phòng chế độ In-Memory/Disk Vector Store nếu máy người dùng chưa bật Weaviate server,
   giúp người dùng chạy thử nghiệm pipeline ngay lập tức mà không bị lỗi gián đoạn!
"""

import numpy as np
import weaviate
from weaviate.classes.config import Configure, Property, DataType
from weaviate.classes.query import MetadataQuery

class WeaviateStore:
    def __init__(
        self,
        collection_name="BankingDocumentChunk",
        host="localhost",
        port=8080,
        grpc_port=50051
    ):
        self.collection_name = collection_name
        self.host = host
        self.port = port
        self.grpc_port = grpc_port
        
        self.client = None
        self.is_connected = False
        
        # Local fallback cache phòng khi Weaviate server chưa được bật
        self._local_chunks = []
        self._local_vectors = None

        self._init_connection()

    def _init_connection(self):
        """Khởi tạo kết nối Weaviate v4"""
        try:
            self.client = weaviate.connect_to_local(
                host=self.host,
                port=self.port,
                grpc_port=self.grpc_port
            )
            if self.client.is_ready():
                self.is_connected = True
                print(f"[Weaviate] Đã kết nối thành công tới Weaviate Server tại {self.host}:{self.port}")
                self._ensure_collection()
            else:
                self._fallback_mode()
        except Exception as e:
            self._fallback_mode(reason=str(e))

    def _fallback_mode(self, reason=""):
        """Kích hoạt chế độ dự phòng an toàn"""
        self.is_connected = False
        self.client = None
        print(f"[Weaviate Store] Weaviate server ({self.host}:{self.port}) chưa khởi chạy.")
        print(f" -> Tự động kích hoạt 'Weaviate In-Memory Vector Store' tương thích 100% để test ngay lập tức.")
        print(f" -> Để chạy Weaviate Docker thực sự, hãy gõ: docker compose -f rag_banking/docker-compose.yml up -d")

    def _ensure_collection(self):
        """Tạo Collection (Schema) trong Weaviate nếu chưa tồn tại"""
        if not self.is_connected or not self.client:
            return
        
        try:
            if not self.client.collections.exists(self.collection_name):
                self.client.collections.create(
                    name=self.collection_name,
                    properties=[
                        Property(name="chunk_id", data_type=DataType.TEXT),
                        Property(name="parent_id", data_type=DataType.TEXT),
                        Property(name="source", data_type=DataType.TEXT),
                        Property(name="breadcrumb", data_type=DataType.TEXT),
                        Property(name="section_title", data_type=DataType.TEXT),
                        Property(name="chunk_type", data_type=DataType.TEXT),
                        Property(name="content", data_type=DataType.TEXT),
                        Property(name="original_text", data_type=DataType.TEXT),
                        Property(name="word_count", data_type=DataType.INT),
                        Property(name="token_count", data_type=DataType.INT),
                        Property(name="is_active", data_type=DataType.BOOL),
                    ],
                    vector_index_config=Configure.VectorIndex.hnsw(
                        distance_metric=Configure.VectorIndex.DistanceMetric.COSINE
                    )
                )
                print(f"[Weaviate] Đã tạo mới Collection '{self.collection_name}'.")
        except Exception as e:
            print(f"[Weaviate] Lỗi kiểm tra collection: {e}")

    def add_chunks(self, chunks, vectors):
        """
        Lưu trữ danh sách chunks cùng vector tương ứng vào Weaviate.
        """
        if not chunks or not vectors:
            return

        # Luôn lưu vào local cache dự phòng
        self._local_chunks = chunks
        self._local_vectors = np.array(vectors, dtype=np.float32)

        # Nếu có kết nối Weaviate Server thực
        if self.is_connected and self.client:
            try:
                collection = self.client.collections.get(self.collection_name)
                # Dùng batch insert_many
                data_objects = []
                for chunk, vec in zip(chunks, vectors):
                    obj = {
                        "properties": {
                            "chunk_id": chunk.get("chunk_id", ""),
                            "parent_id": chunk.get("parent_id", "") or "",
                            "source": chunk.get("source", ""),
                            "breadcrumb": chunk.get("breadcrumb", ""),
                            "section_title": chunk.get("section_title", ""),
                            "chunk_type": chunk.get("chunk_type", ""),
                            "content": chunk.get("content", ""),
                            "original_text": chunk.get("original_text", chunk.get("content", "")),
                            "word_count": int(chunk.get("word_count", 0)),
                            "token_count": int(chunk.get("token_count", 0)),
                            "is_active": bool(chunk.get("is_active", True)),
                        },
                        "vector": vec
                    }
                    data_objects.append(obj)

                # Nạp theo batch
                with collection.batch.dynamic() as batch:
                    for obj in data_objects:
                        batch.add_object(
                            properties=obj["properties"],
                            vector=obj["vector"]
                        )
                print(f"[Weaviate] Đã nạp thành công {len(chunks)} chunks vào Weaviate Server!")
            except Exception as e:
                print(f"[Weaviate] Lỗi nạp dữ liệu lên server: {e}. Đang dùng local cache.")

    def search_vector(self, query_vector, top_k=5):
        """
        Tìm kiếm vector lân cận (Dense Vector Search) qua Weaviate.
        Trả về: list các tuple (chunk, cosine_similarity_score)
        """
        # 1. Nếu có Weaviate Server thực
        if self.is_connected and self.client:
            try:
                collection = self.client.collections.get(self.collection_name)
                response = collection.query.near_vector(
                    near_vector=query_vector,
                    limit=top_k,
                    return_metadata=MetadataQuery(distance=True)
                )
                results = []
                for obj in response.objects:
                    # Chuyển đổi Weaviate cosine distance sang similarity: sim = 1 - distance
                    dist = obj.metadata.distance if obj.metadata and obj.metadata.distance is not None else 1.0
                    similarity = 1.0 - dist
                    chunk_dict = dict(obj.properties)
                    results.append((chunk_dict, float(similarity)))
                return results
            except Exception as e:
                print(f"[Weaviate] Lỗi truy vấn server: {e}. Chuyển sang local vector scan.")

        # 2. Local Fallback Vector Search (Cosine Similarity tính toán qua NumPy)
        if self._local_vectors is None or len(self._local_chunks) == 0:
            return []

        q_vec = np.array(query_vector, dtype=np.float32)
        q_norm = np.linalg.norm(q_vec)
        if q_norm == 0:
            return []

        # Chuẩn hóa vector và tính cosine similarity
        doc_norms = np.linalg.norm(self._local_vectors, axis=1)
        doc_norms[doc_norms == 0] = 1e-10
        cosine_sims = np.dot(self._local_vectors, q_vec) / (doc_norms * q_norm)

        # Lấy top_k chỉ số cao nhất
        top_indices = np.argsort(cosine_sims)[::-1][:top_k]
        results = []
        for idx in top_indices:
            results.append((self._local_chunks[idx], float(cosine_sims[idx])))

        return results

    def close(self):
        """Đóng kết nối client an toàn"""
        if self.client:
            try:
                self.client.close()
            except Exception:
                pass
            self.client = None
