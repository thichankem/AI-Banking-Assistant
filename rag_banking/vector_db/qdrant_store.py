"""
Module Qdrant Vector Database & Hybrid Store (Localhost - Zero Docker)
======================================================================
Chuyên trách:
1. Lưu trữ Dense Vector 1024 chiều (Cosine Distance HNSW) vào đĩa cứng cục bộ.
2. Lọc Metadata cực mạnh (Filterable HNSW: theo source, breadcrumb, section_title, chunk_type).
3. Tích hợp sẵn BM25 Okapi Keyword Search (k1=1.5, b=0.75).
4. Hybrid Search tự động: Hợp nhất Dense + BM25 qua giải thuật RRF (Reciprocal Rank Fusion).

NGUYÊN TẮC THIẾT KẾ:
1. 0% REGEX LAW: Tuyệt đối không dùng regex methods.
2. ZERO TERNARY: Ưu tiên if/else tường minh, dễ đọc.
3. NO __INIT__.PY: Import trực tiếp file module.
"""

import math
import uuid
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    HnswConfigDiff,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue
)


class QdrantBankingStore:
    """
    Cơ sở dữ liệu Vector & Tìm kiếm Lai (Hybrid Search) chạy trực tiếp trên Localhost.
    """

    def __init__(
        self,
        collection_name="banking_docs_1024",
        dimension=1024,
        storage_path="rag_banking/vector_db/qdrant_db",
        use_memory=False

    ):
        self.collection_name = collection_name
        self.dimension = dimension
        self.storage_path = storage_path
        self.use_memory = use_memory

        # 1. Khởi tạo Qdrant Client (chạy file cục bộ hoặc memory)
        if use_memory:
            self.client = QdrantClient(":memory:")
        else:
            Path(storage_path).mkdir(parents=True, exist_ok=True)
            self.client = QdrantClient(path=storage_path)

        # 2. Khởi tạo Collection HNSW Cosine
        self._init_collection()

        # 3. Bộ nhớ đệm BM25 Okapi cục bộ
        self._bm25_corpus = []
        self._bm25_tokenized = []
        self._bm25_breadcrumb_tokens = []
        self._bm25_doc_len = []
        self._bm25_avgdl = 0.0
        self._bm25_idf = {}
        self.k1 = 1.5
        self.b = 0.75

        # Tự động nạp lại BM25 từ đĩa nếu collection đã có dữ liệu trước đó
        if not use_memory:
            self._load_bm25_from_existing_store()

    def _init_collection(self):
        """Tạo Collection với khoảng cách Cosine và chỉ mục HNSW nếu chưa tồn tại"""
        existing_collections = [c.name for c in self.client.get_collections().collections]
        if self.collection_name not in existing_collections:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.dimension,
                    distance=Distance.COSINE
                ),
                hnsw_config=HnswConfigDiff(
                    m=16,
                    ef_construct=100
                )
            )

    def _load_bm25_from_existing_store(self):
        """Khôi phục lại chỉ mục BM25 từ các payload đã lưu trong Qdrant trên đĩa"""
        try:
            pts_count = self.count()
            if pts_count > 0:
                records, _ = self.client.scroll(
                    collection_name=self.collection_name,
                    limit=max(pts_count + 100, 10000),
                    with_payload=True,
                    with_vectors=False
                )
                chunks = [r.payload for r in records if r.payload]
                if chunks:
                    self._bm25_corpus = list(chunks)
                    self._build_bm25_index(self._bm25_corpus)
                    print(f"📦 [Qdrant Store] Đã tự động khôi phục BM25 cho {len(chunks)} chunks từ kho lưu trữ đĩa!")
        except Exception as e:
            pass

    # -------------------------------------------------------------------------
    # 1. NẠP DỮ LIỆU (INGESTION)
    # -------------------------------------------------------------------------
    def add_chunks(self, chunks, batch_size=64):
        """
        Nạp danh sách chunks đã có trường 'vector' vào Qdrant và lập chỉ mục BM25.
        """
        if not chunks:
            return

        points = []
        for idx, chunk in enumerate(chunks):
            vec = chunk.get("vector")
            if not vec:
                continue

            pt_id = chunk.get("chunk_id") or str(idx)
            try:
                point_uuid = str(uuid.UUID(pt_id))
            except Exception:
                point_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, str(pt_id)))

            # Đóng gói Payload sạch sẽ
            payload = {
                "chunk_id": chunk.get("chunk_id", str(idx)),
                "parent_id": chunk.get("parent_id", ""),
                "source": chunk.get("source", ""),
                "breadcrumb": chunk.get("breadcrumb", ""),
                "section_title": chunk.get("section_title", ""),
                "chunk_type": chunk.get("chunk_type", "docx_text"),
                "content": chunk.get("content", ""),
                "original_text": chunk.get("original_text", chunk.get("content", "")),
                "word_count": chunk.get("word_count", 0),
                "token_count": chunk.get("token_count", 0),
                "is_active": chunk.get("is_active", True)
            }
            if "images" in chunk and chunk["images"]:
                payload["images"] = chunk["images"]

            points.append(
                PointStruct(
                    id=point_uuid,
                    vector=[float(x) for x in vec[:self.dimension]],
                    payload=payload
                )
            )

        # Upsert theo từng batch vào Qdrant
        for i in range(0, len(points), batch_size):
            self.client.upsert(
                collection_name=self.collection_name,
                points=points[i : i + batch_size]
            )

        # Lập chỉ mục BM25 đồng thời (tích lũy không ghi đè)
        existing_ids = {c.get("chunk_id") for c in self._bm25_corpus if c.get("chunk_id")}
        if existing_ids:
            new_items = [c for c in chunks if c.get("chunk_id") not in existing_ids]
        else:
            new_items = list(chunks)
        self._bm25_corpus.extend(new_items)
        self._build_bm25_index(self._bm25_corpus)
        print(f"✅ [Qdrant Store] Đã nạp thành công {len(points)} chunks vào Vector DB (Tổng BM25: {len(self._bm25_corpus)} chunks)!")

    # -------------------------------------------------------------------------
    # 2. XÂY DỰNG CHỈ MỤC BM25 OKAPI
    # -------------------------------------------------------------------------
    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Tách từ vựng chữ và số thuần túy (0% Regex Law)."""
        if not text:
            return []
        cleaned_chars = []
        for ch in text.lower():
            if ch.isalnum() or ch == "_":
                cleaned_chars.append(ch)
            else:
                cleaned_chars.append(" ")
        return "".join(cleaned_chars).split()

    def _build_bm25_index(self, chunks):
        """Xây dựng bảng tần suất từ, tập hợp tokens Breadcrumb và IDF cho BM25"""
        self._bm25_corpus = list(chunks)
        self._bm25_tokenized = []
        self._bm25_breadcrumb_tokens = []
        self._bm25_doc_len = []

        word_doc_freq = {}
        for chunk in chunks:
            text = (chunk.get("content") or chunk.get("original_text") or "").strip()
            tokens = self._tokenize(text)
            self._bm25_tokenized.append(tokens)
            doc_len = len(tokens)
            self._bm25_doc_len.append(doc_len)

            # Lập chỉ mục tokens cho Breadcrumb & Section Title (phục vụ BM25F Field-weighting)
            bc_str = f"{chunk.get('breadcrumb', '')} {chunk.get('section_title', '')}"
            bc_tokens = set(self._tokenize(bc_str))
            self._bm25_breadcrumb_tokens.append(bc_tokens)

            for w in set(tokens):
                word_doc_freq[w] = word_doc_freq.get(w, 0) + 1

        total_docs = len(chunks)
        if total_docs > 0:
            self._bm25_avgdl = sum(self._bm25_doc_len) / total_docs
        else:
            self._bm25_avgdl = 0.0

        # Tính IDF chuẩn Okapi
        self._bm25_idf = {}
        for word, freq in word_doc_freq.items():
            self._bm25_idf[word] = math.log(((total_docs - freq + 0.5) / (freq + 0.5)) + 1.0)

    # -------------------------------------------------------------------------
    # 3. CƠ CHẾ LỌC METADATA (PAYLOAD FILTER)
    # -------------------------------------------------------------------------
    @staticmethod
    def _build_qdrant_filter(filter_dict=None):
        """Chuyển đổi dictionary điều kiện sang Qdrant Filter"""
        if not filter_dict:
            return None

        conditions = []
        for key, val in filter_dict.items():
            if val is not None:
                conditions.append(FieldCondition(key=key, match=MatchValue(value=val)))

        if conditions:
            return Filter(must=conditions)
        return None

    # -------------------------------------------------------------------------
    # 4. TÌM KIẾM VECTOR (DENSE SEARCH)
    # -------------------------------------------------------------------------
    def search_vector(self, query_vector, top_k=5, filter_dict=None):
        """
        Tìm kiếm Top-K chunks theo Cosine Similarity trên Qdrant HNSW Index kèm bộ lọc Metadata.
        """
        qdrant_filter = self._build_qdrant_filter(filter_dict)

        try:
            res = self.client.query_points(
                collection_name=self.collection_name,
                query=[float(x) for x in query_vector[:self.dimension]],
                query_filter=qdrant_filter,
                limit=top_k
            )
            results = []
            for hit in res.points:
                chunk = dict(hit.payload)
                score = float(hit.score)  # Cosine Similarity [-1.0, 1.0]
                results.append((chunk, score))
            return results
        except Exception as e:
            print(f"⚠️ [Qdrant Store] Lỗi search_vector: {e}")
            return []

    # -------------------------------------------------------------------------
    # 5. TÌM KIẾM TỪ KHÓA (BM25 OKAPI SEARCH CÓ FIELD-WEIGHTED BREADCRUMB BOOST)
    # -------------------------------------------------------------------------
    def search_bm25(self, query_text, top_k=5, filter_dict=None, breadcrumb_boost=1.5):
        """
        Tìm kiếm từ khóa chính xác qua thuật toán Okapi BM25 kèm lọc Metadata
        và đánh giá điểm cao hơn cho Breadcrumb (BM25F / Field-weighting).
        """
        if not self._bm25_corpus or not query_text:
            return []

        q_tokens = self._tokenize(query_text)
        if not q_tokens:
            return []

        scored_items = []
        for i, doc_tokens in enumerate(self._bm25_tokenized):
            chunk = self._bm25_corpus[i]

            # Kiểm tra bộ lọc metadata nếu có
            if filter_dict:
                match_all = True
                for k, v in filter_dict.items():
                    if chunk.get(k) != v:
                        match_all = False
                        break
                if not match_all:
                    continue

            doc_len = self._bm25_doc_len[i]
            if doc_len == 0:
                continue

            # Đếm tần suất từ
            tf = {}
            for t in doc_tokens:
                tf[t] = tf.get(t, 0) + 1

            bc_set = set()
            if i < len(self._bm25_breadcrumb_tokens):
                bc_set = self._bm25_breadcrumb_tokens[i]

            score = 0.0
            for term in q_tokens:
                term_matched = False
                if term in tf:
                    freq = tf[term]
                    idf = self._bm25_idf.get(term, 0.0)
                    denom = freq + self.k1 * (1.0 - self.b + self.b * (doc_len / self._bm25_avgdl))
                    term_score = idf * (freq * (self.k1 + 1.0)) / denom

                    # Đánh giá điểm cao hơn cho Breadcrumb nếu term xuất hiện trong Breadcrumb
                    if term in bc_set:
                        score += term_score * breadcrumb_boost
                    else:
                        score += term_score
                    term_matched = True

                # Nếu từ khóa nằm trực tiếp trong breadcrumb nhưng thân chunk không lặp lại:
                if not term_matched and term in bc_set:
                    idf = self._bm25_idf.get(term, 1.0)
                    score += idf * (breadcrumb_boost * 0.5)

            if score > 0.0:
                scored_items.append((chunk, float(score)))

        # Sắp xếp điểm BM25 giảm dần
        scored_items.sort(key=lambda x: x[1], reverse=True)
        return scored_items[:top_k]

    # -------------------------------------------------------------------------
    # 6. TÌM KIẾM LAI (HYBRID SEARCH: DENSE + BM25 + RRF)
    # -------------------------------------------------------------------------
    def search_hybrid(
        self,
        query_text,
        query_vector,
        top_k=5,
        candidate_k=15,
        alpha=0.5,
        rrf_k=60,
        filter_dict=None,
        breadcrumb_boost=1.5
    ):
        """
        Tìm kiếm lai đồng thời:
        1. Dense Vector (hiểu ngữ nghĩa sâu) qua Qdrant HNSW.
        2. Sparse BM25 (bắt chính xác mã thẻ, số thông tư, lãi suất) kèm ưu tiên Breadcrumb.
        3. Hợp nhất thứ hạng bằng Reciprocal Rank Fusion (RRF).
        """
        # 1. Gọi song song
        dense_hits = self.search_vector(query_vector, top_k=candidate_k, filter_dict=filter_dict)
        bm25_hits = self.search_bm25(
            query_text,
            top_k=candidate_k,
            filter_dict=filter_dict,
            breadcrumb_boost=breadcrumb_boost
        )

        # 2. Hợp nhất RRF
        candidates = {}
        dense_weight = alpha
        bm25_weight = 1.0 - alpha

        # Ghi nhận thứ hạng Dense
        for rank, (chunk, score) in enumerate(dense_hits, start=1):
            c_id = chunk.get("chunk_id", str(id(chunk)))
            candidates[c_id] = {
                "chunk": chunk,
                "dense_rank": rank,
                "dense_score": round(score, 4),
                "bm25_rank": None,
                "bm25_score": 0.0,
                "rrf_score": 0.0
            }

        # Ghi nhận thứ hạng BM25
        for rank, (chunk, score) in enumerate(bm25_hits, start=1):
            c_id = chunk.get("chunk_id", str(id(chunk)))
            if c_id not in candidates:
                candidates[c_id] = {
                    "chunk": chunk,
                    "dense_rank": None,
                    "dense_score": 0.0,
                    "bm25_rank": rank,
                    "bm25_score": round(score, 4),
                    "rrf_score": 0.0
                }
            else:
                candidates[c_id]["bm25_rank"] = rank
                candidates[c_id]["bm25_score"] = round(score, 4)

        # Tính toán điểm RRF: Score = w1 / (k + rank1) + w2 / (k + rank2)
        for c_id, item in candidates.items():
            rrf = 0.0
            if item["dense_rank"] is not None:
                rrf += dense_weight / (rrf_k + item["dense_rank"])
            if item["bm25_rank"] is not None:
                rrf += bm25_weight / (rrf_k + item["bm25_rank"])
            item["rrf_score"] = round(rrf, 6)

        # Sắp xếp kết quả theo RRF giảm dần và lấy Top-K
        sorted_candidates = sorted(candidates.values(), key=lambda x: x["rrf_score"], reverse=True)
        return sorted_candidates[:top_k]

    def count(self):
        """Đếm tổng số bản ghi vector hiện có"""
        try:
            return self.client.get_collection(self.collection_name).points_count
        except Exception:
            return 0

    def close(self):
        """Giải phóng tài nguyên"""
        if self.client:
            try:
                self.client.close()
            except Exception:
                pass


# Alias chuẩn hóa
QdrantVectorStore = QdrantBankingStore
