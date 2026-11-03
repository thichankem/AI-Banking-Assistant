"""
Module Qdrant Vector Database & Hybrid Store (Localhost - Zero Docker)
======================================================================
Chuyên trách:
1. Lưu trữ Dense Vector 1024 chiều (Cosine Distance HNSW) vào đĩa cứng cục bộ.
2. Lọc Metadata cực mạnh (Filterable HNSW: theo source, chunk_type, parent_id, business_metadata.domain, business_metadata.category).
3. Tích hợp sẵn BM25F Okapi Keyword Search đa trường (content, section_title, breadcrumb, table_data, images) kèm Inverted Index.
4. Hybrid Search tự động: Hợp nhất Dense + BM25 qua giải thuật RRF (Reciprocal Rank Fusion k=60).

NGUYÊN TẮC THIẾT KẾ:
1. 0% REGEX LAW: Tuyệt đối không sử dụng regex methods hay thư viện biểu thức chính quy.
2. ZERO TERNARY: Bắt buộc dùng if/else tường minh từng dòng, không dùng toán tử 3 ngôi.
3. NO TYPING LIBRARY: Tuyệt đối không sử dụng kiểu từ thư viện bổ trợ, thuần túy dùng kiểu gốc.
4. NO __INIT__.PY: Import trực tiếp file module.
"""

import gc
import math
import shutil
import uuid
import warnings
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    HnswConfigDiff,
    PointStruct,
    Filter,
    FilterSelector,
    FieldCondition,
    MatchValue,
    MatchAny,
    PayloadSchemaType
)


class QdrantBankingStore:
    """
    Cơ sở dữ liệu Vector & Tìm kiếm Lai (Hybrid Search) chạy trực tiếp trên Localhost.
    """

    def __init__(
        self,
        collection_name="banking_docs_1024",
        dimension=1024,
        storage_path="agent_banking/vector_db/qdrant_db",
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
            try:
                self.client = QdrantClient(path=storage_path)
            except Exception as e:
                print(f"⚠️ [Qdrant Storage Lock] Không thể mở độc quyền '{storage_path}' -> Chuyển sang In-Memory mode an toàn!")
                self.client = QdrantClient(":memory:")
                self.use_memory = True

        # 2. Khởi tạo Collection HNSW Cosine và Payload Indexes
        self._init_collection()

        # 3. Bộ nhớ đệm BM25F Inverted Index cục bộ
        self._bm25_corpus = []
        self._bm25_inverted_index = {}
        self._bm25_doc_len = []
        self._bm25_doc_norm = []
        self._bm25_avgdl = 0.0
        self._bm25_idf = {}
        self.k1 = 1.5
        self.b = 0.75

        # Tự động nạp lại BM25 từ đĩa nếu collection đã có dữ liệu trước đó
        if not use_memory:
            self._load_bm25_from_existing_store()

    def _init_collection(self):
        """Tạo Collection với khoảng cách Cosine và chỉ mục HNSW nếu chưa tồn tại, kèm payload indexes"""
        existing_collections = []
        for c in self.client.get_collections().collections:
            existing_collections.append(c.name)

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
            self._create_payload_indexes()
        else:
            self._create_payload_indexes()

    def recreate_collection(self):
        """
        Xóa sạch collection hiện tại (Clean Rebuild) và tái khởi tạo với HNSW 1024 chiều,
        tạo lại 5 keyword payload indexes, đồng thời xóa sạch bộ đệm BM25 trong bộ nhớ.
        Bảo đảm xóa sạch 100% vector trên cả bộ nhớ RAM và tệp lưu trữ đĩa (storage.sqlite).
        """
        try:
            existing_collections = []
            for c in self.client.get_collections().collections:
                existing_collections.append(c.name)

            if self.collection_name in existing_collections:
                try:
                    self.client.delete(
                        collection_name=self.collection_name,
                        points_selector=FilterSelector(filter=Filter())
                    )
                except Exception as e_del:
                    print(f"⚠️ [Qdrant Store] Cảnh báo khi xóa sạch điểm: {e_del}")

                self.client.delete_collection(self.collection_name)
        except Exception as e:
            print(f"⚠️ [Qdrant Store] Lỗi khi xóa collection '{self.collection_name}': {e}")

        # Dọn dẹp tệp vật lý phòng thủ đa tầng (Defense-in-depth trên đĩa)
        if not self.use_memory:
            try:
                gc.collect()
                col_dir = Path(self.storage_path) / "collection" / self.collection_name
                if col_dir.exists():
                    shutil.rmtree(col_dir, ignore_errors=True)
            except Exception:
                pass

        # Khởi tạo lại collection mới
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

        # Tạo lại 5 payload indexes
        self._create_payload_indexes()

        # Xóa sạch bộ đệm BM25
        self._bm25_corpus = []
        self._bm25_inverted_index = {}
        self._bm25_doc_len = []
        self._bm25_doc_norm = []
        self._bm25_avgdl = 0.0
        self._bm25_idf = {}

        return True

    def _create_payload_indexes(self):
        """Khởi tạo 5 chỉ mục Payload (Keyword index) phục vụ truy vấn có điều kiện cực nhanh."""
        fields = [
            "source",
            "chunk_type",
            "parent_id",
            "business_metadata.domain",
            "business_metadata.category"
        ]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            for field in fields:
                try:
                    self.client.create_payload_index(
                        collection_name=self.collection_name,
                        field_name=field,
                        field_schema=PayloadSchemaType.KEYWORD
                    )
                except Exception as e:
                    pass

    def _load_bm25_from_existing_store(self):
        """Khôi phục lại chỉ mục BM25 từ các payload đã lưu trong Qdrant trên đĩa"""
        try:
            pts_count = self.count()
            if pts_count > 0:
                limit_val = pts_count + 100
                if limit_val < 10000:
                    limit_val = 10000
                records, _ = self.client.scroll(
                    collection_name=self.collection_name,
                    limit=limit_val,
                    with_payload=True,
                    with_vectors=False
                )
                chunks = []
                for r in records:
                    if r.payload:
                        chunks.append(r.payload)
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
        Nạp danh sách chunks đã có trường 'vector' vào Qdrant và lập chỉ mục BM25F.
        Đảm bảo 100% chunks tuân thủ Target Universal Schema 13 trường.
        """
        if not chunks:
            return

        points = []
        new_payloads = []
        for idx, chunk in enumerate(chunks):
            vec = chunk.get("vector")
            if not vec:
                continue

            pt_id = chunk.get("chunk_id")
            if not pt_id:
                pt_id = str(idx)

            try:
                point_uuid = str(uuid.UUID(str(pt_id)))
            except Exception:
                point_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, str(pt_id)))

            # Chuẩn hóa chunk_type theo Target Universal Schema
            raw_type = chunk.get("chunk_type")
            if raw_type in ("text", "table", "multimodal"):
                chunk_type = raw_type
            else:
                chunk_type = "text"

            # Trích xuất metadata an toàn
            b_meta = chunk.get("business_metadata")
            if not isinstance(b_meta, dict):
                b_meta = {}

            imgs = chunk.get("images")
            if not isinstance(imgs, list):
                imgs = []

            prev_c = chunk.get("prev_chunk_id")
            if prev_c is None:
                prev_c = ""
            else:
                prev_c = str(prev_c)

            next_c = chunk.get("next_chunk_id")
            if next_c is None:
                next_c = ""
            else:
                next_c = str(next_c)

            # Đóng gói Payload chính xác 13 trường của Target Universal Schema
            # Loại bỏ hoàn toàn các trường legacy: original_text, word_count, token_count, is_active
            payload = {
                "chunk_id": str(pt_id),
                "parent_id": str(chunk.get("parent_id") or ""),
                "source": str(chunk.get("source") or ""),
                "chunk_type": chunk_type,
                "breadcrumb": str(chunk.get("breadcrumb") or ""),
                "section_title": str(chunk.get("section_title") or ""),
                "content": str(chunk.get("content") or ""),
                "parent_content": str(chunk.get("parent_content") or ""),
                "business_metadata": b_meta,
                "prev_chunk_id": prev_c,
                "next_chunk_id": next_c,
                "table_data": chunk.get("table_data"),
                "images": imgs
            }

            points.append(
                PointStruct(
                    id=point_uuid,
                    vector=[float(x) for x in vec[:self.dimension]],
                    payload=payload
                )
            )
            new_payloads.append(payload)

        # Upsert theo từng batch vào Qdrant
        total_points = len(points)
        for i in range(0, total_points, batch_size):
            batch_slice = points[i : i + batch_size]
            self.client.upsert(
                collection_name=self.collection_name,
                points=batch_slice
            )

        # Đồng bộ và lập chỉ mục BM25F (True Upsert đồng bộ với Qdrant)
        corpus_id_to_idx = {}
        for idx, c in enumerate(self._bm25_corpus):
            cid = c.get("chunk_id")
            if cid:
                corpus_id_to_idx[cid] = idx

        for p in new_payloads:
            cid = p.get("chunk_id")
            if cid and cid in corpus_id_to_idx:
                existing_idx = corpus_id_to_idx[cid]
                self._bm25_corpus[existing_idx] = p
            else:
                new_idx = len(self._bm25_corpus)
                self._bm25_corpus.append(p)
                if cid:
                    corpus_id_to_idx[cid] = new_idx

        self._build_bm25_index(self._bm25_corpus)
        print(f"✅ [Qdrant Store] Đã nạp thành công {total_points} chunks vào Vector DB (Tổng BM25: {len(self._bm25_corpus)} chunks)!")

    upsert_chunks = add_chunks

    # -------------------------------------------------------------------------
    # 2. XÂY DỰNG CHỈ MỤC BM25F ĐA TRƯỜNG & INVERTED INDEX TỐI ƯU HÓA
    # -------------------------------------------------------------------------
    @staticmethod
    def _tokenize(text):
        """Tách từ vựng chữ và số thuần túy (0% Regex Law)."""
        if not text:
            return []
        cleaned_chars = []
        for ch in str(text).lower():
            if ch.isalnum() or ch == "_":
                cleaned_chars.append(ch)
            else:
                cleaned_chars.append(" ")
        return "".join(cleaned_chars).split()

    @staticmethod
    def _extract_table_text(table_data):
        """Trích xuất dữ liệu bảng thành chuỗi văn bản cho BM25F."""
        if not table_data:
            return ""
        if isinstance(table_data, str):
            return table_data
        if isinstance(table_data, dict):
            parts = []
            for k, v in table_data.items():
                parts.append(str(k))
                parts.append(str(v))
            return " ".join(parts)
        if isinstance(table_data, list):
            parts = []
            for item in table_data:
                if isinstance(item, dict):
                    for k, v in item.items():
                        parts.append(str(k))
                        parts.append(str(v))
                else:
                    parts.append(str(item))
            return " ".join(parts)
        return str(table_data)

    @staticmethod
    def _extract_images_text(images):
        """Trích xuất chú thích và mô tả hình ảnh thành chuỗi văn bản cho BM25F."""
        if not images:
            return ""
        if isinstance(images, str):
            return images
        if isinstance(images, dict):
            cap = str(images.get("caption", ""))
            desc = str(images.get("description", ""))
            return f"{cap} {desc}".strip()
        if isinstance(images, list):
            parts = []
            for img in images:
                if isinstance(img, dict):
                    cap = str(img.get("caption", ""))
                    desc = str(img.get("description", ""))
                    txt = f"{cap} {desc}".strip()
                    if txt:
                        parts.append(txt)
                elif isinstance(img, str):
                    parts.append(img)
            return " ".join(parts)
        return ""

    def _build_bm25_index(self, chunks):
        """
        Xây dựng chỉ mục BM25F đa trường kèm cấu trúc Inverted Index tối ưu hóa.
        Các trường và trọng số:
        - content: 1.0
        - section_title: 2.0 (boost)
        - breadcrumb: 1.5 (boost)
        - table_data: 1.0
        - images: 1.0
        """
        self._bm25_corpus = list(chunks)
        self._bm25_inverted_index = {}
        self._bm25_doc_len = []
        self._bm25_doc_norm = []
        self._bm25_idf = {}

        total_docs = len(chunks)
        if total_docs == 0:
            self._bm25_avgdl = 0.0
            return

        for idx, chunk in enumerate(chunks):
            content_text = chunk.get("content") or ""
            section_title_text = chunk.get("section_title") or ""
            breadcrumb_text = chunk.get("breadcrumb") or ""
            table_text = self._extract_table_text(chunk.get("table_data"))
            images_text = self._extract_images_text(chunk.get("images"))

            c_tokens = self._tokenize(content_text)
            st_tokens = self._tokenize(section_title_text)
            bc_tokens = self._tokenize(breadcrumb_text)
            td_tokens = self._tokenize(table_text)
            img_tokens = self._tokenize(images_text)

            doc_tf = {}
            for t in c_tokens:
                doc_tf[t] = doc_tf.get(t, 0.0) + 1.0
            for t in st_tokens:
                doc_tf[t] = doc_tf.get(t, 0.0) + 2.0
            for t in bc_tokens:
                doc_tf[t] = doc_tf.get(t, 0.0) + 1.5
            for t in td_tokens:
                doc_tf[t] = doc_tf.get(t, 0.0) + 1.0
            for t in img_tokens:
                doc_tf[t] = doc_tf.get(t, 0.0) + 1.0

            doc_len = (
                len(c_tokens) * 1.0
                + len(st_tokens) * 2.0
                + len(bc_tokens) * 1.5
                + len(td_tokens) * 1.0
                + len(img_tokens) * 1.0
            )
            self._bm25_doc_len.append(doc_len)

            for term, freq in doc_tf.items():
                if term not in self._bm25_inverted_index:
                    self._bm25_inverted_index[term] = {}
                self._bm25_inverted_index[term][idx] = freq

        sum_doc_len = sum(self._bm25_doc_len)
        if total_docs > 0:
            self._bm25_avgdl = sum_doc_len / float(total_docs)
        else:
            self._bm25_avgdl = 0.0

        # Tính IDF chuẩn Okapi
        for term, posting in self._bm25_inverted_index.items():
            freq_docs = len(posting)
            self._bm25_idf[term] = math.log(
                ((total_docs - freq_docs + 0.5) / (freq_docs + 0.5)) + 1.0
            )

        # Tiền tính toán hệ số chuẩn hóa độ dài tài liệu cho truy vấn siêu tốc (<4ms)
        for d_len in self._bm25_doc_len:
            if self._bm25_avgdl > 0.0:
                norm_val = self.k1 * (1.0 - self.b + self.b * (d_len / self._bm25_avgdl))
            else:
                norm_val = 1.0
            self._bm25_doc_norm.append(norm_val)

    # -------------------------------------------------------------------------
    # 3. CƠ CHẾ LỌC METADATA (PAYLOAD FILTER & NESTED RESOLVER)
    # -------------------------------------------------------------------------
    @staticmethod
    def _get_nested_field(chunk, key):
        """
        Truy xuất giá trị từ dictionary có hỗ trợ đường dẫn dấu chấm (dot-delimited).
        Ví dụ: 'business_metadata.domain' -> chunk['business_metadata']['domain']
        """
        if not isinstance(chunk, dict):
            return None
        if not key:
            return None
        if not isinstance(key, str):
            return None
        if "." not in key:
            return chunk.get(key)

        parts = key.split(".")
        curr = chunk
        for part in parts:
            if isinstance(curr, dict) and part in curr:
                curr = curr[part]
            else:
                return None
        return curr

    @staticmethod
    def _build_qdrant_filter(filter_dict=None):
        """Chuyển đổi dictionary điều kiện sang Qdrant Filter, hỗ trợ giá trị đơn hoặc danh sách."""
        if not filter_dict:
            return None
        if not isinstance(filter_dict, dict):
            return None

        conditions = []
        for key, val in filter_dict.items():
            if val is not None:
                if isinstance(key, str) and key.strip():
                    clean_k = key.strip()
                    if isinstance(val, (list, tuple, set)):
                        clean_v = [x for x in val if x is not None]
                        conditions.append(FieldCondition(key=clean_k, match=MatchAny(any=clean_v)))
                    else:
                        conditions.append(FieldCondition(key=clean_k, match=MatchValue(value=val)))

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
    # 5. TÌM KIẾM TỪ KHÓA (BM25F OKAPI SEARCH SỬ DỤNG INVERTED INDEX)
    # -------------------------------------------------------------------------
    def search_bm25(self, query_text, top_k=5, filter_dict=None, breadcrumb_boost=1.5):
        """
        Tìm kiếm từ khóa chính xác qua thuật toán Okapi BM25 đa trường (BM25F)
        sử dụng Inverted Index và tiền tính toán tf để đạt độ trễ < 4ms.
        Hỗ trợ lọc metadata đồng bộ bao gồm cả nested fields (business_metadata.domain).
        """
        if not self._bm25_corpus:
            return []
        if not query_text:
            return []

        q_tokens = self._tokenize(query_text)
        if not q_tokens:
            return []

        # Chuẩn bị bộ lọc metadata (bỏ qua giá trị None để đồng bộ tuyệt đối với Qdrant Filter)
        parsed_filter = []
        if filter_dict and isinstance(filter_dict, dict):
            for fk, fv in filter_dict.items():
                if fv is not None:
                    if isinstance(fk, str) and fk.strip():
                        clean_fk = fk.strip()
                        if isinstance(fv, (list, tuple, set)):
                            clean_fv = [x for x in fv if x is not None]
                            parsed_filter.append((clean_fk, clean_fv, True))
                        else:
                            parsed_filter.append((clean_fk, fv, False))

        scores = {}
        filter_cache = {}

        for term in q_tokens:
            posting = self._bm25_inverted_index.get(term)
            if not posting:
                continue

            idf = self._bm25_idf.get(term, 0.0)
            if idf <= 0.0:
                continue

            term_weight = idf * (self.k1 + 1.0)
            for doc_idx, freq in posting.items():
                # Kiểm tra bộ lọc metadata
                if parsed_filter:
                    if doc_idx not in filter_cache:
                        chunk = self._bm25_corpus[doc_idx]
                        match_all = True
                        for fk, target_val, is_coll in parsed_filter:
                            curr_val = self._get_nested_field(chunk, fk)
                            if curr_val is None:
                                match_all = False
                                break

                            if is_coll:
                                if isinstance(curr_val, (list, tuple, set)):
                                    has_match = False
                                    for item in curr_val:
                                        if item in target_val:
                                            has_match = True
                                            break
                                    if not has_match:
                                        match_all = False
                                        break
                                else:
                                    if curr_val not in target_val:
                                        match_all = False
                                        break
                            else:
                                if isinstance(curr_val, (list, tuple, set)):
                                    if target_val not in curr_val:
                                        match_all = False
                                        break
                                else:
                                    if curr_val != target_val:
                                        match_all = False
                                        break

                        filter_cache[doc_idx] = match_all

                    if not filter_cache[doc_idx]:
                        continue

                denom = freq + self._bm25_doc_norm[doc_idx]
                contrib = (freq * term_weight) / denom
                scores[doc_idx] = scores.get(doc_idx, 0.0) + contrib

        if not scores:
            return []

        # Sắp xếp top_k điểm cao nhất
        sorted_indices = sorted(scores.keys(), key=lambda i: scores[i], reverse=True)[:top_k]
        results = []
        for idx in sorted_indices:
            results.append((self._bm25_corpus[idx], float(scores[idx])))
        return results

    # -------------------------------------------------------------------------
    # 6. TÌM KIẾM LAI (HYBRID SEARCH: DENSE + BM25F + RRF)
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
        2. Sparse BM25F (bắt chính xác mã thẻ, số thông tư, lãi suất) qua Inverted Index.
        3. Hợp nhất thứ hạng bằng Reciprocal Rank Fusion (RRF k=60).
        """
        # 1. Gọi truy vấn
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
            c_id = chunk.get("chunk_id")
            if not c_id:
                c_id = str(id(chunk))
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
            c_id = chunk.get("chunk_id")
            if not c_id:
                c_id = str(id(chunk))
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
