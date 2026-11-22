"""
Qdrant Vector Database & Hybrid Store Module (Localhost - Zero Docker)
======================================================================
Responsibilities:
1. Stores dense vectors (Cosine Distance HNSW) to local persistent disk.
2. Metadata filtering (Filterable HNSW: source, chunk_type, parent_id, business_metadata.domain, business_metadata.category).
3. Multi-field BM25F Okapi Keyword Search (content, section_title, breadcrumb, table_data, images) with Inverted Index.
4. Hybrid Search: Fuses Dense + BM25 via Reciprocal Rank Fusion (RRF k=60).

DESIGN INVARIANTS:
1. 0% REGEX LAW: No regex methods or import re.
2. ZERO TERNARY: Explicit if/else statements only.
4. NO __INIT__.PY: Direct module imports.
"""

import gc
import hashlib
import json
import math
import os
import shutil
import struct
import uuid
import warnings
from pathlib import Path

from agent_banking.chunking.universal.planning.schema import VALID_CHUNK_TYPES, payload_index_key
from agent_banking.core.search_text import search_key
from agent_banking.vector_db.config.settings import VectorDBConfig

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

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


def publication_scope(output_dir):
    """Identify a snapshot stream without storing its local path in the index."""
    path = os.path.normcase(str(Path(output_dir).resolve()))
    return hashlib.sha256(path.encode("utf-8")).hexdigest()


def mark_publication_chunks(chunks, scope):
    return [{**chunk, "business_metadata": {**chunk["business_metadata"], "publication_scopes": [scope]}}
            for chunk in chunks]


def _publication_owners(payload):
    metadata = payload.get("business_metadata")
    if isinstance(metadata, dict):
        owners = metadata.get("publication_scopes")
        if isinstance(owners, list):
            return {owner for owner in owners if isinstance(owner, str)}
    return set()


def _index_payload(chunk, fallback_id):
    chunk_type = chunk.get("chunk_type")
    if chunk_type not in VALID_CHUNK_TYPES:
        chunk_type = "text"
    metadata = chunk.get("business_metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    images = chunk.get("images")
    if not isinstance(images, list):
        images = []
    return {
        "chunk_id": str(chunk.get("chunk_id") or fallback_id),
        "parent_id": str(chunk.get("parent_id") or ""),
        "source": str(chunk.get("source") or ""),
        "chunk_type": chunk_type,
        "breadcrumb": str(chunk.get("breadcrumb") or ""),
        "section_title": str(chunk.get("section_title") or ""),
        "content": str(chunk.get("content") or ""),
        "parent_content": str(chunk.get("parent_content") or ""),
        "business_metadata": {key: value for key, value in metadata.items() if key != "publication_revision"},
        "prev_chunk_id": str(chunk.get("prev_chunk_id") or ""),
        "next_chunk_id": str(chunk.get("next_chunk_id") or ""),
        "table_data": chunk.get("table_data"),
        "images": images,
    }


def _set_publication_revision(payload, vector):
    metadata = payload["business_metadata"]
    if not (_publication_owners(payload) or metadata.get("pipeline_fingerprint")):
        return
    fields = {**payload, "business_metadata": {key: value for key, value in metadata.items()
                                              if key != "publication_scopes"}}
    encoded = json.dumps(fields, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    # Local cache and Qdrant persist vectors at float32 precision.
    stored_vector = struct.pack(f"<{len(vector)}f", *vector)
    metadata["publication_revision"] = "publication:" + hashlib.sha256(encoded + stored_vector).hexdigest()


class QdrantBankingStore:
    """
    Vector database & Hybrid Search store running locally.
    """

    def __init__(
        self,
        collection_name=VectorDBConfig.QDRANT_COLLECTION,
        dimension=VectorDBConfig.QDRANT_VECTOR_SIZE,
        storage_path=None,
        use_memory=False
    ):
        if collection_name:
            self.collection_name = collection_name
        else:
            self.collection_name = VectorDBConfig.QDRANT_COLLECTION
        if dimension is not None:
            self.dimension = dimension
        else:
            self.dimension = VectorDBConfig.QDRANT_VECTOR_SIZE
        self.use_memory = use_memory

        resolved_raw = storage_path
        if not resolved_raw:
            resolved_raw = VectorDBConfig.QDRANT_STORAGE_PATH
        raw_path = Path(resolved_raw)
        if raw_path.is_absolute():
            self.storage_path = str(raw_path)
        else:
            self.storage_path = str((PROJECT_ROOT / raw_path).resolve())

        # 1. Initialize Qdrant Client (local path or in-memory)
        if use_memory:
            self.client = QdrantClient(":memory:")
        else:
            Path(self.storage_path).mkdir(parents=True, exist_ok=True)
            try:
                self.client = QdrantClient(path=self.storage_path)
            except Exception as e:
                print(f"⚠️ [Qdrant Storage Lock] Không thể mở độc quyền '{self.storage_path}' -> Chuyển sang In-Memory mode an toàn!")
                self.client = QdrantClient(":memory:")
                self.use_memory = True

        # 2. Initialize HNSW Cosine Collection and Payload Indexes
        self._init_collection()

        # 3. Local in-memory BM25F Inverted Index cache
        self._bm25_corpus = []
        self._bm25_inverted_index = {}
        self._bm25_doc_len = []
        self._bm25_doc_norm = []
        self._bm25_avgdl = 0.0
        self._bm25_idf = {}
        self.k1 = VectorDBConfig.VECTOR_DB_BM25_K1
        self.b = VectorDBConfig.VECTOR_DB_BM25_B
        self._bm25_breadcrumb_boost = VectorDBConfig.VECTOR_DB_BREADCRUMB_BOOST

        # Automatically reload BM25 from disk if collection pre-exists
        if not use_memory:
            self._load_bm25_from_existing_store()

    @staticmethod
    def _resolve_distance_metric():
        metric_map = {
            "cosine": Distance.COSINE,
            "dot": Distance.DOT,
            "euclid": Distance.EUCLID,
            "manhattan": Distance.MANHATTAN,
        }
        key = VectorDBConfig.QDRANT_DISTANCE_METRIC.strip().lower()
        return metric_map.get(key, Distance.COSINE)

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
                    distance=self._resolve_distance_metric()
                ),
                hnsw_config=HnswConfigDiff(
                    m=VectorDBConfig.QDRANT_HNSW_M,
                    ef_construct=VectorDBConfig.QDRANT_HNSW_EF_CONSTRUCT
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

        # Multi-tier physical cleanup (Defense-in-depth on disk)
        if not self.use_memory:
            try:
                gc.collect()
                col_dir = Path(self.storage_path) / "collection" / self.collection_name
                if col_dir.exists():
                    shutil.rmtree(col_dir, ignore_errors=True)
            except Exception:
                pass

        # Recreate fresh collection
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=self.dimension,
                distance=self._resolve_distance_metric()
            ),
            hnsw_config=HnswConfigDiff(
                m=VectorDBConfig.QDRANT_HNSW_M,
                ef_construct=VectorDBConfig.QDRANT_HNSW_EF_CONSTRUCT
            )
        )

        # Rebuild payload indexes
        self._create_payload_indexes()

        # Clear BM25 cache
        self._bm25_corpus = []
        self._bm25_inverted_index = {}
        self._bm25_doc_len = []
        self._bm25_doc_norm = []
        self._bm25_avgdl = 0.0
        self._bm25_idf = {}

        return True

    def _create_payload_indexes(self):
        """Khởi tạo các chỉ mục Payload (Keyword index) phục vụ truy vấn có điều kiện cực nhanh."""
        fields = VectorDBConfig.VECTOR_DB_PAYLOAD_INDEX_FIELDS
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
                limit_val = pts_count + VectorDBConfig.VECTOR_DB_SCROLL_LIMIT_BUFFER
                min_limit = VectorDBConfig.VECTOR_DB_MIN_SCROLL_LIMIT
                if limit_val < min_limit:
                    limit_val = min_limit
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
    # 1. DATA INGESTION
    # -------------------------------------------------------------------------
    def index_key(self, chunk, fallback_id=""):
        payload = _index_payload(chunk, fallback_id)
        vector = [float(value) for value in chunk["vector"][:self.dimension]]
        _set_publication_revision(payload, vector)
        return payload_index_key(payload)

    def add_chunks(self, chunks, batch_size=VectorDBConfig.VECTOR_DB_UPSERT_BATCH_SIZE):
        """
        Nạp danh sách chunks đã có trường 'vector' vào Qdrant và lập chỉ mục BM25F.
        Đảm bảo 100% chunks tuân thủ Target Universal Schema 13 trường.
        """
        if not chunks:
            return set()

        points = []
        new_payloads = []
        for idx, chunk in enumerate(chunks):
            vec = chunk.get("vector")
            if not vec:
                continue

            vector = [float(value) for value in vec[:self.dimension]]
            payload = _index_payload(chunk, str(idx))
            _set_publication_revision(payload, vector)
            point_key = payload_index_key(payload)
            try:
                point_uuid = str(uuid.UUID(point_key))
            except ValueError:
                point_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, point_key))

            points.append(
                PointStruct(
                    id=point_uuid,
                    vector=vector,
                    payload=payload
                )
            )
            new_payloads.append(payload)

        # Upsert in batches into Qdrant
        total_points = len(points)
        for i in range(0, total_points, batch_size):
            batch_slice = points[i : i + batch_size]
            existing_points = self.client.retrieve(
                collection_name=self.collection_name, ids=[point.id for point in batch_slice],
                with_payload=["business_metadata.publication_scopes"], with_vectors=False)
            existing_owners = {record.id: _publication_owners(record.payload or {}) for record in existing_points}
            for index, point in enumerate(batch_slice, start=i):
                payload = new_payloads[index]
                owners = existing_owners.get(point.id, set()) | _publication_owners(payload)
                if owners:
                    payload["business_metadata"] = {**payload["business_metadata"], "publication_scopes": sorted(owners)}
                    point.payload = payload
                    existing_owners[point.id] = owners
            self.client.upsert(
                collection_name=self.collection_name,
                points=batch_slice
            )

        # Synchronize and index BM25F (True Upsert synchronized with Qdrant)
        corpus_id_to_idx = {}
        for idx, c in enumerate(self._bm25_corpus):
            cid = payload_index_key(c)
            if cid:
                corpus_id_to_idx[cid] = idx

        for p in new_payloads:
            cid = payload_index_key(p)
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
        return {payload_index_key(payload) for payload in new_payloads}

    upsert_chunks = add_chunks

    def chunk_ids(self, *, scope=None, index_keys=False):
        identifiers = set()
        offset = None
        payload_fields = ["chunk_id"]
        if index_keys:
            payload_fields.append("business_metadata.publication_revision")
        if scope is not None:
            payload_fields.append("business_metadata.publication_scopes")
        scroll_limit = VectorDBConfig.VECTOR_DB_SCROLL_BATCH_SIZE
        while True:
            records, offset = self.client.scroll(collection_name=self.collection_name, offset=offset, limit=scroll_limit,
                                                  with_payload=payload_fields, with_vectors=False)
            for record in records:
                payload = record.payload
                if payload and (scope is None or scope in _publication_owners(payload)):
                    if index_keys:
                        identifier = payload_index_key(payload)
                    else:
                        identifier = payload.get("chunk_id")
                    identifiers.add(identifier)
            if offset is None:
                return identifiers

    def prune_chunks(self, active_chunk_ids, sources=None, *, scope=None, preserve_sources=(), active_index_keys=None):
        """Remove obsolete versions after their replacements have been upserted.

        With a publication scope, `sources` identifies legacy sources eligible
        for migration. Other explicitly owned snapshots stay outside that scope.
        """
        from agent_banking.chunking.universal.extraction.sources import legacy_source_resolver, source_key

        active = set(active_chunk_ids)
        preserved = {source_key(source) for source in preserve_sources}
        current_sources = {source_key(source) for source in sources or () if isinstance(source, str)}
        resolve_legacy = legacy_source_resolver(current_sources | preserved)

        def stale(payload):
            if active_index_keys is None and payload.get("chunk_id") in active:
                return False
            if active_index_keys is not None and payload_index_key(payload) in active_index_keys:
                return False
            source = payload.get("source")
            if source_key(source) in preserved:
                return False
            owners = _publication_owners(payload)
            if scope is None:
                if resolve_legacy(source) in preserved:
                    return False
                return sources is None or source_key(source) in current_sources
            if owners:
                return scope in owners
            return resolve_legacy(source) in current_sources

        obsolete = []
        shared = {}
        retained_points = {}
        offset = None
        payload_fields = ["chunk_id", "source", "business_metadata.publication_revision"]
        if scope is not None:
            payload_fields.append("business_metadata.publication_scopes")
        scroll_limit = VectorDBConfig.VECTOR_DB_SCROLL_BATCH_SIZE
        mutate_batch = VectorDBConfig.VECTOR_DB_MUTATE_BATCH_SIZE
        while True:
            records, offset = self.client.scroll(collection_name=self.collection_name, offset=offset,
                                                 limit=scroll_limit, with_payload=payload_fields, with_vectors=False)
            for record in records:
                payload = record.payload or {}
                if stale(payload):
                    remaining_owners = tuple(sorted(_publication_owners(payload) - {scope}))
                    if scope is not None and remaining_owners:
                        shared.setdefault(remaining_owners, []).append(record.id)
                        retained_points[payload_index_key(payload)] = list(remaining_owners)
                    else:
                        obsolete.append(record.id)
            if offset is None:
                break
        for owners, identifiers in shared.items():
            for start in range(0, len(identifiers), mutate_batch):
                self.client.set_payload(
                    collection_name=self.collection_name, payload={"publication_scopes": list(owners)},
                    points=identifiers[start:start + mutate_batch], key="business_metadata", wait=True)
        for start in range(0, len(obsolete), mutate_batch):
            self.client.delete(collection_name=self.collection_name, points_selector=obsolete[start:start + mutate_batch], wait=True)
        remaining = []
        for item in self._bm25_corpus:
            owners = retained_points.get(payload_index_key(item))
            if owners:
                item["business_metadata"]["publication_scopes"] = owners
            if not stale(item):
                remaining.append(item)
        if len(remaining) != len(self._bm25_corpus):
            self._build_bm25_index(remaining)
        return len(obsolete)

    def prune_publication_scopes(self, active_by_scope):
        """Prune explicitly owned snapshots in one scan after replacement upserts.

        Other owners and unowned legacy records are retained. Unlike calling
        prune_chunks once per document, this rebuilds BM25 only once.
        """
        if not active_by_scope:
            return 0
        if any(not isinstance(scope, str) or not scope for scope in active_by_scope):
            raise ValueError("Batch pruning requires explicit nonempty publication scopes")
        active = {scope: set(keys) for scope, keys in active_by_scope.items()}

        def remaining_owners(payload):
            owners = _publication_owners(payload)
            key = payload_index_key(payload)
            removed = {scope for scope in owners if scope in active and key not in active[scope]}
            if not removed:
                return None
            return owners - removed

        obsolete = []
        shared = {}
        offset = None
        fields = ["chunk_id", "source", "business_metadata.publication_revision", "business_metadata.publication_scopes"]
        scroll_limit = VectorDBConfig.VECTOR_DB_SCROLL_BATCH_SIZE
        mutate_batch = VectorDBConfig.VECTOR_DB_MUTATE_BATCH_SIZE
        while True:
            records, offset = self.client.scroll(collection_name=self.collection_name, offset=offset,
                                                 limit=scroll_limit, with_payload=fields, with_vectors=False)
            for record in records:
                owners = remaining_owners(record.payload or {})
                if owners is None:
                    continue
                if owners:
                    shared.setdefault(tuple(sorted(owners)), []).append(record.id)
                else:
                    obsolete.append(record.id)
            if offset is None:
                break
        for owners, identifiers in shared.items():
            for start in range(0, len(identifiers), mutate_batch):
                self.client.set_payload(collection_name=self.collection_name,
                                        payload={"publication_scopes": list(owners)},
                                        points=identifiers[start:start + mutate_batch], key="business_metadata", wait=True)
        for start in range(0, len(obsolete), mutate_batch):
            self.client.delete(collection_name=self.collection_name, points_selector=obsolete[start:start + mutate_batch], wait=True)
        remaining = []
        for item in self._bm25_corpus:
            owners = remaining_owners(item)
            if owners is None:
                remaining.append(item)
            elif owners:
                metadata = {**item["business_metadata"], "publication_scopes": sorted(owners)}
                remaining.append({**item, "business_metadata": metadata})
        if obsolete or shared:
            self._build_bm25_index(remaining)
        return len(obsolete)

    # -------------------------------------------------------------------------
    # 2. BUILD MULTI-FIELD BM25F INDEX & OPTIMIZED INVERTED INDEX
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
        original = "".join(cleaned_chars).split()
        folded = search_key(str(text)).split()
        return original + ["~" + word for word in folded]

    @staticmethod
    def _flatten_table_value(value):
        """Flatten nested rows/cells lists into readable strings without metadata noise."""
        if value is None:
            return []
        if isinstance(value, str):
            text = value.strip()
            if text:
                return [text]
            return []
        if isinstance(value, (int, float, bool)):
            return [str(value)]
        if isinstance(value, dict):
            for key in VectorDBConfig.VECTOR_DB_TABLE_DICT_VALUE_KEYS:
                if key in value and value[key] is not None:
                    text = str(value[key]).strip()
                    if text:
                        return [text]
            return []
        if isinstance(value, (list, tuple)):
            out = []
            for item in value:
                out.extend(QdrantBankingStore._flatten_table_value(item))
            return out
        return [str(value)]

    @staticmethod
    def _extract_table_text(table_data):
        """Trích xuất dữ liệu bảng thành chuỗi văn bản sạch cho BM25F."""
        if not table_data:
            return ""
        if isinstance(table_data, str):
            return table_data
        if isinstance(table_data, dict):
            parts = []
            for meta_key in VectorDBConfig.VECTOR_DB_TABLE_META_KEYS:
                val = table_data.get(meta_key)
                if val is not None and str(val).strip():
                    parts.append(str(val).strip())

            headers = table_data.get("headers")
            if headers:
                parts.extend(QdrantBankingStore._flatten_table_value(headers))
            elif table_data.get("header_grid"):
                parts.extend(QdrantBankingStore._flatten_table_value(table_data.get("header_grid")))

            rows = table_data.get("rows")
            if rows:
                parts.extend(QdrantBankingStore._flatten_table_value(rows))
            elif table_data.get("cells"):
                parts.extend(QdrantBankingStore._flatten_table_value(table_data.get("cells")))

            if parts:
                return " ".join(parts)

            skip_keys = VectorDBConfig.VECTOR_DB_TABLE_SKIP_KEYS
            for k, v in table_data.items():
                if str(k) in skip_keys:
                    continue
                parts.append(str(k))
                parts.extend(QdrantBankingStore._flatten_table_value(v))
            return " ".join(parts)
        if isinstance(table_data, list):
            return " ".join(QdrantBankingStore._flatten_table_value(table_data))
        return str(table_data)

    @staticmethod
    def _extract_images_text(images):
        """Trích xuất chú thích và mô tả hình ảnh thành chuỗi văn bản cho BM25F."""
        if not images:
            return ""
        if isinstance(images, str):
            return images

        if isinstance(images, dict):
            img_list = [images]
        elif isinstance(images, list):
            img_list = images
        else:
            return ""

        parts = []
        for img in img_list:
            if isinstance(img, dict):
                seen = set()
                img_parts = []
                for field in VectorDBConfig.VECTOR_DB_IMAGE_TEXT_FIELDS:
                    raw_val = img.get(field)
                    if raw_val is not None:
                        val = str(raw_val).strip()
                        if val and val not in seen:
                            seen.add(val)
                            img_parts.append(val)
                evidence = img.get("context_evidence")
                if isinstance(evidence, list):
                    for ev in evidence:
                        if isinstance(ev, dict):
                            quote = ev.get("quote")
                            if quote is not None:
                                q_str = str(quote).strip()
                                if q_str and q_str not in seen:
                                    seen.add(q_str)
                                    img_parts.append(q_str)
                if img_parts:
                    parts.append(" ".join(img_parts))
            elif isinstance(img, str):
                text = img.strip()
                if text:
                    parts.append(text)
        return " ".join(parts)

    def _build_bm25_index(self, chunks, breadcrumb_boost=None):
        """
        Xây dựng chỉ mục BM25F đa trường kèm cấu trúc Inverted Index tối ưu hóa.
        Các trường và trọng số:
        - content: 1.0
        - section_title: 2.0 (boost)
        - breadcrumb: breadcrumb_boost (mặc định 1.5)
        - table_data: 1.0
        - images: 1.0
        """
        if breadcrumb_boost is not None:
            self._bm25_breadcrumb_boost = float(breadcrumb_boost)
        bc_boost = self._bm25_breadcrumb_boost

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
            source_name = str(chunk.get("source") or "").replace("\\", "/").rsplit("/", 1)[-1]
            if "." in source_name:
                source_stem = source_name.rsplit(".", 1)[0]
            else:
                source_stem = source_name
            doc_name = str(chunk.get("document_name") or "").strip()
            meta = chunk.get("business_metadata")
            if not isinstance(meta, dict):
                meta = {}
            prov = meta.get("provenance")
            page_label = ""
            if isinstance(prov, dict):
                locs = prov.get("locations")
                if isinstance(locs, list):
                    page_nums = []
                    for loc in locs:
                        if isinstance(loc, dict):
                            p_val = loc.get("page_number")
                            if type(p_val) is int and p_val > 0 and p_val not in page_nums:
                                page_nums.append(p_val)
                    if page_nums:
                        page_nums.sort()
                        page_label = "Trang " + " ".join(str(p) for p in page_nums)
            meta_labels = []
            for label in (doc_name, source_stem, page_label):
                if label and label.casefold() not in breadcrumb_text.casefold():
                    if label not in meta_labels:
                        meta_labels.append(label)
            if meta_labels:
                label_prefix = " | ".join(meta_labels)
                if breadcrumb_text:
                    breadcrumb_text = label_prefix + " > " + breadcrumb_text
                else:
                    breadcrumb_text = label_prefix
            table_text = self._extract_table_text(chunk.get("table_data"))
            retrieval = meta.get("retrieval", {})
            if "embedding_content" in retrieval:
                content_text = retrieval["embedding_content"]
                if chunk.get("chunk_type") == "table":
                    table_text = ""
            raw_content = str(chunk.get("content") or "")
            parent_text = str(chunk.get("parent_content") or "")
            if len(raw_content) < 350 and len(parent_text) > len(raw_content) + 40:
                content_text = str(content_text) + "\n" + parent_text[:900]
            images_text = self._extract_images_text(chunk.get("images"))

            c_tokens = self._tokenize(content_text)
            st_tokens = self._tokenize(section_title_text)
            bc_tokens = self._tokenize(breadcrumb_text)
            td_tokens = self._tokenize(table_text)
            img_tokens = self._tokenize(images_text)

            weights = VectorDBConfig.VECTOR_DB_BM25_FIELD_WEIGHTS
            w_content = float(weights["content"])
            w_title = float(weights["section_title"])
            w_table = float(weights["table_data"])
            w_images = float(weights["images"])

            doc_tf = {}
            for t in c_tokens:
                doc_tf[t] = doc_tf.get(t, 0.0) + w_content
            for t in st_tokens:
                doc_tf[t] = doc_tf.get(t, 0.0) + w_title
            for t in bc_tokens:
                doc_tf[t] = doc_tf.get(t, 0.0) + bc_boost
            for t in td_tokens:
                doc_tf[t] = doc_tf.get(t, 0.0) + w_table
            for t in img_tokens:
                doc_tf[t] = doc_tf.get(t, 0.0) + w_images

            doc_len = (
                len(c_tokens) * w_content
                + len(st_tokens) * w_title
                + len(bc_tokens) * bc_boost
                + len(td_tokens) * w_table
                + len(img_tokens) * w_images
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

        # Compute standard Okapi IDF
        for term, posting in self._bm25_inverted_index.items():
            freq_docs = len(posting)
            self._bm25_idf[term] = math.log(
                ((total_docs - freq_docs + 0.5) / (freq_docs + 0.5)) + 1.0
            )

        # Precompute document length normalization for fast queries (<4ms)
        for d_len in self._bm25_doc_len:
            if self._bm25_avgdl > 0.0:
                norm_val = self.k1 * (1.0 - self.b + self.b * (d_len / self._bm25_avgdl))
            else:
                norm_val = 1.0
            self._bm25_doc_norm.append(norm_val)

    # -------------------------------------------------------------------------
    # 3. METADATA FILTERING MECHANISM (PAYLOAD FILTER & NESTED RESOLVER)
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
                        if clean_v:
                            conditions.append(FieldCondition(key=clean_k, match=MatchAny(any=clean_v)))
                    else:
                        conditions.append(FieldCondition(key=clean_k, match=MatchValue(value=val)))

        if conditions:
            return Filter(must=conditions)
        return None

    # -------------------------------------------------------------------------
    # 4. VECTOR SEARCH (DENSE SEARCH)
    # -------------------------------------------------------------------------
    def search_vector(self, query_vector, top_k=VectorDBConfig.VECTOR_DB_DEFAULT_TOP_K, filter_dict=None):
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
    # 5. KEYWORD SEARCH (BM25F OKAPI SEARCH USING INVERTED INDEX)
    # -------------------------------------------------------------------------
    def search_bm25(self, query_text, top_k=VectorDBConfig.VECTOR_DB_DEFAULT_TOP_K, filter_dict=None, breadcrumb_boost=None):
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

        if breadcrumb_boost is not None:
            target_boost = float(breadcrumb_boost)
            if abs(target_boost - self._bm25_breadcrumb_boost) > 1e-6:
                self._build_bm25_index(self._bm25_corpus, breadcrumb_boost=target_boost)

        # Prepare metadata filter (ignore None values to sync with Qdrant Filter)
        parsed_filter = []
        if filter_dict and isinstance(filter_dict, dict):
            for fk, fv in filter_dict.items():
                if fv is not None:
                    if isinstance(fk, str) and fk.strip():
                        clean_fk = fk.strip()
                        if isinstance(fv, (list, tuple, set)):
                            clean_fv = [x for x in fv if x is not None]
                            if clean_fv:
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
                # Check metadata filter
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

        # Sort top-k highest scores
        sorted_indices = sorted(scores.keys(), key=lambda i: scores[i], reverse=True)[:top_k]
        results = []
        for idx in sorted_indices:
            results.append((self._bm25_corpus[idx], float(scores[idx])))
        return results

    # -------------------------------------------------------------------------
    # 6. HYBRID SEARCH (DENSE + BM25F + RRF)
    # -------------------------------------------------------------------------
    def search_hybrid(
        self,
        query_text,
        query_vector,
        top_k=VectorDBConfig.VECTOR_DB_DEFAULT_TOP_K,
        candidate_k=VectorDBConfig.VECTOR_DB_DEFAULT_CANDIDATE_K,
        alpha=VectorDBConfig.VECTOR_DB_DEFAULT_ALPHA,
        rrf_k=VectorDBConfig.VECTOR_DB_DEFAULT_RRF_K,
        filter_dict=None,
        breadcrumb_boost=None
    ):
        """
        Tìm kiếm lai đồng thời:
        1. Dense Vector (hiểu ngữ nghĩa sâu) qua Qdrant HNSW.
        2. Sparse BM25F (bắt chính xác mã thẻ, số thông tư, lãi suất) qua Inverted Index.
        3. Hợp nhất thứ hạng bằng Reciprocal Rank Fusion (RRF).
        """
        # 1. Execute queries
        dense_hits = self.search_vector(query_vector, top_k=candidate_k, filter_dict=filter_dict)
        bm25_hits = self.search_bm25(
            query_text,
            top_k=candidate_k,
            filter_dict=filter_dict,
            breadcrumb_boost=breadcrumb_boost
        )

        # 2. RRF Merge
        candidates = {}
        dense_weight = alpha
        bm25_weight = 1.0 - alpha
        score_decimals = VectorDBConfig.VECTOR_DB_SCORE_ROUND_DECIMALS
        rrf_decimals = VectorDBConfig.VECTOR_DB_RRF_ROUND_DECIMALS

        # Record Dense ranking
        for rank, (chunk, score) in enumerate(dense_hits, start=1):
            c_id = payload_index_key(chunk)
            if not c_id:
                c_id = str(id(chunk))
            candidates[c_id] = {
                "chunk": chunk,
                "dense_rank": rank,
                "dense_score": round(score, score_decimals),
                "bm25_rank": None,
                "bm25_score": 0.0,
                "rrf_score": 0.0
            }

        # Record BM25 ranking
        for rank, (chunk, score) in enumerate(bm25_hits, start=1):
            c_id = payload_index_key(chunk)
            if not c_id:
                c_id = str(id(chunk))
            if c_id not in candidates:
                candidates[c_id] = {
                    "chunk": chunk,
                    "dense_rank": None,
                    "dense_score": 0.0,
                    "bm25_rank": rank,
                    "bm25_score": round(score, score_decimals),
                    "rrf_score": 0.0
                }
            else:
                candidates[c_id]["bm25_rank"] = rank
                candidates[c_id]["bm25_score"] = round(score, score_decimals)

        # Compute RRF score: Score = w1 / (k + rank1) + w2 / (k + rank2)
        for c_id, item in candidates.items():
            rrf = 0.0
            if item["dense_rank"] is not None:
                rrf += dense_weight / (rrf_k + item["dense_rank"])
            if item["bm25_rank"] is not None:
                rrf += bm25_weight / (rrf_k + item["bm25_rank"])
            item["rrf_score"] = round(rrf, rrf_decimals)

        # Sort results descending by RRF and take Top-K
        sorted_candidates = sorted(candidates.values(), key=lambda x: x["rrf_score"], reverse=True)
        return sorted_candidates[:top_k]

    def count(self):
        """Count total stored vector records."""
        try:
            return self.client.get_collection(self.collection_name).points_count
        except Exception:
            return 0

    def close(self):
        """Release client resources."""
        if self.client:
            try:
                self.client.close()
            except Exception:
                pass


# Standardized Alias
QdrantVectorStore = QdrantBankingStore
