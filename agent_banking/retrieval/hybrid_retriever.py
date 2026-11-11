"""
Hybrid Retriever Module (Dense Vector & Okapi BM25 + Reciprocal Rank Fusion)
=============================================================================
Core Capabilities:
1. Universal Query Contract: Accepts ParsedQuery, dict, or str.
2. Dual-channel intelligent routing:
   - Dense Vector channel: expanded_query -> normalized_query -> raw_query.
   - Sparse Okapi BM25 channel: keywords_str -> normalized_query -> raw_query.
3. Smart Suggested Filters: Caller filter_dict > suggested_filters.
4. Resilience & Zero-Result Fallback: Automatically falls back to unconstrained search
   if initial filters return zero candidates, reusing computed embeddings.
5. Reciprocal Rank Fusion (RRF k=60, w_dense=0.5, w_bm25=0.5).
6. Preserves all 13 fields of the Target Universal Schema with outer chunk_id.

DESIGN INVARIANTS:
1. 0% REGEX LAW: No regex library or regex methods.
2. ZERO TERNARY: Explicit if/else statements only.
3. NO TYPING LIBRARY: Native types only.
4. NO __INIT__.PY: Direct module imports.
"""

import time

try:
    from agent_banking.query_parser.rag_query_parser import ParsedQuery
except ImportError:
    try:
        from ..query_parser.rag_query_parser import ParsedQuery
    except Exception:
        ParsedQuery = None


REQUIRED_UNIVERSAL_SCHEMA_KEYS = (
    "chunk_id",
    "parent_id",
    "source",
    "chunk_type",
    "breadcrumb",
    "section_title",
    "content",
    "parent_content",
    "business_metadata",
    "prev_chunk_id",
    "next_chunk_id",
    "table_data",
    "images",
)


class HybridRetriever:
    """
    Hybrid retriever combining dense vectors and sparse Okapi BM25
    using Reciprocal Rank Fusion (RRF k=60).
    """

    def __init__(
        self,
        vector_store=None,
        embedding_model=None,
        bm25_search=None,
        rrf_k=60,
        dense_weight=0.5,
        bm25_weight=None,
        top_k=5,
        embedder=None
    ):
        actual_embedder = None
        actual_store = None

        if embedder is not None:
            actual_embedder = embedder
        elif embedding_model is not None:
            actual_embedder = embedding_model

        if vector_store is not None:
            if hasattr(vector_store, "embed_query") and not hasattr(vector_store, "search_vector"):
                actual_embedder = vector_store
                if embedding_model is not None:
                    actual_store = embedding_model
            else:
                actual_store = vector_store

        self.vector_store = actual_store
        self.embedding_model = actual_embedder
        self.embedder = actual_embedder
        self.bm25_search = bm25_search
        self.rrf_k = rrf_k
        self.dense_weight = float(dense_weight)

        if bm25_weight is not None:
            self.bm25_weight = float(bm25_weight)
        else:
            self.bm25_weight = 1.0 - self.dense_weight

        if top_k is not None:
            self.top_k = int(top_k)
        else:
            self.top_k = 5

        self.last_query_vector = None

    @staticmethod
    def _normalize_chunk(raw_chunk):
        """
        Normalize chunk to preserve exact 13 fields of Target Universal Schema.
        Loại bỏ hoàn toàn các trường legacy (original_text, word_count, token_count, is_active).
        """
        if not isinstance(raw_chunk, dict):
            raw_dict = {}
        else:
            raw_dict = raw_chunk

        chunk_id = raw_dict.get("chunk_id")
        if chunk_id is not None:
            clean_chunk_id = str(chunk_id)
        else:
            clean_chunk_id = str(id(raw_dict))

        parent_id = raw_dict.get("parent_id")
        if parent_id is not None:
            clean_parent_id = str(parent_id)
        else:
            clean_parent_id = ""

        source = raw_dict.get("source")
        if source is not None:
            clean_source = str(source)
        else:
            clean_source = ""

        chunk_type = raw_dict.get("chunk_type")
        if chunk_type is not None:
            clean_chunk_type = str(chunk_type)
        else:
            clean_chunk_type = "text"

        breadcrumb = raw_dict.get("breadcrumb")
        if breadcrumb is not None:
            clean_breadcrumb = str(breadcrumb)
        else:
            clean_breadcrumb = ""

        section_title = raw_dict.get("section_title")
        if section_title is not None:
            clean_section_title = str(section_title)
        else:
            clean_section_title = ""

        content = raw_dict.get("content")
        if content is None:
            content = raw_dict.get("original_text")
        if content is None:
            content = raw_dict.get("text")
        if content is not None:
            clean_content = str(content)
        else:
            clean_content = ""

        parent_content = raw_dict.get("parent_content")
        if parent_content is not None:
            clean_parent_content = str(parent_content)
        else:
            clean_parent_content = ""

        business_metadata = raw_dict.get("business_metadata")
        if isinstance(business_metadata, dict):
            clean_business_metadata = dict(business_metadata)
        else:
            clean_business_metadata = {}

        prev_chunk_id = raw_dict.get("prev_chunk_id")
        if prev_chunk_id is not None:
            str_prev = str(prev_chunk_id).strip()
            if len(str_prev) > 0:
                clean_prev_chunk_id = str_prev
            else:
                clean_prev_chunk_id = None
        else:
            clean_prev_chunk_id = None

        next_chunk_id = raw_dict.get("next_chunk_id")
        if next_chunk_id is not None:
            str_next = str(next_chunk_id).strip()
            if len(str_next) > 0:
                clean_next_chunk_id = str_next
            else:
                clean_next_chunk_id = None
        else:
            clean_next_chunk_id = None

        table_data = raw_dict.get("table_data")
        if table_data is not None:
            clean_table_data = table_data
        else:
            clean_table_data = None

        images = raw_dict.get("images")
        if isinstance(images, list):
            clean_images = list(images)
        else:
            clean_images = []

        return {
            "chunk_id": clean_chunk_id,
            "parent_id": clean_parent_id,
            "source": clean_source,
            "chunk_type": clean_chunk_type,
            "breadcrumb": clean_breadcrumb,
            "section_title": clean_section_title,
            "content": clean_content,
            "parent_content": clean_parent_content,
            "business_metadata": clean_business_metadata,
            "prev_chunk_id": clean_prev_chunk_id,
            "next_chunk_id": clean_next_chunk_id,
            "table_data": clean_table_data,
            "images": clean_images
        }

    @staticmethod
    def _normalize_filter_dict(filters):
        """
        Chuẩn hóa các khóa bộ lọc sang schema thực tế của Qdrant và In-Memory BM25.
        Tự động ánh xạ 'domain' -> 'business_metadata.domain' và 'category' -> 'business_metadata.category'.
        """
        if filters is None:
            return None
        if not isinstance(filters, dict):
            return None
        if len(filters) == 0:
            return None

        normalized = {}
        for k, v in filters.items():
            if v is not None:
                clean_k = str(k).strip()
                if clean_k == "domain":
                    if "business_metadata.domain" not in filters:
                        clean_k = "business_metadata.domain"
                elif clean_k == "category":
                    if "business_metadata.category" not in filters:
                        clean_k = "business_metadata.category"
                normalized[clean_k] = v

        if len(normalized) == 0:
            return None
        return normalized

    def retrieve(
        self,
        query,
        top_k=None,
        candidate_k=None,
        filter_dict=None,
        breadcrumb_boost=1.5
    ):
        """
        Execute Hybrid Retrieval and fuse rankings using Reciprocal Rank Fusion (RRF k=60).
        Supports 3 query formats: ParsedQuery, dict, str (Universal Query Contract).
        Applies smart suggested_filters with automatic zero-result fallback.
        Preserves 100% of the 13 fields in Target Universal Schema with outer chunk_id.
        """
        # Validate input boundary conditions
        if query is None:
            return []
        if isinstance(query, str):
            if len(query.strip()) == 0:
                return []

        # 1. Determine effective Top-K and Candidate-K
        if top_k is not None:
            effective_top_k = int(top_k)
        else:
            effective_top_k = self.top_k

        if effective_top_k <= 0:
            return []

        if candidate_k is not None:
            effective_candidate_k = int(candidate_k)
        else:
            candidate_calc = effective_top_k * 3
            if candidate_calc > 15:
                effective_candidate_k = candidate_calc
            else:
                effective_candidate_k = 15

        if effective_candidate_k < effective_top_k:
            effective_candidate_k = effective_top_k

        # 2. Process Universal Query Contract and extract optimal query string
        raw_query_text = ""
        dense_query_text = ""
        bm25_query_text = ""
        suggested_filters = {}

        if ParsedQuery is not None and isinstance(query, ParsedQuery):
            if query.raw_query:
                raw_query_text = str(query.raw_query).strip()
            else:
                raw_query_text = ""

            # Dense channel: expanded_query -> normalized_query -> raw_query
            if query.expanded_query and str(query.expanded_query).strip():
                dense_query_text = str(query.expanded_query).strip()
            elif query.normalized_query and str(query.normalized_query).strip():
                dense_query_text = str(query.normalized_query).strip()
            else:
                dense_query_text = raw_query_text

            # BM25 channel: keywords_str -> normalized_query -> raw_query
            if query.keywords_str and str(query.keywords_str).strip():
                bm25_query_text = str(query.keywords_str).strip()
            elif query.normalized_query and str(query.normalized_query).strip():
                bm25_query_text = str(query.normalized_query).strip()
            else:
                bm25_query_text = raw_query_text

            if hasattr(query, "suggested_filters"):
                if isinstance(query.suggested_filters, dict):
                    suggested_filters = dict(query.suggested_filters)

        elif isinstance(query, dict):
            if query.get("raw_query"):
                raw_query_text = str(query["raw_query"]).strip()
            elif query.get("query"):
                raw_query_text = str(query["query"]).strip()
            else:
                raw_query_text = ""

            exp_q = query.get("expanded_query")
            norm_q = query.get("normalized_query")
            kws_q = query.get("keywords_str")

            if not kws_q and isinstance(query.get("keywords"), list):
                clean_kws = []
                for kw in query["keywords"]:
                    if kw:
                        clean_kws.append(str(kw).strip())
                kws_q = " ".join(clean_kws)

            # Dense channel: expanded_query -> normalized_query -> raw_query
            if exp_q and str(exp_q).strip():
                dense_query_text = str(exp_q).strip()
            elif norm_q and str(norm_q).strip():
                dense_query_text = str(norm_q).strip()
            else:
                dense_query_text = raw_query_text

            # BM25 channel: keywords_str -> normalized_query -> raw_query
            if kws_q and str(kws_q).strip():
                bm25_query_text = str(kws_q).strip()
            elif norm_q and str(norm_q).strip():
                bm25_query_text = str(norm_q).strip()
            else:
                bm25_query_text = raw_query_text

            s_filt = query.get("suggested_filters")
            if isinstance(s_filt, dict):
                suggested_filters = dict(s_filt)

        else:
            raw_query_text = str(query).strip()
            dense_query_text = raw_query_text
            bm25_query_text = raw_query_text

        if len(raw_query_text) == 0:
            if len(dense_query_text) > 0:
                raw_query_text = dense_query_text
            elif len(bm25_query_text) > 0:
                raw_query_text = bm25_query_text

        # 3. Determine metadata filters (Caller filter_dict > suggested_filters)
        active_filter = None
        if filter_dict is not None and isinstance(filter_dict, dict):
            if len(filter_dict) > 0:
                active_filter = dict(filter_dict)

        if active_filter is None:
            if suggested_filters is not None and isinstance(suggested_filters, dict):
                if len(suggested_filters) > 0:
                    active_filter = dict(suggested_filters)

        norm_filter = self._normalize_filter_dict(active_filter)

        # 4. Dense Vector Search (Cosine HNSW 1024-dim)
        t_dense_start = time.time()
        if self.embedding_model is not None and hasattr(self.embedding_model, "embed_query"):
            query_vector = self.embedding_model.embed_query(dense_query_text)
            self.last_query_vector = query_vector
        else:
            query_vector = []
            self.last_query_vector = None

        if self.vector_store is not None and hasattr(self.vector_store, "search_vector"):
            dense_results = self.vector_store.search_vector(
                query_vector=query_vector,
                top_k=effective_candidate_k,
                filter_dict=norm_filter
            )
        else:
            dense_results = []
        t_dense_duration = (time.time() - t_dense_start) * 1000

        # 5. Sparse BM25 Okapi Search (BM25F Multi-field with Breadcrumb Boost)
        t_bm25_start = time.time()
        if self.bm25_search is not None and hasattr(self.bm25_search, "search"):
            bm25_results = self.bm25_search.search(
                bm25_query_text,
                top_k=effective_candidate_k,
                filter_dict=norm_filter
            )
        elif self.vector_store is not None and hasattr(self.vector_store, "search_bm25"):
            try:
                bm25_results = self.vector_store.search_bm25(
                    query_text=bm25_query_text,
                    top_k=effective_candidate_k,
                    filter_dict=norm_filter,
                    breadcrumb_boost=breadcrumb_boost
                )
            except TypeError:
                bm25_results = self.vector_store.search_bm25(
                    query_text=bm25_query_text,
                    top_k=effective_candidate_k,
                    filter_dict=norm_filter
                )
        else:
            bm25_results = []
        t_bm25_duration = (time.time() - t_bm25_start) * 1000

        # 6. Filter Zero-Result Fallback mechanism
        fallback_unfiltered = False
        if len(dense_results) == 0 and len(bm25_results) == 0 and norm_filter is not None:
            fallback_unfiltered = True
            if self.vector_store is not None and hasattr(self.vector_store, "search_vector"):
                dense_results = self.vector_store.search_vector(
                    query_vector=query_vector,
                    top_k=effective_candidate_k,
                    filter_dict=None
                )
            if self.bm25_search is not None and hasattr(self.bm25_search, "search"):
                bm25_results = self.bm25_search.search(
                    bm25_query_text,
                    top_k=effective_candidate_k,
                    filter_dict=None
                )
            elif self.vector_store is not None and hasattr(self.vector_store, "search_bm25"):
                try:
                    bm25_results = self.vector_store.search_bm25(
                        query_text=bm25_query_text,
                        top_k=effective_candidate_k,
                        filter_dict=None,
                        breadcrumb_boost=breadcrumb_boost
                    )
                except TypeError:
                    bm25_results = self.vector_store.search_bm25(
                        query_text=bm25_query_text,
                        top_k=effective_candidate_k,
                        filter_dict=None
                    )

        # 7. Reciprocal Rank Fusion (RRF k=60) merge
        candidates = {}

        # Record Dense results
        for rank, item in enumerate(dense_results, start=1):
            if isinstance(item, tuple):
                raw_chunk, score = item
            else:
                raw_chunk, score = item, 0.0

            norm_chunk = self._normalize_chunk(raw_chunk)
            chunk_id = norm_chunk["chunk_id"]

            if chunk_id not in candidates:
                candidates[chunk_id] = {
                    "chunk_id": chunk_id,
                    "chunk": norm_chunk,
                    "dense_rank": rank,
                    "dense_score": float(score),
                    "bm25_rank": None,
                    "bm25_score": None,
                    "rrf_score": 0.0
                }
            else:
                candidates[chunk_id]["dense_rank"] = rank
                candidates[chunk_id]["dense_score"] = float(score)

        # Record BM25 results
        for rank, item in enumerate(bm25_results, start=1):
            if isinstance(item, tuple):
                raw_chunk, score = item
            else:
                raw_chunk, score = item, 0.0

            norm_chunk = self._normalize_chunk(raw_chunk)
            chunk_id = norm_chunk["chunk_id"]

            if chunk_id not in candidates:
                candidates[chunk_id] = {
                    "chunk_id": chunk_id,
                    "chunk": norm_chunk,
                    "dense_rank": None,
                    "dense_score": None,
                    "bm25_rank": rank,
                    "bm25_score": float(score),
                    "rrf_score": 0.0
                }
            else:
                candidates[chunk_id]["bm25_rank"] = rank
                candidates[chunk_id]["bm25_score"] = float(score)

        # Compute RRF scores:
        # RRF(d) = (w_dense / (k + rank_dense)) + (w_bm25 / (k + rank_bm25))
        for item in candidates.values():
            rrf = 0.0
            if item["dense_rank"] is not None:
                rrf += self.dense_weight / (self.rrf_k + item["dense_rank"])
            if item["bm25_rank"] is not None:
                rrf += self.bm25_weight / (self.rrf_k + item["bm25_rank"])
            item["rrf_score"] = rrf

        # Sort candidates descending by RRF score
        sorted_candidates = sorted(candidates.values(), key=lambda x: x["rrf_score"], reverse=True)

        # Extract exact Top-K candidates and format return objects
        top_candidates = []
        for rank, item in enumerate(sorted_candidates[:effective_top_k], start=1):
            if item["dense_score"] is not None:
                dense_score_val = round(item["dense_score"], 4)
            else:
                dense_score_val = None

            if item["bm25_score"] is not None:
                bm25_score_val = round(item["bm25_score"], 4)
            else:
                bm25_score_val = None

            top_candidates.append({
                "candidate_rank": rank,
                "chunk_id": item["chunk_id"],
                "chunk": item["chunk"],
                "rrf_score": round(item["rrf_score"], 6),
                "dense_rank": item["dense_rank"],
                "dense_score": dense_score_val,
                "bm25_rank": item["bm25_rank"],
                "bm25_score": bm25_score_val,
                "retrieval_meta": {
                    "dense_latency_ms": round(t_dense_duration, 2),
                    "bm25_latency_ms": round(t_bm25_duration, 2),
                    "rrf_k": self.rrf_k,
                    "dense_weight": self.dense_weight,
                    "bm25_weight": self.bm25_weight,
                    "dense_query_used": dense_query_text,
                    "bm25_query_used": bm25_query_text,
                    "raw_query": raw_query_text,
                    "filter_applied": active_filter,
                    "fallback_unfiltered": fallback_unfiltered
                }
            })

        return top_candidates
