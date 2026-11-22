"""
Banking Document Embedder Module.
======================================================================
Core Responsibilities:
1. Receives chunk lists from Chunker (13-field Universal Schema).
2. Formats Type-Aware Virtual Passages for text, table, and multimodal content.
3. Checks local persistent SQLite WAL cache on formatted Virtual Passages.
4. Generates dense vectors via configured cloud API or deterministic fallback.
5. Normalizes L2-norm and attaches vector to each chunk.

DESIGN PRINCIPLES (GEMINI.md & AGENTS.md):
1. 0% REGEX LAW: String methods only, no regex imports or functions.
2. ZERO TERNARY LAW: Explicit if/else statements only, no ternary operators.
4. NO __INIT__.PY: Direct module imports.
"""

import os
import math
import struct
import sqlite3
import hashlib
from pathlib import Path
from agent_banking.core.models import model_primary
from agent_banking.embedding.config.settings import EmbeddingConfig
from agent_banking.tracing.tracer import trace_span

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class BankingEmbedder:
    """
    High-performance, lightweight dense embedding pipeline for banking RAG.
    """

    def __init__(
        self,
        model_name=None,
        dimension=None,
        api_key=None,
        embedding_url=None,
        cache_db_path=None
    ):
        try:
            primary = model_primary("embedding")
            cfg_model = primary.model
            cfg_dim = primary.dimension
            cfg_key = primary.api_key
            cfg_url = primary.url
        except Exception:
            cfg_model = EmbeddingConfig.EMBEDDING_MODEL_1024
            cfg_dim = EmbeddingConfig.EMBEDDING_DIM
            cfg_key = ""
            cfg_url = EmbeddingConfig.EMBEDDING_URL

        self.model_name = model_name
        if self.model_name is None:
            if cfg_model:
                self.model_name = cfg_model
            else:
                self.model_name = EmbeddingConfig.EMBEDDING_MODEL_1024

        self.dimension = dimension
        if self.dimension is None:
            if cfg_dim:
                self.dimension = cfg_dim
            else:
                self.dimension = EmbeddingConfig.EMBEDDING_DIM

        self.api_key = api_key
        if self.api_key is None:
            self.api_key = cfg_key

        self.embedding_url = embedding_url
        if self.embedding_url is None:
            if cfg_url:
                self.embedding_url = cfg_url
            else:
                self.embedding_url = EmbeddingConfig.EMBEDDING_URL

        self.cache_enabled = EmbeddingConfig.EMBEDDING_CACHE_ENABLED
        effective_cache_path = cache_db_path
        if not effective_cache_path:
            effective_cache_path = EmbeddingConfig.EMBEDDING_CACHE_DB_PATH

        # Resolve absolute SQLite cache path based on project root
        raw_path = Path(effective_cache_path)
        if raw_path.is_absolute():
            self.cache_path = raw_path
        else:
            self.cache_path = (PROJECT_ROOT / raw_path).resolve()

        self.last_cache_hits = 0
        self.last_cache_misses = 0
        if self.cache_enabled:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self._init_cache()

    # -------------------------------------------------------------------------
    # 1. LOCAL SQLITE WAL CACHE LAYER
    # -------------------------------------------------------------------------
    def _init_cache(self):
        """Initialize SQLite cache table with WAL PRAGMAs for high throughput."""
        conn = sqlite3.connect(self.cache_path, timeout=EmbeddingConfig.EMBEDDING_SQLITE_TIMEOUT_SEC)
        try:
            conn.execute(f"PRAGMA journal_mode={EmbeddingConfig.EMBEDDING_SQLITE_JOURNAL_MODE};")
            conn.execute(f"PRAGMA synchronous={EmbeddingConfig.EMBEDDING_SQLITE_SYNCHRONOUS};")
            conn.execute(f"PRAGMA busy_timeout={EmbeddingConfig.EMBEDDING_SQLITE_BUSY_TIMEOUT_MS};")
            conn.execute(f"PRAGMA cache_size={EmbeddingConfig.EMBEDDING_SQLITE_CACHE_SIZE_KB};")
            conn.execute(f"PRAGMA mmap_size={EmbeddingConfig.EMBEDDING_SQLITE_MMAP_SIZE_BYTES};")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    hash_key TEXT PRIMARY KEY,
                    vector BLOB NOT NULL
                );
            """)
            conn.commit()
        finally:
            conn.close()

    def _get_hash(self, virtual_passage):
        """Compute deterministic SHA-256 hash over formatted virtual passage."""
        if virtual_passage is not None:
            clean_text = str(virtual_passage).strip()
        else:
            clean_text = ""
        key_version = EmbeddingConfig.EMBEDDING_CACHE_KEY_VERSION
        key_material = f"{key_version}:{self.embedding_url}:{self.model_name}:{self.dimension}:{self._passage_limit()}:{clean_text}"
        return hashlib.sha256(key_material.encode("utf-8")).hexdigest()

    def _get_cached_vector(self, virtual_passage):
        """
        Retrieve cached vector for a virtual passage string.
        Supports backward compatibility across prefixed and unprefixed representations.
        """
        if not self.cache_enabled:
            return None
        if virtual_passage is None:
            return None

        clean_str = str(virtual_passage).strip()
        if not clean_str:
            return None

        passage_prefix = EmbeddingConfig.EMBEDDING_PASSAGE_PREFIX
        prefix_len = len(passage_prefix)
        h = self._get_hash(clean_str)
        conn = sqlite3.connect(self.cache_path, timeout=EmbeddingConfig.EMBEDDING_SQLITE_TIMEOUT_SEC)
        try:
            conn.execute(f"PRAGMA busy_timeout={EmbeddingConfig.EMBEDDING_SQLITE_BUSY_TIMEOUT_MS};")
            row = conn.execute("SELECT vector FROM cache WHERE hash_key = ?", (h,)).fetchone()
            if row:
                return list(struct.unpack(f"{self.dimension}f", row[0]))

            # Fallback 1: Query without passage: prefix
            if not clean_str.startswith(passage_prefix):
                h_fallback = self._get_hash(f"{passage_prefix} {clean_str}")
                row_fb = conn.execute("SELECT vector FROM cache WHERE hash_key = ?", (h_fallback,)).fetchone()
                if row_fb:
                    return list(struct.unpack(f"{self.dimension}f", row_fb[0]))
            else:
                # Fallback 2: Cached raw text without passage: prefix
                raw_fallback = clean_str[prefix_len:].strip()
                h_raw = self._get_hash(raw_fallback)
                row_raw = conn.execute("SELECT vector FROM cache WHERE hash_key = ?", (h_raw,)).fetchone()
                if row_raw:
                    return list(struct.unpack(f"{self.dimension}f", row_raw[0]))
        finally:
            conn.close()
        return None

    def _get_cached_vectors_batch(self, hashes):
        """Query batch of vectors from SQLite WAL by hash keys."""
        if not self.cache_enabled or not hashes:
            return {}
        results = {}
        unique_hashes = list(set(hashes))
        batch_size = EmbeddingConfig.EMBEDDING_CACHE_QUERY_BATCH_SIZE

        conn = sqlite3.connect(self.cache_path, timeout=EmbeddingConfig.EMBEDDING_SQLITE_TIMEOUT_SEC)
        try:
            conn.execute(f"PRAGMA busy_timeout={EmbeddingConfig.EMBEDDING_SQLITE_BUSY_TIMEOUT_MS};")
            for i in range(0, len(unique_hashes), batch_size):
                sub_hashes = unique_hashes[i:i + batch_size]
                placeholders = ",".join("?" for _ in sub_hashes)
                cursor = conn.execute(
                    f"SELECT hash_key, vector FROM cache WHERE hash_key IN ({placeholders})",
                    sub_hashes
                )
                for h, blob in cursor.fetchall():
                    results[h] = list(struct.unpack(f"{self.dimension}f", blob))
        finally:
            conn.close()
        return results

    def _save_cache_batch(self, items):
        """Persist batch of binary BLOB vectors into SQLite WAL."""
        if not self.cache_enabled or not items:
            return
        records = []
        for h, vec in items:
            blob = struct.pack(f"{self.dimension}f", *vec)
            records.append((h, blob))

        conn = sqlite3.connect(self.cache_path, timeout=EmbeddingConfig.EMBEDDING_SQLITE_TIMEOUT_SEC)
        try:
            conn.execute(f"PRAGMA busy_timeout={EmbeddingConfig.EMBEDDING_SQLITE_BUSY_TIMEOUT_MS};")
            conn.executemany("INSERT OR REPLACE INTO cache (hash_key, vector) VALUES (?, ?);", records)
            conn.commit()
        finally:
            conn.close()

    # -------------------------------------------------------------------------
    # 2. VIRTUAL PASSAGE FORMATTING (Type-Aware & Lossless)
    # -------------------------------------------------------------------------
    @staticmethod
    def _extract_image_descriptions(images):
        """Extract captions and descriptions from image metadata list."""
        if not images or not isinstance(images, list):
            return ""
        descriptions = []
        for img in images:
            if isinstance(img, dict):
                raw_cap = None
                for key in EmbeddingConfig.EMBEDDING_IMAGE_CAPTION_FIELDS:
                    candidate = img.get(key)
                    if candidate is not None and str(candidate).strip():
                        raw_cap = candidate
                        break
                if raw_cap is not None:
                    cap = str(raw_cap).strip()
                    if cap and cap not in descriptions:
                        descriptions.append(cap)
        if descriptions:
            return " | ".join(descriptions)
        return ""

    def format_virtual_passage(self, chunk):
        """
        Format Type-Aware Virtual Passage for dense retrieval.
        Categorizes by 3 chunk types: 'text', 'table', 'multimodal'.
        Preserves 100% of raw content in chunk['content'].
        Prevents double prefixing of context tags and passage: prefix.
        """
        if not isinstance(chunk, dict):
            return ""

        metadata = chunk.get("business_metadata")
        if not isinstance(metadata, dict):
            metadata = {}
        if metadata.get("schema_version") == "2.0":
            from agent_banking.chunking.universal.text.token_budget import format_passage_fields
            retrieval = dict(metadata.get("retrieval", {}))
            return format_passage_fields(chunk.get("content", ""), retrieval)

        # Retrieve raw content from content or original_text
        raw_content = chunk.get("content")
        if raw_content is None:
            raw_content = chunk.get("original_text")
        if raw_content is None:
            raw_content = ""
        raw_content = str(raw_content).strip()

        passage_prefix = EmbeddingConfig.EMBEDDING_PASSAGE_PREFIX
        prefix_len = len(passage_prefix)
        legacy_markers = EmbeddingConfig.EMBEDDING_LEGACY_PREFIX_MARKERS
        labels = EmbeddingConfig.EMBEDDING_PASSAGE_LABELS

        # Clean duplicate prefix markers
        clean_content = raw_content
        changed = True
        while changed:
            changed = False
            if clean_content.startswith(passage_prefix):
                clean_content = clean_content[prefix_len:].strip()
                changed = True
            else:
                has_prefix = False
                for marker in legacy_markers:
                    if clean_content.startswith(marker):
                        has_prefix = True
                        break

                if has_prefix:
                    end_idx = clean_content.find("]")
                    if end_idx != -1:
                        clean_content = clean_content[end_idx + 1:].strip()
                        changed = True

        # Extract breadcrumb and section title
        breadcrumb = chunk.get("breadcrumb")
        if breadcrumb is None:
            breadcrumb = ""
        breadcrumb = str(breadcrumb).strip()

        section_title = chunk.get("section_title")
        if section_title is None:
            section_title = ""
        section_title = str(section_title).strip()

        # Resolve chunk type
        chunk_type = chunk.get("chunk_type")
        if chunk_type is None:
            chunk_type = "text"
        chunk_type = str(chunk_type).lower().strip()

        table_data = chunk.get("table_data")
        images = chunk.get("images")
        if images is None:
            images = []

        # Branch logic by chunk type
        if chunk_type == "table" or table_data:
            # Table branch
            header_parts = []
            table_lbl = labels["table_header"]
            if section_title:
                header_parts.append(f"{table_lbl}: {section_title}")
            else:
                header_parts.append(table_lbl)

            if breadcrumb:
                header_parts.append(f"{labels['context_header']}: {breadcrumb}")

            tag_header = " | ".join(header_parts)
            prefix = f"{passage_prefix} [{tag_header}]\n"

            # Enrich column headers if available
            col_context = ""
            cols_lbl = labels["columns_tag"]
            if isinstance(table_data, dict):
                cols = table_data.get("headers")
                if not cols:
                    cols = table_data.get("columns")
                if cols and isinstance(cols, list):
                    clean_cols = [str(c).strip() for c in cols if c is not None and str(c).strip()]
                    if clean_cols:
                        col_context = f"[{cols_lbl}: {', '.join(clean_cols)}]\n"
            elif isinstance(table_data, list) and len(table_data) > 0 and isinstance(table_data[0], list):
                clean_cols = [str(c).strip() for c in table_data[0] if c is not None and str(c).strip()]
                if clean_cols:
                    col_context = f"[{cols_lbl}: {', '.join(clean_cols)}]\n"

            body = f"{col_context}{clean_content}"

            img_desc = self._extract_image_descriptions(images)
            if img_desc:
                body = f"{body}\n[{labels['illustration_tag']}: {img_desc}]"

            return f"{prefix}{body}".strip()

        elif chunk_type in ("multimodal", "image", "chart") or (images and len(images) > 0 and not clean_content):
            # Multimodal branch
            header_parts = []
            mm_lbl = labels["multimodal_header"]
            if section_title:
                header_parts.append(f"{mm_lbl}: {section_title}")
            else:
                header_parts.append(mm_lbl)

            if breadcrumb:
                header_parts.append(f"{labels['context_header']}: {breadcrumb}")

            tag_header = " | ".join(header_parts)
            prefix = f"{passage_prefix} [{tag_header}]\n"

            img_desc = self._extract_image_descriptions(images)
            body_parts = []
            if clean_content:
                body_parts.append(clean_content)
            if img_desc and img_desc not in clean_content:
                body_parts.append(f"[{labels['image_description_tag']}: {img_desc}]")

            body = "\n".join(body_parts)
            return f"{prefix}{body}".strip()

        else:
            # Text branch
            header_parts = []
            if breadcrumb:
                header_parts.append(f"{labels['context_header']}: {breadcrumb}")
            if section_title:
                header_parts.append(f"{labels['section_header']}: {section_title}")

            if header_parts:
                tag_header = " | ".join(header_parts)
                prefix = f"{passage_prefix} [{tag_header}]\n"
            else:
                prefix = f"{passage_prefix} "

            parent_content = chunk.get("parent_content")
            if parent_content is None:
                parent_content = ""
            parent_content = str(parent_content).strip()

            clean_parent = parent_content
            if clean_parent.startswith(passage_prefix):
                clean_parent = clean_parent[prefix_len:].strip()

            max_parent_chars = EmbeddingConfig.EMBEDDING_MAX_PARENT_SUMMARY_CHARS
            if clean_parent and len(clean_parent) <= max_parent_chars and clean_parent != clean_content:
                body = f"{clean_content}\n[{labels['parent_summary_tag']}: {clean_parent}]"
            else:
                body = clean_content

            img_desc = self._extract_image_descriptions(images)
            if img_desc:
                body = f"{body}\n[{labels['attached_images_tag']}: {img_desc}]"

            return f"{prefix}{body}".strip()

    # -------------------------------------------------------------------------
    # 3. L2 NORMALIZATION
    # -------------------------------------------------------------------------
    @staticmethod
    def _l2_normalize(vec):
        if not EmbeddingConfig.EMBEDDING_NORMALIZE:
            return list(vec)
        norm = math.sqrt(sum(x * x for x in vec))
        if norm > EmbeddingConfig.EMBEDDING_L2_NORM_EPSILON:
            return [x / norm for x in vec]
        return vec

    # -------------------------------------------------------------------------
    # 4. API CALL & LOCAL FALLBACK
    # -------------------------------------------------------------------------
    def _call_api(self, texts, input_type="passage"):
        if not texts:
            return []
        self._validate_passages(texts)

        # 1. Attempt cloud endpoint if configured
        if not (self.embedding_url and self.api_key and self.model_name):
            raise RuntimeError("Cloud embedding model, endpoint and credential are required")
        from agent_banking.chunking.universal.models.text_client import cloud_json

        eps = EmbeddingConfig.EMBEDDING_L2_NORM_EPSILON

        def validate(body):
            items = sorted(body.get("data", []), key=lambda item: item.get("index", -1))
            if [item.get("index") for item in items] != list(range(len(texts))):
                raise ValueError("Incomplete or misindexed embedding response")
            results = []
            for item in items:
                vector = item.get("embedding", [])
                if len(vector) != self.dimension or not all(isinstance(value, (int, float)) and math.isfinite(value) for value in vector):
                    raise ValueError("Embedding response has invalid dimension or nonfinite values")
                squared_norm = sum(value * value for value in vector)
                if not math.isfinite(squared_norm) or squared_norm <= eps:
                    raise ValueError("Embedding API returned an invalid or zero vector norm")
                results.append(self._l2_normalize(vector))
            return results

        payload = {
            "model": self.model_name,
            "input": texts,
            "dimensions": self.dimension,
            "encoding_format": EmbeddingConfig.EMBEDDING_ENCODING_FORMAT,
            "input_text_truncate": EmbeddingConfig.EMBEDDING_INPUT_TEXT_TRUNCATE,
            "input_type": input_type,
        }
        body = cloud_json(self.embedding_url, self.api_key, payload,
                          model_module="embedding", validate=validate, dimension=self.dimension)
        return validate(body)

    def _passage_limit(self):
        raw_limit = os.getenv("MAX_PASSAGE_TOKENS")
        if raw_limit is not None and raw_limit.strip() != "":
            limit = int(raw_limit.strip())
        else:
            limit = EmbeddingConfig.EMBEDDING_MAX_PASSAGE_TOKENS
        min_limit = EmbeddingConfig.EMBEDDING_MIN_PASSAGE_TOKENS_LIMIT
        max_limit = EmbeddingConfig.EMBEDDING_MAX_PASSAGE_TOKENS_LIMIT
        if not min_limit <= limit <= max_limit:
            raise ValueError(f"MAX_PASSAGE_TOKENS must be between {min_limit} and {max_limit}")
        return limit

    def _validate_passages(self, texts):
        max_tokens = self._passage_limit()
        try:
            from agent_banking.chunking.universal.text.token_budget import get_tokenizer
            tokenizer = get_tokenizer()
        except (ImportError, OSError, RuntimeError):
            return
        counts = [len(tokenizer.encode(str(text)).ids) for text in texts]
        if any(count > max_tokens for count in counts):
            raise ValueError("Passage exceeds " + str(max_tokens) + " tokens; rechunk before embedding (max=" + str(max(counts)) + ")")

    def _passage_windows(self, passage):
        from agent_banking.chunking.universal.text.boundaries import safe_boundaries
        from agent_banking.chunking.universal.text.token_budget import TokenBudget, get_tokenizer

        limit = self._passage_limit()
        budget = TokenBudget(limit, limit, get_tokenizer())
        if budget.count(passage) <= limit:
            return [(passage, 1)]
        passage_prefix = EmbeddingConfig.EMBEDDING_PASSAGE_PREFIX
        prefix = passage_prefix + "\n"
        text = passage.removeprefix(passage_prefix).lstrip("\n")
        # Embedding windows can split math notation while the original chunk keeps it intact.
        spans = budget.split(text, lambda fragment: budget.count(prefix + fragment) <= limit,
                             align_words=True, boundaries=safe_boundaries(text, protect_markup=False))
        return [(prefix + fragment, budget.count(fragment)) for _, _, fragment in spans if fragment.strip()]

    # -------------------------------------------------------------------------
    # 5. PUBLIC APPLICATION INTERFACE
    # -------------------------------------------------------------------------
    def embed_text(self, text, is_query=False):
        """Generate embedding vector for a single text or query."""
        if text is None:
            text = ""

        if is_query:
            input_type = "query"
            query_prefix = EmbeddingConfig.EMBEDDING_QUERY_PREFIX
            if text.startswith(query_prefix):
                virtual_passage = text
            else:
                virtual_passage = f"{query_prefix}{text}"
        else:
            input_type = "passage"
            chunk_mock = {"content": text, "chunk_type": "text"}
            virtual_passage = self.format_virtual_passage(chunk_mock)

        self._validate_passages([virtual_passage])
        if is_query:
            with trace_span(
                "query_embedding",
                run_type="embedding",
                model=self.model_name,
                dimension=self.dimension,
            ) as span_details:
                cached = self._get_cached_vector(virtual_passage)
                if cached:
                    span_details["cache_hit"] = True
                    span_details["result"] = "cache_hit"
                    return cached
                span_details["cache_hit"] = False
                vectors = self._call_api([virtual_passage], input_type=input_type)
                if vectors:
                    h = self._get_hash(virtual_passage)
                    self._save_cache_batch([(h, vectors[0])])
                    span_details["result"] = "api_ok"
                    return vectors[0]
                span_details["result"] = "fallback_zero"
                return [0.0] * self.dimension

        cached = self._get_cached_vector(virtual_passage)
        if cached:
            return cached

        vectors = self._call_api([virtual_passage], input_type=input_type)
        if vectors:
            h = self._get_hash(virtual_passage)
            self._save_cache_batch([(h, vectors[0])])
            return vectors[0]
        return [0.0] * self.dimension

    def embed_query(self, query):
        """Generate embedding vector for a search query."""
        return self.embed_text(query, is_query=True)

    def embed_chunks(self, chunks, batch_size=EmbeddingConfig.BATCH_SIZE):
        """
        Embed a list of chunks, attaching dense vector to each chunk.
        Preserves 100% of chunk content in chunk['content'].
        """
        if not chunks:
            return []
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        from agent_banking.chunking.universal.planning.schema import require_index_ready

        require_index_ready(chunks)

        self.last_cache_hits = 0
        self.last_cache_misses = 0

        # Step 1: Format virtual passage and compute hash
        passages = []
        hashes = []
        for c in chunks:
            vp = self.format_virtual_passage(c)
            passages.append(vp)
            hashes.append(self._get_hash(vp))

        # Step 2: Query batch from SQLite cache
        cached_map = self._get_cached_vectors_batch(hashes)

        to_fetch_indices = []
        to_fetch_passages = []
        to_fetch_hashes = []

        for idx, (c, vp, h) in enumerate(zip(chunks, passages, hashes)):
            if h in cached_map:
                c["vector"] = cached_map[h]
                self.last_cache_hits += 1
            else:
                to_fetch_indices.append(idx)
                to_fetch_passages.append(vp)
                to_fetch_hashes.append(h)
                self.last_cache_misses += 1

        # Step 3: Call API for chunks not present in cache
        if to_fetch_passages:
            from itertools import groupby

            indexed = zip(to_fetch_indices, to_fetch_passages, to_fetch_hashes)
            for _, group in groupby(indexed, key=lambda item: chunks[item[0]].get("source")):
                entries = list(group)
                windows = [(index, text, weight) for index, passage, _ in entries
                           for text, weight in self._passage_windows(passage)]
                combined = {index: [0.0] * self.dimension for index, _, _ in entries}
                for i in range(0, len(windows), batch_size):
                    batch = windows[i:i + batch_size]
                    vectors = self._call_api([text for _, text, _ in batch], input_type="passage")
                    if len(vectors) != len(batch):
                        raise ValueError("Embedding response omitted one or more passage windows")
                    for (index, _, weight), vector in zip(batch, vectors):
                        for dimension, value in enumerate(vector):
                            combined[index][dimension] += value * weight
                cache_to_save = []
                for index, _, digest in entries:
                    vector = self._l2_normalize(combined[index])
                    if not any(vector):
                        raise ValueError("Cannot embed a passage without searchable text")
                    chunks[index]["vector"] = vector
                    cache_to_save.append((digest, vector))
                self._save_cache_batch(cache_to_save)

        return chunks

    def embed_documents(self, texts, batch_size=EmbeddingConfig.BATCH_SIZE):
        """Generate vectors for a list of raw text documents."""
        if not texts:
            return []
        chunks = [{"content": t, "chunk_type": "text"} for t in texts]
        self.embed_chunks(chunks, batch_size=batch_size)
        results = []
        for c in chunks:
            v = c.get("vector")
            if v:
                results.append(v)
            else:
                results.append([0.0] * self.dimension)
        return results


# Backward compatibility aliases
BankingEmbeddingPipeline = BankingEmbedder
EmbeddingModel = BankingEmbedder
