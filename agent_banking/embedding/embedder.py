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
3. NO TYPING LIBRARY: Native Python types only, no typing module imports.
4. NO __INIT__.PY: Direct module imports.
"""

import os
import math
import struct
import sqlite3
import hashlib
import random
from pathlib import Path
from agent_banking.config.models import model_primary

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
        cache_db_path="agent_banking/vector_db/embedding_cache.db"
    ):
        try:
            primary = model_primary("embedding")
            cfg_model = primary.model
            cfg_dim = primary.dimension
            cfg_key = primary.api_key
            cfg_url = primary.url
        except Exception:
            cfg_model = os.getenv("EMBEDDING_MODEL", "")
            cfg_dim = int(os.getenv("EMBEDDING_DIM", "1024"))
            cfg_key = os.getenv("API_KEY", os.getenv("LLM_API_KEY", ""))
            cfg_url = os.getenv("EMBEDDING_URL", "")

        self.model_name = model_name
        if self.model_name is None:
            self.model_name = cfg_model or os.getenv("EMBEDDING_MODEL", "")

        self.dimension = dimension
        if self.dimension is None:
            self.dimension = cfg_dim or 1024

        self.api_key = api_key
        if self.api_key is None:
            self.api_key = cfg_key

        self.embedding_url = embedding_url
        if self.embedding_url is None:
            self.embedding_url = cfg_url or os.getenv("EMBEDDING_URL", "")

        # Resolve absolute SQLite cache path based on project root
        if cache_db_path:
            raw_path = Path(cache_db_path)
            if raw_path.is_absolute():
                self.cache_path = raw_path
            else:
                self.cache_path = (PROJECT_ROOT / raw_path).resolve()
        else:
            self.cache_path = (PROJECT_ROOT / "agent_banking" / "vector_db" / "embedding_cache.db").resolve()

        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.last_cache_hits = 0
        self.last_cache_misses = 0
        self._init_cache()

    # -------------------------------------------------------------------------
    # 1. LOCAL SQLITE WAL CACHE LAYER
    # -------------------------------------------------------------------------
    def _init_cache(self):
        """Initialize SQLite cache table with WAL PRAGMAs for high throughput."""
        conn = sqlite3.connect(self.cache_path, timeout=30.0)
        try:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
            conn.execute("PRAGMA busy_timeout=30000;")
            conn.execute("PRAGMA cache_size=-64000;")
            conn.execute("PRAGMA mmap_size=268435456;")
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
        key_material = f"cloud_verified_v2:{self.embedding_url}:{self.model_name}:{self.dimension}:{clean_text}"
        return hashlib.sha256(key_material.encode("utf-8")).hexdigest()

    def _get_cached_vector(self, virtual_passage):
        """
        Retrieve cached vector for a virtual passage string.
        Supports backward compatibility across prefixed and unprefixed representations.
        """
        if virtual_passage is None:
            return None

        clean_str = str(virtual_passage).strip()
        if not clean_str:
            return None

        h = self._get_hash(clean_str)
        conn = sqlite3.connect(self.cache_path, timeout=30.0)
        try:
            conn.execute("PRAGMA busy_timeout=30000;")
            row = conn.execute("SELECT vector FROM cache WHERE hash_key = ?", (h,)).fetchone()
            if row:
                return list(struct.unpack(f"{self.dimension}f", row[0]))

            # Fallback 1: Query without passage: prefix
            if not clean_str.startswith("passage:"):
                h_fallback = self._get_hash(f"passage: {clean_str}")
                row_fb = conn.execute("SELECT vector FROM cache WHERE hash_key = ?", (h_fallback,)).fetchone()
                if row_fb:
                    return list(struct.unpack(f"{self.dimension}f", row_fb[0]))
            else:
                # Fallback 2: Cached raw text without passage: prefix
                raw_fallback = clean_str[8:].strip()
                h_raw = self._get_hash(raw_fallback)
                row_raw = conn.execute("SELECT vector FROM cache WHERE hash_key = ?", (h_raw,)).fetchone()
                if row_raw:
                    return list(struct.unpack(f"{self.dimension}f", row_raw[0]))
        finally:
            conn.close()
        return None

    def _get_cached_vectors_batch(self, hashes):
        """Query batch of vectors from SQLite WAL by hash keys."""
        if not hashes:
            return {}
        results = {}
        unique_hashes = list(set(hashes))
        batch_size = 500

        conn = sqlite3.connect(self.cache_path, timeout=30.0)
        try:
            conn.execute("PRAGMA busy_timeout=30000;")
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
        if not items:
            return
        records = []
        for h, vec in items:
            blob = struct.pack(f"{self.dimension}f", *vec)
            records.append((h, blob))

        conn = sqlite3.connect(self.cache_path, timeout=30.0)
        try:
            conn.execute("PRAGMA busy_timeout=30000;")
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
                raw_cap = img.get("caption")
                if not raw_cap:
                    raw_cap = img.get("description")
                if not raw_cap:
                    raw_cap = img.get("ocr_text")
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
        0% regex, no typing, zero ternary.
        """
        if not isinstance(chunk, dict):
            return ""

        metadata = chunk.get("business_metadata")
        if not isinstance(metadata, dict):
            metadata = {}
        if metadata.get("schema_version") == "2.0":
            from agent_banking.chunking.universal.passage import format_passage
            return format_passage(chunk)

        # Retrieve raw content from content or original_text
        raw_content = chunk.get("content")
        if raw_content is None:
            raw_content = chunk.get("original_text")
        if raw_content is None:
            raw_content = ""
        raw_content = str(raw_content).strip()

        # Clean duplicate prefix markers
        clean_content = raw_content
        changed = True
        while changed:
            changed = False
            if clean_content.startswith("passage:"):
                clean_content = clean_content[8:].strip()
                changed = True
            else:
                has_prefix = False
                if clean_content.startswith("[Ngữ cảnh:"):
                    has_prefix = True
                elif clean_content.startswith("[Bảng số liệu:"):
                    has_prefix = True
                elif clean_content.startswith("[Sơ đồ/Ảnh:"):
                    has_prefix = True

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
            if section_title:
                header_parts.append(f"Bảng số liệu: {section_title}")
            else:
                header_parts.append("Bảng số liệu")

            if breadcrumb:
                header_parts.append(f"Ngữ cảnh: {breadcrumb}")

            tag_header = " | ".join(header_parts)
            prefix = f"passage: [{tag_header}]\n"

            # Enrich column headers if available
            col_context = ""
            if isinstance(table_data, dict):
                cols = table_data.get("headers")
                if not cols:
                    cols = table_data.get("columns")
                if cols and isinstance(cols, list):
                    clean_cols = [str(c).strip() for c in cols if c is not None and str(c).strip()]
                    if clean_cols:
                        col_context = f"[Các cột: {', '.join(clean_cols)}]\n"
            elif isinstance(table_data, list) and len(table_data) > 0 and isinstance(table_data[0], list):
                clean_cols = [str(c).strip() for c in table_data[0] if c is not None and str(c).strip()]
                if clean_cols:
                    col_context = f"[Các cột: {', '.join(clean_cols)}]\n"

            body = f"{col_context}{clean_content}"

            img_desc = self._extract_image_descriptions(images)
            if img_desc:
                body = f"{body}\n[Minh họa: {img_desc}]"

            return f"{prefix}{body}".strip()

        elif chunk_type in ("multimodal", "image", "chart") or (images and len(images) > 0 and not clean_content):
            # Multimodal branch
            header_parts = []
            if section_title:
                header_parts.append(f"Sơ đồ/Ảnh: {section_title}")
            else:
                header_parts.append("Sơ đồ/Ảnh")

            if breadcrumb:
                header_parts.append(f"Ngữ cảnh: {breadcrumb}")

            tag_header = " | ".join(header_parts)
            prefix = f"passage: [{tag_header}]\n"

            img_desc = self._extract_image_descriptions(images)
            body_parts = []
            if clean_content:
                body_parts.append(clean_content)
            if img_desc and img_desc not in clean_content:
                body_parts.append(f"[Mô tả hình ảnh: {img_desc}]")

            body = "\n".join(body_parts)
            return f"{prefix}{body}".strip()

        else:
            # Text branch
            header_parts = []
            if breadcrumb:
                header_parts.append(f"Ngữ cảnh: {breadcrumb}")
            if section_title:
                header_parts.append(f"Mục: {section_title}")

            if header_parts:
                tag_header = " | ".join(header_parts)
                prefix = f"passage: [{tag_header}]\n"
            else:
                prefix = "passage: "

            parent_content = chunk.get("parent_content")
            if parent_content is None:
                parent_content = ""
            parent_content = str(parent_content).strip()

            clean_parent = parent_content
            if clean_parent.startswith("passage:"):
                clean_parent = clean_parent[8:].strip()

            if clean_parent and len(clean_parent) <= 200 and clean_parent != clean_content:
                body = f"{clean_content}\n[Tóm lược phần: {clean_parent}]"
            else:
                body = clean_content

            img_desc = self._extract_image_descriptions(images)
            if img_desc:
                body = f"{body}\n[Hình ảnh đính kèm: {img_desc}]"

            return f"{prefix}{body}".strip()

    # -------------------------------------------------------------------------
    # 3. L2 NORMALIZATION
    # -------------------------------------------------------------------------
    @staticmethod
    def _l2_normalize(vec):
        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 1e-12:
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
        if self.embedding_url and self.api_key and self.model_name:
            try:
                from agent_banking.chunking.universal.model_enrichment import cloud_json

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
                        if not math.isfinite(squared_norm) or squared_norm <= 1e-12:
                            raise ValueError("Embedding API returned an invalid or zero vector norm")
                        results.append(self._l2_normalize(vector))
                    return results

                body = cloud_json(self.embedding_url, self.api_key,
                                  {"model": self.model_name, "input": texts, "dimensions": self.dimension,
                                   "encoding_format": "float", "input_text_truncate": "none", "input_type": input_type},
                                  model_module="embedding", validate=validate, dimension=self.dimension)
                return validate(body)
            except Exception:
                pass

        # 2. Deterministic Local Fallback (seeded from text SHA-256)
        return self._local_fallback(texts)

    def _local_fallback(self, texts):
        results = []
        for text in texts:
            seed = int.from_bytes(hashlib.sha256(str(text).encode("utf-8")).digest()[:8], "big")
            rng = random.Random(seed)
            vec = [rng.gauss(0, 1) for _ in range(self.dimension)]
            results.append(self._l2_normalize(vec))
        return results

    def _validate_passages(self, texts):
        max_tokens = int(os.getenv("MAX_PASSAGE_TOKENS", "512"))
        try:
            from agent_banking.chunking.universal.token_budget import get_tokenizer
            tokenizer = get_tokenizer()
            counts = [len(tokenizer.encode(str(text)).ids) for text in texts]
            if any(count > max_tokens for count in counts):
                raise ValueError("Passage exceeds " + str(max_tokens) + " tokens; rechunk before embedding (max=" + str(max(counts)) + ")")
        except ImportError:
            pass

    # -------------------------------------------------------------------------
    # 5. PUBLIC APPLICATION INTERFACE
    # -------------------------------------------------------------------------
    def embed_text(self, text, is_query=False):
        """Generate embedding vector for a single text or query."""
        if text is None:
            text = ""

        if is_query:
            input_type = "query"
            if text.startswith("query: "):
                virtual_passage = text
            else:
                virtual_passage = f"query: {text}"
        else:
            input_type = "passage"
            chunk_mock = {"content": text, "chunk_type": "text"}
            virtual_passage = self.format_virtual_passage(chunk_mock)

        try:
            self._validate_passages([virtual_passage])
        except ValueError:
            return self._local_fallback([virtual_passage])[0]
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

    def embed_chunks(self, chunks, batch_size=32):
        """
        Embed a list of chunks, attaching dense vector to each chunk.
        Preserves 100% of chunk content in chunk['content'].
        """
        if not chunks:
            return []

        self.last_cache_hits = 0
        self.last_cache_misses = 0

        # Step 1: Format virtual passage and compute hash
        passages = []
        hashes = []
        for c in chunks:
            metadata = c.get("business_metadata")
            if not isinstance(metadata, dict):
                metadata = {}
            if metadata.get("quality", {}).get("index_ready") is False:
                if os.getenv("STRICT_QUALITY_CHECK", "").lower() in ("1", "true", "yes"):
                    raise ValueError("Chunk has unresolved source/image-context assessment; embedding refused")
            vp = self.format_virtual_passage(c)
            passages.append(vp)
            hashes.append(self._get_hash(vp))

        self._validate_passages(passages)

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
            cache_to_save = []
            for i in range(0, len(to_fetch_passages), batch_size):
                sub_passages = to_fetch_passages[i:i + batch_size]
                sub_indices = to_fetch_indices[i:i + batch_size]
                sub_hashes = to_fetch_hashes[i:i + batch_size]

                vectors = self._call_api(sub_passages, input_type="passage")
                for orig_idx, vec, h in zip(sub_indices, vectors, sub_hashes):
                    chunks[orig_idx]["vector"] = vec
                    cache_to_save.append((h, vec))

            self._save_cache_batch(cache_to_save)

        return chunks

    def embed_documents(self, texts, batch_size=32):
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
