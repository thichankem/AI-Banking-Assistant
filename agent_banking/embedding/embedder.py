"""
Module Banking Document Embedder (Tối Giản - Gọn Nhẹ - Chuẩn Xác)
======================================================================
Chức năng cốt lõi duy nhất:
1. Nhận danh sách chunks từ Chunker (chuẩn 13 trường Universal Schema).
2. Định dạng Type-Aware Virtual Passage (multilingual-e5-large) cho text, table, multimodal.
3. Kiểm tra bộ nhớ đệm (SQLite WAL Cache) trên chính Virtual Passage đã format.
4. Gửi batch lên FPT Cloud API (multilingual-e5-large); lỗi API được báo rõ.
5. Chuẩn hóa L2-norm và gắn trực tiếp vector 1024 chiều vào từng chunk.

NGUYÊN TẮC THIẾT KẾ (GEMINI.md & AGENTS.md):
1. 0% REGEX LAW: Tuyệt đối không dùng regex methods hay import re.
2. ZERO TERNARY LAW: Bắt buộc dùng if/else tường minh từng dòng, không dùng ?: hay toán tử 3 ngôi.
3. NO TYPING LIBRARY: Không import typing hay from typing import ...
4. NO __INIT__.PY: Import trực tiếp file module.
"""

import math
import struct
import sqlite3
import hashlib
from pathlib import Path
from agent_banking.config.models import model_primary

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class BankingEmbedder:
    """
    Bộ tạo vector nhúng tối giản và hiệu năng cao cho RAG Banking.
    """

    def __init__(
        self,
        model_name=None,
        dimension=None,
        api_key=None,
        embedding_url=None,
        cache_db_path="agent_banking/vector_db/embedding_cache.db"
    ):
        primary = model_primary("embedding")
        self.model_name = model_name or primary.model
        self.dimension = primary.dimension
        if dimension is not None:
            self.dimension = dimension
        self.api_key = primary.api_key
        if api_key is not None:
            self.api_key = api_key
        self.embedding_url = embedding_url or primary.url

        # Giải quyết đường dẫn cache SQLite tuyệt đối chuẩn xác theo project root
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
    # 1. TẦNG CACHE CỤC BỘ SQLITE WAL (Xác định & Bất biến)
    # -------------------------------------------------------------------------
    def _init_cache(self):
        """Khởi tạo bảng cache SQLite với các PRAGMA WAL tối ưu hiệu năng cao."""
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
        """Tính SHA-256 xác định trên chính Virtual Passage gửi tới mô hình."""
        if virtual_passage is not None:
            clean_text = str(virtual_passage).strip()
        else:
            clean_text = ""
        key_material = f"cloud_verified_v2:{self.embedding_url}:{self.model_name}:{self.dimension}:{clean_text}"
        return hashlib.sha256(key_material.encode("utf-8")).hexdigest()

    def _get_cached_vector(self, virtual_passage):
        """
        Lấy vector 1024-d từ cache theo chuỗi virtual passage.
        Hỗ trợ backwards compatibility: nếu đầu vào là raw text chưa có tiền tố passage:,
        sẽ tự động tra cứu fallback với tiền tố 'passage: '.
        Ngược lại, nếu đầu vào có tiền tố 'passage: ', sẽ thử tra cứu fallback với chuỗi thô.
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

            # Fallback 1: Caller gọi bằng raw text chưa có tiền tố 'passage: '
            if not clean_str.startswith("passage:"):
                h_fallback = self._get_hash(f"passage: {clean_str}")
                row_fb = conn.execute("SELECT vector FROM cache WHERE hash_key = ?", (h_fallback,)).fetchone()
                if row_fb:
                    return list(struct.unpack(f"{self.dimension}f", row_fb[0]))
            else:
                # Fallback 2: Caller gọi bằng 'passage: ...' nhưng cache lưu raw text
                raw_fallback = clean_str[8:].strip()
                h_raw = self._get_hash(raw_fallback)
                row_raw = conn.execute("SELECT vector FROM cache WHERE hash_key = ?", (h_raw,)).fetchone()
                if row_raw:
                    return list(struct.unpack(f"{self.dimension}f", row_raw[0]))
        finally:
            conn.close()
        return None

    def _get_cached_vectors_batch(self, hashes):
        """Truy vấn hàng loạt vector từ SQLite WAL theo danh sách hash key."""
        if not hashes:
            return {}
        result = {}
        conn = sqlite3.connect(self.cache_path, timeout=30.0)
        try:
            conn.execute("PRAGMA busy_timeout=30000;")
            batch_size = 500
            for i in range(0, len(hashes), batch_size):
                sub_hashes = hashes[i:i + batch_size]
                placeholders = ",".join("?" for _ in sub_hashes)
                cursor = conn.execute(
                    f"SELECT hash_key, vector FROM cache WHERE hash_key IN ({placeholders})",
                    sub_hashes
                )
                for h_key, blob in cursor.fetchall():
                    result[h_key] = list(struct.unpack(f"{self.dimension}f", blob))
        finally:
            conn.close()
        return result

    def _save_cache_batch(self, items):
        """Lưu hàng loạt vector nhị phân 4096-byte BLOB vào SQLite WAL."""
        if not items:
            return
        rows = []
        for h, vec in items:
            if len(vec) == self.dimension:
                blob = struct.pack(f"{self.dimension}f", *vec)
                rows.append((h, blob))
        if not rows:
            return
        conn = sqlite3.connect(self.cache_path, timeout=30.0)
        try:
            conn.execute("PRAGMA busy_timeout=30000;")
            conn.executemany("INSERT OR REPLACE INTO cache (hash_key, vector) VALUES (?, ?)", rows)
            conn.commit()
        finally:
            conn.close()

    @staticmethod
    def _extract_image_descriptions(images_list):
        """Trích xuất chuỗi chú thích / mô tả từ danh sách hình ảnh."""
        if not isinstance(images_list, list):
            return ""
        img_descs = []
        for img in images_list:
            if isinstance(img, dict):
                caption = str(img.get("caption") or "").strip()
                desc = str(img.get("description") or "").strip()
                if caption:
                    if desc:
                        if caption != desc:
                            img_descs.append(f"Chú thích: {caption} | Mô tả: {desc}")
                        else:
                            img_descs.append(f"Chú thích: {caption}")
                    else:
                        img_descs.append(f"Chú thích: {caption}")
                elif desc:
                    img_descs.append(f"Mô tả: {desc}")
        if len(img_descs) > 0:
            return "; ".join(img_descs)
        return ""

    def format_virtual_passage(self, chunk):
        """
        Định dạng đoạn văn ảo (Virtual Passage) cho mô hình multilingual-e5-large.
        Hỗ trợ phân loại type-aware theo 3 loại chunk: 'text', 'table', 'multimodal'.
        Bảo toàn 100% nội dung gốc trong chunk['content'] (không bao giờ thay đổi chunk['content']).
        Chống lặp kép tiền tố [Ngữ cảnh: và tiền tố passage:.
        Tuân thủ tuyệt đối: 0% regex, không dùng typing, zero ternary (if/else tường minh).
        """
        if not isinstance(chunk, dict):
            return ""

        metadata = chunk.get("business_metadata")
        if not isinstance(metadata, dict):
            metadata = {}
        if metadata.get("schema_version") == "2.0":
            from agent_banking.chunking.universal.passage import format_passage
            return format_passage(chunk)

        # Lấy nội dung thô từ content hoặc original_text
        raw_content = chunk.get("content")
        if raw_content is None:
            raw_content = chunk.get("original_text")
        if raw_content is None:
            raw_content = ""
        raw_content = str(raw_content).strip()

        # Loại bỏ tiền tố passage: và các tiền tố ngữ cảnh nếu đã có sẵn (tránh lặp kép)
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
                    end_idx = clean_content.find("]\n")
                    if end_idx != -1:
                        clean_content = clean_content[end_idx + 2:].strip()
                        changed = True
                    else:
                        end_crlf_idx = clean_content.find("]\r\n")
                        if end_crlf_idx != -1:
                            clean_content = clean_content[end_crlf_idx + 3:].strip()
                            changed = True
                        else:
                            bracket_idx = clean_content.find("]")
                            if bracket_idx != -1:
                                clean_content = clean_content[bracket_idx + 1:].strip()
                                changed = True

        bc = str(chunk.get("breadcrumb") or "").strip()
        sec_title = str(chunk.get("section_title") or "").strip()

        # Xác định loại chunk (chunk_type)
        raw_c_type = str(chunk.get("chunk_type") or "").strip().lower()
        images_list = chunk.get("images")
        if images_list is None:
            images_list = []
        elif not isinstance(images_list, list):
            images_list = []

        table_data = chunk.get("table_data")

        # Phân nhánh theo loại chunk
        if raw_c_type in ("text", "paragraph", "heading", "sentence", "body"):
            c_type = "text"
        elif "table" in raw_c_type or raw_c_type in ("data_table", "grid", "spreadsheet") or bool(table_data):
            c_type = "table"
        elif "multimodal" in raw_c_type or raw_c_type in ("image", "figure", "chart", "diagram", "picture", "caption") or bool(images_list):
            c_type = "multimodal"
        else:
            c_type = "text"

        # ---------------------------------------------------------------------
        # Nhánh 1: Bảng biểu (Table)
        # Template: passage: [Bảng số liệu: {section_title} | Ngữ cảnh: {breadcrumb}]\n{content}
        # ---------------------------------------------------------------------
        if c_type == "table":
            header_parts = []
            if sec_title:
                header_parts.append(f"Bảng số liệu: {sec_title}")
            else:
                header_parts.append("Bảng số liệu")

            if bc:
                header_parts.append(f"Ngữ cảnh: {bc}")

            header_str = " | ".join(header_parts)

            # Trích xuất cấu trúc cột từ table_data nếu có
            col_header_str = ""
            if isinstance(table_data, dict):
                headers = table_data.get("headers")
                if isinstance(headers, list):
                    if len(headers) > 0:
                        non_empty = [str(h).strip() for h in headers if str(h).strip()]
                        if len(non_empty) > 0:
                            col_header_str = "Các cột: " + ", ".join(non_empty)
            elif isinstance(table_data, list):
                if len(table_data) > 0:
                    first_row = table_data[0]
                    if isinstance(first_row, list):
                        if len(first_row) > 0:
                            non_empty = [str(h).strip() for h in first_row if str(h).strip()]
                            if len(non_empty) > 0:
                                col_header_str = "Các cột: " + ", ".join(non_empty)

            if col_header_str:
                if col_header_str not in clean_content:
                    if clean_content:
                        body_content = f"{col_header_str}\n{clean_content}"
                    else:
                        body_content = col_header_str
                else:
                    body_content = clean_content
            else:
                body_content = clean_content

            # Làm giàu chú thích ảnh từ images nếu có (bảng biểu có hình minh họa/sơ đồ)
            img_desc_str = self._extract_image_descriptions(images_list)
            if img_desc_str:
                if img_desc_str not in body_content:
                    if body_content:
                        body_content = f"{body_content}\n[Chú thích/Mô tả: {img_desc_str}]"
                    else:
                        body_content = f"[Chú thích/Mô tả: {img_desc_str}]"

            if body_content:
                passage = f"passage: [{header_str}]\n{body_content}"
            else:
                passage = f"passage: [{header_str}]"

        # ---------------------------------------------------------------------
        # Nhánh 2: Đa phương thức (Multimodal / Images)
        # Template: passage: [Sơ đồ/Ảnh: {section_title} | Ngữ cảnh: {breadcrumb}]\n{content}
        # ---------------------------------------------------------------------
        elif c_type == "multimodal":
            header_parts = []
            if sec_title:
                header_parts.append(f"Sơ đồ/Ảnh: {sec_title}")
            else:
                header_parts.append("Sơ đồ/Ảnh")

            if bc:
                header_parts.append(f"Ngữ cảnh: {bc}")

            header_str = " | ".join(header_parts)

            # Trích xuất chú thích ảnh / mô tả từ images nếu có
            img_desc_str = self._extract_image_descriptions(images_list)

            body_content = clean_content
            if img_desc_str:
                if img_desc_str not in clean_content:
                    if body_content:
                        body_content = f"{body_content}\n[Chú thích/Mô tả: {img_desc_str}]"
                    else:
                        body_content = f"[Chú thích/Mô tả: {img_desc_str}]"

            if body_content:
                passage = f"passage: [{header_str}]\n{body_content}"
            else:
                passage = f"passage: [{header_str}]"

        # ---------------------------------------------------------------------
        # Nhánh 3: Văn bản chuẩn (Text)
        # Template: passage: [Ngữ cảnh: {breadcrumb} | Mục: {section_title}]\n{content}
        # ---------------------------------------------------------------------
        else:
            header_parts = []
            if bc:
                header_parts.append(f"Ngữ cảnh: {bc}")
            if sec_title:
                header_parts.append(f"Mục: {sec_title}")

            body_content = clean_content

            # Làm giàu ngữ cảnh từ parent_content nếu súc tích (<= 200 ký tự) và khác content
            raw_parent = chunk.get("parent_content")
            if raw_parent is not None:
                clean_parent = str(raw_parent).strip()
                changed_p = True
                while changed_p:
                    changed_p = False
                    if clean_parent.startswith("passage:"):
                        clean_parent = clean_parent[8:].strip()
                        changed_p = True
                    else:
                        has_p_prefix = False
                        if clean_parent.startswith("[Ngữ cảnh:"):
                            has_p_prefix = True
                        elif clean_parent.startswith("[Bảng số liệu:"):
                            has_p_prefix = True
                        elif clean_parent.startswith("[Sơ đồ/Ảnh:"):
                            has_p_prefix = True

                        if has_p_prefix:
                            end_idx = clean_parent.find("]\n")
                            if end_idx != -1:
                                clean_parent = clean_parent[end_idx + 2:].strip()
                                changed_p = True
                            else:
                                end_crlf_idx = clean_parent.find("]\r\n")
                                if end_crlf_idx != -1:
                                    clean_parent = clean_parent[end_crlf_idx + 3:].strip()
                                    changed_p = True
                                else:
                                    bracket_idx = clean_parent.find("]")
                                    if bracket_idx != -1:
                                        clean_parent = clean_parent[bracket_idx + 1:].strip()
                                        changed_p = True

                if clean_parent:
                    if clean_parent != clean_content:
                        if clean_parent != sec_title:
                            if len(clean_parent) <= 200:
                                if clean_parent not in clean_content:
                                    if body_content:
                                        body_content = f"{body_content}\nBối cảnh: {clean_parent}"
                                    else:
                                        body_content = f"Bối cảnh: {clean_parent}"

            if len(header_parts) > 0:
                header_str = " | ".join(header_parts)
                if body_content:
                    passage = f"passage: [{header_str}]\n{body_content}"
                else:
                    passage = f"passage: [{header_str}]"
            else:
                if body_content:
                    passage = f"passage: {body_content}"
                else:
                    passage = "passage: "

        return passage

    # -------------------------------------------------------------------------
    # 3. CHUẨN HÓA L2 (Để tính Cosine bằng Dot Product)
    # -------------------------------------------------------------------------
    @staticmethod
    def _l2_normalize(vec):
        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 1e-12:
            return [x / norm for x in vec]
        return vec

    # -------------------------------------------------------------------------
    # 4. GỌI API EMBEDDING (FPT Cloud -> Local Fallback)
    # -------------------------------------------------------------------------
    def _call_api(self, texts, input_type="passage"):
        if not texts:
            return []
        self._validate_passages(texts)
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

    def _validate_passages(self, texts):
        if "e5" in self.model_name.lower():
            from agent_banking.chunking.universal.token_budget import get_tokenizer
            tokenizer = get_tokenizer()
            counts = [len(tokenizer.encode(str(text)).ids) for text in texts]
            if any(count > 512 for count in counts):
                raise ValueError("E5 passage exceeds 512 tokens; rechunk before embedding (max=" + str(max(counts)) + ")")

    # -------------------------------------------------------------------------
    # 5. GIAO DIỆN CHÍNH CHO ỨNG DỤNG
    # -------------------------------------------------------------------------
    def embed_text(self, text, is_query=False):
        """Tạo vector nhúng cho 1 đoạn văn bản hoặc 1 câu hỏi query."""
        if text is None:
            text = ""

        if is_query:
            input_type = "query"
            if "e5" in self.model_name.lower():
                if text.startswith("query: "):
                    virtual_passage = text
                else:
                    virtual_passage = f"query: {text}"
            else:
                virtual_passage = text
        else:
            input_type = "passage"
            chunk_mock = {"content": text, "chunk_type": "text"}
            virtual_passage = self.format_virtual_passage(chunk_mock)

        self._validate_passages([virtual_passage])
        cached = self._get_cached_vector(virtual_passage)
        if cached:
            return cached

        vectors = self._call_api([virtual_passage], input_type=input_type)
        if vectors:
            h = self._get_hash(virtual_passage)
            self._save_cache_batch([(h, vectors[0])])
            return vectors[0]
        raise RuntimeError("Embedding API returned no vector")

    def embed_query(self, query):
        """Tạo vector cho câu hỏi tìm kiếm."""
        return self.embed_text(query, is_query=True)

    def embed_chunks(self, chunks, batch_size=32):
        """
        Nhận danh sách chunks từ Chunker, gắn vector 1024 chiều vào từng chunk.
        Bảo toàn 100% nội dung gốc trong chunk['content'] (không bao giờ thay đổi chunk['content']).
        """
        if not chunks:
            return []

        self.last_cache_hits = 0
        self.last_cache_misses = 0

        # Bước 1: Format virtual passage và tính hash cho từng chunk
        passages = []
        hashes = []
        for c in chunks:
            metadata = c.get("business_metadata")
            if not isinstance(metadata, dict):
                metadata = {}
            if metadata.get("quality", {}).get("index_ready") is False:
                raise ValueError("Chunk has unresolved source/image-context assessment; embedding refused")
            vp = self.format_virtual_passage(c)
            passages.append(vp)
            hashes.append(self._get_hash(vp))

        self._validate_passages(passages)

        # Bước 2: Truy vấn batch từ SQLite cache
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

        # Bước 3: Gọi API cho những chunk chưa có trong cache
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
        """Tạo danh sách vector cho danh sách văn bản."""
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


# Alias tiện dụng
BankingEmbeddingPipeline = BankingEmbedder
EmbeddingModel = BankingEmbedder
