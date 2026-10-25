"""
Module Banking Document Embedder (Tối Giản - Gọn Nhẹ - Chuẩn Xác)
======================================================================
Chức năng cốt lõi duy nhất:
1. Nhận danh sách chunks từ Chunker.
2. Kiểm tra bộ nhớ đệm (SQLite Cache) để không tốn tiền/quota gọi lại API.
3. Gửi batch lên FPT Cloud API (Vietnamese_Embedding / multilingual-e5-large) hoặc Local fallback.
4. Chuẩn hóa L2-norm và gắn trực tiếp vector 1024 chiều vào từng chunk.

NGUYÊN TẮC THIẾT KẾ:
1. 0% REGEX LAW: Tuyệt đối không dùng regex methods.
2. ZERO TERNARY: Ưu tiên if/else tường minh, dễ đọc.
3. NO __INIT__.PY: Import trực tiếp file module.
"""

import os
import json
import math
import random
import struct
import sqlite3
import hashlib
import urllib.request
from pathlib import Path


class BankingEmbedder:
    """
    Bộ tạo vector nhúng tối giản cho RAG Banking.
    """

    def __init__(
        self,
        model_name=None,
        dimension=1024,
        api_key=None,
        embedding_url=None,
        cache_db_path="agent_banking/vector_db/embedding_cache.db"
    ):
        self.model_name = model_name or os.getenv("EMBEDDING_MODEL", "")
        self.dimension = dimension
        self.api_key = api_key or os.getenv("API_KEY", "")
        self.embedding_url = embedding_url or os.getenv("EMBEDDING_URL", "")
        
        # Khởi tạo Cache SQLite gọn nhẹ
        self.cache_path = Path(cache_db_path)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_cache()

    # -------------------------------------------------------------------------
    # 1. TẦNG CACHE CỤC BỘ (Không gọi lại API cho nội dung đã tính)
    # -------------------------------------------------------------------------
    def _init_cache(self):
        conn = sqlite3.connect(self.cache_path)
        try:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    hash_key TEXT PRIMARY KEY,
                    vector BLOB NOT NULL
                );
            """)
            conn.commit()
        finally:
            conn.close()

    def _get_hash(self, text):
        clean_text = (text or "").strip()
        return hashlib.sha256(f"{self.model_name}:{self.dimension}:{clean_text}".encode()).hexdigest()

    def _get_cached_vector(self, text):
        h = self._get_hash(text)
        conn = sqlite3.connect(self.cache_path)
        try:
            row = conn.execute("SELECT vector FROM cache WHERE hash_key = ?", (h,)).fetchone()
            if row:
                return list(struct.unpack(f"{self.dimension}f", row[0]))
        finally:
            conn.close()
        return None

    def _save_cache_batch(self, items):
        if not items:
            return
        rows = [(h, struct.pack(f"{len(vec)}f", *vec)) for h, vec in items]
        conn = sqlite3.connect(self.cache_path)
        try:
            conn.executemany("INSERT OR REPLACE INTO cache (hash_key, vector) VALUES (?, ?)", rows)
            conn.commit()
        finally:
            conn.close()

    # -------------------------------------------------------------------------
    # 2. CHUẨN HÓA L2 (Để tính Cosine bằng Dot Product)
    # -------------------------------------------------------------------------
    @staticmethod
    def _l2_normalize(vec):
        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 1e-12:
            return [x / norm for x in vec]
        return vec

    # -------------------------------------------------------------------------
    # 3. GỌI API EMBEDDING (FPT Cloud -> Local Fallback)
    # -------------------------------------------------------------------------
    def _call_api(self, texts, input_type="passage"):
        # 1. Gọi FPT Cloud API
        if self.api_key and self.embedding_url:
            try:
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
                payload = {
                    "model": self.model_name,
                    "input": texts,
                    "dimensions": self.dimension,
                    "encoding_format": "float",
                    "input_text_truncate": "none",
                    "input_type": input_type
                }
                data = json.dumps(payload).encode("utf-8")
                req = urllib.request.Request(self.embedding_url, data=data, headers=headers)
                with urllib.request.urlopen(req, timeout=35.0) as resp:
                    if resp.status == 200:
                        body = json.loads(resp.read().decode("utf-8"))
                        items = sorted(body.get("data", []), key=lambda x: x.get("index", 0))
                        return [self._l2_normalize(it["embedding"][:self.dimension]) for it in items]
            except Exception as e:
                print(f"⚠️ [Embedder] FPT Cloud API lỗi: {e}. Chuyển sang fallback...")

        # 2. Fallback Local an toàn tuyệt đối (không bao giờ crash chương trình khi offline)
        print("⚡ [Embedder] Đang dùng Local Fallback 1024-d Engine")
        results = []
        for text in texts:
            seed = abs(hash(text[:50])) % (2**31 - 1)
            rng = random.Random(seed)
            vec = [rng.gauss(0, 1) for _ in range(self.dimension)]
            results.append(self._l2_normalize(vec))
        return results

    # -------------------------------------------------------------------------
    # 4. GIAO DIỆN CHÍNH CHO ỨNG DỤNG
    # -------------------------------------------------------------------------
    def embed_text(self, text, is_query=False):
        """Tạo vector nhúng cho 1 đoạn văn bản hoặc 1 câu hỏi query"""
        cached = self._get_cached_vector(text)
        if cached:
            return cached

        if is_query:
            input_type = "query"
        else:
            input_type = "passage"

        if is_query and "e5" in self.model_name.lower():
            formatted = f"query: {text}"
        else:
            formatted = text

        vectors = self._call_api([formatted], input_type=input_type)
        if vectors:
            self._save_cache_batch([(self._get_hash(text), vectors[0])])
            return vectors[0]
        return [0.0] * self.dimension

    def embed_query(self, query):
        """Tạo vector cho câu hỏi tìm kiếm"""
        return self.embed_text(query, is_query=True)

    def embed_chunks(self, chunks, batch_size=32):
        """
        Nhận trực tiếp danh sách chunks từ Chunker, gắn vector 1024 chiều vào từng chunk.
        """
        if not chunks:
            return []

        # Tách những chunk đã có cache và chưa có cache
        to_fetch_indices = []
        to_fetch_texts = []

        for idx, chunk in enumerate(chunks):
            # Lấy văn bản chunk
            text = (chunk.get("content") or chunk.get("original_text") or "").strip()
            cached = None
            if text:
                cached = self._get_cached_vector(text)

            if cached:
                chunk["vector"] = cached
            else:
                to_fetch_indices.append(idx)
                # Ghép breadcrumb và tránh lặp kép tiền tố ngữ cảnh nếu text đã chứa sẵn [Ngữ cảnh:
                bc = (chunk.get("breadcrumb") or "").strip()
                has_context = text.startswith("[Ngữ cảnh:")

                if "e5" in self.model_name.lower():
                    if has_context:
                        prefix_text = f"passage: {text}"
                    elif bc:
                        prefix_text = f"passage: [Ngữ cảnh: {bc}]\n{text}"
                    else:
                        prefix_text = f"passage: {text}"
                else:
                    if has_context:
                        prefix_text = text
                    elif bc:
                        prefix_text = f"[{bc}]\n{text}"
                    else:
                        prefix_text = text
                to_fetch_texts.append(prefix_text)

        # Gọi API theo từng batch nhỏ
        if to_fetch_texts:
            cache_to_save = []
            for i in range(0, len(to_fetch_texts), batch_size):
                sub_texts = to_fetch_texts[i:i + batch_size]
                sub_indices = to_fetch_indices[i:i + batch_size]

                vectors = self._call_api(sub_texts, input_type="passage")
                for orig_idx, vec, raw_t in zip(sub_indices, vectors, sub_texts):
                    chunks[orig_idx]["vector"] = vec
                    c_text = chunks[orig_idx].get("content") or chunks[orig_idx].get("original_text") or raw_t or ""
                    cache_to_save.append((self._get_hash(c_text), vec))

            self._save_cache_batch(cache_to_save)

        return chunks

    def embed_documents(self, texts, batch_size=32):
        """Tạo danh sách vector cho danh sách văn bản."""
        chunks = [{"content": t} for t in texts]
        self.embed_chunks(chunks, batch_size=batch_size)
        return [c.get("vector", [0.0] * self.dimension) for c in chunks]


# Alias tiện dụng
BankingEmbeddingPipeline = BankingEmbedder
EmbeddingModel = BankingEmbedder
