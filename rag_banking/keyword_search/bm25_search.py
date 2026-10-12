"""
Giải thuật BM25 Okapi (Keyword Search)
Phục vụ tìm kiếm từ khóa chính xác, mã sản phẩm, số liệu, tên gói vay, biểu phí ngân hàng.
"""

import math
import re

class BM25Search:
    def __init__(self, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.chunks = []
        self.corpus_size = 0
        self.tokenized_corpus = []
        self.doc_lengths = []
        self.avg_doc_len = 0.0
        self.idf = {}

    def _tokenize(self, text):
        """Tách từ đơn giản, chuyển chữ thường và lọc ký tự đặc biệt"""
        return re.findall(r"\w+", text.lower())

    def index(self, chunks):
        """
        Lập chỉ mục BM25 cho danh sách chunks tài liệu.
        """
        self.chunks = chunks
        self.corpus_size = len(chunks)
        self.tokenized_corpus = []
        self.doc_lengths = []

        if self.corpus_size == 0:
            self.avg_doc_len = 0.0
            self.idf = {}
            return

        for chunk in chunks:
            # Ưu tiên tìm kiếm trên full content (gồm cả breadcrumb + text/table)
            text = chunk.get("content", "")
            tokens = self._tokenize(text)
            self.tokenized_corpus.append(tokens)
            self.doc_lengths.append(len(tokens))

        self.avg_doc_len = sum(self.doc_lengths) / self.corpus_size

        # Tính toán Document Frequency (df) cho mỗi từ
        doc_freq = {}
        for tokens in self.tokenized_corpus:
            for word in set(tokens):
                doc_freq[word] = doc_freq.get(word, 0) + 1

        # Tính toán Inverse Document Frequency (IDF) theo công thức Okapi BM25
        self.idf = {}
        for word, freq in doc_freq.items():
            # Công thức BM25 chuẩn: log(((N - n + 0.5) / (n + 0.5)) + 1)
            score = math.log(((self.corpus_size - freq + 0.5) / (freq + 0.5)) + 1.0)
            self.idf[word] = max(score, 0.0001)

    def search(self, query, top_k=5):
        """
        Tìm kiếm các chunks phù hợp nhất với câu truy vấn theo điểm BM25.
        Trả về: list các tuple (chunk, bm25_score)
        """
        if self.corpus_size == 0:
            return []

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        scores = [0.0] * self.corpus_size

        for i, doc_tokens in enumerate(self.tokenized_corpus):
            doc_len = self.doc_lengths[i]
            if doc_len == 0:
                continue

            # Đếm tần suất từ (Term Frequency) trong văn bản
            tf = {}
            for token in doc_tokens:
                tf[token] = tf.get(token, 0) + 1

            doc_score = 0.0
            for q_term in query_tokens:
                if q_term not in tf:
                    continue

                term_freq = tf[q_term]
                idf_val = self.idf.get(q_term, 0.0)

                # Công thức thành phần BM25 Okapi
                numerator = term_freq * (self.k1 + 1.0)
                denominator = term_freq + self.k1 * (1.0 - self.b + self.b * (doc_len / self.avg_doc_len))
                doc_score += idf_val * (numerator / denominator)

            scores[i] = doc_score

        # Lấy top_k tài liệu có điểm số cao nhất
        scored_pairs = list(zip(self.chunks, scores))
        scored_pairs.sort(key=lambda x: x[1], reverse=True)

        return scored_pairs[:top_k]
