"""
Module Syllable Budget Manager (Chuẩn hóa Đơn vị Âm tiết Tiếng Việt & Phân tách Động)
========================================================================================
Quản lý ngân sách âm tiết tiếng Việt chuẩn hóa cố định (MIN=100, MAX=400, OVERLAP=50, CEILING=500).
Tích hợp:
1. Phân tách đàn hồi (Elastic Window Splitting [250, 420]): Tìm ranh giới ngắt câu/đoạn
   tối ưu nhất dựa trên trọng số dấu câu (\\n\\n, . ? !, ; :, ,).
2. Tiền tố ngữ cảnh (Contextual Header Prepending): Tự động gắn [Ngữ cảnh: {breadcrumb} > {section_title}]
   vào đầu nội dung để neo thực thể/đại từ nhân xưng cho Dense & BM25 embeddings.
3. Hợp đồng 7 trường (7-Field Schema Contract): Đảm bảo tuyệt đối 100% chunk sinh ra
   chỉ chứa đúng và đủ 7 khóa.
4. 0% REGEX LAW: Toàn bộ xử lý bằng String & Unicode properties.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

try:
    from rag_banking.chunking.unified_chunk.canonical_resolver import split_sentences_banking
except (ImportError, ValueError):
    from .canonical_resolver import split_sentences_banking


# =============================================================================
# BỘ HẰNG SỐ NGÂN SÁCH ÂM TIẾT CỐ ĐỊNH CHO TOÀN BỘ HỆ THỐNG
# =============================================================================
MIN_SYLLABLES: int = 100
MAX_SYLLABLES: int = 400
OVERLAP_SYLLABLES: int = 0  # Tuyệt đối không overlap content (0% content overlap)
CEILING_SYLLABLES: int = 500


def prepend_context_header(content: str, breadcrumb: str, section_title: str) -> str:
    """
    Tự động chèn tiền tố ngữ cảnh: [Ngữ cảnh: {breadcrumb} > {section_title}]
    vào dòng đầu tiên của nội dung chunk (0% Regex).
    """
    clean_bc = breadcrumb.strip()
    clean_st = section_title.strip()
    clean_content = content.strip()

    prefix = ""
    if clean_bc and clean_st and clean_st != clean_bc:
        prefix = f"[Ngữ cảnh: {clean_bc} > {clean_st}]\n"
    elif clean_bc:
        prefix = f"[Ngữ cảnh: {clean_bc}]\n"
    elif clean_st:
        prefix = f"[Ngữ cảnh: {clean_st}]\n"

    # Tránh chèn trùng lặp nếu nội dung đã có tiền tố ngữ cảnh
    if prefix and clean_content.startswith("[Ngữ cảnh:"):
        return clean_content

    return prefix + clean_content


def build_7_field_chunk(
    chunk_id: str,
    source: str,
    chunk_type: str,
    breadcrumb: str,
    section_title: str,
    content: str,
    word_count: Optional[int] = None
) -> dict:
    """
    Hàm khởi tạo duy nhất tuân thủ tuyệt đối Hợp đồng 7 trường (7-Field Schema Contract):
    `chunk_id`, `source`, `chunk_type`, `breadcrumb`, `section_title`, `content`, `word_count`.
    """
    final_content = prepend_context_header(content, breadcrumb, section_title)
    return {
        "chunk_id": chunk_id,
        "source": source,
        "chunk_type": chunk_type,
        "breadcrumb": breadcrumb,
        "section_title": section_title,
        "content": final_content,
        "word_count": word_count if word_count is not None else len(final_content.split())
    }


class SyllableBudgetManager:
    """
    Trình quản lý và kiểm định ngân sách âm tiết văn bản tiếng Việt (0% Regex).
    """

    def __init__(
        self,
        min_syllables: int = MIN_SYLLABLES,
        max_syllables: int = MAX_SYLLABLES,
        overlap_syllables: int = OVERLAP_SYLLABLES,
        ceiling_syllables: int = CEILING_SYLLABLES,
        **kwargs
    ):
        self.min_syllables = min_syllables
        self.max_syllables = max_syllables
        self.overlap_syllables = overlap_syllables
        self.ceiling_syllables = ceiling_syllables

    def count_syllables(self, text: str) -> int:
        """Đếm số âm tiết / từ thuần túy không dùng regex."""
        return len(text.split())

    def evaluate_budget_status(self, text: str) -> dict:
        """
        Đánh giá trạng thái độ dài của văn bản so với ngân sách âm tiết chuẩn.
        """
        syllables = self.count_syllables(text)
        if syllables < self.min_syllables:
            status = "under_budget"
        elif syllables <= self.max_syllables:
            status = "golden_zone"
        else:
            status = "over_budget"

        return {
            "syllables": syllables,
            "status": status,
            "min_syllables": self.min_syllables,
            "max_syllables": self.max_syllables,
            "ceiling_syllables": self.ceiling_syllables
        }

    def find_optimal_elastic_split_point(
        self,
        words: list[str],
        window_start: int = 250,
        window_end: int = 420
    ) -> int:
        """
        Tìm ranh giới ngắt đàn hồi tối ưu nhất trong cửa sổ [250, 420] từ
        dựa trên thang điểm trọng số dấu câu (0% Regex):
        - Đoạn xuống dòng kép (\\n\\n): 10.0 điểm
        - Dấu ngắt câu dứt điểm (. ! ?): 8.0 điểm
        - Dấu ngắt ý chính (; :): 5.0 điểm
        - Dấu phẩy (,): 2.0 điểm
        - Khoảng trắng thông thường: 1.0 điểm
        """
        total_words = len(words)
        if total_words <= window_end:
            return total_words

        effective_start = min(window_start, total_words - 1)
        effective_end = min(window_end, total_words)

        best_idx = effective_end
        best_score = -1.0

        for idx in range(effective_start, effective_end):
            w = words[idx]
            score = 1.0

            if "\n\n" in w:
                score = 10.0
            elif any(w.endswith(punct) for punct in (".", "!", "?")):
                score = 8.0
            elif any(w.endswith(punct) for punct in (";", ":")):
                score = 5.0
            elif w.endswith(","):
                score = 2.0

            # Ưu tiên các điểm ngắt gần biên 350-380 từ hơn
            position_bias = 1.0 - (abs(idx - 350) / 350.0) * 0.2
            total_candidate_score = score * position_bias

            if total_candidate_score > best_score:
                best_score = total_candidate_score
                best_idx = idx + 1

        return best_idx

    def split_text_elastic(
        self,
        text: str,
        window_start: int = 250,
        window_end: int = 420,
        ceiling: int = 450,
        overlap: int = 0
    ) -> list[str]:
        """
        Cắt văn bản dài thành các đoạn đạt chuẩn bằng cửa sổ ngắt đàn hồi.
        Không cắt thô bạo tại 400 mà tìm điểm ngắt câu tự nhiên trong [250, 420].
        Trần tuyệt đối không vượt quá ceiling (450 âm tiết).
        Tuyệt đối không overlap nội dung giữa các chunk (0% content overlap, overlap=0).
        """
        words = text.split()
        if len(words) <= ceiling:
            return [text.strip()] if text.strip() else []

        chunks: list[str] = []
        start_idx = 0

        while start_idx < len(words):
            remaining_words = len(words) - start_idx
            if remaining_words <= ceiling:
                chunk_words = words[start_idx:]
                chunks.append(" ".join(chunk_words))
                break

            # Cửa sổ tìm kiếm
            win_s = min(window_start, remaining_words - 1)
            win_e = min(window_end, remaining_words)

            slice_words = words[start_idx:start_idx + win_e + 20]
            relative_split_point = self.find_optimal_elastic_split_point(
                slice_words,
                window_start=win_s,
                window_end=win_e
            )

            # Đảm bảo không vượt quá trần ceiling
            relative_split_point = min(relative_split_point, ceiling)
            end_idx = start_idx + relative_split_point

            chunk_text = " ".join(words[start_idx:end_idx]).strip()
            if chunk_text:
                chunks.append(chunk_text)

            # Tiến index không gối đầu nội dung (0% content overlap)
            if overlap <= 0:
                next_start = end_idx
            else:
                next_start = end_idx - overlap
                if next_start <= start_idx:
                    next_start = end_idx
            start_idx = next_start

        return chunks

    def split_sentences_by_syllable_budget(
        self,
        sentences: list[str],
        max_syllables: Optional[int] = None,
        overlap_syllables: Optional[int] = None
    ) -> list[str]:
        """
        Gom nhóm các câu thành các đoạn văn đạt ngân sách âm tiết mục tiêu kèm gối đầu (overlap).
        """
        target_max = max_syllables or self.max_syllables
        target_overlap = overlap_syllables or self.overlap_syllables

        if not sentences:
            return []

        full_text = " ".join(sentences)
        return self.split_text_elastic(
            text=full_text,
            window_start=250,
            window_end=target_max,
            ceiling=self.ceiling_syllables,
            overlap=target_overlap
        )


__all__ = [
    "MIN_SYLLABLES",
    "MAX_SYLLABLES",
    "OVERLAP_SYLLABLES",
    "CEILING_SYLLABLES",
    "SyllableBudgetManager",
    "prepend_context_header",
    "build_7_field_chunk"
]
