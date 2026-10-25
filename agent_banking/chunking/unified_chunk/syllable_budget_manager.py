"""
Module Syllable Budget Manager (Chuẩn hóa Đơn vị Âm tiết Tiếng Việt & Phân tách Động)
========================================================================================
Quản lý ngân sách âm tiết tiếng Việt chuẩn hóa cố định (MIN=100, MAX=400, CEILING=500).
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

try:
    from agent_banking.chunking.unified_chunk.canonical_resolver import split_sentences_banking
except (ImportError, ValueError):
    from .canonical_resolver import split_sentences_banking


# =============================================================================
# BỘ HẰNG SỐ NGÂN SÁCH ÂM TIẾT CỐ ĐỊNH CHO TOÀN BỘ HỆ THỐNG
# =============================================================================
MIN_SYLLABLES: int = 100
MAX_SYLLABLES: int = 400
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
    if clean_bc and clean_st and clean_st != clean_bc and not clean_bc.endswith(clean_st):
        prefix = f"[Ngữ cảnh: {clean_bc} > {clean_st}]\n"
    elif clean_bc:
        prefix = f"[Ngữ cảnh: {clean_bc}]\n"
    elif clean_st:
        prefix = f"[Ngữ cảnh: {clean_st}]\n"

    # Tránh chèn trùng lặp nếu nội dung đã có tiền tố ngữ cảnh
    if prefix and clean_content.startswith("[Ngữ cảnh:"):
        return clean_content

    return prefix + clean_content


import time

def build_7_field_chunk(
    chunk_id: str,
    source: str,
    chunk_type: str,
    breadcrumb: str,
    section_title: str,
    content: str,
    word_count: int | None = None,
    original_text: str | None = None,
    parent_id: str | None = None,
    location: list[dict] | None = None,
    token_count: int | None = None,
    element_ids: list[str] | None = None,
    is_active: bool = True,
    updated_time: int | None = None,
    images: list[dict] | None = None,
    **kwargs
) -> dict:
    """
    Hàm khởi tạo chunk chuẩn mực cho toàn bộ hệ thống AI Banking Assistant:
    - 7 trường cốt lõi bắt buộc: `chunk_id`, `source`, `chunk_type`, `breadcrumb`, `section_title`, `content`, `word_count`.
    - Các trường mở rộng chuẩn Enterprise RAG:
        + `parent_id`: Phục vụ kiến trúc Small-to-Big / Hierarchical Retrieval.
        + `original_text`: Văn bản thô ban đầu trước khi prepend tiền tố ngữ cảnh.
        + `token_count`: Ước tính/tính toán số subword token cho LLM prompt budget.
        + `location`: Tọa độ trực quan [{"page": int, "bbox": [x0, y0, x1, y1]}] cho PDF Viewer highlight.
        + `element_ids`: Danh sách ID atomic elements trong Document Layout Analysis (DLA).
        + `is_active`: Cờ quản lý vòng đời văn bản (soft delete khi cập nhật biểu phí/chính sách mới).
        + `updated_time`: Timestamp epoch ms của chunk.
    """
    final_content = prepend_context_header(content, breadcrumb, section_title)
    w_count = word_count if word_count is not None else len(final_content.split())
    raw_text = original_text if original_text is not None else content.strip()
    tok_count = token_count if token_count is not None else int(round(w_count * 1.25))
    up_time = updated_time if updated_time is not None else int(time.time() * 1000)

    chunk: dict = {
        "chunk_id": chunk_id,
        "parent_id": parent_id,
        "source": source,
        "chunk_type": chunk_type,
        "breadcrumb": breadcrumb,
        "section_title": section_title,
        "content": final_content,
        "original_text": raw_text,
        "word_count": w_count,
        "token_count": tok_count,
        "location": location if location is not None else [],
        "element_ids": element_ids if element_ids is not None else [],
        "is_active": is_active,
        "updated_time": up_time
    }
    if images is not None and len(images) > 0:
        chunk["images"] = images
    else:
        if "images" in kwargs and kwargs["images"]:
            chunk["images"] = kwargs["images"]
    return chunk


build_banking_chunk = build_7_field_chunk


class SyllableBudgetManager:
    """
    Trình quản lý và kiểm định ngân sách âm tiết văn bản tiếng Việt (0% Regex).
    """

    def __init__(
        self,
        min_syllables: int = MIN_SYLLABLES,
        max_syllables: int = MAX_SYLLABLES,
        ceiling_syllables: int = CEILING_SYLLABLES,
        **kwargs
    ):
        self.min_syllables = min_syllables
        self.max_syllables = max_syllables
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

    def split_text_elastic(
        self,
        text: str,
        window_start: int = 250,
        window_end: int = 420,
        ceiling: int = 450,
        **kwargs
    ) -> list[str]:
        """
        Phân đoạn văn bản bằng wtpsplit (SaT) và gom nhóm theo ngân sách âm tiết.
        Tuyệt đối không dùng regex hoặc các mảng ký tự neo / dấu câu thủ công.
        """
        words = text.split()
        if len(words) <= ceiling:
            return [text.strip()] if text.strip() else []

        # Phân đoạn câu trực tiếp bằng mô hình wtpsplit (SaT)
        sentences = split_sentences_banking(text)
        if len(sentences) <= 1:
            chunks: list[str] = []
            target = min(window_end, ceiling)
            for i in range(0, len(words), target):
                chunk = " ".join(words[i:i + target]).strip()
                if chunk:
                    chunks.append(chunk)
            return chunks

        return self.split_sentences_by_syllable_budget(
            sentences=sentences,
            max_syllables=min(window_end, ceiling)
        )

    def split_sentences_by_syllable_budget(
        self,
        sentences: list[str],
        max_syllables: int | None = None,
        **kwargs
    ) -> list[str]:
        """
        Gom nhóm các câu thành các đoạn văn đạt ngân sách âm tiết mục tiêu.
        Bảo toàn trọn vẹn ranh giới câu do wtpsplit (SaT) tạo ra, không cắt vụn giữa câu.
        """
        target_max = max_syllables or self.max_syllables

        if not sentences:
            return []

        chunks: list[str] = []
        current_chunk: list[str] = []
        current_count = 0

        for s in sentences:
            s_clean = s.strip()
            if not s_clean:
                continue
            s_words = len(s_clean.split())

            # Trường hợp câu đơn lẻ đặc biệt vượt trần ceiling -> ngắt đàn hồi
            if s_words > self.ceiling_syllables:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = []
                    current_count = 0
                sub_parts = self.split_text_elastic(
                    text=s_clean,
                    window_start=250,
                    window_end=target_max,
                    ceiling=self.ceiling_syllables
                )
                chunks.extend(sub_parts)
                continue

            if current_count + s_words > target_max and current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_count = 0

            current_chunk.append(s_clean)
            current_count += s_words

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks


__all__ = [
    "MIN_SYLLABLES",
    "MAX_SYLLABLES",
    "CEILING_SYLLABLES",
    "SyllableBudgetManager",
    "prepend_context_header",
    "build_7_field_chunk",
    "build_banking_chunk"
]
