"""
Module Syllable Budget Manager (Chuẩn hóa Đơn vị Âm tiết Tiếng Việt & Cấu hình Tham số)
========================================================================================
Giải quyết bài toán: Sự lệch đơn vị giữa DOCX (min_section_size = 120 từ, max_chunk_size = 800 ký tự)
và Structure Chunker (120 âm tiết, 400 âm tiết).

Chuẩn hóa 100%:
- Quy đổi TOÀN BỘ điều kiện đo lường kích thước văn bản về một đơn vị duy nhất:
  `count_syllables_vietnamese` (0% Regex Engine - Pure String Tokenization).
- Thiết lập 3 Profile chuẩn hóa phục vụ Grid Search:
  1. Profile A (Ngắn / Precision / QA FAQ): (min = 100, max = 200 âm tiết).
  2. Profile B (Trung bình / Pháp lý / Ngân hàng): (min = 150, max = 400 âm tiết).
  3. Profile C (Parent-Child Roll-up): Chunk con 200 âm tiết liên kết tới Chunk cha 700 âm tiết qua parent_id.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

try:
    from rag_banking.chunking.unified_chunk.canonical_resolver import (
        count_syllables_vietnamese,
        count_words,
        split_sentences_banking
    )
except (ImportError, ValueError):
    from ...unified_chunk.canonical_resolver import (
        count_syllables_vietnamese,
        count_words,
        split_sentences_banking
    )


@dataclass(frozen=True)
class SyllableProfile:
    """Cấu hình tham số ngân sách âm tiết cho một chiến lược phân đoạn."""
    profile_id: str
    name: str
    min_syllables: int
    max_syllables: int
    overlap_syllables: int
    is_hierarchical: bool = False
    parent_min_syllables: int = 400
    parent_max_syllables: int = 700
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "name": self.name,
            "min_syllables": self.min_syllables,
            "max_syllables": self.max_syllables,
            "overlap_syllables": self.overlap_syllables,
            "is_hierarchical": self.is_hierarchical,
            "parent_min_syllables": self.parent_min_syllables if self.is_hierarchical else None,
            "parent_max_syllables": self.parent_max_syllables if self.is_hierarchical else None,
            "description": self.description
        }


# =============================================================================
# 3 PROFILE CHUẨN ĐƯỢC THIẾT LẬP CHO HỆ THỐNG NGÂN HÀNG
# =============================================================================

PROFILE_A = SyllableProfile(
    profile_id="profile_a",
    name="Profile A (Ngắn - Precision / QA FAQ)",
    min_syllables=100,
    max_syllables=200,
    overlap_syllables=30,
    is_hierarchical=False,
    description="Tối ưu cho việc truy xuất câu hỏi ngắn, tra cứu định nghĩa, biểu phí và hạn mức chính xác."
)

PROFILE_B = SyllableProfile(
    profile_id="profile_b",
    name="Profile B (Trung bình - Pháp lý / Ngân hàng)",
    min_syllables=150,
    max_syllables=400,
    overlap_syllables=50,
    is_hierarchical=False,
    description="Cấu hình vàng cho điều khoản hợp đồng tín dụng, nghị quyết HĐQT, quy tắc bảo hiểm."
)

PROFILE_C = SyllableProfile(
    profile_id="profile_c",
    name="Profile C (Parent-Child Roll-up)",
    min_syllables=80,
    max_syllables=200,
    overlap_syllables=30,
    is_hierarchical=True,
    parent_min_syllables=400,
    parent_max_syllables=700,
    description="Chunk con 200 âm tiết phục vụ retrieve chính xác, liên kết chunk cha 700 âm tiết để roll-up ngữ cảnh."
)

STANDARD_PROFILES: dict[str, SyllableProfile] = {
    "profile_a": PROFILE_A,
    "profile_b": PROFILE_B,
    "profile_c": PROFILE_C
}


class SyllableBudgetManager:
    """
    Trình quản lý và kiểm định ngân sách âm tiết văn bản tiếng Việt (0% Regex).
    """

    def __init__(self, profile: SyllableProfile = PROFILE_B):
        self.profile = profile

    def count_syllables(self, text: str) -> int:
        """Đếm số âm tiết tiếng Việt thuần túy không dùng regex."""
        return count_syllables_vietnamese(text)

    def evaluate_budget_status(self, text: str) -> dict[str, Any]:
        """
        Đánh giá trạng thái độ dài của văn bản so với ngân sách của Profile hiện tại:
        - 'under_budget': < min_syllables (nguy cơ phân mảnh).
        - 'golden_zone': min_syllables <= count <= max_syllables (vùng lý tưởng).
        - 'over_budget': > max_syllables (cần cắt nhỏ có gối đầu).
        """
        syllables = self.count_syllables(text)
        if syllables < self.profile.min_syllables:
            status = "under_budget"
        elif syllables <= self.profile.max_syllables:
            status = "golden_zone"
        else:
            status = "over_budget"

        return {
            "syllables": syllables,
            "status": status,
            "profile_id": self.profile.profile_id,
            "min_syllables": self.profile.min_syllables,
            "max_syllables": self.profile.max_syllables
        }

    def split_sentences_by_syllable_budget(
        self,
        sentences: list[str],
        max_syllables: int | None = None,
        overlap_syllables: int | None = None
    ) -> list[str]:
        """
        Gom nhóm các câu thành các đoạn văn đạt ngân sách âm tiết mục tiêu kèm gối đầu (overlap).
        """
        target_max = max_syllables or self.profile.max_syllables
        target_overlap = overlap_syllables or self.profile.overlap_syllables

        if not sentences:
            return []

        chunks: list[str] = []
        current_sentences: list[str] = []
        current_syllables = 0

        for sent in sentences:
            sent_syl = self.count_syllables(sent)
            if not sent_syl:
                continue

            # Nếu một câu đơn lẻ vượt quá ngân sách max_syllables: chia cứng theo từ
            if sent_syl > target_max:
                if current_sentences:
                    chunks.append(" ".join(current_sentences))
                    current_sentences = []
                    current_syllables = 0

                words = sent.split()
                sub_words: list[str] = []
                for w in words:
                    sub_words.append(w)
                    if len(sub_words) >= target_max:
                        chunks.append(" ".join(sub_words))
                        # Giữ lại overlap
                        sub_words = sub_words[-target_overlap:] if target_overlap > 0 else []
                if sub_words:
                    chunks.append(" ".join(sub_words))
                continue

            # Nếu thêm câu này vào không vượt trần
            if current_syllables + sent_syl <= target_max:
                current_sentences.append(sent)
                current_syllables += sent_syl
            else:
                # Phát hành chunk hiện tại
                chunks.append(" ".join(current_sentences))

                # Lùi lại các câu cuối để tạo overlap âm tiết
                overlap_collector: list[str] = []
                overlap_count = 0
                for prev_sent in reversed(current_sentences):
                    ps_syl = self.count_syllables(prev_sent)
                    if overlap_count + ps_syl <= target_overlap:
                        overlap_collector.insert(0, prev_sent)
                        overlap_count += ps_syl
                    else:
                        break

                current_sentences = overlap_collector + [sent]
                current_syllables = sum(self.count_syllables(s) for s in current_sentences)

        if current_sentences:
            chunks.append(" ".join(current_sentences))

        return chunks
