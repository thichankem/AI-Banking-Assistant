"""
Hệ thống Phân đoạn Tài liệu dựa trên Cấu trúc (Document Structure-Based Chunking Engine)
Kiến trúc phân đoạn nhận thức cấu trúc và độ dài phục vụ RAG Banking:

Bao gồm 3 chiến lược cốt lõi:
1. Strategy 1: Document_structure-based chunking (Cắt theo cấu trúc sub-section + ngân sách âm tiết).
2. Strategy 2: Hierarchical chunking (Phân đoạn đa tầng Parent - Child với cơ chế roll-up).
3. Strategy 3: Section_length_aware chunking (Nhận thức độ dài mục con, gộp mục ngắn để chống phân mảnh vỡ vụn - Less Fragmentation).

Đạt chuẩn 3 tiêu chí thông tin (Information Criteria):
- Preservation (Bảo toàn ngữ cảnh tiêu đề, breadcrumbs)
- Less Fragmentation (Hạn chế tối đa chunk quá ngắn < 80 từ)
- Correctness (Cắt đúng ranh giới ngữ nghĩa, bảo toàn bảng biểu)
"""

from __future__ import annotations

import unicodedata
from pathlib import Path

from .canonical_resolver import (
    split_sentences_banking,
    group_sentences_banking
)
from .syllable_budget_manager import (
    MIN_SYLLABLES,
    MAX_SYLLABLES
)


class DocumentElement:
    """Đại diện cho một phần tử thị giác/cấu trúc của tài liệu (Visual / Structural Object)."""

    def __init__(
        self,
        element_type: str,  # 'heading', 'text', 'table', 'list_item'
        content: str,
        level: int = 0,
        metadata: dict | None = None
    ):
        self.element_type = element_type
        self.content = content.strip()
        self.level = level
        self.metadata = metadata or {}
        self.syllable_count = len(self.content.split())

    def __repr__(self) -> str:
        return f"<Element {self.element_type} (lvl={self.level}, len={self.syllable_count}): {self.content[:30]}...>"


class SubSection:
    """Đại diện cho một phân đoạn cấu trúc gồm tiêu đề và các phần tử nội dung trực thuộc."""

    def __init__(self, title: str, level: int, breadcrumb: str):
        self.title = title
        self.level = level
        self.breadcrumb = breadcrumb
        self.elements: list[DocumentElement] = []

    def total_syllables(self) -> int:
        """Tổng số âm tiết của toàn bộ phần tử trong phân đoạn."""
        return sum(el.syllable_count for el in self.elements)


def _build_structure_chunk(
    chunk_id: str,
    strategy: str,
    source: str,
    chunk_type: str,
    breadcrumb: str,
    section_title: str,
    content: str,
    raw_content: str | None = None,
    hierarchy_level: str | None = None,
    parent_id: str | None = None
) -> dict:
    """Helper chuẩn hóa schema cho chunk cấu trúc."""
    chunk: dict = {
        "chunk_id": chunk_id,
        "strategy": strategy,
        "source": source,
        "chunk_type": chunk_type,
        "breadcrumb": breadcrumb,
        "section_title": section_title,
        "content": content,
        "syllable_count": len(content.split()),
        "word_count": len(content.split())
    }
    if hierarchy_level:
        chunk["hierarchy_level"] = hierarchy_level
    if parent_id:
        chunk["parent_id"] = parent_id
    return chunk


class DocumentStructureChunker:
    """
    Bộ phân đoạn tài liệu dựa trên cấu trúc chuyên sâu cho tài liệu ngân hàng.
    """

    def __init__(
        self,
        chunk_size: int = MAX_SYLLABLES,        # Ngân sách âm tiết mục tiêu cho mỗi chunk (400)
        min_section_size: int = MIN_SYLLABLES,  # Ngưỡng tối thiểu để kích hoạt Section_length_aware gộp mục (100)
        **kwargs
    ):
        self.chunk_size = chunk_size
        self.min_section_size = min_section_size

    # -------------------------------------------------------------------------
    # BƯỚC 1: PARSING VĂN BẢN THÀNH DOCUMENT STRUCTURE & SUB-SECTIONS
    # -------------------------------------------------------------------------

    def parse_structure(self, text: str, source_name: str = "doc") -> list[SubSection]:
        """Chuyển đổi văn bản cấu trúc phân cấp thành danh sách SubSection có quan hệ thứ bậc."""
        text = unicodedata.normalize("NFC", text)
        lines = text.split("\n")

        active_headers: dict[int, str] = {}
        sub_sections: list[SubSection] = []

        # Section khởi đầu nếu chưa có tiêu đề
        current_section = SubSection(title=Path(source_name).stem, level=0, breadcrumb=Path(source_name).stem)
        sub_sections.append(current_section)

        def get_current_breadcrumb() -> str:
            sorted_lvls = sorted(active_headers.keys())
            titles = [active_headers[lvl] for lvl in sorted_lvls if active_headers[lvl]]
            if titles:
                return " > ".join(titles)
            return Path(source_name).stem

        i = 0
        total_lines = len(lines)
        while i < total_lines:
            line = lines[i]
            stripped = line.strip()

            if not stripped:
                i += 1
                continue

            # 1. Phát hiện Tiêu đề (# H1, ## H2, ...)
            if stripped.startswith("#"):
                hash_part = stripped.split()[0]
                if all(c == "#" for c in hash_part) and len(hash_part) <= 6:
                    level = len(hash_part)
                    title = stripped[level:].strip()

                    active_headers[level] = title
                    # Xóa các cấp con sâu hơn
                    for lvl in [l for l in active_headers.keys() if l > level]:
                        del active_headers[lvl]

                    breadcrumb = get_current_breadcrumb()
                    current_section = SubSection(title=title, level=level, breadcrumb=breadcrumb)
                    sub_sections.append(current_section)
                    i += 1
                    continue

            # 2. Phát hiện Bảng biểu phân tách cột
            if stripped.startswith("|") and stripped.endswith("|") and i + 1 < total_lines:
                next_stripped = lines[i + 1].strip()
                if next_stripped.startswith("|") and next_stripped.endswith("|") and "-" in next_stripped:
                    table_lines = [line]
                    i += 1
                    while i < total_lines and lines[i].strip().startswith("|") and lines[i].strip().endswith("|"):
                        table_lines.append(lines[i])
                        i += 1

                    table_content = "\n".join(table_lines)
                    current_section.elements.append(
                        DocumentElement(element_type="table", content=table_content, level=current_section.level)
                    )
                    continue

            # 3. Văn bản thường / Danh sách
            current_section.elements.append(
                DocumentElement(element_type="text", content=line, level=current_section.level)
            )
            i += 1

        # Lọc bỏ các sub_section rỗng không có element
        return [s for s in sub_sections if s.elements]

    parse_markdown_to_structure = parse_structure

    # -------------------------------------------------------------------------
    # STRATEGY 1: DOCUMENT_STRUCTURE-BASED CHUNKING
    # -------------------------------------------------------------------------

    def chunk_strategy_1_structure_based(self, sub_sections: list[SubSection], source_name: str) -> list[dict]:
        """
        Chiến lược 1:
        Cắt theo cấu trúc sub-section với ngân sách âm tiết chunk_size.
        """
        chunks: list[dict] = []
        chunk_idx = 0

        for section in sub_sections:
            current_elements: list[DocumentElement] = []
            current_syllables = 0

            def emit_chunk(elems: list[DocumentElement], chunk_type: str = "text"):
                nonlocal chunk_idx
                if not elems:
                    return
                combined_text = "\n".join(e.content for e in elems).strip()
                if not combined_text:
                    return

                full_content = f"[{section.breadcrumb}]\n{combined_text}"
                chunks.append(_build_structure_chunk(
                    chunk_id=f"{Path(source_name).stem}_s1_{chunk_idx}",
                    strategy="strategy_1_structure_based",
                    source=source_name,
                    chunk_type=chunk_type,
                    breadcrumb=section.breadcrumb,
                    section_title=section.title,
                    content=full_content,
                    raw_content=combined_text
                ))
                chunk_idx += 1

            for el in section.elements:
                # Nếu phần tử là bảng: flush chunk hiện tại và phát hành riêng chunk bảng
                if el.element_type == "table":
                    if current_elements:
                        emit_chunk(current_elements)
                        current_elements = []
                        current_syllables = 0
                    emit_chunk([el], chunk_type="table")
                    continue

                # Nếu phần tử quá dài vượt quá chunk_size
                if el.syllable_count > self.chunk_size:
                    if current_elements:
                        emit_chunk(current_elements)
                        current_elements = []
                        current_syllables = 0

                    sentences = split_sentences_banking(el.content)
                    sub_parts = group_sentences_banking(sentences, max_words=self.chunk_size)
                    for part in sub_parts:
                        sub_el = DocumentElement(element_type="text", content=part)
                        emit_chunk([sub_el])
                    continue

                # Nếu thêm vào vẫn nhỏ hơn ngân sách chunk_size
                if current_syllables + el.syllable_count <= self.chunk_size:
                    current_elements.append(el)
                    current_syllables += el.syllable_count
                else:
                    # Vượt quá ngân sách -> yield chunk hiện tại và mở chunk mới
                    emit_chunk(current_elements)
                    current_elements = [el]
                    current_syllables = el.syllable_count

            if current_elements:
                emit_chunk(current_elements)

        return chunks

    # -------------------------------------------------------------------------
    # STRATEGY 3: SECTION_LENGTH_AWARE CHUNKING (LESS FRAGMENTATION)
    # -------------------------------------------------------------------------

    def chunk_strategy_3_section_length_aware(self, sub_sections: list[SubSection], source_name: str) -> list[dict]:
        """
        Chiến lược 3 (Section_length_aware):
        Tự động nhận thức độ dài của từng sub-section. Nếu các sub-section con
        quá ngắn (< min_section_size), gộp chúng lại với nhau để triệt tiêu
        hiện tượng phân mảnh vụn vặt (Less Fragmentation), đảm bảo chunk đạt
        độ sâu ngữ nghĩa tối ưu.
        """
        # 1. Giai đoạn tiền xử lý: Gộp các sub-section ngắn cùng phân cấp
        merged_sections: list[SubSection] = []
        buffer_section: SubSection | None = None

        for sec in sub_sections:
            if buffer_section is None:
                buffer_section = sec
                continue

            # Kiểm tra xem có thể gộp sec vào buffer_section không
            # Điều kiện: Tổng âm tiết < chunk_size VÀ buffer_section quá ngắn
            can_merge = (
                (buffer_section.total_syllables() < self.min_section_size or sec.total_syllables() < self.min_section_size)
                and (buffer_section.total_syllables() + sec.total_syllables() <= self.chunk_size)
            )

            if can_merge:
                # Gộp elements và mở rộng breadcrumb
                buffer_section.elements.extend(sec.elements)
                if sec.title not in buffer_section.breadcrumb:
                    buffer_section.breadcrumb = f"{buffer_section.breadcrumb} & {sec.title}"
            else:
                merged_sections.append(buffer_section)
                buffer_section = sec

        if buffer_section:
            merged_sections.append(buffer_section)

        # 2. Sau khi đã gộp chống phân mảnh, áp dụng cấu trúc sinh chunk
        return self.chunk_strategy_1_structure_based(merged_sections, source_name)

    # -------------------------------------------------------------------------
    # STRATEGY 2: HIERARCHICAL CHUNKING (PARENT - CHILD LINKING)
    # -------------------------------------------------------------------------

    def chunk_strategy_2_hierarchical(self, sub_sections: list[SubSection], source_name: str) -> list[dict]:
        """
        Chiến lược 2 (Hierarchical Chunking):
        Sinh song song:
        - Child Chunks: Chi tiết từng đoạn phục vụ tìm kiếm chính xác (High Precision).
        - Parent Chunks: Bao quát toàn bộ sub-section lớn để Roll-up khi retrieve đủ ngưỡng.
        Liên kết giữa Child và Parent qua `parent_id`.
        """
        all_chunks: list[dict] = []
        parent_idx = 0
        child_idx = 0

        for sec in sub_sections:
            parent_id = f"{Path(source_name).stem}_parent_{parent_idx}"
            parent_content = "\n".join(e.content for e in sec.elements).strip()

            # Tạo Child Chunks từ section này
            child_chunks = self.chunk_strategy_1_structure_based([sec], source_name)
            child_ids: list[str] = []

            for c in child_chunks:
                c_id = f"{Path(source_name).stem}_child_{child_idx}"
                c["chunk_id"] = c_id
                c["strategy"] = "strategy_2_hierarchical"
                c["hierarchy_level"] = "child"
                c["parent_id"] = parent_id
                child_ids.append(c_id)
                all_chunks.append(c)
                child_idx += 1

            # Chỉ lưu parent_chunk nếu section có nhiều hơn 1 child chunk (để phục vụ roll-up)
            if len(child_chunks) > 1:
                parent_chunk = _build_structure_chunk(
                    chunk_id=parent_id,
                    strategy="strategy_2_hierarchical",
                    source=source_name,
                    chunk_type="parent_overview",
                    breadcrumb=sec.breadcrumb,
                    section_title=sec.title,
                    content=f"[{sec.breadcrumb} - TỔNG QUAN MỤC]\n{parent_content}",
                    raw_content=parent_content,
                    hierarchy_level="parent"
                )
                parent_chunk["child_chunk_ids"] = child_ids
                all_chunks.append(parent_chunk)
                parent_idx += 1

        return all_chunks
