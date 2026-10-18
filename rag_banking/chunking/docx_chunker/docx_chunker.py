"""
Bộ phân đoạn tài liệu Microsoft Word (.docx) chuyên dụng cho ngân hàng.
========================================================================
Nâng cấp Kiến trúc Thế hệ mới (Next-Gen Architecture):
1. Tự động nhận diện bảng biểu phức tạp (chứa ô gộp w:gridSpan, w:vMerge hoặc đa tiêu đề).
2. Kết xuất bảng phức tạp sang hình ảnh thị giác và định tuyến qua VLMTableExtractor (Gemma 4:12B Vision).
3. Gom cụm bảng biểu bằng TableFormatter (Stateful Group-Row & Footnote Atomicity Injection).
4. Chuẩn hóa tiền tố [Ngữ cảnh: {breadcrumb} > {section_title}] cho toàn bộ chunk.
5. Tuân thủ tuyệt đối Hợp đồng 7 trường (7-Field Schema Contract) và 0% REGEX LAW.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import docx
from docx.table import Table

logger = logging.getLogger("docx_chunker")

from ..unified_chunk.canonical_resolver import (
    clean_and_dehyphenate_text,
    split_sentences_banking
)
from ..unified_chunk.syllable_budget_manager import (
    SyllableBudgetManager,
    prepend_context_header,
    build_7_field_chunk
)
from ..pdf_chunker.vlm_table_extractor import (
    VLMTableExtractor,
    render_docx_table_to_image
)
from ..pdf_chunker.table_formatter import TableFormatter


def is_complex_docx_table(table: Table) -> bool:
    """
    Kiểm tra xem bảng biểu Word có cấu trúc phức tạp hay không (0% Regex):
    - Chứa ô gộp theo cột (`w:gridSpan`)
    - Chứa ô gộp theo hàng (`w:vMerge`)
    - Có nhiều hơn 1 hàng tiêu đề (hàng 1 và hàng 2 đều in đậm hoặc có nền)
    """
    try:
        xml_str = table._tbl.xml
        if "w:gridSpan" in xml_str or "w:vMerge" in xml_str:
            return True
    except Exception:
        pass

    rows = table.rows
    if len(rows) > 2:
        # Kiểm tra nếu 2 hàng đầu tiên đều có chữ in đậm -> Bảng đa tầng tiêu đề
        try:
            row0_bold = all(
                any(r.bold for p in c.paragraphs for r in p.runs)
                for c in rows[0].cells if c.text.strip()
            )
            row1_bold = all(
                any(r.bold for p in c.paragraphs for r in p.runs)
                for c in rows[1].cells if c.text.strip()
            )
            if row0_bold and row1_bold:
                return True
        except Exception:
            pass

    return False


class DOCXChunker:
    """
    Bộ phân đoạn DOCX nhận thức cấu trúc và bảng biểu ngân hàng thế hệ mới.
    """

    def __init__(
        self,
        vlm_extractor: Optional[VLMTableExtractor] = None,
        table_formatter: Optional[TableFormatter] = None,
        syllable_manager: Optional[SyllableBudgetManager] = None,
        max_chunk_size: int = 800,
        min_section_size: int = 100,
        chunk_overlap: int = 0,
        table_rows_per_chunk: int = 4,
        **kwargs
    ):
        self.vlm_extractor = vlm_extractor or VLMTableExtractor()
        self.table_formatter = table_formatter or TableFormatter()
        self.syllable_manager = syllable_manager or SyllableBudgetManager()
        self.max_chunk_size = max_chunk_size
        self.min_section_size = min_section_size
        self.chunk_overlap = chunk_overlap
        self.table_rows_per_chunk = table_rows_per_chunk

    def chunk_docx(self, docx_path: str | Path) -> list[dict]:
        """
        Đọc tệp DOCX và chuyển đổi thành danh sách các chunk chuẩn 7 trường.
        """
        path = Path(docx_path)
        if not path.exists():
            return []

        try:
            doc = docx.Document(str(path))
        except Exception as e:
            logger.error(f"[DOCXChunker] Không thể đọc tệp {path.name}: {e}")
            return []

        chunks: list[dict] = []
        counter = 0

        # Quản lý tiêu đề và đường dẫn Breadcrumbs
        doc_title = path.stem.replace("_", " ").title()
        current_headings: list[str] = [doc_title]
        current_section = doc_title

        text_buffer: list[str] = []
        buffer_words = 0

        def get_breadcrumb() -> str:
            return " > ".join(current_headings) if current_headings else doc_title

        def flush_text_buffer() -> None:
            nonlocal counter, buffer_words
            if not text_buffer:
                buffer_words = 0
                return

            raw_text = "\n".join(text_buffer).strip()
            text_buffer.clear()
            buffer_words = 0
            if not raw_text:
                return

            clean_text = clean_and_dehyphenate_text(raw_text)
            breadcrumb = get_breadcrumb()

            # Phân tách văn bản bằng SyllableBudgetManager với 0% content overlap (trần sau tiền tố <= 450)
            split_chunks = self.syllable_manager.split_text_elastic(
                text=clean_text,
                window_start=200,
                window_end=380,
                ceiling=420,
                overlap=0
            )

            for group in (split_chunks or [clean_text]):
                if not group.strip():
                    continue
                chunk_obj = build_7_field_chunk(
                    chunk_id=f"{path.stem}_docx_c{counter}",
                    source=path.name,
                    chunk_type="docx_text",
                    breadcrumb=breadcrumb,
                    section_title=current_section,
                    content=group.strip()
                )
                chunks.append(chunk_obj)
                counter += 1

        # Duyệt qua toàn bộ phần tử trong body (theo đúng thứ tự xuất hiện)
        for element in doc.element.body:
            tag = element.tag.split("}")[-1] if "}" in element.tag else element.tag

            if tag == "p":
                # Xử lý đoạn văn
                p = docx.text.paragraph.Paragraph(element, doc)
                text = p.text.strip()
                if not text:
                    continue

                style_name = (p.style.name or "").lower()
                style_id = (getattr(p.style, "style_id", "") or "").lower()
                is_heading = (
                    "heading" in style_name
                    or "heading" in style_id
                    or "title" in style_name
                    or "title" in style_id
                    or (all(run.bold for run in p.runs if run.text.strip()) and len(text) < 120)
                )

                if is_heading:
                    # Gặp tiêu đề mới -> Đóng gói văn bản trước đó
                    flush_text_buffer()
                    current_section = text
                    # Cập nhật danh sách headings
                    if "1" in style_name or "title" in style_name:
                        current_headings = [doc_title, text]
                    elif "2" in style_name:
                        current_headings = current_headings[:2] + [text]
                    else:
                        current_headings = current_headings[:3] + [text]
                else:
                    text_buffer.append(text)
                    buffer_words += len(text.split())
                    if buffer_words >= 350:
                        flush_text_buffer()

            elif tag == "tbl":
                # Xử lý bảng biểu
                flush_text_buffer()
                table = Table(element, doc)
                tbl_chunks, counter = self._process_docx_table(
                    table=table,
                    path=path,
                    breadcrumb=get_breadcrumb(),
                    section_title=current_section,
                    counter=counter
                )
                chunks.extend(tbl_chunks)

        # Xả phần văn bản cuối cùng trong bộ đệm
        flush_text_buffer()
        return chunks

    def _process_docx_table(
        self,
        table: Table,
        path: Path,
        breadcrumb: str,
        section_title: str,
        counter: int
    ) -> tuple[list[dict], int]:
        """Trích xuất và định dạng bảng biểu Word (phức tạp hoặc đơn giản)."""
        if not table.rows:
            return [], counter

        # Bảng phức tạp (ô gộp hoặc đa tầng tiêu đề): Render sang ảnh lưới -> VLM
        if is_complex_docx_table(table) and self.vlm_extractor.is_available():
            try:
                tbl_img = render_docx_table_to_image(table)
                vlm_res = self.vlm_extractor.extract_table(tbl_img)
                table_chunks = self.table_formatter.format_vlm_result_to_chunks(
                    vlm_result=vlm_res,
                    source=path.name,
                    chunk_id_prefix=f"{path.stem}_tbl",
                    breadcrumb=breadcrumb,
                    section_title=section_title,
                    chunk_type="docx_table",
                    start_index=counter
                )
                if table_chunks:
                    return table_chunks, counter + len(table_chunks)
            except Exception as e:
                logger.debug(f"[DOCXChunker] VLM table extraction failed: {e}")

        # Bảng đơn giản hoặc fallback: Trích xuất ma trận ô và định dạng chuẩn
        headers = [c.text.strip().replace("\n", " ") for c in table.rows[0].cells]
        headers = [h or f"Cột_{i+1}" for i, h in enumerate(headers)]

        matrix: list[list[str]] = []
        for row in table.rows[1:]:
            vals = [c.text.strip().replace("\n", " ") for c in row.cells]
            matrix.append(vals)

        table_chunks = self.table_formatter.format_matrix_to_chunks(
            headers=headers,
            matrix=matrix,
            source=path.name,
            chunk_id_prefix=f"{path.stem}_tbl",
            breadcrumb=breadcrumb,
            section_title=section_title,
            chunk_type="docx_table",
            start_index=counter
        )
        return table_chunks, counter + len(table_chunks)


__all__ = ["DOCXChunker", "is_complex_docx_table"]
