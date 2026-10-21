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
from collections import Counter
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


def get_docx_heading_level(p: docx.text.paragraph.Paragraph, body_font_size: float = 11.0) -> int | None:
    """
    Xác định cấp độ Heading bằng OpenXML AST và Font Size Histogram (0% Regex, 0% Keywords, 0% Anchors).
    """
    s_name = (p.style.name or "").lower()
    s_id = (getattr(p.style, "style_id", "") or "").lower()

    # 1. Kiểm tra outlineLvl trong OpenXML pPr hoặc Style Definition
    outlineLvl = None
    if p._p.pPr is not None:
        ol = p._p.pPr.find(docx.oxml.ns.qn('w:outlineLvl'))
        if ol is not None:
            outlineLvl = ol.get(docx.oxml.ns.qn('w:val'))
    if outlineLvl is None and hasattr(p.style, '_element') and p.style._element.pPr is not None:
        ol = p.style._element.pPr.find(docx.oxml.ns.qn('w:outlineLvl'))
        if ol is not None:
            outlineLvl = ol.get(docx.oxml.ns.qn('w:val'))

    if outlineLvl is not None:
        try:
            return int(outlineLvl) + 1
        except Exception:
            pass

    # 2. Kiểm tra Style Name / Style ID chuẩn OpenXML
    if "title" in s_name or "title" in s_id:
        return 1
    if "subtitle" in s_name or "subtitle" in s_id:
        return 2
    if "heading" in s_name or "heading" in s_id:
        digits = "".join(ch for ch in (s_name + s_id) if ch.isdigit())
        if digits:
            return int(digits[0])
        return 1

    # 3. Typography fallback: Font Size Histogram (áp dụng cho đoạn văn không có Heading Style)
    # Loại trừ list items có numPr
    has_numpr = False
    if p._p.pPr is not None and p._p.pPr.find(docx.oxml.ns.qn('w:numPr')) is not None:
        has_numpr = True

    if not has_numpr:
        txt = p.text.strip()
        p_size = body_font_size
        for r in p.runs:
            if r.font and r.font.size:
                p_size = r.font.size.pt
                break

        if p_size >= body_font_size * 1.5:
            return 1
        elif p_size >= body_font_size * 1.25 and len(txt) < 140:
            return 2
        elif p_size >= body_font_size * 1.15 and len(txt) < 120:
            return 3

    return None


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
        table_rows_per_chunk: int = 4,
        **kwargs
    ):
        self.vlm_extractor = vlm_extractor or VLMTableExtractor()
        self.table_formatter = table_formatter or TableFormatter()
        self.syllable_manager = syllable_manager or SyllableBudgetManager()
        self.max_chunk_size = max_chunk_size
        self.min_section_size = min_section_size
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

        # Thu thập phân phối cỡ chữ (Character-Weighted Font Size Histogram)
        font_sizes: list[float] = []
        for p in doc.paragraphs:
            txt = p.text.strip()
            if not txt:
                continue
            for r in p.runs:
                if r.font and r.font.size:
                    font_sizes.extend([r.font.size.pt] * len(r.text.strip()))

        body_font_size = Counter(font_sizes).most_common(1)[0][0] if font_sizes else 11.0

        # Quản lý tiêu đề và đường dẫn Breadcrumbs đa tầng (Hierarchical Stack - 0% Regex)
        doc_title = path.stem.replace("_", " ").title()
        heading_stack: list[tuple[int, str]] = [(0, doc_title)]
        current_section = doc_title
        current_heading_eid: Optional[str] = None

        text_buffer: list[str] = []
        current_eids: list[tuple[str, str]] = []
        elem_idx = 0
        tbl_idx = 0
        buffer_words = 0

        def get_breadcrumb() -> str:
            return " > ".join(t for _, t in heading_stack) if heading_stack else doc_title

        def map_elements_to_groups(
            chunk_eid_texts: list[tuple[str, str]],
            split_groups: list[str],
            heading_eid: Optional[str] = None
        ) -> list[list[str]]:
            """
            Phân bổ element_ids vào từng group/sub-chunk bằng Giao khoảng ký tự (Character-Span Overlap).
            TUYỆT ĐỐI 0% REGEX, 0% KEYWORDS, 0% HARDCODED ANCHORS:
            - Dựa trên tọa độ ký tự tương đối [start, end] của từng element trong khối văn bản.
            - Giao khoảng: max(e_start, g_start) < min(e_end, g_end).
            """
            if len(split_groups) == 1:
                eids = [eid for eid, _ in chunk_eid_texts]
                if heading_eid and heading_eid not in eids:
                    eids.insert(0, heading_eid)
                return [eids]

            elem_spans: list[tuple[str, int, int]] = []
            clean_elements: list[str] = []
            pos = 0
            for eid, etxt in chunk_eid_texts:
                c_txt = clean_and_dehyphenate_text(etxt).strip()
                if not c_txt:
                    continue
                start = pos
                end = pos + len(c_txt)
                elem_spans.append((eid, start, end))
                clean_elements.append(c_txt)
                pos = end + 1

            full_clean = "\n".join(clean_elements)
            group_eids_list: list[list[str]] = [[] for _ in split_groups]
            cursor = 0

            for g_idx, group in enumerate(split_groups):
                g_clean = group.strip()
                if not g_clean:
                    continue

                g_pos = full_clean.find(g_clean, cursor)
                if g_pos == -1:
                    g_prefix = g_clean[:50]
                    g_pos = full_clean.find(g_prefix, cursor)
                    if g_pos == -1:
                        g_pos = cursor

                g_start = g_pos
                g_end = g_start + len(g_clean)
                cursor = g_end

                for eid, e_start, e_end in elem_spans:
                    if max(e_start, g_start) < min(e_end, g_end):
                        if eid not in group_eids_list[g_idx]:
                            group_eids_list[g_idx].append(eid)

                # Fallback hình học: Nếu group không có element nào trùng lặp, lấy element gần cursor nhất
                if not group_eids_list[g_idx] and elem_spans:
                    closest_eid = min(elem_spans, key=lambda s: abs(s[1] - g_start))[0]
                    group_eids_list[g_idx].append(closest_eid)

            # Tiêu đề mục (heading) luôn gắn chặt vào sub-chunk đầu tiên (group 0)
            if heading_eid and heading_eid not in group_eids_list[0]:
                group_eids_list[0].insert(0, heading_eid)

            return group_eids_list

        def flush_text_buffer() -> None:
            nonlocal counter, buffer_words, current_eids, current_heading_eid
            if not text_buffer and not current_heading_eid:
                buffer_words = 0
                return

            raw_text = "\n".join(text_buffer).strip()
            text_buffer.clear()
            buffer_words = 0
            chunk_eid_texts = list(current_eids)
            current_eids.clear()
            h_eid = current_heading_eid
            current_heading_eid = None

            if not raw_text:
                if h_eid:
                    chunk_obj = build_7_field_chunk(
                        chunk_id=f"{path.stem}_docx_c{counter}",
                        source=path.name,
                        chunk_type="docx_text",
                        breadcrumb=get_breadcrumb(),
                        section_title=current_section,
                        content=current_section,
                        original_text=current_section,
                        parent_id=f"{path.stem}_sec_{counter}",
                        element_ids=[h_eid]
                    )
                    chunks.append(chunk_obj)
                    counter += 1
                return

            clean_text = clean_and_dehyphenate_text(raw_text)
            breadcrumb = get_breadcrumb()

            # Phân tách văn bản bằng SyllableBudgetManager (trần sau tiền tố <= 450)
            split_groups = [
                g.strip() for g in (self.syllable_manager.split_text_elastic(
                    text=clean_text,
                    window_start=200,
                    window_end=380,
                    ceiling=420
                ) or [clean_text]) if g.strip()
            ]

            if not split_groups:
                return

            group_eids_list = map_elements_to_groups(
                chunk_eid_texts=chunk_eid_texts,
                split_groups=split_groups,
                heading_eid=h_eid
            )

            for group, g_eids in zip(split_groups, group_eids_list):
                chunk_obj = build_7_field_chunk(
                    chunk_id=f"{path.stem}_docx_c{counter}",
                    source=path.name,
                    chunk_type="docx_text",
                    breadcrumb=breadcrumb,
                    section_title=current_section,
                    content=group,
                    original_text=group,
                    parent_id=f"{path.stem}_sec_{counter}",
                    element_ids=g_eids
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

                eid = f"{path.name}_e{elem_idx}"
                elem_idx += 1

                lvl = get_docx_heading_level(p, body_font_size)

                if lvl is not None:
                    # Gặp tiêu đề mới -> Đóng gói văn bản trước đó
                    flush_text_buffer()
                    current_heading_eid = eid
                    while heading_stack and heading_stack[-1][0] >= lvl:
                        heading_stack.pop()
                    heading_stack.append((lvl, text))
                    current_section = text
                else:
                    text_buffer.append(text)
                    current_eids.append((eid, text))
                    buffer_words += len(text.split())
                    if buffer_words >= 350:
                        flush_text_buffer()

            elif tag == "tbl":
                # Xử lý bảng biểu
                flush_text_buffer()
                table = Table(element, doc)
                tbl_eid = f"{path.name}_tbl_{tbl_idx}"
                tbl_idx += 1
                tbl_chunks, counter = self._process_docx_table(
                    table=table,
                    path=path,
                    breadcrumb=get_breadcrumb(),
                    section_title=current_section,
                    counter=counter,
                    table_eid=tbl_eid
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
        counter: int,
        table_eid: Optional[str] = None
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
                    if table_eid:
                        for tc in table_chunks:
                            tc["element_ids"] = [table_eid]
                    return table_chunks, counter + len(table_chunks)
            except Exception as e:
                logger.debug(f"[DOCXChunker] VLM table extraction failed: {e}")

        # Bảng đơn giản hoặc fallback: Trích xuất ma trận ô và định dạng chuẩn
        raw_headers = [c.text.strip().replace("\n", " ") for c in table.rows[0].cells]
        last_hdr = ""
        headers: list[str] = []
        for h in raw_headers:
            if h:
                last_hdr = h
                headers.append(h)
            elif last_hdr:
                headers.append(f"{last_hdr} (Chi tiết)")
            else:
                headers.append("Nội dung")

        # Khử trùng lặp tiêu đề
        seen: dict[str, int] = {}
        unique_headers: list[str] = []
        for h in headers:
            if h not in seen:
                seen[h] = 1
                unique_headers.append(h)
            else:
                seen[h] += 1
                unique_headers.append(f"{h} ({seen[h]})")

        matrix: list[list[str]] = []
        for row in table.rows[1:]:
            vals = [c.text.strip().replace("\n", " ") for c in row.cells]
            matrix.append(vals)

        table_chunks = self.table_formatter.format_matrix_to_chunks(
            headers=unique_headers,
            matrix=matrix,
            source=path.name,
            chunk_id_prefix=f"{path.stem}_tbl",
            breadcrumb=breadcrumb,
            section_title=section_title,
            chunk_type="docx_table",
            start_index=counter
        )
        if table_eid:
            for tc in table_chunks:
                tc["element_ids"] = [table_eid]
        return table_chunks, counter + len(table_chunks)


__all__ = ["DOCXChunker", "is_complex_docx_table"]
