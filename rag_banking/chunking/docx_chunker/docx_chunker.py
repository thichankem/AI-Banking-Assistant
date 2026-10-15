"""
Module DOCX Layout-Aware & Advanced Table/Hierarchy Chunker
============================================================
Bộ phân đoạn tài liệu Microsoft Word (.docx) ngân hàng nhận thức bố cục và cấu trúc DLA:
1. Global Document Font Size Profiling (0% Regex / Rule-based Keywords): Quét phân phối kích cỡ chữ toàn văn bản
   (Character-weighted Font Histogram), so sánh tương quan kích cỡ chữ để tự động phân cấp Heading Level 1, 2, 3...
2. Comprehensive Coverage: Quét trọn vẹn Textbox (<w:txbxContent>), Bảng lồng (Nested Tables), Ô gộp (Merged Cells).
3. Multi-Level Table Header (Header 2 tầng): Tự động phát hiện và ghép tiêu đề phân cấp.
4. Table Atomicity & Row Batching: Chia bảng theo hàng kèm lặp lại Header ở mọi chunk; tuyệt đối không cắt vụn bảng.
5. Strict Document Flow: Luôn flush buffer text trước khi xử lý bảng để giữ nguyên thứ tự tài liệu.
6. Section-Length-Aware Anti-Fragmentation: Tự động gộp các mục con còi cọc (< 120 từ), đưa chunk về dải vàng (150-450 từ).
"""

from __future__ import annotations
from pathlib import Path
from collections import Counter

import docx
from docx.text.paragraph import Paragraph
from docx.table import Table, _Cell
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

try:
    from ..unified_chunk.canonical_resolver import (
        count_words,
        split_sentences_banking,
        group_sentences_banking,
        clean_and_dehyphenate_text
    )
except (ImportError, ValueError):
    from rag_banking.chunking.unified_chunk.canonical_resolver import (
        count_words,
        split_sentences_banking,
        group_sentences_banking,
        clean_and_dehyphenate_text
    )


def _build_docx_chunk(
    chunk_id: str,
    source: str,
    chunk_type: str,
    breadcrumb: str,
    section_title: str,
    content: str
) -> dict:
    """Helper chuẩn hóa việc tạo đối tượng chunk DOCX, đảm bảo Schema tinh gọn."""
    return {
        "chunk_id": chunk_id,
        "source": source,
        "chunk_type": chunk_type,
        "breadcrumb": breadcrumb,
        "section_title": section_title,
        "content": content,
        "word_count": count_words(content)
    }


class DOCXChunker:
    """
    Bộ phân đoạn tài liệu DOCX chuyên dụng cho ngân hàng theo giải thuật DLA & Advanced Table/Hierarchy.
    """
    MAX_HEADING_LEVEL: int = 6  # Khống chế tối đa 6 cấp heading (H1..H6), không tính tên tệp (title)

    def __init__(
        self,
        max_chunk_size: int = 800,       # Trần kích thước ký tự (~400 từ)
        min_section_size: int = 120,     # Ngưỡng từ tối thiểu dải vàng (150-450 từ)
        chunk_overlap: int = 100,        # Độ gối đầu khi phải chia nhỏ đoạn văn bản dài
        table_rows_per_chunk: int = 4    # Số hàng bảng biểu tối đa trong một chunk
    ):
        self.max_chunk_size = max_chunk_size
        self.min_section_size = min_section_size
        self.chunk_overlap = chunk_overlap
        self.table_rows_per_chunk = table_rows_per_chunk

    def chunk_docx(self, docx_path: str | Path) -> list[dict]:
        """
        Đọc và phân đoạn toàn diện một tệp DOCX ngân hàng:
        Hỗ trợ heading in đậm không dùng style, bảng 2 tầng, textbox, ô gộp, bảng lồng nhau.
        """
        docx_path = Path(docx_path)
        if not docx_path.exists():
            print(f"[DOCX Chunker] Không tìm thấy tệp: {docx_path}")
            return []

        try:
            doc = docx.Document(str(docx_path))
        except Exception as e:
            print(f"[DOCX Chunker] Không thể mở tệp {docx_path.name}: {e}")
            return []

        chunks: list[dict] = []
        chunk_counter = 0

        # Thứ bậc tiêu đề hoạt động: {1: "Tiêu đề lớn", 2: "Tiêu đề con", ...}
        active_headers: dict[int, str] = {}
        text_buffer: list[str] = []

        def get_current_breadcrumb() -> str:
            sorted_levels = sorted(active_headers.keys())
            titles = [active_headers[lvl] for lvl in sorted_levels if active_headers[lvl]]
            return " > ".join(titles) if titles else docx_path.stem

        def get_current_section_title() -> str:
            if not active_headers:
                return "Tổng quan"
            max_lvl = max(active_headers.keys())
            return active_headers[max_lvl]

        def emit_text_chunks(force_flush: bool = False) -> None:
            """
            Đóng gói văn bản trong buffer thành chunk(s).
            Áp dụng Section-Length-Aware: Chỉ emit khi đủ min_section_size hoặc bị ép buộc (force_flush).
            """
            nonlocal chunk_counter
            if not text_buffer:
                return

            raw_text = "\n".join(text_buffer).strip()
            total_words = count_words(raw_text)

            # Nếu chưa đủ min_section_size và không bị force_flush -> giữ lại gộp tiếp
            if not force_flush and total_words < self.min_section_size:
                return

            text_buffer.clear()
            if not raw_text:
                return

            clean_raw = clean_and_dehyphenate_text(raw_text)
            breadcrumb = get_current_breadcrumb()
            section_title = get_current_section_title()
            header_prefix = f"[Breadcrumbs: {breadcrumb}]"

            # 1. Văn bản ngắn vừa trong max_chunk_size
            if len(clean_raw) <= self.max_chunk_size:
                full_content = f"{header_prefix}\n{clean_raw}"
                chunks.append(_build_docx_chunk(
                    chunk_id=f"{docx_path.stem}_docx_c{chunk_counter}",
                    source=docx_path.name,
                    chunk_type="docx_text",
                    breadcrumb=breadcrumb,
                    section_title=section_title,
                    content=full_content
                ))
                chunk_counter += 1
            else:
                # 2. Văn bản dài -> Cắt bằng bộ tách câu an toàn
                sentences = split_sentences_banking(clean_raw)
                sub_parts = group_sentences_banking(sentences, self.max_chunk_size, self.chunk_overlap)
                for part in sub_parts:
                    sub_prefix = f"[Breadcrumbs: {breadcrumb}]"
                    full_content = f"{sub_prefix}\n{part}"
                    chunks.append(_build_docx_chunk(
                        chunk_id=f"{docx_path.stem}_docx_c{chunk_counter}",
                        source=docx_path.name,
                        chunk_type="docx_text",
                        breadcrumb=breadcrumb,
                        section_title=section_title,
                        content=full_content
                    ))
                    chunk_counter += 1

        # 1. Trích xuất Textbox trong tài liệu (<w:txbxContent>)
        textbox_texts = self._extract_textbox_content(doc)
        if textbox_texts:
            for tb_text in textbox_texts:
                text_buffer.append(f"> [Ghi chú / Textbox]: {tb_text}")

        # 2. Lập hồ sơ tương quan kích cỡ chữ toàn văn bản (Global Document Font Profiling)
        font_profile = self._profile_document_fonts(doc)

        # 3. Duyệt các phần tử cấp khối trong Body theo thứ tự tài liệu
        for child in doc.element.body:
            # 3.1. Đoạn văn bản (<w:p>)
            if child.tag.endswith("p"):
                p = Paragraph(child, doc)
                text = p.text.strip()
                if not text:
                    continue

                style_name = p.style.name if p.style else ""
                heading_level = self._detect_heading_level(p, style_name, text, font_profile)

                # Khi gặp Tiêu đề (Heading)
                if heading_level is not None:
                    if heading_level == 1:
                        emit_text_chunks(force_flush=True)
                    else:
                        current_words = count_words("\n".join(text_buffer))
                        if current_words >= self.min_section_size:
                            emit_text_chunks(force_flush=True)
                        else:
                            if text_buffer:
                                text_buffer.append(f"\n### {text}")

                    active_headers[heading_level] = text
                    levels_to_remove = [lvl for lvl in active_headers.keys() if lvl > heading_level]
                    for lvl in levels_to_remove:
                        del active_headers[lvl]
                    continue

                # Xử lý đoạn thường, bullets hoặc numbered list
                if "List Bullet" in style_name:
                    formatted_line = f"- {text}" if not text.startswith(("-", "*", "•")) else text
                    text_buffer.append(formatted_line)
                elif "List Number" in style_name:
                    text_buffer.append(text)
                else:
                    text_buffer.append(text)

                current_chars = sum(len(line) + 1 for line in text_buffer)
                if current_chars >= self.max_chunk_size:
                    emit_text_chunks(force_flush=True)

            # 3.2. Bảng biểu (<w:tbl>)
            elif child.tag.endswith("tbl"):
                # BẮT BUỘC: Flush buffer text trước khi xử lý bảng
                emit_text_chunks(force_flush=True)

                table = Table(child, doc)
                table_chunks = self._extract_table_chunks(
                    table=table,
                    docx_path=docx_path,
                    breadcrumb=get_current_breadcrumb(),
                    section_title=get_current_section_title(),
                    start_chunk_idx=chunk_counter
                )
                chunks.extend(table_chunks)
                chunk_counter += len(table_chunks)

        # Flush buffer còn lại cuối cùng
        emit_text_chunks(force_flush=True)

        return chunks

    def _get_doc_default_font_size(self, doc: docx.Document) -> float:
        """Trích xuất kích cỡ chữ mặc định của tài liệu từ OpenXML docDefaults."""
        try:
            docDefaults = doc.styles.element.find(qn("w:docDefaults"))
            if docDefaults is not None:
                rPrDefault = docDefaults.find(qn("w:rPrDefault"))
                if rPrDefault is not None:
                    rPr = rPrDefault.find(qn("w:rPr"))
                    if rPr is not None:
                        sz = rPr.find(qn("w:sz"))
                        if sz is not None and sz.get(qn("w:val")):
                            return float(sz.get(qn("w:val"))) / 2.0
        except Exception:
            pass
        return 11.0

    def _get_run_font_size(self, run, p: Paragraph, default_sz: float) -> float:
        """Trích xuất kích cỡ chữ thực tế (point) của run văn bản."""
        if run.font.size is not None:
            return run.font.size.pt
        if hasattr(run, "_element") and run._element is not None and run._element.rPr is not None:
            el_sz = run._element.rPr.find(qn("w:sz"))
            if el_sz is not None and el_sz.get(qn("w:val")):
                return float(el_sz.get(qn("w:val"))) / 2.0
        if p.style and p.style.font and p.style.font.size is not None:
            return p.style.font.size.pt
        return default_sz

    def _profile_document_fonts(self, doc: docx.Document) -> dict:
        """
        Khảo sát phân phối kích cỡ chữ trên toàn bộ tệp DOCX (Character-weighted Font Histogram):
        1. Tính tổng số lượng ký tự theo từng kích cỡ chữ trong toàn bộ văn bản và bảng biểu.
        2. Xác định base_size: Cỡ chữ thân bài có mật độ xuất hiện lớn nhất (Mode).
        3. Phân cấp các cỡ chữ lớn hơn base_size theo thứ tự giảm dần thành Heading Level 1, 2, 3...
        """
        default_sz = self._get_doc_default_font_size(doc)
        font_char_counts = Counter()

        for p in doc.paragraphs:
            for r in p.runs:
                t = r.text.strip()
                if not t:
                    continue
                sz = self._get_run_font_size(r, p, default_sz)
                font_char_counts[sz] += len(t)

        for t in doc.tables:
            for row in t.rows:
                for cell in row.cells:
                    for p in cell.paragraphs:
                        for r in p.runs:
                            t = r.text.strip()
                            if not t:
                                continue
                            sz = self._get_run_font_size(r, p, default_sz)
                            font_char_counts[sz] += len(t)

        if not font_char_counts:
            return {
                "base_size": 11.0,
                "heading_levels": {},
                "default_sz": 11.0,
                "larger_sizes": []
            }

        base_size = font_char_counts.most_common(1)[0][0]
        # Các cỡ chữ lớn hơn cỡ chữ thân bài ít nhất 0.4pt (loại trừ sai số số thực)
        # Khống chế tối đa (MAX_HEADING_LEVEL - 1) cỡ chữ lớn để dành cấp cuối cùng cho tiểu mục in đậm
        max_font_headings = max(self.MAX_HEADING_LEVEL - 1, 1)
        larger_sizes = sorted([sz for sz in font_char_counts.keys() if sz >= base_size + 0.4], reverse=True)[:max_font_headings]
        heading_levels = {sz: min(idx + 1, self.MAX_HEADING_LEVEL) for idx, sz in enumerate(larger_sizes)}

        return {
            "base_size": base_size,
            "heading_levels": heading_levels,
            "default_sz": default_sz,
            "larger_sizes": larger_sizes
        }

    def _detect_heading_level(
        self,
        p: Paragraph,
        style_name: str,
        text: str,
        font_profile: dict | None = None
    ) -> int | None:
        """
        Nhận diện cấp độ tiêu đề dựa trên so sánh tương quan cỡ chữ toàn tài liệu (0% Regex):
        Khống chế tối đa 6 cấp heading (H1..H6), không tính tên file (title):
        1. Word Styles chuẩn (Title, Subtitle, Heading 1..9).
        2. Thuộc tính OpenXML Outline Level (w:outlineLvl).
        3. Tương quan kích cỡ chữ toàn văn bản (Global Font Size Profiling):
           - Kích cỡ lớn nhất so với thân bài -> Level 1.
           - Kích cỡ lớn thứ nhì -> Level 2...
           - Cỡ chữ bằng thân bài nhưng in đậm toàn bộ -> Sub-heading (Level kế tiếp, tối đa 6).
        """
        style_lower = style_name.lower()

        # 1. Qua Style Word
        if "title" in style_lower:
            return 1
        if "subtitle" in style_lower:
            return 2
        if "heading" in style_lower:
            digits = [c for c in style_name if c.isdigit()]
            lvl = int("".join(digits)) if digits else 1
            return min(lvl, self.MAX_HEADING_LEVEL)

        # 2. Qua OpenXML Outline Level
        try:
            if hasattr(p, "_element") and p._element is not None:
                pPr = p._element.pPr
                if pPr is not None and hasattr(pPr, "outlineLvl") and pPr.outlineLvl is not None:
                    lvl_val = pPr.outlineLvl.val
                    if lvl_val is not None:
                        return min(int(lvl_val) + 1, self.MAX_HEADING_LEVEL)
        except Exception:
            pass

        first_line = text.split("\n")[0].strip()
        tokens = first_line.split()
        if not tokens or len(first_line) > 140:
            return None

        non_empty_runs = [r for r in p.runs if r.text.strip()]
        if not non_empty_runs:
            return None

        if not font_profile:
            return None

        default_sz = font_profile.get("default_sz", 11.0)
        run_sizes = [self._get_run_font_size(r, p, default_sz) for r in non_empty_runs]
        p_sz = max(run_sizes)

        # 3.1. So sánh với các kích cỡ chữ lớn hơn cỡ chữ chung của file
        for sz, lvl in font_profile.get("heading_levels", {}).items():
            if abs(p_sz - sz) < 0.3:
                return min(lvl, self.MAX_HEADING_LEVEL)

        # 3.2. Cỡ chữ bằng cỡ chữ chung nhưng in đậm toàn bộ (Sub-heading, tối đa Level 6)
        is_all_bold = all(r.bold for r in non_empty_runs)
        if is_all_bold and len(first_line) < 100:
            if 1 <= len(tokens) <= 15 and not first_line.endswith((".", ";", ":")):
                sub_lvl = len(font_profile.get("larger_sizes", [])) + 1
                return min(sub_lvl, self.MAX_HEADING_LEVEL)

        return None

    def _extract_textbox_content(self, doc: docx.Document) -> list[str]:
        """Trích xuất nội dung văn bản bên trong Textbox (<w:txbxContent>)."""
        textbox_texts: list[str] = []
        try:
            for el in doc.element.body.iter():
                if el.tag.endswith("txbxContent"):
                    tb_paras = []
                    for p_el in el.iter():
                        if p_el.tag.endswith("p"):
                            p = Paragraph(p_el, doc)
                            if p.text.strip():
                                tb_paras.append(p.text.strip())
                    if tb_paras:
                        textbox_texts.append("\n".join(tb_paras))
        except Exception:
            pass
        return textbox_texts

    def _extract_table_chunks(
        self,
        table: Table,
        docx_path: Path,
        breadcrumb: str,
        section_title: str,
        start_chunk_idx: int
    ) -> list[dict]:
        """
        Trích xuất bảng biểu trong DOCX thành Linearized Table Chunks kèm tên cột:
        1. Nhận diện Header 2 tầng (Hierarchical Multi-level Header).
        2. Xử lý Ô gộp (Merged Cells) không bị lặp text.
        3. Duyệt đệ quy Bảng lồng nhau (Nested Tables).
        4. Chia theo Batch rows kèm LẶP LẠI TOÀN BỘ HEADER ở mỗi chunk.
        """
        if not table.rows:
            return []

        header_names = self._detect_table_headers(table)
        data_row_strs: list[str] = []

        start_row_idx = 2 if self._is_two_level_header(table) else 1

        for r_idx in range(start_row_idx, len(table.rows)):
            row = table.rows[r_idx]
            # Xử lý ô gộp: lọc bỏ cell trùng XML element
            unique_cells: list[_Cell] = []
            seen_tcs: set[int] = set()
            for cell in row.cells:
                tc_id = id(cell._tc)
                if tc_id not in seen_tcs:
                    seen_tcs.add(tc_id)
                    unique_cells.append(cell)

            row_items = []
            for c_idx, cell in enumerate(unique_cells):
                cell_val = self._extract_cell_text_with_nested_tables(cell)
                if not cell_val:
                    continue
                col_name = header_names[c_idx] if c_idx < len(header_names) else f"Cột_{c_idx+1}"
                row_items.append(f"{col_name}: {cell_val}")

            if not row_items:
                continue

            row_line = " | ".join(row_items)
            data_row_strs.append(f"* Hàng {r_idx}: {row_line}")

        if not data_row_strs:
            return []

        table_breadcrumb = f"{breadcrumb} > Bảng biểu" if breadcrumb else f"{docx_path.stem} > Bảng biểu"
        table_chunks: list[dict] = []

        # Chia bảng theo batch rows và lặp lại header ở mỗi chunk
        batch_size = self.table_rows_per_chunk
        total_rows = len(data_row_strs)

        for i in range(0, total_rows, batch_size):
            batch_data = data_row_strs[i:i + batch_size]

            linearized_content = (
                f"[Breadcrumbs: {table_breadcrumb}]\n"
                f"Cột dữ liệu: {', '.join(header_names)}\n"
                f"Chi tiết từng hàng:\n" + "\n".join(batch_data)
            )

            table_chunks.append(_build_docx_chunk(
                chunk_id=f"{docx_path.stem}_docx_tbl_{start_chunk_idx + len(table_chunks)}",
                source=docx_path.name,
                chunk_type="docx_table",
                breadcrumb=table_breadcrumb,
                section_title=section_title,
                content=linearized_content
            ))

        return table_chunks

    def _is_two_level_header(self, table: Table) -> bool:
        """Kiểm tra bảng có cấu trúc header 2 tầng hay không."""
        if len(table.rows) < 3:
            return False
        row0_cells = [c.text.strip() for c in table.rows[0].cells]
        row1_cells = [c.text.strip() for c in table.rows[1].cells]
        has_duplicates_or_empty = len(row0_cells) != len(set(row0_cells)) or any(c == "" for c in row0_cells)
        has_row1_content = any(c != "" for c in row1_cells)
        return has_duplicates_or_empty and has_row1_content

    def _detect_table_headers(self, table: Table) -> list[str]:
        """Tự động xây dựng danh sách tên cột, hỗ trợ ghép header 2 tầng."""
        if not table.rows:
            return []

        row0_cells = [cell.text.strip().replace("\n", " ") for cell in table.rows[0].cells]

        if self._is_two_level_header(table):
            row1_cells = [cell.text.strip().replace("\n", " ") for cell in table.rows[1].cells]
            headers = []
            for i in range(max(len(row0_cells), len(row1_cells))):
                top = row0_cells[i] if i < len(row0_cells) else ""
                sub = row1_cells[i] if i < len(row1_cells) else ""
                if top and sub and top != sub:
                    headers.append(f"{top} - {sub}")
                elif sub:
                    headers.append(sub)
                elif top:
                    headers.append(top)
                else:
                    headers.append(f"Cột_{i+1}")
            return headers

        # Header 1 tầng đơn giản
        return [name if name else f"Cột_{i+1}" for i, name in enumerate(row0_cells)]

    def _extract_cell_text_with_nested_tables(self, cell: _Cell) -> str:
        """Trích xuất text của cell và duyệt đệ quy các bảng lồng nhau bên trong cell nếu có."""
        parts = []
        cell_text = cell.text.strip().replace("\n", " ")
        if cell_text:
            parts.append(cell_text)

        if hasattr(cell, "tables") and cell.tables:
            for nested_tbl in cell.tables:
                nested_rows = []
                for n_r in nested_tbl.rows:
                    n_vals = [c.text.strip().replace("\n", " ") for c in n_r.cells if c.text.strip()]
                    if n_vals:
                        nested_rows.append(" | ".join(n_vals))
                if nested_rows:
                    parts.append("[Bảng lồng]: " + " ; ".join(nested_rows))

        return " ".join(parts)
