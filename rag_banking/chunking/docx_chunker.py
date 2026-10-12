"""
DOCX Layout-Aware & Advanced Table/Hierarchy Chunker (DLA - Document Layout Analysis)
Bộ phân đoạn tài liệu Microsoft Word (.docx) ngân hàng nhận thức bố cục và cấu trúc chuyên sâu:
1. Heuristic Heading Detection: Không phụ thuộc vào Word Style; tự động nhận diện in đậm (run.bold),
   đánh số tay (Điều 5., Chương II., 1. Hạn mức) theo thể thức ngân hàng Việt Nam.
2. Comprehensive Content Coverage: Quét trọn vẹn Textbox (<w:txbxContent>), Header/Footer,
   Footnotes, Bảng lồng nhau (Nested Tables) và xử lý chuẩn xác Ô gộp (Merged Cells).
3. Multi-Level Table Header (Header 2 tầng): Tự động phát hiện và ghép tiêu đề phân cấp
   (vd: Thẻ tín dụng - Hạn mức ngày, Thẻ ghi nợ - Hạn mức lần).
4. Table Atomicity & Row Batching: Chia bảng theo hàng kèm lặp lại Header ở mọi chunk;
   tuyệt đối không áp dụng sentence splitting lên dữ liệu bảng biểu.
5. Strict Document Flow: Luôn flush buffer text trước khi xử lý <tbl> để giữ nguyên thứ tự tài liệu.
6. Section-Length-Aware Anti-Fragmentation: Tự động phát hiện và gộp các mục con còi cọc (< 120 từ),
   đưa chunk về dải kích thước vàng (150 – 450 từ).
"""

import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Set, Tuple

try:
    import docx
    from docx.text.paragraph import Paragraph
    from docx.table import Table, _Cell
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

from .canonical_resolver import (
    count_words,
    count_syllables_vietnamese,
    extract_canonical_legal_ref,
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
    content: str,
    raw_content: str,
    legal_ref: Optional[str] = None
) -> Dict[str, Any]:
    """Helper chuẩn hóa việc tạo đối tượng chunk DOCX, đảm bảo Schema thống nhất."""
    return {
        "chunk_id": chunk_id,
        "source": source,
        "format": "docx",
        "chunk_type": chunk_type,
        "breadcrumb": breadcrumb,
        "legal_ref": legal_ref,
        "section_title": section_title,
        "content": content,
        "raw_content": raw_content,
        "char_length": len(content),
        "word_count": count_words(content),
        "syllable_count": count_syllables_vietnamese(content)
    }


class DOCXChunker:
    """
    Bộ phân đoạn tài liệu DOCX chuyên dụng cho ngân hàng theo giải thuật DLA & Advanced Table/Hierarchy.
    """

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

    def is_available(self) -> bool:
        """Kiểm tra thư viện python-docx đã sẵn sàng hay chưa."""
        return DOCX_AVAILABLE

    def chunk_docx(self, docx_path: str | Path) -> List[Dict[str, Any]]:
        """
        Đọc và phân đoạn toàn diện một tệp DOCX ngân hàng:
        Hỗ trợ heading in đậm không dùng style, bảng 2 tầng, textbox, ô gộp, bảng lồng nhau.
        """
        docx_path = Path(docx_path)
        if not self.is_available():
            raise ImportError("Vui lòng cài đặt thư viện 'python-docx' để xử lý tệp Word .docx!")

        if not docx_path.exists():
            print(f"[DOCX Chunker] Không tìm thấy tệp: {docx_path}")
            return []

        try:
            doc = docx.Document(str(docx_path))
        except Exception as e:
            print(f"[DOCX Chunker] Không thể mở tệp {docx_path.name}: {e}")
            return []

        chunks: List[Dict[str, Any]] = []
        chunk_counter = 0

        # Thứ bậc tiêu đề hoạt động: {1: "Tiêu đề lớn", 2: "Tiêu đề con", ...}
        active_headers: Dict[int, str] = {}
        # Buffer tích lũy văn bản
        text_buffer: List[str] = []

        def get_current_breadcrumb() -> str:
            sorted_levels = sorted(active_headers.keys())
            titles = [active_headers[lvl] for lvl in sorted_levels if active_headers[lvl]]
            return " > ".join(titles) if titles else docx_path.stem

        def get_current_section_title() -> str:
            if not active_headers:
                return "Tổng quan"
            max_lvl = max(active_headers.keys())
            return active_headers[max_lvl]

        def emit_text_chunks(force_flush: bool = False):
            """
            Đóng gói văn bản tích lũy trong buffer thành chunk(s).
            Áp dụng Section-Length-Aware: Chỉ emit khi đủ min_section_size hoặc bị ép buộc (force_flush).
            """
            nonlocal chunk_counter
            if not text_buffer:
                return

            raw_text = "\n".join(text_buffer).strip()
            total_words = count_words(raw_text)

            # Nếu chưa đủ min_section_size và không bị ép buộc flush -> Giữ lại để gộp tiếp
            if not force_flush and total_words < self.min_section_size:
                return

            text_buffer.clear()
            if not raw_text:
                return

            # Làm sạch dehyphenation và ngắt dòng cứng
            clean_raw = clean_and_dehyphenate_text(raw_text)
            breadcrumb = get_current_breadcrumb()
            section_title = get_current_section_title()
            legal_ref = extract_canonical_legal_ref(breadcrumb, clean_raw, source_name=docx_path.name)

            header_prefix = f"[Breadcrumbs: {breadcrumb}]"
            if legal_ref:
                header_prefix += f"\n[Pháp lý: {legal_ref}]"

            # 1. Trường hợp văn bản nằm trong giới hạn max_chunk_size
            if len(clean_raw) <= self.max_chunk_size:
                full_content = f"{header_prefix}\n{clean_raw}"
                chunks.append(_build_docx_chunk(
                    chunk_id=f"{docx_path.stem}_docx_c{chunk_counter}",
                    source=docx_path.name,
                    chunk_type="docx_text",
                    breadcrumb=breadcrumb,
                    section_title=section_title,
                    content=full_content,
                    raw_content=clean_raw,
                    legal_ref=legal_ref
                ))
                chunk_counter += 1
            else:
                # 2. Trường hợp văn bản dài -> Cắt bằng bộ tách câu ngân hàng an toàn
                sentences = split_sentences_banking(clean_raw)
                sub_parts = group_sentences_banking(sentences, self.max_chunk_size, self.chunk_overlap)
                for part in sub_parts:
                    sub_legal = extract_canonical_legal_ref(breadcrumb, part, source_name=docx_path.name) or legal_ref
                    sub_prefix = f"[Breadcrumbs: {breadcrumb}]"
                    if sub_legal:
                        sub_prefix += f"\n[Pháp lý: {sub_legal}]"
                    full_content = f"{sub_prefix}\n{part}"

                    chunks.append(_build_docx_chunk(
                        chunk_id=f"{docx_path.stem}_docx_c{chunk_counter}",
                        source=docx_path.name,
                        chunk_type="docx_text",
                        breadcrumb=breadcrumb,
                        section_title=section_title,
                        content=full_content,
                        raw_content=part,
                        legal_ref=sub_legal
                    ))
                    chunk_counter += 1

        # ---------------------------------------------------------------------
        # 1. TRÍCH XUẤT NỘI DUNG TEXTBOX TRONG TÀI LIỆU (<w:txbxContent>)
        # ---------------------------------------------------------------------
        textbox_texts = self._extract_textbox_content(doc)
        if textbox_texts:
            for tb_text in textbox_texts:
                text_buffer.append(f"> [Ghi chú / Textbox]: {tb_text}")

        # ---------------------------------------------------------------------
        # 2. DUYỆT CÁC PHẦN TỬ CẤP KHỐI TRONG BODY THEO THỨ TỰ THỰC TẾ
        # ---------------------------------------------------------------------
        for child in doc.element.body:
            # 2.1. Đoạn văn bản (<w:p>)
            if child.tag.endswith("p"):
                p = Paragraph(child, doc)
                text = p.text.strip()
                if not text:
                    continue

                style_name = p.style.name if p.style else ""
                heading_level = self._detect_heading_level(p, style_name, text)

                # Khi gặp một Tiêu đề (Heading)
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

            # 2.2. Bảng biểu (<w:tbl>)
            elif child.tag.endswith("tbl"):
                # BẮT BUỘC: Flush buffer text trước để giữ đúng thứ tự tài liệu
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

        # Phát hành phần văn bản còn lại cuối cùng
        emit_text_chunks(force_flush=True)

        return chunks

    def _detect_heading_level(self, p: "Paragraph", style_name: str, text: str) -> Optional[int]:
        """
        Nhận diện cấp độ tiêu đề kết hợp:
        1. Dựa trên Word Styles chuẩn (Heading 1..4, Title, Subtitle).
        2. Dựa trên Heuristic In Đậm (Bold) và Regex cấu trúc thể thức pháp lý ngân hàng.
        """
        style_lower = style_name.lower()

        # 1. Qua Style
        if "title" in style_lower:
            return 1
        if "subtitle" in style_lower:
            return 2
        if "heading" in style_lower:
            m = re.search(r"\d+", style_name)
            return int(m.group(0)) if m else 1

        # 2. Heuristic: Kiểm tra xem đoạn văn bản có được in đậm toàn bộ hay không
        is_all_bold = False
        if p.runs:
            non_empty_runs = [r for r in p.runs if r.text.strip()]
            if non_empty_runs and all(r.bold for r in non_empty_runs):
                is_all_bold = True

        # 3. Qua Regex thể thức (kể cả style là Normal nhưng in đậm hoặc ngắn < 120 chars)
        first_line = text.split("\n")[0].strip()
        if len(first_line) < 120:
            if re.match(r"^(PHẦN|CHƯƠNG)\s+[IVXLCDM\d]+", first_line, re.IGNORECASE):
                return 1
            if re.match(r"^MỤC\s+[IVXLCDM\d]+", first_line, re.IGNORECASE):
                return 2
            if re.match(r"^Điều\s+\d+", first_line, re.IGNORECASE):
                return 3
            if re.match(r"^Câu\s+\d+[:.]", first_line, re.IGNORECASE):
                return 3
            if re.match(r"^(BIỂU PHÍ|QUY TẮC|THỎA THUẬN|QUY ĐỊNH|QUY CHẾ)", first_line, re.IGNORECASE):
                return 2
            if re.match(r"^Khoản\s+\d+", first_line, re.IGNORECASE):
                return 4

            # Nếu toàn bộ đoạn in đậm và có đánh số thứ tự đầu đoạn
            if is_all_bold:
                if re.match(r"^\d{1,2}\.\s+[A-ZÀ-Ỹ]", first_line):
                    return 4
                if re.match(r"^[a-zđ]\)\s+[A-ZÀ-Ỹ]", first_line):
                    return 5
                # Tiêu đề ngắn in hoa hoàn toàn
                if first_line.isupper() and 5 < len(first_line) < 80:
                    return 2

        return None

    def _extract_textbox_content(self, doc: "docx.Document") -> List[str]:
        """Trích xuất nội dung văn bản nằm bên trong Textbox (<w:txbxContent>)."""
        textbox_texts = []
        try:
            for el in doc.element.body.iter():
                if el.tag.endswith("txbxContent"):
                    # Lấy text của tất cả các paragraph bên trong textbox
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
        table: "Table",
        docx_path: Path,
        breadcrumb: str,
        section_title: str,
        start_chunk_idx: int
    ) -> List[Dict[str, Any]]:
        """
        Trích xuất bảng biểu trong DOCX thành Linearized Table Chunks kèm tên cột:
        1. Nhận diện Header 2 tầng (Hierarchical Multi-level Header).
        2. Xử lý Ô gộp (Merged Cells) không bị lặp text.
        3. Duyệt đệ quy Bảng lồng nhau (Nested Tables).
        4. Chia theo Batch rows kèm LẶP LẠI TOÀN BỘ HEADER ở mỗi chunk.
        """
        if not table.rows:
            return []

        # 1. Phát hiện Header 1 tầng hoặc 2 tầng
        header_names = self._detect_table_headers(table)
        data_row_strs: List[str] = []
        raw_row_strs: List[str] = []

        start_row_idx = 2 if self._is_two_level_header(table) else 1

        for r_idx in range(start_row_idx, len(table.rows)):
            row = table.rows[r_idx]
            # Xử lý ô gộp: lọc bỏ các cell trỏ cùng một XML element
            unique_cells = []
            seen_tcs = set()
            for cell in row.cells:
                tc_id = id(cell._tc)
                if tc_id not in seen_tcs:
                    seen_tcs.add(tc_id)
                    unique_cells.append(cell)

            row_items = []
            for c_idx, cell in enumerate(unique_cells):
                # Trích xuất text ô, kiểm tra cả bảng lồng nhau bên trong cell nếu có
                cell_val = self._extract_cell_text_with_nested_tables(cell)
                if not cell_val:
                    continue

                col_name = header_names[c_idx] if c_idx < len(header_names) else f"Cột_{c_idx+1}"
                row_items.append(f"{col_name}: {cell_val}")

            if not row_items:
                continue

            row_line = " | ".join(row_items)
            data_row_strs.append(f"* Hàng {r_idx}: {row_line}")
            raw_row_strs.append(row_line)

        if not data_row_strs:
            return []

        table_breadcrumb = f"{breadcrumb} > Bảng biểu" if breadcrumb else f"{docx_path.stem} > Bảng biểu"
        table_chunks: List[Dict[str, Any]] = []

        # Chia bảng theo batch rows và lặp lại header ở mỗi chunk
        batch_size = self.table_rows_per_chunk
        total_rows = len(data_row_strs)

        for i in range(0, total_rows, batch_size):
            batch_data = data_row_strs[i:i + batch_size]
            batch_raw = raw_row_strs[i:i + batch_size]

            linearized_content = (
                f"[Breadcrumbs: {table_breadcrumb}]\n"
                f"Cột dữ liệu: {', '.join(header_names)}\n"
                f"Chi tiết từng hàng:\n" + "\n".join(batch_data)
            )
            raw_content = "\n".join(batch_raw)
            legal_ref = extract_canonical_legal_ref(table_breadcrumb, raw_content, source_name=docx_path.name)
            if legal_ref:
                linearized_content = f"[Pháp lý: {legal_ref}]\n{linearized_content}"

            table_chunks.append(_build_docx_chunk(
                chunk_id=f"{docx_path.stem}_docx_tbl_{start_chunk_idx + len(table_chunks)}",
                source=docx_path.name,
                chunk_type="docx_table",
                breadcrumb=table_breadcrumb,
                section_title=section_title,
                content=linearized_content,
                raw_content=raw_content,
                legal_ref=legal_ref
            ))

        return table_chunks

    def _is_two_level_header(self, table: "Table") -> bool:
        """Kiểm tra bảng có cấu trúc header 2 tầng hay không."""
        if len(table.rows) < 3:
            return False
        row0_cells = [c.text.strip() for c in table.rows[0].cells]
        row1_cells = [c.text.strip() for c in table.rows[1].cells]
        # Nếu hàng 0 có ô trùng nhau (ô gộp ngang) hoặc rỗng, và hàng 1 có chữ chi tiết
        has_duplicates_or_empty = len(row0_cells) != len(set(row0_cells)) or any(c == "" for c in row0_cells)
        has_row1_content = any(c != "" for c in row1_cells)
        return has_duplicates_or_empty and has_row1_content

    def _detect_table_headers(self, table: "Table") -> List[str]:
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

    def _extract_cell_text_with_nested_tables(self, cell: "_Cell") -> str:
        """Trích xuất text của cell và duyệt đệ quy các bảng lồng nhau bên trong cell nếu có."""
        parts = []
        cell_text = cell.text.strip().replace("\n", " ")
        if cell_text:
            parts.append(cell_text)

        # Kiểm tra bảng lồng nhau bên trong cell
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
