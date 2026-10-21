"""
Module Table Formatter (Universal Footnote Inlining & Stateful Group-Row)
==========================================================================
Chuyển đổi dữ liệu bảng biểu từ Vision LLM / ma trận ô thành các chunk đạt chuẩn:
1. Universal Footnote-to-Cell Inlining (0% Regex):
   - Hàn gắn trực tiếp các chú thích có ký hiệu neo (anchor) vào ô dữ liệu tương ứng:
     `- <Thực thể> | <Cột k>: <Giá trị ô> [Chú thích <anchor>: <nội dung>]`
   - Khắc phục triệt để hiện tượng đứt gãy liên kết ngữ nghĩa giữa số liệu và chú thích trong RAG.
2. Chú thích chung (General Footnotes):
   - Các chú thích toàn bảng không neo hoặc chưa được nhúng nội dòng được gắn vào khối chú thích chung.
3. Stateful Group-Row Batching:
   - Lặp lại đầy đủ ngữ cảnh, tiêu đề cột và tên nhóm khi một bảng/nhóm trải dài qua nhiều chunk.
4. Hợp đồng 7 trường (7-Field Schema Contract) & Profile B (150-400 âm tiết, trần 450):
   - `chunk_id`, `source`, `chunk_type`, `breadcrumb`, `section_title`, `content`, `word_count`.
5. 0% REGEX LAW: Tuyệt đối không dùng `import re` hay regex methods.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional, Union

from ..unified_chunk.syllable_budget_manager import build_7_field_chunk
from .vlm_table_extractor import TableExtractionResult, RowGroup, TableRow, FootnoteItem


class TableFormatType(str, Enum):
    LINEAR_BULLET = "linear_bullet"
    MARKDOWN = "markdown"
    MARKDOWN_TABLE = "markdown_table"
    COMPACT_HTML = "compact_html"
    HTML = "html"


# -----------------------------------------------------------------------------
# 1. THUẬT TOÁN SO KHỚP KÝ HIỆU NEO RANH GIỚI TUYỆT ĐỐI (0% REGEX)
# -----------------------------------------------------------------------------
def contains_anchor_exact(cell_text: str, anchor: str) -> bool:
    """
    Kiểm tra ký hiệu neo có nằm trong ô với ranh giới chính xác tuyệt đối (0% Regex):
    - (*) không khớp nhầm vào (**)
    - * không khớp nhầm vào (*)
    - (1) không khớp nhầm vào (10)
    - [1] không khớp nhầm vào [10]
    - 1 không khớp nhầm vào 10 hoặc 100
    """
    if not anchor or not cell_text:
        return False

    anchor_clean = anchor.strip()
    if not anchor_clean or anchor_clean not in cell_text:
        return False

    idx = 0
    anchor_len = len(anchor_clean)

    while True:
        pos = cell_text.find(anchor_clean, idx)
        if pos == -1:
            break

        start = pos
        end = pos + anchor_len
        is_valid = True

        # 1. Ranh giới phía trước (preceding boundary)
        if start > 0:
            prev_char = cell_text[start - 1]
            if anchor_clean.startswith("*") and (prev_char == "*" or prev_char == "("):
                is_valid = False
            elif anchor_clean.startswith("(") and prev_char == "(":
                is_valid = False
            elif anchor_clean.startswith("[") and prev_char == "[":
                is_valid = False
            elif anchor_clean.isdigit() and prev_char.isdigit():
                is_valid = False

        # 2. Ranh giới phía sau (following boundary)
        if end < len(cell_text):
            next_char = cell_text[end]
            if anchor_clean.endswith("*") and (next_char == "*" or next_char == ")"):
                is_valid = False
            elif anchor_clean.endswith(")") and next_char == ")":
                is_valid = False
            elif anchor_clean.endswith("]") and next_char == "]":
                is_valid = False
            elif anchor_clean.isdigit() and next_char.isdigit():
                is_valid = False

        if is_valid:
            return True

        idx = pos + 1

    return False


# -----------------------------------------------------------------------------
# 2. TABLE FORMATTER CLASS
# -----------------------------------------------------------------------------
class TableFormatter:
    """
    Bộ định dạng và phân đoạn bảng biểu đa lĩnh vực (Universal Table Formatter).
    Tích hợp Semantic Footnote Inlining và Stateful Group-Row Batching.
    """

    def __init__(
        self,
        default_batch_size: int = 6,
        target_syllable_budget: int = 350,
        max_syllable_budget: int = 420,
        footnote_prefix: str | None = None,
        group_prefix_template: str | None = None,
        inline_label: str = "Chú thích",
        config: object | None = None,
        **kwargs
    ):
        self.default_batch_size = default_batch_size
        self.target_syllable_budget = target_syllable_budget
        self.max_syllable_budget = max_syllable_budget
        self.inline_label = inline_label

        if footnote_prefix is not None:
            self.footnote_prefix = footnote_prefix
        elif config is not None and getattr(config, "table_footnote_prefix", None):
            self.footnote_prefix = config.table_footnote_prefix
        else:
            self.footnote_prefix = "\n[Ghi chú / Chú thích bảng]:\n"

        if group_prefix_template is not None:
            self.group_prefix_template = group_prefix_template
        elif config is not None and getattr(config, "table_group_prefix_template", None):
            self.group_prefix_template = config.table_group_prefix_template
        else:
            self.group_prefix_template = "[Nhóm / Danh mục: {group_name}]"

    def _format_row_bullet_with_inlining(
        self,
        row_header: str,
        cells: list[str],
        column_headers: list[str],
        footnotes: list[FootnoteItem]
    ) -> tuple[str, set[str]]:
        """
        Định dạng 1 hàng dữ liệu sang dạng bullet (linearized_bullet), đồng thời
        tự động nhúng trực tiếp nội dung footnote vào ô chứa ký hiệu neo:
        `- <Tên thực thể> | <Tên cột>: <Giá trị> [Chú thích <anchor>: <nội dung>]`
        """
        consumed_anchors: set[str] = set()
        pairs: list[str] = []

        # Kiểm tra xem row_header có chứa anchor nào không
        clean_row_header = row_header.strip()
        for fn in footnotes:
            if fn.anchor_symbol and contains_anchor_exact(clean_row_header, fn.anchor_symbol):
                clean_row_header = f"{clean_row_header} [{self.inline_label} {fn.anchor_symbol}: {fn.content}]"
                consumed_anchors.add(fn.anchor_symbol)

        # Duyệt qua từng ô dữ liệu
        for idx, val in enumerate(cells):
            clean_val = val.strip()
            if not clean_val or clean_val == "-":
                continue

            col_name = column_headers[idx].strip() if idx < len(column_headers) else ""

            # Nhúng các footnote có anchor xuất hiện trong ô này hoặc tiêu đề cột tương ứng
            cell_inlined = clean_val
            for fn in footnotes:
                if not fn.anchor_symbol:
                    continue
                in_cell = contains_anchor_exact(clean_val, fn.anchor_symbol)
                in_col = contains_anchor_exact(col_name, fn.anchor_symbol) if col_name else False
                if in_cell or in_col:
                    cell_inlined = f"{cell_inlined} [{self.inline_label} {fn.anchor_symbol}: {fn.content}]"
                    consumed_anchors.add(fn.anchor_symbol)

            if col_name:
                pairs.append(f"{col_name}: {cell_inlined}")
            else:
                pairs.append(cell_inlined)

        if pairs:
            val_str = " | ".join(pairs)
            if clean_row_header:
                return f"- {clean_row_header} | {val_str}", consumed_anchors
            return f"- {val_str}", consumed_anchors
        elif clean_row_header:
            return f"- {clean_row_header}", consumed_anchors
        return "", consumed_anchors

    def _format_row_bullet(
        self,
        row_header: str,
        cells: list[str],
        column_headers: list[str]
    ) -> str:
        """Định dạng hàng cơ bản không nhúng footnote (dùng cho backward compatibility)."""
        res, _ = self._format_row_bullet_with_inlining(row_header, cells, column_headers, [])
        return res

    def format_vlm_result_to_chunks(
        self,
        vlm_result: Union[TableExtractionResult, dict],
        source: str,
        chunk_id_prefix: str,
        breadcrumb: str,
        section_title: str,
        chunk_type: str = "pdf_table",
        start_index: int = 0,
        page_num: Optional[int] = None,
        bbox: Optional[list[float]] = None,
        parent_id: Optional[str] = None
    ) -> list[dict]:
        """
        Phân đoạn kết quả trích xuất VLM thành danh sách các chunk chuẩn 7 trường.
        Thực hiện Semantic Footnote Inlining, Stateful Group-Row Batching và bảo toàn chú thích chung.
        """
        if isinstance(vlm_result, dict):
            parsed = TableExtractionResult.from_dict(vlm_result)
        else:
            parsed = vlm_result

        table_title = parsed.table_title
        col_headers = parsed.column_headers
        row_groups = parsed.row_groups

        # Chuẩn hóa footnotes sang list[FootnoteItem]
        raw_footnotes = parsed.footnotes
        normalized_footnotes: list[FootnoteItem] = []
        for fn in raw_footnotes:
            if isinstance(fn, FootnoteItem):
                normalized_footnotes.append(fn)
            elif isinstance(fn, dict):
                normalized_footnotes.append(FootnoteItem.from_dict(fn))
            elif isinstance(fn, str):
                normalized_footnotes.append(FootnoteItem.from_dict(fn))

        if not row_groups:
            if table_title or normalized_footnotes:
                fn_text = "\n".join(f"  * {str(fn)}" for fn in normalized_footnotes)
                content = f"[Bảng: {table_title or 'Bảng dữ liệu'}]\n{fn_text}".strip()
                chunk_obj = build_7_field_chunk(
                    chunk_id=f"{chunk_id_prefix}_c{start_index}",
                    source=source,
                    chunk_type=chunk_type,
                    breadcrumb=breadcrumb,
                    section_title=section_title,
                    content=content,
                    original_text=content,
                    parent_id=parent_id or f"{chunk_id_prefix}_parent",
                    location=[{"page": page_num, "bbox": bbox}] if page_num is not None else []
                )
                return [chunk_obj]
            return []

        # BƯỚC 1: Format toàn bộ các hàng và xác định các anchor đã được nhúng nội dòng
        all_consumed_anchors: set[str] = set()
        formatted_groups: list[tuple[Optional[str], list[str]]] = []

        for group in row_groups:
            g_name = group.group_name
            g_lines: list[str] = []
            for row in group.rows:
                row_str, consumed = self._format_row_bullet_with_inlining(
                    row_header=row.row_header,
                    cells=row.cells,
                    column_headers=col_headers,
                    footnotes=normalized_footnotes
                )
                all_consumed_anchors.update(consumed)
                if row_str:
                    g_lines.append(row_str)
                elif row.row_header or row.cells:
                    # Đảm bảo không bỏ sót dòng nếu _format_row_bullet_with_inlining bỏ qua
                    fallback_r = f"- {row.row_header} | {' | '.join(row.cells)}".strip(" | -")
                    if fallback_r:
                        g_lines.append(f"- {fallback_r}")
            if g_lines:
                formatted_groups.append((g_name, g_lines))

        # BƯỚC 2: Gom các chú thích chung (không có anchor, hoặc có anchor nhưng chưa được nhúng vào ô nào)
        general_footnotes: list[FootnoteItem] = []
        for fn in normalized_footnotes:
            if not fn.anchor_symbol or (fn.anchor_symbol not in all_consumed_anchors):
                general_footnotes.append(fn)

        footnote_block = ""
        if general_footnotes:
            fn_lines = [f"  * {str(fn)}" for fn in general_footnotes if str(fn).strip()]
            if fn_lines:
                footnote_block = self.footnote_prefix + "\n".join(fn_lines)

        col_header_line = f"Cột: {' | '.join(col_headers)}" if col_headers else ""

        chunks: list[dict] = []
        counter = start_index

        # BƯỚC 3: Đóng gói các hàng theo ngân sách Profile B (150-400 âm tiết, trần 450)
        for group_name, row_lines in formatted_groups:
            current_row_lines: list[str] = []

            for row_str in row_lines:
                test_lines = current_row_lines + [row_str]
                test_body = "\n".join(test_lines)
                estimated_words = len(test_body.split())

                # Khi vượt ngưỡng tối đa (Profile B max ~420 words) và đã có ít nhất 2 dòng
                if estimated_words > self.max_syllable_budget and len(current_row_lines) >= 2:
                    chunk_text = self._assemble_table_chunk_content(
                        breadcrumb=breadcrumb,
                        section_title=section_title,
                        table_title=table_title,
                        col_header_line=col_header_line,
                        group_name=group_name,
                        row_lines=current_row_lines,
                        footnote_block=footnote_block
                    )
                    table_loc = [{"page": page_num, "bbox": bbox}] if page_num is not None else []
                    p_id = parent_id or f"{chunk_id_prefix}_parent"
                    chunk_obj = build_7_field_chunk(
                        chunk_id=f"{chunk_id_prefix}_c{counter}",
                        source=source,
                        chunk_type=chunk_type,
                        breadcrumb=breadcrumb,
                        section_title=section_title,
                        content=chunk_text,
                        original_text=chunk_text,
                        parent_id=p_id,
                        location=table_loc
                    )
                    chunks.append(chunk_obj)
                    counter += 1
                    current_row_lines = [row_str]
                else:
                    current_row_lines.append(row_str)

            # Đóng gói phần còn lại của nhóm
            if current_row_lines:
                chunk_text = self._assemble_table_chunk_content(
                    breadcrumb=breadcrumb,
                    section_title=section_title,
                    table_title=table_title,
                    col_header_line=col_header_line,
                    group_name=group_name,
                    row_lines=current_row_lines,
                    footnote_block=footnote_block
                )
                table_loc = [{"page": page_num, "bbox": bbox}] if page_num is not None else []
                p_id = parent_id or f"{chunk_id_prefix}_parent"
                chunk_obj = build_7_field_chunk(
                    chunk_id=f"{chunk_id_prefix}_c{counter}",
                    source=source,
                    chunk_type=chunk_type,
                    breadcrumb=breadcrumb,
                    section_title=section_title,
                    content=chunk_text,
                    original_text=chunk_text,
                    parent_id=p_id,
                    location=table_loc
                )
                chunks.append(chunk_obj)
                counter += 1

        return chunks

    def _assemble_table_chunk_content(
        self,
        breadcrumb: str,
        section_title: str,
        table_title: Optional[str],
        col_header_line: str,
        group_name: Optional[str],
        row_lines: list[str],
        footnote_block: str
    ) -> str:
        """
        Lắp ráp toàn bộ các thành phần của 1 chunk bảng biểu đạt Hợp đồng 7 trường:
        1. Contextual Header: [Ngữ cảnh: {breadcrumb} > {section_title}]
        2. Tiêu đề bảng biểu
        3. Tiêu đề cột làm phẳng
        4. Tên nhóm chỉ tiêu (Stateful Group-Row)
        5. Các dòng dữ liệu (đã nhúng footnote nội dòng)
        6. Chú thích chung (nếu có)
        """
        parts: list[str] = []

        # 1. Header ngữ cảnh phả hệ
        if breadcrumb and section_title and section_title != breadcrumb and not breadcrumb.endswith(section_title):
            parts.append(f"[Ngữ cảnh: {breadcrumb} > {section_title}]")
        elif breadcrumb:
            parts.append(f"[Ngữ cảnh: {breadcrumb}]")
        elif section_title:
            parts.append(f"[Ngữ cảnh: {section_title}]")

        # 2. Tiêu đề bảng
        clean_tbl_title = table_title.strip() if table_title else ""
        if clean_tbl_title and clean_tbl_title.lower() not in ("null", "none", "string or null", "string"):
            parts.append(f"Bảng biểu: {clean_tbl_title}")

        # 3. Tiêu đề cột
        if col_header_line:
            parts.append(col_header_line)

        # 4. Nhóm chỉ tiêu / Danh mục (Stateful)
        clean_group = group_name.strip() if group_name else ""
        if clean_group and clean_group.lower() not in ("null", "none", "string or null", "string"):
            parts.append(self.group_prefix_template.format(group_name=clean_group))

        # 5. Dòng dữ liệu
        parts.extend(row_lines)

        # 6. Chú thích chung
        if footnote_block:
            parts.append(footnote_block)

        return "\n".join(parts).strip()

    def format_matrix_to_chunks(
        self,
        headers: list[str],
        matrix: list[list[str]],
        source: str,
        chunk_id_prefix: str,
        breadcrumb: str,
        section_title: str,
        chunk_type: str = "table",
        start_index: int = 0,
        footnotes: Optional[list[Union[str, FootnoteItem]]] = None,
        row_groups: Optional[list[RowGroup]] = None,
        table_title: Optional[str] = None,
        page_num: Optional[int] = None,
        bbox: Optional[list[float]] = None,
        parent_id: Optional[str] = None
    ) -> list[dict]:
        """Hỗ trợ phân đoạn trực tiếp từ ma trận ô 2D hoặc danh sách RowGroup chuẩn Topo học."""
        if row_groups is not None:
            active_groups = row_groups
        else:
            table_rows: list[TableRow] = []
            for r in matrix:
                if not r:
                    continue
                r_hdr = r[0] if len(r) > 0 else ""
                cells = r[1:] if len(r) > 1 else []
                table_rows.append(TableRow(row_header=r_hdr, cells=cells))
            active_groups = [RowGroup(group_name=None, rows=table_rows)]

        fn_items: list[FootnoteItem] = []
        if footnotes:
            for f in footnotes:
                if isinstance(f, FootnoteItem):
                    fn_items.append(f)
                else:
                    fn_items.append(FootnoteItem.from_dict(f))

        vlm_res = TableExtractionResult(
            table_title=table_title,
            column_headers=headers[1:] if len(headers) > 1 else headers,
            row_groups=active_groups,
            footnotes=fn_items
        )
        return self.format_vlm_result_to_chunks(
            vlm_result=vlm_res,
            source=source,
            chunk_id_prefix=chunk_id_prefix,
            breadcrumb=breadcrumb,
            section_title=section_title,
            chunk_type=chunk_type,
            start_index=start_index,
            page_num=page_num,
            bbox=bbox,
            parent_id=parent_id
        )

    # =========================================================================
    # CÁC HÀM TƯƠNG THÍCH NGƯỢC (BACKWARD COMPATIBILITY)
    # =========================================================================
    def format(self, text: str, format_type: object = TableFormatType.MARKDOWN_TABLE) -> str:
        return text

    @staticmethod
    def format_table(headers: list[str], rows: list[list[str]], format_type: str = "markdown_table") -> str:
        lines: list[str] = []
        for r_idx, row in enumerate(rows, start=1):
            desc = " | ".join(f"{h}: {v}" for h, v in zip(headers, row) if v)
            if desc:
                lines.append(f"* Hàng {r_idx}: {desc}")
        return "\n".join(lines)


__all__ = ["TableFormatType", "TableFormatter", "contains_anchor_exact"]
