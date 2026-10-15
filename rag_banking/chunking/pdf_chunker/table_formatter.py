"""
Module Table Formatter (Bộ Chuyển đổi và Thử nghiệm 3 Định dạng Bảng Biểu Ngân hàng)
=====================================================================================
Giải quyết bài toán tối ưu biểu diễn bảng biểu phức tạp cho bộ nạp RAG và LLM.

Hỗ trợ 3 định dạng cấu trúc theo yêu cầu nghiên cứu:
1. `linearized_bullet`:
   - Tuyến tính hóa từng hàng theo cấu trúc câu tự nhiên:
     `* Hàng X: Cột A là Val A | Cột B là Val B`
   - Phù hợp nhất cho Dense Vector Embeddings và Dense Retrieval (ngữ cảnh hàng độc lập).
2. `markdown_table` (Batched Header Retention):
   - Bảng Markdown chuẩn có cơ chế tự động lặp lại Header sau mỗi lô N hàng (mặc định N = 10).
   - Bảo toàn ngữ cảnh định danh cột cho các bảng biểu dài, chống việc LLM bị mất dấu tiêu đề cột.
3. `compact_html` (Zero-style Minimalist HTML):
   - Bảng HTML tối giản (`<table><thead><tr><th>...</th></tr></thead><tbody><tr><td>...</td></tr></tbody>`).
   - Giữ nguyên vẹn 100% cấu trúc ô gộp đa tầng (`rowspan`, `colspan`), loại bỏ toàn bộ thuộc tính
     thừa (style, class, id) để tiết kiệm tối đa số lượng token trong context window.

TUÂN THỦ 100% SCHEMA CONTRACT:
- Bắt buộc có: `chunk_id`, `source`, `chunk_type`, `breadcrumb`, `section_title`, `content`, `word_count`.
- Tuyệt đối không sinh trường rác ('ref', 'legal_ref', 'raw_content', 'char_length', 'format').
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

try:
    from rag_banking.chunking.unified_chunk.canonical_resolver import count_words
except (ImportError, ValueError):
    from ...unified_chunk.canonical_resolver import count_words


class TableFormatType(str, Enum):
    """3 định dạng bảng biểu thử nghiệm."""
    LINEARIZED_BULLET = "linearized_bullet"
    MARKDOWN_TABLE = "markdown_table"
    COMPACT_HTML = "compact_html"


@dataclass
class TableCell:
    """Đại diện cho một ô trong bảng, hỗ trợ thuộc tính ô gộp."""
    text: str
    rowspan: int = 1
    colspan: int = 1

    def to_clean_text(self) -> str:
        return " ".join(self.text.split())


@dataclass
class NormalizedTable:
    """Cấu trúc dữ liệu bảng biểu chuẩn hóa trung gian (Intermediate Representation)."""
    headers: list[str] = field(default_factory=list)
    rows: list[list[TableCell]] = field(default_factory=list)
    caption: str = ""

    @classmethod
    def from_markdown(cls, md_text: str, caption: str = "") -> NormalizedTable:
        """Parse chuỗi bảng Markdown thành NormalizedTable."""
        lines = [line.strip() for line in md_text.strip().split("\n") if line.strip()]
        if not lines:
            return cls(caption=caption)

        headers: list[str] = []
        rows: list[list[TableCell]] = []

        # Tách dòng có chứa '|'
        table_lines = [l for l in lines if l.startswith("|") and l.endswith("|")]
        if not table_lines:
            return cls(caption=caption)

        def split_row(row_str: str) -> list[str]:
            raw_cells = row_str.strip("|").split("|")
            return [" ".join(c.strip().split()) for c in raw_cells]

        # Dòng 0: Header
        raw_headers = split_row(table_lines[0])
        headers = raw_headers

        start_row_idx = 1
        # Nếu dòng 1 là separator (--- | ---)
        if len(table_lines) > 1 and all(c in "-: |" for c in table_lines[1]):
            start_row_idx = 2

        for r_line in table_lines[start_row_idx:]:
            # Bỏ qua dòng separator bị lặp lại
            if all(c in "-: |" for c in r_line):
                continue
            cell_strs = split_row(r_line)
            # Khắc phục độ dài không đều
            if len(cell_strs) < len(headers):
                cell_strs.extend([""] * (len(headers) - len(cell_strs)))
            elif len(cell_strs) > len(headers):
                cell_strs = cell_strs[:len(headers)]

            row_cells = [TableCell(text=c) for c in cell_strs]
            rows.append(row_cells)

        return cls(headers=headers, rows=rows, caption=caption)

    @classmethod
    def from_matrix(
        cls,
        headers: list[str],
        matrix: list[list[Any]],
        caption: str = ""
    ) -> NormalizedTable:
        """Khởi tạo từ danh sách ma trận thuần túy."""
        clean_headers = [" ".join(str(h).split()) for h in headers]
        rows: list[list[TableCell]] = []
        for r in matrix:
            row_cells: list[TableCell] = []
            for item in r:
                if isinstance(item, TableCell):
                    row_cells.append(item)
                elif isinstance(item, dict):
                    row_cells.append(TableCell(
                        text=str(item.get("text", "")),
                        rowspan=int(item.get("rowspan", 1)),
                        colspan=int(item.get("colspan", 1))
                    ))
                else:
                    row_cells.append(TableCell(text=str(item)))
            rows.append(row_cells)
        return cls(headers=clean_headers, rows=rows, caption=caption)


class TableFormatter:
    """
    Bộ chuyển đổi định dạng bảng sang 3 chuẩn biểu diễn (Bullet, Markdown, Compact HTML)
    kèm cơ chế đóng gói Chunk chuẩn Schema Contract.
    """

    def __init__(self, default_batch_size: int = 10):
        self.default_batch_size = default_batch_size

    def format(
        self,
        table: NormalizedTable | str,
        format_type: TableFormatType | str = TableFormatType.LINEARIZED_BULLET,
        batch_size: int | None = None
    ) -> str:
        """
        Chuyển đổi bảng sang định dạng chỉ định.
        """
        if isinstance(table, str):
            table = NormalizedTable.from_markdown(table)

        fmt = TableFormatType(format_type)
        if fmt == TableFormatType.LINEARIZED_BULLET:
            return self.to_linearized_bullet(table)
        elif fmt == TableFormatType.MARKDOWN_TABLE:
            return self.to_markdown_table_batched(table, batch_size=batch_size or self.default_batch_size)
        elif fmt == TableFormatType.COMPACT_HTML:
            return self.to_compact_html(table)
        else:
            raise ValueError(f"Định dạng bảng không hợp lệ: {format_type}")

    # -------------------------------------------------------------------------
    # 1. Định dạng 1: Linearized Bullet (* Hàng X: Cột A là ..., Cột B là ...)
    # -------------------------------------------------------------------------
    def to_linearized_bullet(self, table: NormalizedTable) -> str:
        """
        Tuyến tính hóa bảng thành danh sách bullet points tự giải thích ngữ nghĩa.
        """
        if not table.rows:
            return ""

        headers = table.headers or [f"Cột {i + 1}" for i in range(len(table.rows[0]))]
        bullet_lines: list[str] = []

        if table.caption:
            bullet_lines.append(f"**Bảng: {table.caption}**")

        for row_idx, row in enumerate(table.rows, start=1):
            cell_descriptions: list[str] = []
            for col_idx, cell in enumerate(row):
                header_name = headers[col_idx] if col_idx < len(headers) else f"Cột {col_idx + 1}"
                val = cell.to_clean_text()
                if not val:
                    val = "Không áp dụng"
                cell_descriptions.append(f"{header_name} là {val}")

            row_text = f"* Hàng {row_idx}: " + " | ".join(cell_descriptions)
            bullet_lines.append(row_text)

        return "\n".join(bullet_lines)

    # -------------------------------------------------------------------------
    # 2. Định dạng 2: Markdown Table với Header lặp lại mỗi batch N hàng
    # -------------------------------------------------------------------------
    def to_markdown_table_batched(
        self,
        table: NormalizedTable,
        batch_size: int = 10
    ) -> str:
        """
        Tạo bảng Markdown chuẩn, lặp lại Header sau mỗi `batch_size` hàng để duy trì context.
        """
        if not table.rows:
            return ""

        headers = table.headers or [f"Cột {i + 1}" for i in range(len(table.rows[0]))]
        header_line = "| " + " | ".join(headers) + " |"
        separator_line = "| " + " | ".join(["---"] * len(headers)) + " |"

        output_lines: list[str] = []
        if table.caption:
            output_lines.append(f"**Bảng: {table.caption}**\n")

        for row_idx, row in enumerate(table.rows):
            # Nếu bắt đầu một batch mới: in Header
            if row_idx % batch_size == 0:
                if row_idx > 0:
                    output_lines.append("")  # Dòng trống ngăn cách batch
                output_lines.append(header_line)
                output_lines.append(separator_line)

            cell_texts = [cell.to_clean_text() for cell in row]
            # Điền thiếu nếu hàng ngắn hơn header
            while len(cell_texts) < len(headers):
                cell_texts.append("")
            row_line = "| " + " | ".join(cell_texts[:len(headers)]) + " |"
            output_lines.append(row_line)

        return "\n".join(output_lines)

    # -------------------------------------------------------------------------
    # 3. Định dạng 3: Compact HTML giữ nguyên rowspan và colspan đa tầng
    # -------------------------------------------------------------------------
    def to_compact_html(self, table: NormalizedTable) -> str:
        """
        Tạo HTML tối giản, loại bỏ toàn bộ styling/whitespace dư thừa nhưng bảo toàn
        100% thuộc tính ô gộp `rowspan` và `colspan` phục vụ cấu trúc đa tầng.
        """
        if not table.rows:
            return ""

        parts: list[str] = ["<table>"]

        if table.caption:
            parts.append(f"<caption>{table.caption}</caption>")

        # Header
        if table.headers:
            parts.append("<thead><tr>")
            for h in table.headers:
                parts.append(f"<th>{h}</th>")
            parts.append("</tr></thead>")

        # Body
        parts.append("<tbody>")
        for row in table.rows:
            parts.append("<tr>")
            for cell in row:
                attrs = []
                if cell.rowspan > 1:
                    attrs.append(f'rowspan="{cell.rowspan}"')
                if cell.colspan > 1:
                    attrs.append(f'colspan="{cell.colspan}"')
                attr_str = (" " + " ".join(attrs)) if attrs else ""
                parts.append(f"<td{attr_str}>{cell.to_clean_text()}</td>")
            parts.append("</tr>")
        parts.append("</tbody></table>")

        return "".join(parts)

    # -------------------------------------------------------------------------
    # 4. Đóng gói Chunk tuân thủ nghiêm ngặt Schema Contract
    # -------------------------------------------------------------------------
    def create_table_chunk(
        self,
        table: NormalizedTable | str,
        chunk_id: str,
        source: str,
        breadcrumb: str,
        section_title: str,
        format_type: TableFormatType | str = TableFormatType.LINEARIZED_BULLET,
        batch_size: int = 10,
        page_start: int | None = None,
        page_end: int | None = None
    ) -> dict[str, Any]:
        """
        Tạo đối tượng chunk bảng biểu tuân thủ nghiêm ngặt Schema Contract:
        - Bắt buộc có: chunk_id, source, chunk_type, breadcrumb, section_title, content, word_count.
        - TUYỆT ĐỐI KHÔNG sinh các trường rác: 'ref', 'legal_ref', 'raw_content', 'char_length', 'format'.
        """
        fmt_str = str(format_type.value if hasattr(format_type, "value") else format_type)
        formatted_table = self.format(table, format_type=fmt_str, batch_size=batch_size)

        full_content = f"[Breadcrumbs: {breadcrumb}]\n{formatted_table}".strip()

        chunk_data: dict[str, Any] = {
            "chunk_id": chunk_id,
            "source": source,
            "chunk_type": f"pdf_table_{fmt_str}",
            "breadcrumb": breadcrumb,
            "section_title": section_title,
            "content": full_content,
            "word_count": count_words(full_content)
        }

        # Bổ sung các trường trang an toàn nếu được cung cấp
        if page_start is not None:
            chunk_data["page_start"] = page_start
            chunk_data["page_end"] = page_end or page_start
            chunk_data["page_numbers"] = list(range(chunk_data["page_start"], chunk_data["page_end"] + 1))

        return chunk_data
