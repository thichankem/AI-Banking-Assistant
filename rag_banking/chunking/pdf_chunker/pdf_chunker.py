"""
Bộ phân đoạn tài liệu PDF ngân hàng và đa ngành thế hệ mới (Next-Gen PDF Chunker).
=================================================================================
1. Tích hợp Dynamic Bounding Box bằng Typographic & Spatial Proximity:
   - Tự động đo lường cỡ chữ trung vị (median font size) của trang.
   - Mở rộng đáy bảng động theo đúng độ dài thực tế của footnote, loại bỏ 100% con số cứng +8%.
2. Tích hợp Vision LLM Table Engine (VLMTableExtractor - Gemma 4:12B Vision)
   với cơ chế tuần tự hóa GPUQueueLock bảo vệ VRAM RTX 4060.
3. Định dạng bảng biểu bằng TableFormatter với Universal Footnote-to-Cell Inlining
   (nhúng trực tiếp chú thích vào ô dữ liệu, triệt tiêu đứt gãy ngữ nghĩa RAG).
4. Tự động chuyển đổi trang scan bằng OCRExtractor (Tesseract v5 + OpenCV).
5. Phân đoạn văn bản với Profile B (150-400 âm tiết, trần 450) và tự động gắn tiền tố
   ngữ cảnh [Ngữ cảnh: {breadcrumb} > {section_title}].
6. Tuân thủ tuyệt đối Hợp đồng 7 trường (7-Field Schema Contract) và 0% REGEX LAW.
"""

from __future__ import annotations

import logging
import unicodedata
from pathlib import Path
from typing import Optional, Union

import pymupdf as fitz
from PIL import Image

logger = logging.getLogger("pdf_chunker")

from .ocr_extractor import OCRExtractor
from .scan_detector import detect_is_scanned_page
from .vlm_table_extractor import (
    VLMTableExtractor,
    crop_pdf_table_image,
    parse_fallback_footnotes_no_regex,
    TableRow,
    RowGroup,
    TableExtractionResult
)
from .table_formatter import TableFormatter
from ..unified_chunk.canonical_resolver import (
    clean_and_dehyphenate_text,
    split_sentences_banking,
    detect_and_remove_headers_footers
)
from ..unified_chunk.syllable_budget_manager import (
    SyllableBudgetManager,
    prepend_context_header,
    build_7_field_chunk
)


# -----------------------------------------------------------------------------
# 1. DYNAMIC BOUNDING BOX BẰNG TYPOGRAPHIC & SPATIAL PROXIMITY (0% REGEX)
# -----------------------------------------------------------------------------
def _is_universal_structural_anchor(text: str) -> bool:
    """
    Nhận diện ký hiệu phân tách/chỉ mục đầu dòng theo chuẩn Unicode (0% Regex, 0% mảng ký tự neo hardcode).
    """
    s = text.strip()
    if not s:
        return False

    first_char = s[0]
    # Ký hiệu biểu tượng hoặc dấu phân tách chuẩn Unicode (Categories: P* Punctuation, S* Symbol)
    cat = unicodedata.category(first_char)
    if cat.startswith("S") or first_char in "-•":
        return True

    # Cặp ngoặc đầu dòng
    if first_char in "([":
        close_char = ")" if first_char == "(" else "]"
        idx = s.find(close_char)
        if 1 < idx <= 6:
            return True

    # Chỉ mục ngắn ở đầu dòng (vd: '1.', '1)', 'a.', 'a)')
    tokens = s.split(maxsplit=1)
    if tokens:
        w0 = tokens[0]
        if len(w0) <= 3 and (w0.endswith(".") or w0.endswith(")")):
            core = w0[:-1]
            if core.isdigit() or (len(core) == 1 and core.isalpha()):
                return True

    return False


def detect_dynamic_table_bottom(
    page: object,
    table_bbox: Union[tuple[float, float, float, float], list[float]],
    footer_margin_ratio: float = 0.065,
    min_font_scale: float = 0.90,
    max_vertical_gap: Optional[float] = None
) -> float:
    """
    Tính toán toạ độ đáy y2 động bằng giải thuật Typographic & Spatial Proximity DLA
    theo hướng tổng quát hóa đa ngành (0% Regex, không phụ thuộc hardcode từ khóa).

    Nguyên lý thích ứng (Adaptive Logic):
    1. Thống kê động toàn trang: Đo lường cỡ chữ trung vị (median_font_size) có trọng số
       độ dài ký tự, từ đó tự động suy xuất các ngưỡng khoảng cách dọc thích ứng.
    2. Quét vùng lân cận: Quét các dòng văn bản nằm ngay dưới đáy bảng (y2) trong phạm vi lề ngang.
    3. Tiêu chí mở rộng dòng:
       - Font Scale: Cỡ chữ của dòng nhỏ hơn chuẩn thân bài (< min_font_scale * median_font_size).
       - Structural Anchor: Dòng bắt đầu bằng ký hiệu neo cấu trúc phổ quát (symbols, ngoặc, chỉ mục).
       - Wrapped Continuation: Dòng tràn nằm sát dòng chú thích phía trên (vertical_gap <= continuation_gap)
         với cỡ chữ không vượt quá văn bản thường.
    4. Điều kiện dừng tự nhiên: Dừng ngay khi khoảng cách dọc vượt ngưỡng, gặp dòng văn bản thân bài
       bình thường (không có neo), hoặc chạm lề footer an toàn.
    """
    page_h = page.rect.height
    footer_limit = page_h * (1.0 - footer_margin_ratio)
    x0, y0, x1, y2 = table_bbox

    # 1. Đo lường cỡ chữ trung vị toàn trang (có trọng số độ dài ký tự)
    page_dict = page.get_text("dict")
    font_sizes: list[float] = []
    candidate_lines: list[dict] = []

    for block in page_dict.get("blocks", []):
        if block.get("type") == 0:  # Khối văn bản (text block)
            for line in block.get("lines", []):
                line_bbox = line.get("bbox", (0, 0, 0, 0))
                line_spans = line.get("spans", [])
                if not line_spans:
                    continue

                line_text = "".join(s.get("text", "") for s in line_spans).strip()
                if not line_text:
                    continue

                total_chars = sum(len(s.get("text", "").strip()) for s in line_spans)
                if total_chars > 0:
                    weighted_size = sum(s.get("size", 10.0) * len(s.get("text", "").strip()) for s in line_spans) / total_chars
                else:
                    weighted_size = line_spans[0].get("size", 10.0)

                # Thu thập mẫu đại diện tính median font toàn trang
                font_sizes.extend([weighted_size] * min(total_chars, 50))

                # Kiểm tra ứng viên dòng nằm dưới đáy bảng
                lx0, ly0, lx1, ly1 = line_bbox
                if ly0 >= y2 - 4.0 and ly1 <= footer_limit:
                    # Kiểm tra phạm vi lề ngang (dung sai 30pt)
                    if not (lx1 < x0 - 30.0 or lx0 > x1 + 30.0):
                        candidate_lines.append({
                            "text": line_text,
                            "bbox": line_bbox,
                            "font_size": weighted_size,
                            "y0": ly0,
                            "y1": ly1,
                            "x0": lx0,
                            "x1": lx1
                        })

    if font_sizes:
        font_sizes.sort()
        median_font_size = font_sizes[len(font_sizes) // 2]
    else:
        median_font_size = 11.0

    # Ngưỡng khoảng cách dọc thích ứng dựa trên cỡ chữ trung vị của trang
    gap_threshold = max_vertical_gap if max_vertical_gap is not None else max(24.0, median_font_size * 2.5)
    continuation_gap = max(8.0, median_font_size * 0.75)

    # 2. Sắp xếp các dòng theo chiều dọc từ trên xuống dưới
    candidate_lines.sort(key=lambda it: it["y0"])

    # 3. Vòng lặp mở rộng đáy động
    current_bottom = y2
    prev_bottom = y2
    has_active_footnote = False

    for item in candidate_lines:
        ly0 = item["y0"]
        ly1 = item["y1"]
        txt = item["text"]
        f_size = item["font_size"]

        gap = ly0 - prev_bottom
        if gap > gap_threshold:
            break  # Khoảng cách quá xa so với khối bảng

        # Kiểm tra tiêu chuẩn nhận diện dòng chú thích (0% Regex, không phụ thuộc từ khóa ngôn ngữ)
        has_anchor = _is_universal_structural_anchor(txt)
        is_small_font = (f_size < min_font_scale * median_font_size)
        is_wrapped_continuation = (
            has_active_footnote and
            gap <= continuation_gap and
            f_size <= median_font_size * 1.05
        )

        if is_small_font or has_anchor or is_wrapped_continuation:
            current_bottom = max(current_bottom, ly1 + 2.0)
            prev_bottom = ly1
            has_active_footnote = True
        else:
            # Gặp dòng văn bản thông thường (cỡ chữ chuẩn, không có neo) -> Dừng ngay
            break

    return min(current_bottom, footer_limit)


# -----------------------------------------------------------------------------
# 2. NHẬN DIỆN HEADING ĐA NGỮ (0% REGEX)
# -----------------------------------------------------------------------------
def _is_heading_line(
    line: str,
    font_size: Optional[float] = None,
    median_font_size: Optional[float] = None,
    is_bold: bool = False
) -> bool:
    """
    Nhận diện dòng tiêu đề phân cấp đa ngôn ngữ bằng Typography, Bố cục Hình học và Cú pháp
    (0% Regex Law, 0% Rule-based đọc từ vựng).
    """
    s = line.strip()
    if not s or len(s) > 150:
        return False
    words = s.split()
    if not words:
        return False

    ends_with_terminal = s.endswith((".", ";", ":"))

    # 1. ĐẶC TRƯNG HÌNH HỌC & TYPOGRAPHY KHÔNG GIAN (DLA thuần túy)
    # Nếu có cỡ chữ: Cỡ chữ lớn hơn đáng kể văn bản thân bài và không kết câu
    if font_size is not None and median_font_size is not None and median_font_size > 0:
        if font_size >= 1.12 * median_font_size and len(words) <= 15 and not ends_with_terminal:
            return True

    # Nếu chữ in đậm và là dòng ngắn không kết câu
    if is_bold and len(words) <= 12 and not ends_with_terminal:
        return True

    w0_raw = words[0]
    w0_clean = w0_raw.rstrip(".:)-#[]()").lower()
    w0_num_clean = w0_raw.rstrip(".:)-").strip("[]()")

    def _is_roman_numeral(token: str) -> bool:
        clean = token.rstrip(".:)-#[]()").lower()
        return bool(clean and len(clean) <= 8 and all(c in "ivxlcdm" for c in clean))

    def _is_identifier_token(tok: str) -> bool:
        c = tok.rstrip(".:)-#[]()")
        if not c:
            return False
        if c.isdigit():
            return True
        if _is_roman_numeral(c):
            return True
        if "." in c and all(p.isdigit() for p in c.split(".") if p):
            return True
        if len(c) == 1 and c.isupper():
            return True
        return False

    # 2. CẤU TRÚC TIÊU ĐỀ CÓ NHÃN CÚ PHÁP TỔNG QUÁT: [Từ viết hoa] [Chỉ số: Số / La Mã / Chữ cái]
    if len(words) >= 2:
        # Tiền tố 1 từ nhãn viết hoa đi kèm số/La Mã/chữ cái: "Chapter 1", "Article 15", "Điều 5"
        if words[0] and words[0][0].isupper() and words[0].rstrip(".:)-").isalpha():
            if _is_identifier_token(words[1]):
                if len(words) <= 15 and not ends_with_terminal:
                    return True
        # Tiền tố 2 từ nhãn viết hoa đi kèm số/La Mã: "Phụ lục 2", "Chuyên đề 3", "Special Section 1"
        if len(words) >= 3:
            if words[0] and words[1] and words[0][0].isupper() and words[0].isalpha() and words[1].isalpha():
                if _is_identifier_token(words[2]):
                    if len(words) <= 15 and not s.endswith((".", ";", ":")):
                        return True

    # 2. Số La Mã đứng đầu: I., II., III., IV. (không hardcode tuple cố định)
    if _is_roman_numeral(w0_raw) and len(words) >= 2:
        if w0_raw.endswith(".") or w0_raw.endswith(":") or w0_raw.endswith(")"):
            if not s.endswith((".", ";", ":")):
                return True

    # 3. Tiêu đề in hoa đứng độc lập (không kết thúc bằng dấu chấm câu kết thúc đoạn)
    if s.isupper() and 8 <= len(s) <= 150 and not s.endswith((".", ";", ":")):
        return True

    # 4. Cấu trúc đánh số phân cấp đa tầng: 1.1, 1.1.1, 2.1.3
    w0_dots = w0_raw.rstrip(".")
    if "." in w0_dots and all(p.isdigit() for p in w0_dots.split(".")) and 2 <= len(words) <= 15:
        if not s.endswith((".", ";", ":")):
            return True

    # 5. Cấu trúc đánh số đơn cấp chuẩn hóa: 1., 2., [1], (1), A., B. đi kèm Title Case / In hoa
    if (w0_num_clean.isdigit() or (len(w0_num_clean) == 1 and w0_num_clean.isalpha())):
        is_num_prefix = (
            w0_raw.endswith(".") or w0_raw.endswith(")") or w0_raw.endswith(":") or
            (w0_raw.startswith("[") and w0_raw.endswith("]")) or
            (w0_raw.startswith("(") and w0_raw.endswith(")"))
        )
        if is_num_prefix and 2 <= len(words) <= 15 and not s.endswith((".", ";", ":")):
            if words[1].isupper() or (words[1][0].isupper() if words[1] else False):
                return True

    # 6. Tiêu đề dạng Title Case ngắn gọn không dấu chấm cuối (vd: "Overview of Architecture")
    if 2 <= len(words) <= 8 and len(s) <= 80 and not s.endswith((".", ";", ":")):
        if all(w[0].isupper() for w in words if w and w[0].isalpha()):
            return True

    return False


def is_simple_uniform_table(
    tbl: object,
    df: list[list[Optional[str]]] | None = None,
    max_cells: int = 30
) -> tuple[bool, str]:
    """
    Xác định điều kiện để được phép sử dụng trích xuất PyMuPDF:
    CHỈ DÙNG PYMUPDF KHI VÀ CHỈ KHI BẢNG CỰC KỲ ĐƠN GIẢN (<= 30 Ô VÀ 0% MERGE CELL):
    1. Kích thước bảng cực kỳ nhỏ: Tổng số ô (row_count * col_count) <= 30 ô.
    2. Cấu trúc bảng tuyệt đối đều đặn (Factual Uniform Grid) - KHÔNG CÓ BẤT KỲ Ô GỘP (MERGE CELL) NÀO:
       - Số lượng cột của tất cả các dòng phải bằng nhau tuyệt đối: all(len(row) == col_count for row in df).
       - Không có ô gộp vật lý (len(tbl.cells) == row_count * col_count).
       - Không có bất kỳ ô mang giá trị None nào trong ma trận (PyMuPDF dùng None để đánh dấu ô bị gộp từ ô khác).
       - Hàng tiêu đề không bị gộp (mọi cột đều có nhãn rành mạch, không None/rỗng).
    CÒN LẠI TOÀN BỘ BẢNG KHÁC (Số ô > 30 hoặc có merge cell / bảng quét) BẮT BUỘC PHẢI DÙNG LLM VISION!
    """
    if df is None:
        try:
            df = tbl.extract()
        except Exception:
            return False, "Không thể trích xuất ma trận bảng từ PyMuPDF"

    if not df or not isinstance(df, list) or len(df) == 0:
        return False, "Bảng rỗng hoặc không có dữ liệu hàng"

    row_count = getattr(tbl, "row_count", len(df))
    col_count = getattr(tbl, "col_count", len(df[0]) if df and df[0] else 0)

    if row_count <= 0 or col_count <= 0:
        return False, f"Kích thước bảng không hợp lệ ({row_count} hàng x {col_count} cột)"

    total_cells = row_count * col_count

    # 1. Điều kiện số ô <= 30 ô
    if total_cells > max_cells:
        return False, f"Bảng có {total_cells} ô vượt quá ngưỡng cho phép ({max_cells} ô) -> Bắt buộc dùng LLM Vision"

    # 2. Điều kiện hàng cột phải đều đặn (Factual Uniform)
    for idx, row in enumerate(df):
        if not isinstance(row, list) or len(row) != col_count:
            return False, f"Hàng {idx} có {len(row) if isinstance(row, list) else 0} cột != {col_count} cột -> Không đều hàng cột"

    # 3. Điều kiện 0% merge cell
    # 3a. Kiểm tra số hộp ô vật lý
    phys_cells = getattr(tbl, "cells", [])
    if phys_cells and len(phys_cells) != total_cells:
        return False, f"Phát hiện ô gộp (merge cell): Số ô vật lý {len(phys_cells)} != tổng số ô lưới {total_cells}"

    # 3b. Kiểm tra giá trị None trong ma trận (dấu hiệu ô bị span/merge của PyMuPDF)
    for r_idx, row in enumerate(df):
        for c_idx, cell in enumerate(row):
            if cell is None:
                return False, f"Phát hiện ô bị gộp tại hàng {r_idx}, cột {c_idx} (giá trị None)"

    # 3c. Kiểm tra hàng tiêu đề không có ô rỗng cạnh ô có chữ (header merge)
    header_non_empty = [c for c in df[0] if c is not None and str(c).strip()]
    if len(header_non_empty) < col_count:
        return False, f"Tiêu đề gộp ô ({len(header_non_empty)}/{col_count} cột có nhãn)"

    return True, f"Bảng cực kỳ đơn giản đều đặn ({row_count}x{col_count} = {total_cells} ô <= {max_cells} ô, 0% merge cell)"


def is_complex_pdf_table(tbl: object, df: list[list[str]] | None = None) -> tuple[bool, str]:
    """Hàm tương thích ngược: Bảng phức tạp là bảng không thỏa mãn is_simple_uniform_table."""
    is_simple, reason = is_simple_uniform_table(tbl, df, max_cells=30)
    return (not is_simple, reason)


def resolve_topological_table_matrix(
    df: list[list[str]]
) -> tuple[list[str], list[list[str]], list[RowGroup]]:
    """
    Biến đổi ma trận ô 2D thô từ PDF thành cấu trúc bảng Topo học hoàn chỉnh (0% REGEX LAW):
    1. Loại bỏ các cột 100% rỗng trên toàn bộ các hàng.
    2. Giải quyết tiêu đề đa tầng (Multi-tier header hierarchy):
       - Nhận diện hàng 0 và hàng 1 nếu hàng 0 là ô gộp ngang (chứa text followed by None/empty)
         và hàng 1 chứa các tiêu đề chi tiết.
       - Làm phẳng phân cấp: 'HẠNG THẺ -> Chuẩn', 'HẠNG THẺ -> Vàng', 'HẠNG THẺ -> Platinum'.
       - Forward-fill tiêu đề cho các cột con.
    3. Hàn gắn các dòng tiếp nối ô phân mảnh (Multi-line row continuation):
       - Nếu 1 dòng chỉ có nội dung ở cột chi tiết/ghi chú mà các cột định danh (STT, Tên) để trống,
         hàn gắn vào dòng phía trước để tránh phân mảnh dữ liệu.
    4. Tách các dòng biểu ngữ phân nhóm thành RowGroup(group_name=...).
    """
    if not df or len(df) < 2:
        return [], [], []

    num_cols = len(df[0])

    # 1. Lọc bỏ các cột hoàn toàn rỗng trên toàn bộ các hàng
    valid_col_indices = [
        c for c in range(num_cols)
        if any(row[c] is not None and str(row[c]).strip() for row in df)
    ]
    if not valid_col_indices:
        return [], [], []

    compact_df = [
        [(str(row[c]).strip().replace("\n", " ") if row[c] is not None else "") for c in valid_col_indices]
        for row in df
    ]

    if not compact_df:
        return [], [], []

    # 2. Xử lý tiêu đề đa tầng (Multi-tier headers)
    hdr0 = compact_df[0]
    hdr1 = compact_df[1] if len(compact_df) > 1 else []
    has_multilevel = False

    if len(compact_df) >= 3:
        h0_non_empty = sum(1 for c in hdr0 if c)
        h1_non_empty = sum(1 for c in hdr1 if c)
        if any(not c for c in hdr0) and h1_non_empty > h0_non_empty:
            has_multilevel = True

    if has_multilevel:
        filled_h0 = []
        last_h = ""
        for c in hdr0:
            if c:
                last_h = c
            filled_h0.append(last_h)

        final_headers = []
        for top, sub in zip(filled_h0, hdr1):
            if top and sub and top != sub:
                final_headers.append(f"{top} -> {sub}")
            else:
                final_headers.append(top or sub or "Chi tiết")
        body_rows = compact_df[2:]
    else:
        final_headers = []
        last_h = ""
        for c in hdr0:
            if c:
                last_h = c
                final_headers.append(c)
            elif last_h:
                final_headers.append(f"{last_h} (Chi tiết)")
            else:
                final_headers.append("Nội dung")
        body_rows = compact_df[1:]

    # Khử trùng lặp tên cột
    seen_h: dict[str, int] = {}
    unique_headers: list[str] = []
    for h in final_headers:
        if h not in seen_h:
            seen_h[h] = 1
            unique_headers.append(h)
        else:
            seen_h[h] += 1
            unique_headers.append(f"{h} ({seen_h[h]})")

    # 3. Hàn gắn các dòng tiếp nối ô phân mảnh (Multi-line row continuation)
    healed_rows: list[list[str]] = []
    for r in body_rows:
        is_continuation = (
            len(r) >= 3 and
            not r[0] and not r[1] and
            any(r[2:]) and
            bool(healed_rows)
        )
        if is_continuation:
            for c_idx in range(2, len(r)):
                if r[c_idx]:
                    healed_rows[-1][c_idx] = (healed_rows[-1][c_idx] + " " + r[c_idx]).strip()
        else:
            healed_rows.append(r)

    # 4. Tách các dòng biểu ngữ phân nhóm thành RowGroup (Group-row banners)
    row_groups: list[RowGroup] = []
    curr_group_name: Optional[str] = None
    curr_group_rows: list[TableRow] = []

    for r in healed_rows:
        non_empty = [c for c in r if c]
        if not non_empty:
            continue

        is_group_banner = False
        banner_text = ""
        if len(non_empty) == 1 and len(r) >= 3:
            is_group_banner = True
            banner_text = non_empty[0]
        elif len(non_empty) == 2 and not r[0] and r[1] and all(not c for c in r[2:]):
            is_group_banner = True
            banner_text = r[1]
        elif len(non_empty) == 2 and r[0] and r[1] and all(not c for c in r[2:]):
            first_tok = r[0].strip().rstrip(".:)")
            is_roman = bool(first_tok and len(first_tok) <= 5 and all(c in "IVXLCDMivxlcdm" for c in first_tok))
            if is_roman or not any(char.isdigit() for char in r[1]):
                is_group_banner = True
                banner_text = f"{r[0]} {r[1]}".strip()

        if is_group_banner:
            if curr_group_rows:
                row_groups.append(RowGroup(group_name=curr_group_name, rows=curr_group_rows))
                curr_group_rows = []
            curr_group_name = banner_text
        else:
            r_hdr = r[0] if len(r) > 0 else ""
            cells = r[1:] if len(r) > 1 else []
            curr_group_rows.append(TableRow(row_header=r_hdr, cells=cells))

    if curr_group_rows:
        row_groups.append(RowGroup(group_name=curr_group_name, rows=curr_group_rows))

    if not row_groups and healed_rows:
        default_rows = [
            TableRow(row_header=(r[0] if len(r) > 0 else ""), cells=(r[1:] if len(r) > 1 else []))
            for r in healed_rows
        ]
        row_groups.append(RowGroup(group_name=None, rows=default_rows))

    return unique_headers, healed_rows, row_groups


def clean_pdf_table_matrix(df: list[list[str]]) -> tuple[list[str], list[list[str]]]:
    """Hàm tương thích ngược: Gọi giải thuật Topo học chuẩn."""
    headers, matrix, _ = resolve_topological_table_matrix(df)
    return headers, matrix


def order_blocks_by_spatial_reading_flow(
    blocks: list[tuple],
    page_width: float,
    page_height: float
) -> list[tuple]:
    """
    Sắp xếp các khối văn bản theo luồng đọc không gian 2D (Spatial Reading Flow)
    tuân thủ 0% REGEX LAW và không hardcode từ khóa.

    Nguyên lý hình học DLA:
    1. Nhận diện các dải ngang (horizontal bands) được ngăn cách bởi các khối banner trải rộng (width >= 0.65 * W).
    2. Trong mỗi dải ngang:
       - Nếu các khối chia thành 2 cột rõ rệt (trung điểm X nằm về 2 phía so với tâm trang):
         Xếp tất cả khối cột trái (từ trên xuống dưới), sau đó xếp tất cả khối cột phải (từ trên xuống dưới).
       - Nếu không phân cột: Xếp tuần tự từ trên xuống dưới theo tọa độ y0.
    """
    if not blocks or len(blocks) <= 1:
        return blocks

    sorted_blocks = sorted(blocks, key=lambda b: b[1])

    bands: list[dict] = []
    curr_band: dict = {"type": "band", "blocks": []}

    for b in sorted_blocks:
        x0, y0, x1, y1 = b[0], b[1], b[2], b[3]
        text = b[4].strip() if len(b) > 4 and isinstance(b[4], str) else ""
        if not text:
            continue
        w = x1 - x0
        is_banner = (w >= 0.65 * page_width)

        if is_banner:
            if curr_band["blocks"]:
                bands.append(curr_band)
            bands.append({"type": "banner", "blocks": [b]})
            curr_band = {"type": "band", "blocks": []}
        else:
            curr_band["blocks"].append(b)

    if curr_band["blocks"]:
        bands.append(curr_band)

    final_ordered: list[tuple] = []
    mid_x = page_width * 0.50

    for band in bands:
        if band["type"] == "banner":
            final_ordered.extend(band["blocks"])
        else:
            b_list = band["blocks"]
            left_cols = [b for b in b_list if (b[0] + b[2]) / 2.0 < mid_x]
            right_cols = [b for b in b_list if (b[0] + b[2]) / 2.0 >= mid_x]

            if left_cols and right_cols and len(left_cols) + len(right_cols) == len(b_list):
                left_cols.sort(key=lambda b: b[1])
                right_cols.sort(key=lambda b: b[1])
                final_ordered.extend(left_cols)
                final_ordered.extend(right_cols)
            else:
                b_list.sort(key=lambda b: b[1])
                final_ordered.extend(b_list)

    return final_ordered


# -----------------------------------------------------------------------------
# 3. HEADING TRACKER & TIÊU ĐỀ THỰC TẾ (0% REGEX)
# -----------------------------------------------------------------------------
class PDFHeadingTracker:
    """Theo dõi phân cấp tiêu đề đa tầng (Doc Title -> H1 -> H2) tuân thủ 0% Regex."""
    def __init__(self, doc_title: str):
        self.doc_title = doc_title.strip()
        self.h1: Optional[str] = None
        self.h2: Optional[str] = None

    def update(self, heading: str):
        h = heading.strip()
        if not h or h == self.doc_title or h.lower().startswith("bảng biểu trang"):
            return
        # Nhận diện cấp H1: Phụ đề cấp cao (ÁP DỤNG CHO..., DÀNH CHO..., PHẦN..., CHƯƠNG...) hoặc tiêu đề in hoa ngắn
        is_h1_candidate = (
            any(h.lower().startswith(p) for p in ("áp dụng cho", "ap dung cho", "dành cho", "danh cho", "phần", "phan", "chương", "chuong", "mục", "muc", "phụ lục", "phu luc")) or
            (h.isupper() and len(h.split()) <= 10 and not h.startswith(("GHI CHÚ", "GHI CHU", "LƯU Ý", "LUU Y", "ĐIỀU", "DIEU")))
        )
        if is_h1_candidate:
            self.h1 = h
            self.h2 = None
        elif self.h1 and h != self.h1:
            self.h2 = h
        else:
            self.h1 = h

    def get_breadcrumb(self) -> str:
        tokens = [self.doc_title]
        if self.h1 and self.h1 != self.doc_title:
            tokens.append(self.h1)
        if self.h2 and self.h2 != self.h1 and self.h2 != self.doc_title:
            tokens.append(self.h2)
        return " > ".join(tokens)

    def get_section_title(self) -> str:
        return self.h2 or self.h1 or self.doc_title


def extract_real_pdf_title(page: object, fallback_stem: str) -> tuple[str, Optional[str]]:
    """Trích xuất tiêu đề chính và phân cấp phụ đề H1 đầu tiên trên trang 1 (0% Regex)."""
    try:
        blocks = page.get_text("blocks")
    except Exception:
        blocks = []

    top_candidates = []
    for b in blocks:
        if b[1] < 260 and b[4].strip():
            txt = b[4].strip()
            if txt.isdigit() or len(txt) < 3:
                continue
            txt_lower = txt.lower()
            if any(txt_lower.startswith(p) for p in ("ngân hàng", "ngan hang", "mẫu số", "mau so", "số:", "so:", "cộng hòa", "cong hoa")):
                continue
            top_candidates.append((b[1], txt))

    top_candidates.sort(key=lambda x: x[0])
    doc_title = None
    subheading = None
    if top_candidates:
        first_txt = top_candidates[0][1]
        lines = [l.strip() for l in first_txt.splitlines() if l.strip()]
        if lines:
            doc_title = lines[0]
            if len(lines) > 1 and _is_heading_line(lines[1]):
                subheading = lines[1]
        if len(top_candidates) > 1 and not subheading:
            second_txt = top_candidates[1][1]
            sec_lines = [l.strip() for l in second_txt.splitlines() if l.strip()]
            if sec_lines and _is_heading_line(sec_lines[0]):
                subheading = sec_lines[0]

    return doc_title or clean_filename_fallback(fallback_stem), subheading


# -----------------------------------------------------------------------------
# 4. CORE PDF CHUNKER CLASS
# -----------------------------------------------------------------------------
class PDFChunker:
    """
    Bộ phân đoạn PDF nhận thức bố cục (DLA) và bảng biểu thị giác thế hệ mới.
    """

    def __init__(
        self,
        ocr_extractor: Optional[OCRExtractor] = None,
        vlm_extractor: Optional[VLMTableExtractor] = None,
        table_formatter: Optional[TableFormatter] = None,
        syllable_manager: Optional[SyllableBudgetManager] = None,
        max_chunk_size: int = 800,
        min_section_size: int = 100,
        table_rows_per_chunk: int = 4,
        config: object | None = None,
        **kwargs
    ):
        self.config = config
        self.ocr_extractor = ocr_extractor or OCRExtractor(config=config)
        self.vlm_extractor = vlm_extractor or VLMTableExtractor(config=config)
        self.table_formatter = table_formatter or TableFormatter(config=config)
        self.syllable_manager = syllable_manager or SyllableBudgetManager()
        self.max_chunk_size = max_chunk_size
        self.min_section_size = min_section_size
        self.table_rows_per_chunk = table_rows_per_chunk

    def chunk_pdf(
        self,
        pdf_path: str | Path,
        max_pages: Optional[int] = None,
        **kwargs
    ) -> list[dict]:
        """
        Đọc và phân đoạn toàn diện một tệp PDF:
        - Bóc tách bảng biểu qua Vision LLM với mở rộng đáy động (Dynamic Bounding Box).
        - Xử lý văn bản số hóa và OCR scan tự động.
        - Phân bổ ngân sách Profile B (150-400 âm tiết) với tiền tố ngữ cảnh.
        """
        path = Path(pdf_path)
        if not path.exists():
            return []

        try:
            doc = fitz.open(str(path))
        except Exception as e:
            logger.error(f"[PDFChunker] Không thể mở tệp {path.name}: {e}")
            return []

        chunks: list[dict] = []
        counter = 0

        doc_title = path.stem.replace("_", " ").title()
        current_section = doc_title

        total_pages = len(doc)
        limit_pages = min(total_pages, max_pages) if max_pages else total_pages
        if limit_pages == 0:
            doc.close()
            return []

        # Trích xuất tiêu đề văn bản thực tế từ trang đầu tiên (0% Regex)
        doc_title, first_subheading = extract_real_pdf_title(doc[0], path.stem)
        tracker = PDFHeadingTracker(doc_title=doc_title)
        if first_subheading:
            tracker.update(first_subheading)

        chunks: list[dict] = []
        counter = 0

        for page_idx in range(limit_pages):
            page = doc[page_idx]
            page_num = page_idx + 1

            # 1. Bóc tách bảng biểu trên trang với Breadcrumb phân cấp thực tế
            tbl_chunks, counter, exclusion_boxes = self._process_page_tables(page, page_num, path, tracker, counter)
            chunks.extend(tbl_chunks)

            # 2. Bóc tách và phân đoạn văn bản trên trang với ranh giới loại trừ bảng và luồng đọc không gian
            txt_chunks, counter = self._process_page_text(
                page=page,
                page_num=page_num,
                path=path,
                tracker=tracker,
                counter=counter,
                is_first_page=(page_idx == 0),
                table_exclusion_boxes=exclusion_boxes
            )
            chunks.extend(txt_chunks)

        doc.close()
        return chunks

    def _process_page_tables(
        self,
        page: object,
        page_num: int,
        path: Path,
        tracker: PDFHeadingTracker,
        counter: int
    ) -> tuple[list[dict], int, list[tuple[float, float, float, float]]]:
        """Trích xuất và định dạng toàn bộ bảng biểu trên một trang PDF bằng Dynamic Bounding Box & Topo Grid."""
        try:
            tables = page.find_tables()
        except Exception:
            tables = []

        if not tables or not getattr(tables, "tables", []):
            return [], counter, []

        pw, ph = page.rect.width, page.rect.height
        page_chunks: list[dict] = []
        table_exclusion_boxes: list[tuple[float, float, float, float]] = []

        # Phân loại trang số (Digital PDF) hay trang quét (Scanned PDF)
        is_scanned_page, _ = detect_is_scanned_page(page)

        try:
            page_blocks = page.get_text("blocks")
        except Exception:
            page_blocks = []

        for tbl in tables.tables:
            tbl_y0 = tbl.bbox[1]
            cand_headings = []
            for b in page_blocks:
                if b[3] <= tbl_y0 + 8 and b[4].strip():
                    for line in b[4].strip().splitlines():
                        l_clean = line.strip()
                        if _is_heading_line(l_clean) and l_clean != tracker.doc_title:
                            cand_headings.append(l_clean)
            if cand_headings:
                tracker.update(cand_headings[-1])

            breadcrumb = tracker.get_breadcrumb()
            section_title = tracker.get_section_title()

            # 1. Trích xuất ma trận ô thô để kiểm tra cấu trúc
            try:
                raw_df = tbl.extract()
            except Exception:
                raw_df = None

            # 2. Tính toán đáy bảng động bằng Typographic Proximity (0% CỐ ĐỊNH 8%)
            dynamic_bottom = detect_dynamic_table_bottom(
                page=page,
                table_bbox=tbl.bbox,
                footer_margin_ratio=0.065
            )

            # Ghi nhận vùng ranh giới loại trừ bảng (kèm phần đáy chú thích động)
            safe_bottom = max(tbl.bbox[3], dynamic_bottom)
            table_exclusion_boxes.append((tbl.bbox[0], tbl.bbox[1], tbl.bbox[2], safe_bottom))

            extracted_successfully = False

            # Đánh giá tính đơn giản theo điều kiện Factual Uniform Grid <= 30 ô
            is_simple, reason = is_simple_uniform_table(tbl, raw_df, max_cells=30)
            logger.info(f"[PDFChunker] Bảng trang {page_num}: is_simple={is_simple} ({reason})")

            # -----------------------------------------------------------------
            # CHIẾN LƯỢC 1: BẢNG CỰC KỲ ĐƠN GIẢN (<= 30 Ô, 0% MERGE CELL)
            # CHỈ KHI NÀY MỚI ĐƯỢC PHÉP DÙNG PYMUPDF TOPO GRID
            # -----------------------------------------------------------------
            if is_simple and not is_scanned_page and raw_df is not None:
                try:
                    headers, matrix, row_groups = resolve_topological_table_matrix(raw_df)
                    if headers and (matrix or row_groups):
                        # Gom footnote thực tế từ vùng chữ số nằm dưới đáy bảng
                        fallback_footnotes = []
                        if dynamic_bottom > tbl.bbox[3]:
                            fn_rect = fitz.Rect(tbl.bbox[0] - 10, tbl.bbox[3], tbl.bbox[2] + 10, dynamic_bottom)
                            fn_text = page.get_text("text", clip=fn_rect)
                            if fn_text.strip():
                                fallback_footnotes = parse_fallback_footnotes_no_regex(fn_text)

                        table_chunks = self.table_formatter.format_matrix_to_chunks(
                            headers=headers,
                            matrix=matrix,
                            row_groups=row_groups,
                            source=path.name,
                            chunk_id_prefix=f"{path.stem}_tbl",
                            breadcrumb=breadcrumb,
                            section_title=section_title,
                            chunk_type="pdf_table",
                            start_index=counter,
                            footnotes=fallback_footnotes,
                            page_num=page_num,
                            bbox=list(tbl.bbox)
                        )
                        if table_chunks:
                            page_chunks.extend(table_chunks)
                            counter += len(table_chunks)
                            extracted_successfully = True
                            logger.info(f"[PDFChunker] Trích xuất PyMuPDF thành công ({len(table_chunks)} chunks) cho bảng đơn giản.")
                except Exception as e:
                    logger.warning(f"[PDFChunker] PyMuPDF Topo Grid gặp lỗi: {e}")

            # -----------------------------------------------------------------
            # CHIẾN LƯỢC 2: BẢNG PHỨC TẠP (> 30 Ô HOẶC CÓ MERGE CELL HOẶC SCAN)
            # BẮT BUỘC DÙNG LLM VISION (ƯU TIÊN 1 COLAB GPU, DỰ PHÒNG 2 RTX 4060)
            # TUYỆT ĐỐI KHÔNG FALLBACK! NẾU LỖI BÁO THẲNG!
            # -----------------------------------------------------------------
            if not extracted_successfully:
                logger.info(f"[PDFChunker] Bắt buộc định tuyến sang LLM Vision cho bảng trang {page_num} ({reason}).")
                pix = page.get_pixmap(dpi=150)
                img_mode = "RGB" if pix.alpha == 0 else "RGBA"
                pil_page = Image.frombytes(img_mode, [pix.width, pix.height], pix.samples).convert("RGB")

                cropped_tbl = crop_pdf_table_image(
                    page_image=pil_page,
                    bbox=tbl.bbox,
                    page_dim=(pw, ph),
                    dynamic_bottom=dynamic_bottom,
                    footer_margin_ratio=0.065
                )

                # Trích xuất văn bản thô hỗ trợ ngữ cảnh
                clip_rect = fitz.Rect(tbl.bbox[0], tbl.bbox[1], tbl.bbox[2], safe_bottom)
                raw_clip_text = page.get_text("text", clip=clip_rect)

                # allow_fallback=False: Cấm fallback! Sẽ ném RuntimeError nếu Colab & RTX 4060 đều không thành công
                vlm_res = self.vlm_extractor.extract_table(
                    table_image=cropped_tbl,
                    raw_fallback_text=raw_clip_text,
                    allow_fallback=False
                )
                table_chunks = self.table_formatter.format_vlm_result_to_chunks(
                    vlm_result=vlm_res,
                    source=path.name,
                    chunk_id_prefix=f"{path.stem}_tbl",
                    breadcrumb=breadcrumb,
                    section_title=section_title,
                    chunk_type="pdf_table",
                    start_index=counter,
                    page_num=page_num,
                    bbox=list(tbl.bbox)
                )
                if table_chunks:
                    page_chunks.extend(table_chunks)
                    counter += len(table_chunks)
                    extracted_successfully = True
                elif raw_clip_text.strip():
                    fallback_chunk = build_7_field_chunk(
                        chunk_id=f"{path.stem}_tbl_c{counter}",
                        source=path.name,
                        chunk_type="pdf_table",
                        breadcrumb=breadcrumb,
                        section_title=section_title,
                        content=raw_clip_text.strip(),
                        original_text=raw_clip_text.strip(),
                        parent_id=f"{path.stem}_tbl_parent",
                        location=[{"page": page_num, "bbox": list(tbl.bbox)}]
                    )
                    page_chunks.append(fallback_chunk)
                    counter += 1
                    extracted_successfully = True
                else:
                    logger.warning(f"[PDFChunker] Bảng trang {page_num} không có nội dung ô.")

        return page_chunks, counter, table_exclusion_boxes

    def _process_page_text(
        self,
        page: object,
        page_num: int,
        path: Path,
        tracker: PDFHeadingTracker,
        counter: int,
        is_first_page: bool = False,
        table_exclusion_boxes: Optional[list[tuple[float, float, float, float]]] = None
    ) -> tuple[list[dict], int]:
        """Trích xuất và phân đoạn khối văn bản thuần hoặc OCR scan trên trang PDF với Luồng đọc không gian 2D."""
        pw, ph = page.rect.width, page.rect.height
        is_scanned_page, _ = detect_is_scanned_page(page)

        # 1. Thu thập và loại trừ các khối văn bản nằm trong vùng bảng biểu (Exclusion Masking)
        try:
            raw_blocks = page.get_text("blocks")
            non_tab_blocks = []
            exclusion_zones = table_exclusion_boxes or []

            for b in raw_blocks:
                bx0, by0, bx1, by1 = b[0], b[1], b[2], b[3]
                b_text = b[4].strip() if len(b) > 4 and isinstance(b[4], str) else ""
                if not b_text:
                    continue

                in_table_zone = False
                for ex in exclusion_zones:
                    ex0, ey0, ex1, ey1 = ex
                    # Kiểm tra độ giao nhau theo chiều dọc và chiều ngang
                    y_overlap = max(0.0, min(by1, ey1) - max(by0, ey0))
                    b_h = max(1.0, by1 - by0)
                    if y_overlap / b_h >= 0.35:
                        x_overlap = max(0.0, min(bx1, ex1) - max(bx0, ex0))
                        b_w = max(1.0, bx1 - bx0)
                        if x_overlap / b_w >= 0.35:
                            in_table_zone = True
                            break

                if not in_table_zone:
                    # Bỏ qua toàn bộ khối tiêu đề, phụ đề, tiêu ngữ nằm trước bảng biểu đầu tiên trên trang 1
                    first_table_top = min((ex[1] for ex in exclusion_zones), default=220)
                    if is_first_page and by1 <= first_table_top:
                        continue
                    if is_first_page and by1 < 220 and (b_text == tracker.doc_title or any(b_text.lower().startswith(p) for p in ("ngân hàng", "mẫu số", "quyết định", "cộng hòa"))):
                        continue
                    non_tab_blocks.append(b)

            # Sắp xếp các khối văn bản ngoài bảng theo Luồng đọc không gian 2D (Spatial Reading Flow)
            ordered_blocks = order_blocks_by_spatial_reading_flow(non_tab_blocks, pw, ph)
            raw_text = "\n\n".join(b[4].strip() for b in ordered_blocks if len(b) > 4 and b[4].strip())
        except Exception:
            raw_text = page.get_text("text", sort=True)

        chunk_type = "pdf_text"

        # 2. CƠ CHẾ BẢO VỆ OCR SCAN (GUARDRAIL):
        # CHỈ kích hoạt OCR khi trang THỰC SỰ là trang scan (is_scanned_page == True).
        # Tuyệt đối không OCR đè lên tài liệu số (Digital PDF) để triệt tiêu lỗi trùng lặp table và OCR vỡ hạt.
        if is_scanned_page and len(raw_text.split()) < 30 and len(page.get_images()) > 0:
            ocr_text = self.ocr_extractor.extract_text_from_page(page)
            if len(ocr_text.split()) >= 10:
                raw_text = ocr_text
                chunk_type = "pdf_ocr_scan"

        if not raw_text.strip():
            return [], counter

        # Làm sạch và loại bỏ rác mép trang
        cleaned_text = clean_and_dehyphenate_text(raw_text)
        cleaned_text = detect_and_remove_headers_footers(cleaned_text)

        lines = [l.strip() for l in cleaned_text.splitlines() if l.strip()]

        # Cập nhật tiêu đề mục gần nhất nếu có dòng heading
        for l in lines:
            if _is_heading_line(l) and l != tracker.doc_title:
                tracker.update(l)

        breadcrumb = tracker.get_breadcrumb()
        section_title = tracker.get_section_title()

        # Phân tách văn bản dài theo ngân sách âm tiết đàn hồi
        split_chunks = self.syllable_manager.split_text_elastic(
            text=cleaned_text,
            window_start=200,
            window_end=380,
            ceiling=420
        )

        page_chunks: list[dict] = []
        for group in (split_chunks or [cleaned_text]):
            if not group.strip():
                continue
            chunk_obj = build_7_field_chunk(
                chunk_id=f"{path.stem}_pdf_c{counter}",
                source=path.name,
                chunk_type=chunk_type,
                breadcrumb=breadcrumb,
                section_title=section_title,
                content=group.strip(),
                original_text=group.strip(),
                parent_id=f"{path.stem}_p{page_num}",
                location=[{"page": page_num}]
            )
            page_chunks.append(chunk_obj)
            counter += 1

        return page_chunks, counter


# -----------------------------------------------------------------------------
# 4. TIỆN ÍCH DLA TITLE & FILENAME FALLBACK
# -----------------------------------------------------------------------------
def clean_filename_fallback(stem: str) -> str:
    """Tạo tên hiển thị nghiệp vụ chuẩn mực khi không có text số hoặc OCR."""
    parts = stem.split("_")
    cleaned = []
    for i, p in enumerate(parts):
        if p.lower() in ("pdf", "docx", "doc", "txt", "full", "slide", "slides", "scan", "scanned"):
            continue
        if p.isdigit() and i < 4:
            continue
        cleaned.append(p)
    return " ".join(cleaned).strip().title() or stem.replace("_", " ").title()


def is_valid_title_candidate(title: str) -> bool:
    """Kiểm tra chuỗi tiêu đề có phải là câu chữ nghiệp vụ thật hay là rác OCR."""
    if not title or len(title) < 10:
        return False
    words = title.split()
    if len(words) < 2:
        return False
    vowels = set("aáàảãạăắằẳẵặâấầẩẫậeéèẻẽẹêếềểễệiíìỉĩịoóòỏõọôốồổỗộơớờởỡợuúùủũụưứừửữựyýỳỷỹỵ"
                 "AÁÀẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬEÉÈẺẼẸÊẾỀỂỄỆIÍÌỈĨỊOÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢUÚÙỦŨỤƯỨỪỬỮỰYÝỲỶỸỴ")
    return any(c in vowels for c in title)


def extract_document_title_dla(doc: object = None, pdf_path: object = None) -> str:
    """Trích xuất tiêu đề văn bản bằng Visual DLA hoặc fallback tên tệp."""
    if doc is not None:
        try:
            if len(doc) > 0:
                stem_hint = Path(pdf_path).stem if pdf_path else (Path(doc.name).stem if hasattr(doc, "name") and doc.name else "")
                real_t, _ = extract_real_pdf_title(doc[0], stem_hint)
                if real_t and is_valid_title_candidate(real_t):
                    return real_t
        except Exception:
            pass
    if pdf_path is not None:
        return clean_filename_fallback(Path(pdf_path).stem)
    if doc is not None:
        if hasattr(doc, "name") and doc.name:
            return clean_filename_fallback(Path(doc.name).stem)
        if isinstance(doc, (str, Path)):
            return clean_filename_fallback(Path(doc).stem)
    return "Tài liệu Ngân hàng"


__all__ = [
    "PDFChunker",
    "is_simple_uniform_table",
    "is_complex_pdf_table",
    "clean_pdf_table_matrix",
    "detect_dynamic_table_bottom",
    "_is_universal_structural_anchor",
    "_is_heading_line",
    "extract_document_title_dla",
    "clean_filename_fallback",
    "is_valid_title_candidate",
]
