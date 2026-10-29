"""
Bộ phân đoạn tài liệu Microsoft Word (.docx) chuyên dụng cho ngân hàng.
========================================================================
Tính năng chuẩn hóa kiến trúc OpenXML AST:
1. Phân tách văn bản theo phân cấp Heading chuẩn OpenXML (Level 1 -> 6) và ngân sách âm tiết.
2. XỬ LÝ DANH SÁCH ĐA TẦNG (Multi-level Lists):
   - Đọc trực tiếp thẻ OpenXML AST w:numPr, w:ilvl và Styles.
   - Thụt lề theo cấp độ phân cấp và chuẩn hóa tiền tố danh sách (- / 1.).
3. DUYỆT BẢNG NÂNG CAO (Advanced Table Traverser):
   - Duyệt đệ quy Bảng lồng Bảng (Nested Tables trong w:tc).
   - Cơ chế Sticky Header: Neo giữ tiêu đề cột cho bảng siêu dài qua nhiều chunks.
   - Khử trùng lặp ô gộp ngang (Colspan) và ô gộp dọc (Rowspan).
4. TRÍCH XUẤT HỘP THÔNG TIN / CẢNH BÁO (w:txbxContent):
   - Nhận diện các khối Textbox/Callout định dạng nổi trong hợp đồng và biểu phí.
5. NHẬN DIỆN ĐỊNH DẠNG VĂN BẢN VÀ HÌNH ẢNH CÓ CHÚ THÍCH:
   - In đậm (**), In nghiêng (*), Gạch chân (<u>), gộp Runs liền kề.
   - CHỈ lưu hình ảnh khi có chú thích (caption) bên dưới (loại trừ icon trang trí qua wp:extent).
6. Tuân thủ tuyệt đối Hợp đồng Chunk 7 trường, 0% REGEX LAW, Zero Ternary.
"""

from __future__ import annotations

import logging
from pathlib import Path

import docx
from docx.table import Table, _Cell

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


def _is_run_bold(r: docx.text.run.Run) -> bool:
    """Kiểm tra Run có định dạng in đậm không (Zero Ternary, 0% Regex)."""
    if r.bold:
        return True
    if hasattr(r, "_r") and r._r is not None and r._r.rPr is not None:
        b_elem = r._r.rPr.find(docx.oxml.ns.qn("w:b"))
        if b_elem is not None:
            val = b_elem.get(docx.oxml.ns.qn("w:val"))
            if val is None:
                return True
            if val.lower() not in ("0", "false", "none"):
                return True
    return False


def _is_run_italic(r: docx.text.run.Run) -> bool:
    """Kiểm tra Run có định dạng in nghiêng không (Zero Ternary, 0% Regex)."""
    if r.italic:
        return True
    if hasattr(r, "_r") and r._r is not None and r._r.rPr is not None:
        i_elem = r._r.rPr.find(docx.oxml.ns.qn("w:i"))
        if i_elem is not None:
            val = i_elem.get(docx.oxml.ns.qn("w:val"))
            if val is None:
                return True
            if val.lower() not in ("0", "false", "none"):
                return True
    return False


def _is_run_underline(r: docx.text.run.Run) -> bool:
    """Kiểm tra Run có định dạng gạch chân không (Zero Ternary, 0% Regex)."""
    if r.underline:
        return True
    if hasattr(r, "_r") and r._r is not None and r._r.rPr is not None:
        u_elem = r._r.rPr.find(docx.oxml.ns.qn("w:u"))
        if u_elem is not None:
            val = u_elem.get(docx.oxml.ns.qn("w:val"))
            if val is None:
                return True
            if val.lower() not in ("0", "false", "none"):
                return True
    return False


def _get_paragraph_list_prefix(p: docx.text.paragraph.Paragraph) -> str:
    """
    Xác định tiền tố danh sách (- hoặc 1.) và mức thụt lề dựa trên OpenXML AST w:numPr và Style.
    Tuân thủ tuyệt đối 0% Regex và Zero Ternary.
    """
    s_name = ""
    if p.style and p.style.name:
        s_name = p.style.name.lower()
    s_id = ""
    if p.style and getattr(p.style, "style_id", None):
        s_id = p.style.style_id.lower()

    ilvl = 0
    has_numpr = False
    if p._p.pPr is not None:
        numPr = p._p.pPr.find(docx.oxml.ns.qn("w:numPr"))
        if numPr is not None:
            has_numpr = True
            ilvl_elem = numPr.find(docx.oxml.ns.qn("w:ilvl"))
            if ilvl_elem is not None:
                val = ilvl_elem.get(docx.oxml.ns.qn("w:val"))
                if val is not None and val.isdigit():
                    ilvl = int(val)

    is_bullet = False
    if "bullet" in s_name or "bullet" in s_id:
        is_bullet = True

    is_number = False
    if "number" in s_name or "number" in s_id or "ordered" in s_name:
        is_number = True

    if not is_bullet and not is_number and has_numpr:
        # Nếu đoạn văn có thuộc tính numPr mà không định nghĩa rõ kiểu, mặc định là bullet
        is_bullet = True

    if not is_bullet and not is_number:
        return ""

    indent = "  " * ilvl
    if is_bullet:
        return indent + "- "
    else:
        return indent + "1. "


def extract_paragraph_formatted_text(p: docx.text.paragraph.Paragraph) -> str:
    """
    Trích xuất văn bản đoạn văn DOCX bảo toàn định dạng Markdown tiêu chuẩn (0% Regex, Zero Ternary):
    - Tự động nhận diện danh sách phân cấp w:numPr và chèn tiền tố thụt lề chuẩn.
    - In đậm (Bold): **text**
    - In nghiêng (Italic): *text*
    - Gạch chân (Underline): <u>text</u>
    - Tự động gộp các Run liền kề cùng định dạng theo chuẩn OpenXML.
    - Xử lý ký tự xuống dòng \\n an toàn trong Markdown.
    """
    list_prefix = _get_paragraph_list_prefix(p)

    if not p.runs:
        raw_text = p.text.strip()
        if not raw_text:
            return ""
        if list_prefix:
            if raw_text.startswith("• "):
                raw_text = raw_text[2:].lstrip()
            else:
                if raw_text.startswith("- "):
                    raw_text = raw_text[2:].lstrip()
            return list_prefix + raw_text
        else:
            if raw_text.startswith("• "):
                return "- " + raw_text[2:].lstrip()
            return raw_text

    raw_runs: list[tuple[str, bool, bool, bool]] = []
    for r in p.runs:
        txt = r.text
        if not txt:
            continue
        is_bold = _is_run_bold(r)
        is_italic = _is_run_italic(r)
        is_underline = _is_run_underline(r)
        raw_runs.append((txt, is_bold, is_italic, is_underline))

    if not raw_runs:
        raw_text = p.text.strip()
        if list_prefix:
            return list_prefix + raw_text
        else:
            return raw_text

    # Gom cụm các runs liền kề có cùng bộ cờ định dạng
    merged_runs: list[tuple[str, bool, bool, bool]] = []
    curr_text = raw_runs[0][0]
    curr_b = raw_runs[0][1]
    curr_i = raw_runs[0][2]
    curr_u = raw_runs[0][3]

    for txt, b, i, u in raw_runs[1:]:
        if b == curr_b and i == curr_i and u == curr_u:
            curr_text += txt
        else:
            merged_runs.append((curr_text, curr_b, curr_i, curr_u))
            curr_text = txt
            curr_b = b
            curr_i = i
            curr_u = u
    merged_runs.append((curr_text, curr_b, curr_i, curr_u))

    formatted_parts: list[str] = []
    for txt, b, i, u in merged_runs:
        if not b and not i and not u:
            formatted_parts.append(txt)
            continue

        leading_ws = ""
        trailing_ws = ""
        clean_t = txt
        if clean_t.startswith(" "):
            leading_ws = " "
            clean_t = clean_t.lstrip(" ")
        if clean_t.endswith(" "):
            trailing_ws = " "
            clean_t = clean_t.rstrip(" ")

        if not clean_t:
            formatted_parts.append(txt)
            continue

        if "\n" in clean_t:
            lines = clean_t.split("\n")
            sub_parts = []
            for line in lines:
                sub_l = line.strip()
                if not sub_l:
                    continue
                item = sub_l
                if u:
                    item = f"<u>{item}</u>"
                if i:
                    item = f"*{item}*"
                if b:
                    item = f"**{item}**"
                sub_parts.append(item)
            item = " ".join(sub_parts)
        else:
            item = clean_t
            if u:
                item = f"<u>{item}</u>"
            if i:
                item = f"*{item}*"
            if b:
                item = f"**{item}**"

        formatted_parts.append(leading_ws + item + trailing_ws)

    result = "".join(formatted_parts).strip()
    if not result:
        result = p.text.strip()

    if list_prefix:
        if result.startswith("• "):
            result = result[2:].lstrip()
        else:
            if result.startswith("- "):
                result = result[2:].lstrip()
            else:
                if result.startswith("* "):
                    result = result[2:].lstrip()
                else:
                    if result.startswith("+ "):
                        result = result[2:].lstrip()
        result = list_prefix + result
    else:
        if result.startswith("• "):
            result = "- " + result[2:].lstrip()
        else:
            if result.startswith("+ "):
                result = "- " + result[2:].lstrip()

    return result


def _extract_cell_content(cell: _Cell, doc: docx.document.Document) -> str:
    """
    Trích xuất toàn bộ nội dung của ô (Cell) bao gồm các đoạn văn bản (w:p)
    và bảng lồng đệ quy (w:tbl) theo chuẩn OpenXML AST (0% Regex, Zero Ternary).
    """
    parts: list[str] = []
    tc_elem = cell._tc

    for child in tc_elem:
        child_tag = child.tag
        if "}" in child_tag:
            child_tag = child_tag.split("}")[-1]

        if child_tag == "p":
            p = docx.text.paragraph.Paragraph(child, doc)
            fmt = extract_paragraph_formatted_text(p)
            if fmt:
                parts.append(fmt)

        elif child_tag == "tbl":
            nested_table = Table(child, doc)
            nested_rows_str: list[str] = []
            for n_row in nested_table.rows:
                seen_sub_tcs: set[int] = set()
                n_cells_txt: list[str] = []
                for n_c in n_row.cells:
                    n_tc_id = id(n_c._tc)
                    if n_tc_id in seen_sub_tcs:
                        continue
                    seen_sub_tcs.add(n_tc_id)
                    sub_cell_txt = _extract_cell_content(n_c, doc)
                    if sub_cell_txt:
                        n_cells_txt.append(sub_cell_txt)

                if n_cells_txt:
                    nested_rows_str.append(" - ".join(n_cells_txt))

            if nested_rows_str:
                nested_formatted = "[Bảng con: " + " | ".join(nested_rows_str) + "]"
                parts.append(nested_formatted)

    res = " ".join(parts).strip()
    return res


def is_sequence_identifier(token: str) -> bool:
    """
    Kiểm tra xem một token có phải là mã định danh thứ tự (Sequence Identifier) theo chuẩn tài liệu khoa học:
    - Chữ số đơn / nhiều chữ số: '1', '24'
    - Chỉ mục phân nhánh phân cấp: '1.1', '2.3.1', '1-A', 'A.1'
    - Chữ số La Mã: 'i', 'ii', 'iii', 'iv', 'v', 'vi', 'vii', 'viii', 'ix', 'x'
    Tuyệt đối tuân thủ 0% REGEX & ZERO TERNARY.
    """
    clean = token.strip().rstrip(":.-,").lower()
    if not clean:
        return False

    if clean.isdigit():
        return True

    if clean in ("i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x"):
        return True

    sep = ""
    if "." in clean:
        sep = "."
    else:
        if "-" in clean:
            sep = "-"

    if sep:
        parts = clean.split(sep)
        if len(parts) >= 2:
            all_valid = True
            for p in parts:
                if not p.isdigit() and not (len(p) == 1 and p.isalnum()):
                    all_valid = False
                    break
            if all_valid:
                return True

    return False


def is_image_caption(p_or_text: docx.text.paragraph.Paragraph | str, style_name: str = "") -> bool:
    """
    Xác định chú thích hình ảnh hoàn toàn dựa trên OpenXML AST và Cấu trúc Cú pháp Định danh:
    1. Tầng OpenXML Field Code: Phát hiện trường SEQ (Word Sequence Field) chuẩn OpenXML (loại trừ SEQ Table).
    2. Tầng OpenXML Style Definition: Nhận diện styleId hoặc pStyle là 'Caption' / 'Chú thích' (loại trừ Table Caption).
    3. Tầng OpenXML Layout AST: Căn giữa hoặc in nghiêng dưới ảnh, độ dài <= 25 từ, không kết thúc bằng '?' hay '!'.
    4. Tầng Cấu trúc Cú pháp Phổ quát: Loại trừ nhãn hành chính, nhận diện sequence identifier hoặc visual nouns.
    Tuyệt đối tuân thủ 0% REGEX & ZERO TERNARY.
    """
    clean_style = style_name.lower()
    p_elem = None
    if isinstance(p_or_text, docx.text.paragraph.Paragraph):
        p_elem = p_or_text._p
        clean_text = p_or_text.text.strip()
        if p_or_text.style and p_or_text.style.name:
            clean_style = p_or_text.style.name.lower()
        if p_or_text.style and getattr(p_or_text.style, "style_id", None):
            s_id = p_or_text.style.style_id.lower()
            if s_id:
                clean_style = clean_style + " " + s_id
    else:
        clean_text = str(p_or_text).strip()

    if not clean_text:
        return False

    words = clean_text.split()
    if len(words) > 35 or len(clean_text) > 200:
        return False

    if "heading" in clean_style or "title" in clean_style:
        if "caption" not in clean_style and "chú thích" not in clean_style:
            return False

    if clean_text.endswith("?"):
        return False

    lower_text = clean_text.lower()
    excluded_prefixes = (
        "lưu ý", "trả lời", "hướng dẫn", "khuyến cáo", "ví dụ", "chú ý", "ghi chú",
        "đối tượng", "thời gian", "phạm vi", "điều kiện", "lãi suất", "mức phí",
        "hồ sơ", "thủ tục", "bước", "mục", "chương", "điều", "phần", "câu", "bài",
        "nguyên tắc", "quy định", "nội dung", "hình thức", "phương thức", "kết quả",
        "giải pháp", "nguyên nhân", "yêu cầu", "trách nhiệm", "quyền hạn", "bảng", "table"
    )
    for prefix in excluded_prefixes:
        if lower_text.startswith(prefix):
            return False

    # === TẦNG 1: OPENXML FIELD CODES ===
    if p_elem is not None:
        seq_simples = p_elem.xpath(".//w:fldSimple[contains(@w:instr, 'SEQ')]")
        if seq_simples:
            instr = (seq_simples[0].get(docx.oxml.ns.qn('w:instr')) or '').lower()
            if "table" not in instr and "bảng" not in instr:
                return True

        seq_instrs = p_elem.xpath(".//w:instrText[contains(text(), 'SEQ')]")
        if seq_instrs:
            itxt = (seq_instrs[0].text or '').lower()
            if "table" not in itxt and "bảng" not in itxt:
                return True

    # === TẦNG 2: OPENXML STYLE DEFINITION ===
    if "caption" in clean_style or "chuthich" in clean_style or "chú thích" in clean_style:
        if "table" not in clean_style and "bảng" not in clean_style:
            return True

    if p_elem is not None and p_elem.pPr is not None:
        pStyle = p_elem.pPr.find(docx.oxml.ns.qn('w:pStyle'))
        if pStyle is not None:
            val = (pStyle.get(docx.oxml.ns.qn('w:val')) or '').lower()
            if "caption" in val or "chuthich" in val or "chú thích" in val:
                if "table" not in val and "bảng" not in val:
                    return True

    # === TẦNG 3: OPENXML TYPOGRAPHY & LAYOUT AST ===
    has_xml_layout_clue = False
    if p_elem is not None:
        is_centered = False
        if p_elem.pPr is not None:
            jc = p_elem.pPr.find(docx.oxml.ns.qn('w:jc'))
            if jc is not None:
                if jc.get(docx.oxml.ns.qn('w:val')) == 'center':
                    is_centered = True

        all_italic = False
        if isinstance(p_or_text, docx.text.paragraph.Paragraph):
            runs_with_text = [r for r in p_or_text.runs if r.text.strip()]
            if runs_with_text:
                italic_count = 0
                for r in runs_with_text:
                    if _is_run_italic(r):
                        italic_count += 1
                if italic_count == len(runs_with_text):
                    all_italic = True

        if is_centered or all_italic:
            if not clean_text.isupper():
                if len(words) <= 25:
                    if not clean_text.endswith("?") and not clean_text.endswith("!"):
                        has_xml_layout_clue = True

    if has_xml_layout_clue:
        return True

    # === TẦNG 4: CẤU TRÚC ĐỊNH DANH PHỔ QUÁT ===
    visual_nouns = (
        "hình", "ảnh", "hình ảnh", "sơ đồ", "biểu đồ", "lưu đồ",
        "minh họa", "hình minh họa", "ảnh chụp", "giao diện", "màn hình",
        "chú thích", "bảng ảnh", "figure", "fig", "image", "chart", "diagram",
        "photo", "illustration", "screenshot"
    )

    w0_clean = words[0].lower().rstrip(":.-,")
    w0_ends_delim = words[0].endswith((":", "-", "—", "–", "."))

    w01_clean = ""
    w01_ends_delim = False
    if len(words) >= 2:
        w01_clean = (words[0] + " " + words[1]).lower().rstrip(":.-,")
        w01_ends_delim = words[1].endswith((":", "-", "—", "–", "."))

    has_visual_prefix = False
    if w0_clean in visual_nouns or w01_clean in visual_nouns:
        has_visual_prefix = True

    if has_visual_prefix and len(words) <= 30:
        if len(words) >= 2:
            w1_clean = words[1].rstrip(":.-,")
            if is_sequence_identifier(w1_clean):
                if words[1].endswith((":", "-", "—", "–", ".")):
                    return True
                if len(words) > 2 and words[2].startswith((":", "-", "—", "–")):
                    return True
        if len(words) >= 3:
            w2_clean = words[2].rstrip(":.-,")
            if is_sequence_identifier(w2_clean):
                if words[2].endswith((":", "-", "—", "–", ".")):
                    return True
                if len(words) > 3 and words[3].startswith((":", "-", "—", "–")):
                    return True

    matched_visual = False
    matched_delim = False

    if w01_clean in visual_nouns:
        matched_visual = True
        matched_delim = w01_ends_delim
    else:
        if w0_clean in visual_nouns:
            if w0_clean in ("hình", "ảnh"):
                if w0_ends_delim:
                    matched_visual = True
                    matched_delim = True
            else:
                matched_visual = True
                matched_delim = w0_ends_delim

    if matched_visual:
        if matched_delim:
            return True
        if len(words) <= 25:
            if len(words) > 15 and clean_text.endswith("."):
                pass
            else:
                return True

    return False


def get_docx_heading_level(p: docx.text.paragraph.Paragraph) -> int | None:
    """
    Xác định cấp độ Heading chính xác từ Level 1 -> 6 theo chuẩn OpenXML AST và Cấu trúc Văn bản.
    Tuyệt đối tuân thủ 0% REGEX & ZERO TERNARY.
    """
    s_name = ""
    if p.style and p.style.name:
        s_name = p.style.name.lower()
    s_id = ""
    if p.style and getattr(p.style, "style_id", None):
        s_id = p.style.style_id.lower()

    if "bullet" in s_name or "bullet" in s_id:
        return None

    txt = p.text.strip()
    if not txt:
        return None

    words = txt.split()
    if len(txt) > 160 or len(words) > 28:
        return None

    lower_txt = txt.lower()

    if is_image_caption(p, s_name):
        return None

    if txt.endswith((".", "...", "!", ";")):
        return None
    if txt.endswith("?"):
        if not lower_txt.startswith("câu "):
            return None

    inline_labels = ("trả lời", "lưu ý", "ghi chú", "ví dụ", "chú ý", "khuyến cáo", "hướng dẫn")
    for label in inline_labels:
        if lower_txt.startswith(label):
            return None

    currency_and_address_keywords = (
        "phường ", "quận ", "huyện ", "tp ", "tp. ", "tỉnh ", "đường ", "phố ",
        " triệu", " tỷ", " vnd", " vnđ", " usd", " đồng"
    )
    for kw in currency_and_address_keywords:
        if kw in lower_txt:
            return None

    if txt.endswith(":"):
        structural_keywords = (
            "chương", "phần", "mục", "điều", "khoản", "điểm", "bước",
            "câu", "bài", "phụ lục", "quy định", "nguyên tắc", "nhiệm vụ"
        )
        is_structural = False
        for kw in structural_keywords:
            if lower_txt.startswith(kw):
                is_structural = True
                break
        if not is_structural:
            return None

    outlineLvl = None
    if p._p.pPr is not None:
        ol = p._p.pPr.find(docx.oxml.ns.qn('w:outlineLvl'))
        if ol is not None:
            outlineLvl = ol.get(docx.oxml.ns.qn('w:val'))

    curr_s = p.style
    visited_styles: set[str] = set()
    while outlineLvl is None and curr_s is not None:
        style_key = getattr(curr_s, "style_id", "") or getattr(curr_s, "name", "")
        if style_key:
            if style_key in visited_styles:
                break
            visited_styles.add(style_key)

        if hasattr(curr_s, '_element') and curr_s._element.pPr is not None:
            ol = curr_s._element.pPr.find(docx.oxml.ns.qn('w:outlineLvl'))
            if ol is not None:
                outlineLvl = ol.get(docx.oxml.ns.qn('w:val'))

        curr_s = getattr(curr_s, "base_style", None)

    if outlineLvl is not None and outlineLvl.isdigit():
        try:
            val_lvl = int(outlineLvl) + 1
            if val_lvl < 1:
                val_lvl = 1
            else:
                if val_lvl > 6:
                    val_lvl = 6
            return val_lvl
        except Exception:
            pass

    if "title" in s_name or "title" in s_id:
        return 1
    if "subtitle" in s_name or "subtitle" in s_id:
        return 2
    if "heading" in s_name or "heading" in s_id:
        digits = "".join(ch for ch in (s_name + s_id) if ch.isdigit())
        if digits:
            h_lvl = int(digits[0])
            if h_lvl < 1:
                h_lvl = 1
            else:
                if h_lvl > 6:
                    h_lvl = 6
            return h_lvl

    has_numpr = False
    ilvl_val = None
    if p._p.pPr is not None:
        numPr = p._p.pPr.find(docx.oxml.ns.qn('w:numPr'))
        if numPr is not None:
            has_numpr = True
            ilvl_elem = numPr.find(docx.oxml.ns.qn('w:ilvl'))
            if ilvl_elem is not None:
                ilvl_val = ilvl_elem.get(docx.oxml.ns.qn('w:val'))

    runs_with_text = [r for r in p.runs if r.text.strip()]
    all_bold = False
    has_underline = False
    if runs_with_text:
        bold_count = 0
        for r in runs_with_text:
            if _is_run_bold(r):
                bold_count += 1
            if _is_run_underline(r):
                has_underline = True

        if bold_count == len(runs_with_text):
            all_bold = True

    if has_numpr:
        is_valid_numpr_heading = False
        if all_bold or txt.isupper():
            if not txt.endswith((".", "...", "!", ";", ":")):
                is_valid_numpr_heading = True

        if is_valid_numpr_heading:
            if ilvl_val is not None and ilvl_val.isdigit():
                calc_lvl = int(ilvl_val) + 2
                if calc_lvl > 6:
                    calc_lvl = 6
                return calc_lvl
            else:
                return 3
        else:
            return None

    if all_bold or has_underline:
        first_raw = words[0]
        first_w = first_raw.lower().rstrip(":.-,)")
        clean_num = first_raw.rstrip(":.-,)")

        if lower_txt.startswith("phần ") or lower_txt.startswith("chương "):
            return 1

        if first_raw.endswith((".", ")", ":", "-")) and first_w in ("i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x"):
            return 2
        if lower_txt.startswith("mục "):
            return 2
        if txt.isupper() and len(words) <= 15:
            return 2

        if lower_txt.startswith("điều ") or lower_txt.startswith("bước ") or lower_txt.startswith("câu ") or lower_txt.startswith("bài "):
            return 3
        if (first_raw.endswith(".") or first_raw.endswith(")")) and clean_num.isdigit():
            return 3

        if "." in clean_num:
            dot_parts = clean_num.split(".")
            all_digits = True
            for dp in dot_parts:
                if not dp.isdigit():
                    all_digits = False
                    break
            if all_digits:
                if len(dot_parts) == 2:
                    return 4
                else:
                    if len(dot_parts) == 3:
                        return 5
                    else:
                        if len(dot_parts) >= 4:
                            return 6

        if (first_raw.endswith(".") or first_raw.endswith(")")) and len(clean_num) == 1 and clean_num.isalpha():
            return 4

        return 3

    return None


class DOCXChunker:
    """
    Bộ phân đoạn DOCX nhận thức cấu trúc văn bản, kiểu chữ, bảng biểu đa tầng và hình ảnh có chú thích.
    """

    def __init__(
        self,
        syllable_manager: SyllableBudgetManager | None = None,
        max_chunk_size: int = 800,
        min_section_size: int = 100,
        images_output_dir: str | Path | None = None,
        **kwargs
    ):
        if syllable_manager is not None:
            self.syllable_manager = syllable_manager
        else:
            self.syllable_manager = SyllableBudgetManager()
        self.max_chunk_size = max_chunk_size
        self.min_section_size = min_section_size
        self.images_output_dir = images_output_dir

    def chunk_docx(
        self,
        docx_path: str | Path,
        images_output_dir: str | Path | None = None
    ) -> list[dict]:
        """
        Đọc tệp DOCX, trích xuất cấu trúc văn bản, danh sách phân cấp, bảng biểu lồng nhau và hình ảnh chú thích.
        """
        path = Path(docx_path)
        if not path.exists():
            return []

        try:
            doc = docx.Document(str(path))
        except Exception as e:
            logger.error(f"[DOCXChunker] Không thể đọc tệp {path.name}: {e}")
            return []

        target_images_dir = images_output_dir
        if target_images_dir is None:
            target_images_dir = self.images_output_dir
        if target_images_dir is None:
            target_images_dir = Path("chunking/output/extracted_images") / path.stem
        else:
            target_images_dir = Path(target_images_dir)

        chunks: list[dict] = []
        counter = 0

        doc_title = path.stem.replace("_", " ").title()
        heading_stack: list[tuple[int, str]] = [(0, doc_title)]
        current_section = doc_title
        pending_heading_eids: list[str] = []

        text_buffer: list[str] = []
        current_eids: list[tuple[str, str]] = []
        images_buffer: list[dict] = []
        elem_idx = 0
        buffer_words = 0
        img_counter = 0

        def get_breadcrumb() -> str:
            if heading_stack:
                return " > ".join(t for _, t in heading_stack)
            else:
                return doc_title

        def map_elements_to_groups(
            chunk_eid_texts: list[tuple[str, str]],
            split_groups: list[str],
            heading_eid: str | None = None
        ) -> list[list[str]]:
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

                if not group_eids_list[g_idx] and elem_spans:
                    closest_eid = min(elem_spans, key=lambda s: abs(s[1] - g_start))[0]
                    group_eids_list[g_idx].append(closest_eid)

            if heading_eid and heading_eid not in group_eids_list[0]:
                group_eids_list[0].insert(0, heading_eid)

            return group_eids_list

        def flush_text_buffer() -> None:
            nonlocal counter, buffer_words, current_eids, pending_heading_eids, images_buffer
            if not text_buffer and not images_buffer:
                buffer_words = 0
                return

            raw_text = "\n".join(text_buffer).strip()
            text_buffer.clear()
            buffer_words = 0
            chunk_eid_texts = list(current_eids)
            current_eids.clear()
            current_images = list(images_buffer)
            images_buffer.clear()

            if not raw_text and not current_images:
                return

            h_eids_to_attach = list(pending_heading_eids)
            pending_heading_eids.clear()

            clean_text = clean_and_dehyphenate_text(raw_text)
            breadcrumb = get_breadcrumb()

            raw_splits = self.syllable_manager.split_text_elastic(
                text=clean_text,
                window_start=200,
                window_end=380,
                ceiling=420
            )
            if not raw_splits:
                raw_splits = [clean_text]

            split_groups = []
            for g in raw_splits:
                g_str = g.strip()
                if g_str:
                    split_groups.append(g_str)

            if not split_groups:
                return

            first_h_eid = None
            if h_eids_to_attach:
                first_h_eid = h_eids_to_attach[0]

            group_eids_list = map_elements_to_groups(
                chunk_eid_texts=chunk_eid_texts,
                split_groups=split_groups,
                heading_eid=first_h_eid
            )

            if len(h_eids_to_attach) > 1:
                for he in reversed(h_eids_to_attach[1:]):
                    if he not in group_eids_list[0]:
                        group_eids_list[0].insert(0, he)

            for g_idx, (group, g_eids) in enumerate(zip(split_groups, group_eids_list)):
                group_images = []
                for img in current_images:
                    cap = img.get("caption", "")
                    if cap and cap in group:
                        group_images.append(img)
                    else:
                        if len(split_groups) == 1:
                            group_images.append(img)

                if g_idx == 0 and not group_images and current_images:
                    assigned = False
                    for other_group in split_groups:
                        for img in current_images:
                            if img.get("caption", "") and img.get("caption", "") in other_group:
                                assigned = True
                                break
                    if not assigned:
                        group_images = current_images

                formatted_content = prepend_context_header(
                    content=group,
                    breadcrumb=breadcrumb,
                    section_title=current_section
                )

                chunk_obj = build_7_field_chunk(
                    chunk_id=f"{path.stem}_docx_c{counter}",
                    source=path.name,
                    chunk_type="docx_text",
                    breadcrumb=breadcrumb,
                    section_title=current_section,
                    content=formatted_content,
                    original_text=group,
                    parent_id=f"{path.stem}_sec_{counter}",
                    element_ids=g_eids,
                    images=group_images
                )
                chunks.append(chunk_obj)
                counter += 1

        body_elements = list(doc.element.body)
        total_elements = len(body_elements)
        skip_indices: set[int] = set()

        for i, element in enumerate(body_elements):
            if i in skip_indices:
                continue

            tag = element.tag
            if "}" in tag:
                tag = tag.split("}")[-1]

            if tag == "p":
                p = docx.text.paragraph.Paragraph(element, doc)
                text = p.text.strip()

                # 1. Trích xuất Textbox / Callout (w:txbxContent)
                txbx_nodes = element.xpath(".//w:txbxContent")
                if txbx_nodes:
                    for txbx in txbx_nodes:
                        txbx_paras: list[str] = []
                        for tp_elem in txbx.xpath(".//w:p"):
                            tp = docx.text.paragraph.Paragraph(tp_elem, doc)
                            t_fmt = extract_paragraph_formatted_text(tp)
                            if t_fmt:
                                txbx_paras.append(t_fmt)
                        if txbx_paras:
                            txbx_txt = " ".join(txbx_paras).strip()
                            if txbx_txt:
                                box_eid = f"{path.name}_e{elem_idx}"
                                elem_idx += 1
                                box_entry = f"> **[Hộp thông tin]:** {txbx_txt}"
                                text_buffer.append(box_entry)
                                current_eids.append((box_eid, box_entry))
                                buffer_words += len(box_entry.split())

                # 2. Kiểm tra lọc icon/bullet trang trí bằng OpenXML Extent (<wp:extent>)
                extents = element.xpath(".//wp:extent")
                is_small_icon = False
                if extents:
                    for ext_elem in extents:
                        cx = ext_elem.get("cx")
                        cy = ext_elem.get("cy")
                        if cx and cy:
                            try:
                                if int(cx) < 600000 or int(cy) < 600000:
                                    is_small_icon = True
                                    break
                            except Exception:
                                pass

                # 3. Trích xuất hình ảnh: DrawingML và VML
                blip_rids: list[str] = []
                if not is_small_icon:
                    for blip in element.xpath(".//a:blip"):
                        rid = blip.get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed")
                        if rid:
                            blip_rids.append(rid)
                    for vimg in element.xpath(".//*[local-name()='imagedata']"):
                        rid = vimg.get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id")
                        if rid:
                            blip_rids.append(rid)

                if blip_rids:
                    caption_text = ""
                    caption_idx = None

                    if text:
                        p_s_name = ""
                        if p.style and p.style.name:
                            p_s_name = p.style.name
                        if is_image_caption(p, p_s_name):
                            caption_text = clean_and_dehyphenate_text(text).strip()
                            caption_idx = i

                    if not caption_text:
                        consecutive_empty_p = 0
                        for next_i in range(i + 1, total_elements):
                            next_elem = body_elements[next_i]
                            next_tag = next_elem.tag
                            if "}" in next_tag:
                                next_tag = next_tag.split("}")[-1]

                            if next_tag != "p":
                                break

                            has_sub_blip = bool(
                                next_elem.xpath(".//a:blip") or
                                next_elem.xpath(".//*[local-name()='imagedata']")
                            )
                            if has_sub_blip:
                                break

                            next_p = docx.text.paragraph.Paragraph(next_elem, doc)
                            nxt_txt = next_p.text.strip()

                            if not nxt_txt:
                                consecutive_empty_p += 1
                                if consecutive_empty_p > 2:
                                    break
                                continue

                            if get_docx_heading_level(next_p) is not None:
                                break

                            p_style_name = ""
                            if next_p.style and next_p.style.name:
                                p_style_name = next_p.style.name

                            if is_image_caption(next_p, p_style_name):
                                caption_text = clean_and_dehyphenate_text(nxt_txt).strip()
                                caption_idx = next_i
                            break

                    if caption_text and caption_idx is not None:
                        for r_id in blip_rids:
                            if r_id in doc.part.related_parts:
                                part = doc.part.related_parts[r_id]
                                if hasattr(part, "blob") and part.blob:
                                    ctype = getattr(part, "content_type", "")
                                    ext = "png"
                                    if "jpeg" in ctype or "jpg" in ctype:
                                        ext = "jpg"
                                    else:
                                        if "webp" in ctype:
                                            ext = "webp"
                                        else:
                                            if "gif" in ctype:
                                                ext = "gif"

                                    img_counter += 1
                                    img_filename = f"{path.stem}_img_{img_counter}.{ext}"
                                    img_save_path = target_images_dir / img_filename
                                    try:
                                        target_images_dir.mkdir(parents=True, exist_ok=True)
                                        img_save_path.write_bytes(part.blob)
                                    except Exception as e:
                                        logger.warning(f"Không thể lưu file ảnh {img_filename}: {e}")

                                    images_buffer.append({
                                        "image_id": f"{path.stem}_img_{img_counter}",
                                        "caption": caption_text,
                                        "image_path": str(img_save_path).replace("\\", "/"),
                                        "content_type": ctype
                                    })

                        if caption_idx != i:
                            skip_indices.add(caption_idx)
                        caption_eid = f"{path.name}_e{elem_idx}"
                        elem_idx += 1
                        caption_entry = f"[Hình ảnh: {caption_text}]"
                        text_buffer.append(caption_entry)
                        current_eids.append((caption_eid, caption_entry))
                        buffer_words += len(caption_entry.split())

                        if caption_idx == i:
                            continue

                if not text:
                    continue

                eid = f"{path.name}_e{elem_idx}"
                elem_idx += 1

                lvl = get_docx_heading_level(p)

                if lvl is not None:
                    if lvl >= 3 and buffer_words < 120 and text_buffer:
                        formatted_h = f"### {text}"
                        text_buffer.append(formatted_h)
                        current_eids.append((eid, formatted_h))
                        buffer_words += len(formatted_h.split())
                        while heading_stack and heading_stack[-1][0] >= lvl:
                            heading_stack.pop()
                        heading_stack.append((lvl, text))
                    else:
                        flush_text_buffer()
                        pending_heading_eids.append(eid)
                        while heading_stack and heading_stack[-1][0] >= lvl:
                            heading_stack.pop()
                        heading_stack.append((lvl, text))
                        current_section = text
                else:
                    formatted_p_text = extract_paragraph_formatted_text(p)
                    text_buffer.append(formatted_p_text)
                    current_eids.append((eid, formatted_p_text))
                    buffer_words += len(formatted_p_text.split())
                    if buffer_words >= 350:
                        flush_text_buffer()

            elif tag == "tbl":
                table = Table(element, doc)
                prev_row_txt = ""
                table_header_row = ""
                is_first_table_row = True
                needs_sticky_header = False

                for row in table.rows:
                    seen_tcs: set[int] = set()
                    row_cells_txt = []
                    for c in row.cells:
                        tc_id = id(c._tc)
                        if tc_id in seen_tcs:
                            continue
                        seen_tcs.add(tc_id)

                        cell_blips = c._tc.xpath(".//a:blip")
                        cell_vimgs = c._tc.xpath(".//*[local-name()='imagedata']")
                        if cell_blips or cell_vimgs:
                            c_cap_text = ""
                            for cp in c.paragraphs:
                                cp_t = cp.text.strip()
                                if cp_t:
                                    cp_s_name = ""
                                    if cp.style and cp.style.name:
                                        cp_s_name = cp.style.name
                                    if is_image_caption(cp, cp_s_name):
                                        c_cap_text = clean_and_dehyphenate_text(cp_t).strip()
                                        break
                            if c_cap_text:
                                c_rids = []
                                for cb in cell_blips:
                                    crid = cb.get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed")
                                    if crid:
                                        c_rids.append(crid)
                                for cv in cell_vimgs:
                                    crid = cv.get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id")
                                    if crid:
                                        c_rids.append(crid)

                                for r_id in c_rids:
                                    if r_id in doc.part.related_parts:
                                        part = doc.part.related_parts[r_id]
                                        if hasattr(part, "blob") and part.blob:
                                            ctype = getattr(part, "content_type", "")
                                            ext = "png"
                                            if "jpeg" in ctype or "jpg" in ctype:
                                                ext = "jpg"
                                            else:
                                                if "webp" in ctype:
                                                    ext = "webp"
                                                else:
                                                    if "gif" in ctype:
                                                        ext = "gif"
                                            img_counter += 1
                                            img_filename = f"{path.stem}_img_{img_counter}.{ext}"
                                            img_save_path = target_images_dir / img_filename
                                            try:
                                                target_images_dir.mkdir(parents=True, exist_ok=True)
                                                img_save_path.write_bytes(part.blob)
                                            except Exception as e:
                                                logger.warning(f"Không thể lưu file ảnh {img_filename}: {e}")

                                            images_buffer.append({
                                                "image_id": f"{path.stem}_img_{img_counter}",
                                                "caption": c_cap_text,
                                                "image_path": str(img_save_path).replace("\\", "/"),
                                                "content_type": ctype
                                            })

                        c_str = _extract_cell_content(c, doc)
                        if c_str:
                            row_cells_txt.append(c_str)

                    row_txt = " | ".join(row_cells_txt).strip()
                    if row_txt:
                        if row_txt == prev_row_txt:
                            continue
                        prev_row_txt = row_txt

                        if is_first_table_row:
                            table_header_row = row_txt
                            is_first_table_row = False

                        # Chèn Sticky Header nếu bảng đã bị ngắt sang chunk mới
                        if needs_sticky_header and table_header_row:
                            sticky_entry = f"[Tiếp tục bảng: {table_header_row}]"
                            sticky_eid = f"{path.name}_e{elem_idx}"
                            elem_idx += 1
                            text_buffer.append(sticky_entry)
                            current_eids.append((sticky_eid, sticky_entry))
                            buffer_words += len(sticky_entry.split())
                            needs_sticky_header = False

                        tbl_eid = f"{path.name}_e{elem_idx}"
                        elem_idx += 1
                        text_buffer.append(row_txt)
                        current_eids.append((tbl_eid, row_txt))
                        buffer_words += len(row_txt.split())

                    if buffer_words >= 350:
                        flush_text_buffer()
                        needs_sticky_header = True

        flush_text_buffer()

        # Dọn dẹp chunk đuôi vi mô (< 35 từ)
        if len(chunks) >= 2:
            last_c = chunks[-1]
            last_wc = len(last_c.get("original_text", "").split())
            if last_wc < 35 and not last_c.get("images"):
                prev_c = chunks[-2]
                last_sec = last_c.get("section_title", "").strip()
                prev_sec = prev_c.get("section_title", "").strip()
                prefix_heading = ""
                if last_sec and last_sec != prev_sec:
                    prefix_heading = f"### {last_sec}\n"

                merged_orig = prev_c["original_text"] + "\n" + prefix_heading + last_c["original_text"]
                merged_content = prev_c["content"] + "\n" + prefix_heading + last_c["original_text"]
                prev_c["original_text"] = merged_orig.strip()
                prev_c["content"] = merged_content.strip()
                prev_c["word_count"] = len(merged_orig.split())
                prev_c["token_count"] = int(prev_c["word_count"] * 1.25)
                for eid in last_c.get("element_ids", []):
                    if eid not in prev_c["element_ids"]:
                        prev_c["element_ids"].append(eid)
                chunks.pop()

        return chunks


__all__ = [
    "DOCXChunker",
    "is_image_caption",
    "extract_paragraph_formatted_text",
    "get_docx_heading_level"
]
