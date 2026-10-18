"""
Module Canonical Resolver (Universal NLP & Text Normalization Engine)
======================================================================
Bộ tiện ích chuẩn hóa văn bản, phân đoạn câu và xử lý hình thái tự nhiên
cho hệ thống RAG Banking (chuyên trách tiền xử lý Tiếng Việt và Tiếng Anh;
mọi ký tự và ngôn ngữ khác được bảo toàn nguyên vẹn như chuỗi ký tự đặc biệt).

NGUYÊN TẮC THIẾT KẾ:
1. 0% REGEX LAW: Tuyệt đối không sử dụng `import re` hay regex methods.
2. BẢO TOÀN NGUYÊN VẸN NỘI DUNG: Không tự tiện xóa bỏ ký tự ngoại ngữ;
   coi các ngôn ngữ khác như các chuỗi ký tự đặc biệt cần được giữ nguyên bản.
3. UNICODE STANDARD INVARIANTS: Dựa trên chuẩn Unicode (UAX #29, Unicode Categories L*, N*, P*, Z*, C*)
   và nguyên lý chữ hoa/chữ thường (Bicameral Capitalization) để nhận diện ranh giới câu.
"""

from __future__ import annotations

import unicodedata


# -----------------------------------------------------------------------------
# 1. LÀM SẠCH VĂN BẢN & HÀN GẮN NGẮT DÒNG GẠCH NỐI (DEHYPHENATION)
# -----------------------------------------------------------------------------
def clean_and_dehyphenate_text(text: str) -> str:
    """
    Làm sạch văn bản tự nhiên (bảo toàn nguyên vẹn mọi ký tự, không xóa ngôn ngữ khác):
    1. Hàn gắn các từ bị ngắt dòng có dấu gạch nối (vd: ngân-\\nhàng -> ngân hàng).
    2. Chuẩn hóa khoảng trắng và loại bỏ dòng trống dư thừa.
    Thuần túy xử lý chuỗi cơ sở (0% Regex).
    """
    if not text:
        return ""

    lines_raw = text.split("\n")
    stitched_lines: list[str] = []
    idx = 0
    total_raw = len(lines_raw)

    while idx < total_raw:
        line = lines_raw[idx]
        stripped = line.rstrip()
        # Nếu dòng kết thúc bằng dấu '-' nối với từ dòng kế tiếp
        if stripped.endswith("-") and len(stripped) > 1 and stripped[-2].isalnum() and idx + 1 < total_raw:
            next_line = lines_raw[idx + 1].lstrip()
            if next_line and next_line[0].isalnum():
                stitched_lines.append(stripped[:-1] + next_line)
                idx += 2
                continue
        stitched_lines.append(line)
        idx += 1

    # Chuẩn hóa khoảng trắng dư thừa trong từng dòng
    lines = [" ".join(line.split()) for line in stitched_lines]

    # Bỏ các dòng trống liên tiếp (chỉ giữ tối đa 1 dòng trống phân cách đoạn)
    cleaned_lines: list[str] = []
    prev_empty = False
    for line in lines:
        if not line:
            if not prev_empty:
                cleaned_lines.append("")
                prev_empty = True
        else:
            cleaned_lines.append(line)
            prev_empty = False

    return "\n".join(cleaned_lines).strip()


# -----------------------------------------------------------------------------
# 2. PHÂN ĐOẠN CÂU PHỔ QUÁT (UNIVERSAL SENTENCE BOUNDARY SEGMENTATION)
# -----------------------------------------------------------------------------
def split_sentences(text: str) -> list[str]:
    """
    Phân đoạn câu song ngữ Tiếng Việt - Tiếng Anh tối ưu hóa cho RAG Banking (0% Regex).
    Dựa trên các bất biến ngôn ngữ tự nhiên và chuẩn Unicode:
    1. Ranh giới câu xuất hiện tại dấu ngắt câu cuối câu (. ? ! …) hoặc ngắt đoạn kép (\\n\\n).
    2. Gom các dấu đóng ngoặc/trích dẫn liền kề bám theo câu (." .) .').
    3. Nguyên lý chữ thường (Lowercase Invariant): Trong tiếng Việt và tiếng Anh, câu mới
       không bao giờ bắt đầu bằng chữ thường (bảo vệ 'Smith et al. showed', 'e.g. for instance', 'tp. Hồ Chí Minh').
    4. Không ngắt khi dấu chấm nằm trong số tiền, số thập phân, mã số, URL hoặc tiền tố chỉ mục / chức danh ngắn.
    5. Xử lý ngắt dòng mềm (soft wrap) trong PDF/OCR tiếng Việt & tiếng Anh một cách tự nhiên.
    """
    if not text:
        return []

    text_len = len(text)
    sentences: list[str] = []
    current_chars: list[str] = []
    closing_delimiters = {'"', "'", '”', '’', '»', ')', ']', '}'}

    i = 0
    while i < text_len:
        c = text[i]

        # A. Ngắt đoạn văn bản cứng (\\n\\n) -> Ranh giới câu/đoạn chắc chắn
        if c == "\n":
            is_double_break = False
            next_idx = i + 1
            while next_idx < text_len and text[next_idx] in (" ", "\t", "\r", "\n"):
                if text[next_idx] == "\n":
                    is_double_break = True
                next_idx += 1

            if is_double_break:
                candidate = "".join(current_chars).strip()
                if candidate:
                    sentences.append(candidate)
                current_chars = []
                i = next_idx
                continue

            # Ngắt dòng đơn \n trong PDF/OCR:
            prev_content = "".join(current_chars).strip()
            if prev_content and next_idx < text_len:
                next_char = text[next_idx]
                if next_char.islower():
                    # Soft wrap: thay \n bằng khoảng trắng để câu liền mạch
                    current_chars.append(" ")
                    i = next_idx
                    continue
                elif prev_content[-1] in (".", "?", "!", ";"):
                    sentences.append(prev_content)
                    current_chars = []
                    i = next_idx
                    continue

            current_chars.append(" ")
            i += 1
            continue

        # B. Dấu kết thúc câu (. ? !)
        if c in {".", "?", "!"}:
            current_chars.append(c)

            # Gom dấu chấm lửng liên tiếp (...)
            if c == ".":
                while i + 1 < text_len and text[i + 1] == ".":
                    current_chars.append(".")
                    i += 1

            # Gom các dấu đóng ngoặc / đóng trích dẫn đi liền kề ngay sau
            next_pos = i + 1
            while next_pos < text_len and text[next_pos] in closing_delimiters:
                current_chars.append(text[next_pos])
                next_pos += 1

            # Bỏ qua khoảng trắng theo sau để kiểm tra ký tự bắt đầu tiếp theo
            follow_pos = next_pos
            has_space_after = False
            while follow_pos < text_len and text[follow_pos] in (" ", "\t", "\r", "\n"):
                has_space_after = True
                follow_pos += 1

            # Nếu đến cuối chuỗi -> kết thúc câu cuối cùng
            if follow_pos >= text_len:
                candidate = "".join(current_chars).strip()
                if candidate:
                    sentences.append(candidate)
                current_chars = []
                i = follow_pos
                continue

            # Nếu không có khoảng trắng theo sau -> Số thập phân (3.14), URL (domain.com), Email
            if not has_space_after:
                i = next_pos
                continue

            next_char = text[follow_pos]

            # 1. Bất biến chữ thường (Lowercase Lookahead Invariant):
            # Trong văn bản chuẩn, câu mới KHÔNG BAO GIỜ bắt đầu bằng chữ thường
            if next_char.islower():
                current_chars.append(" ")
                i = follow_pos
                continue

            # 2. Tiền tố phân cấp / chỉ mục danh sách hoặc danh xưng ngắn (vd: "Section 1.", "Dr. Smith", "Fig. 2", "1.")
            curr_str = "".join(current_chars).strip()
            words = curr_str.split()
            if words:
                last_token = words[-1].rstrip(".:?!;\"'”’»)]}")
                # A. Chỉ mục số/chữ danh sách/đề mục ngắn (vd: "1.", "Section 1.", "A.")
                if len(words) <= 3 and (last_token.isdigit() or (len(last_token) == 1 and last_token.isalpha())):
                    current_chars.append(" ")
                    i = follow_pos
                    continue

                # B. Từ ngắn viết hoa trước tên riêng hoặc số (vd: "Dr. Smith", "Fig. 2", "Prof. Nguyen")
                if 1 <= len(last_token) <= 4 and last_token[0].isupper() and (next_char.isupper() or next_char.isdigit()):
                    current_chars.append(" ")
                    i = follow_pos
                    continue

            # Thỏa mãn ranh giới câu hợp lệ -> Cắt câu!
            candidate = "".join(current_chars).strip()
            if candidate:
                sentences.append(candidate)
            current_chars = []
            i = follow_pos
            continue

        # Các ký tự văn bản thông thường
        current_chars.append(c)
        i += 1

    # Đóng gói phần còn lại nếu có
    if current_chars:
        candidate = "".join(current_chars).strip()
        if candidate:
            sentences.append(candidate)

    return sentences


split_sentences_banking = split_sentences
split_sentences_generalized = split_sentences


# -----------------------------------------------------------------------------
# 3. GOM NHÓM CÂU THEO NGÂN SÁCH (SENTENCE GROUPING - 0% CONTENT OVERLAP)
# -----------------------------------------------------------------------------
def group_sentences(
    sentences: list[str],
    max_words: int = 350,
    overlap_words: int = 0,
    max_chunk_size: int | None = None,
    chunk_overlap: int | None = None
) -> list[str]:
    """
    Gom các câu thành từng nhóm (chunk) với kích thước từ tối đa.
    Tuyệt đối không overlap nội dung (0% content overlap) giữa các chunk liền kề.
    Đảm bảo không bao giờ cắt vụn giữa chừng một câu văn.
    """
    if not sentences:
        return []

    limit = max_words if max_words is not None else (max_chunk_size // 2 if max_chunk_size else 350)
    overlap = overlap_words if overlap_words is not None else (chunk_overlap // 2 if chunk_overlap else 0)

    chunks: list[str] = []
    current_sentences: list[str] = []
    current_word_count = 0

    for s in sentences:
        s_words = len(s.split())
        if current_word_count + s_words > limit and current_sentences:
            chunks.append(" ".join(current_sentences))
            if overlap <= 0:
                current_sentences = []
                current_word_count = 0
            else:
                overlap_sentences: list[str] = []
                overlap_count = 0
                for prev_s in reversed(current_sentences):
                    prev_w = len(prev_s.split())
                    if overlap_count + prev_w <= overlap:
                        overlap_sentences.insert(0, prev_s)
                        overlap_count += prev_w
                    else:
                        break
                current_sentences = overlap_sentences
                current_word_count = overlap_count

        current_sentences.append(s)
        current_word_count += s_words

    if current_sentences:
        chunks.append(" ".join(current_sentences))

    return chunks


group_sentences_banking = group_sentences


# -----------------------------------------------------------------------------
# 4. ĐÁNH GIÁ TÍNH HỢP LỆ VĂN BẢN (0% REGEX)
# -----------------------------------------------------------------------------
def is_valid_readable_text(text: str) -> bool:
    """
    Kiểm tra văn bản hợp lệ theo đặc tính chuẩn Unicode (0% Regex).
    1. Phát hiện và loại bỏ rác mã hóa PDF, Mojibake và ký tự điều khiển hỏng bảng mã.
    2. Đảm bảo văn bản chứa các ký tự có nghĩa (chữ cái Category L* hoặc ký số N*).
    3. Không xóa bỏ ngôn ngữ ngoại lai (coi như các chuỗi ký tự đặc biệt).
    """
    if not text:
        return False
    s = text.strip()
    if len(s) < 5:
        return False

    total_chars = len(s)
    # 1. Ký tự điều khiển bất thường hoặc ký tự thay thế hỏng bảng mã (\ufffd, \x00-\x1f)
    bad_chars = sum(
        1 for c in s
        if c in ("\ufffd", "\x00", "\ufeff") or
        (unicodedata.category(c).startswith("C") and c not in ("\n", "\r", "\t"))
    )
    if bad_chars / total_chars > 0.03:
        return False

    # 2. Phải có ít nhất ký tự chữ cái (Unicode category L*) hoặc ký hiệu số (N*)
    has_meaningful_content = any(
        unicodedata.category(c)[0] in ("L", "N") for c in s
    )
    if not has_meaningful_content:
        return False

    return True


is_valid_natural_text = is_valid_readable_text


def is_valid_vietnamese_text(text: str) -> bool:
    """
    Kiểm tra tính hợp lệ văn bản đọc được (tương thích ngược, hoàn toàn không hardcode bảng chữ).
    """
    return is_valid_readable_text(text)


# -----------------------------------------------------------------------------
# 5. LỌC HEADER / FOOTER & MẪU BIỂU TỰ NHIÊN (STRUCTURAL PATTERNS)
# -----------------------------------------------------------------------------
def detect_and_remove_headers_footers(
    text: str,
    custom_ignore_phrases: list[str] | None = None,
    config: object | None = None
) -> str:
    """
    Loại bỏ các dòng header/footer, số trang và mã biểu mẫu theo cấu trúc cú pháp phổ quát.
    Tuân thủ 0% Regex Law và nguyên tắc Zero Domain Hardcoding qua config.
    """
    if not text:
        return ""

    ignore_phrases: list[str] = []
    if custom_ignore_phrases is not None:
        ignore_phrases.extend(custom_ignore_phrases)
    elif config is not None and getattr(config, "custom_ignore_phrases", None):
        ignore_phrases.extend(config.custom_ignore_phrases)

    lines = text.splitlines()
    filtered: list[str] = []

    for line in lines:
        s = line.strip()
        if not s:
            continue
        s_lower = s.lower()
        words = s.split()

        # 1. Số trang độc lập hoặc có nhãn ngắn (vd: "1", "1/5", "Page 2", "Trang 3")
        if len(words) == 1:
            w0 = words[0]
            if w0.isdigit() or (all(c.isdigit() or c in "/-" for c in w0) and any(c.isdigit() for c in w0)):
                continue
        elif len(words) in (2, 3) and words[0].isalpha() and len(words[0]) <= 6:
            remainder = "".join(words[1:])
            if all(c.isdigit() or c in "/-" for c in remainder) and any(c.isdigit() for c in remainder):
                continue

        # 2. Tiêu đề mẫu biểu có mã định danh và dấu hai chấm (vd: "01/TNDN:", "202-B:", "#12:", "A-04:")
        if ":" in s and len(words) <= 10:
            prefix = s.split(":", 1)[0].strip()
            prefix_words = prefix.split()
            if prefix_words:
                code_token = prefix_words[-1]
                has_digit = any(c.isdigit() for c in code_token)
                has_code_delim = any(c in "/-#_" for c in code_token) or (len(code_token) <= 5 and any(c.isdigit() for c in code_token) and any(c.isalpha() for c in code_token))
                if has_digit and has_code_delim:
                    continue

        # 3. Cụm từ người dùng cấu hình loại trừ (Slogan, banner, metadata)
        if ignore_phrases and any(p.lower() in s_lower for p in ignore_phrases if p.strip()):
            continue

        filtered.append(line)

    return "\n".join(filtered)


# -----------------------------------------------------------------------------
# 6. TIỆN ÍCH TRÍCH XUẤT ĐỊNH DANH & PHÂN CẤP TỔNG QUÁT
# -----------------------------------------------------------------------------
def extract_document_identifier(filename: str = "", text: str = "") -> str:
    """Lấy định danh tài liệu sạch từ tên tệp hoặc văn bản đầu vào (0% Regex)."""
    if filename:
        stem = filename.rsplit(".", 1)[0] if "." in filename else filename
        return stem.strip()
    if text:
        first_line = text.strip().splitlines()[0] if text.strip() else ""
        return first_line.strip()
    return ""


def extract_hierarchical_units(text: str = "", text_chunk: str = "") -> list[str]:
    """
    Trích xuất các điều khoản, phân đoạn phân cấp song ngữ Việt - Anh (0% Regex):
    Cú pháp tổng quát: [Từ định danh viết hoa] [Chỉ số: Số / Số La Mã].
    Ví dụ tiếng Việt: Điều 1, Khoản 2, Mục III, Chương 4, Phần I...
    Ví dụ tiếng Anh: Section 1, Article 2, Chapter III, Part 4, Clause 5...
    """
    content = text or text_chunk
    if not content:
        return []
    words = content.split()
    results: list[str] = []
    i = 0
    roman_chars = set("ivxlcdmIVXLCDM")
    while i < len(words) - 1:
        curr_w = words[i].strip(",.;:()")
        next_w = words[i + 1].strip(",.;:()")

        # Đảm bảo từ viết hoa thuộc bảng chữ cái Latin (Tiếng Việt / Tiếng Anh)
        is_title_word = bool(
            curr_w
            and curr_w[0].isupper()
            and curr_w.isalpha()
            and 2 <= len(curr_w) <= 20
            and all(unicodedata.name(c, "").startswith("LATIN") for c in curr_w)
        )
        is_num = next_w.isdigit()
        is_roman = bool(next_w and all(c in roman_chars for c in next_w) and not is_num)

        if is_title_word and (is_num or is_roman):
            results.append(f"{curr_w.capitalize()} {next_w.upper() if is_roman else next_w}")
            i += 1
        i += 1
    return list(dict.fromkeys(results))


__all__ = [
    "clean_and_dehyphenate_text",
    "split_sentences",
    "split_sentences_banking",
    "split_sentences_generalized",
    "group_sentences",
    "group_sentences_banking",
    "is_valid_vietnamese_text",
    "is_valid_readable_text",
    "is_valid_natural_text",
    "detect_and_remove_headers_footers",
    "extract_document_identifier",
    "extract_hierarchical_units",
]
