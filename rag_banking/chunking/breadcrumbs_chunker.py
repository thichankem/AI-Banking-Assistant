"""
Breadcrumbs Chunking (Phân đoạn bảo toàn đường dẫn phân cấp)
Chuyên trách phân tách tài liệu ngân hàng & pháp lý mà vẫn bảo toàn 100% ngữ cảnh phân cấp:
1. Breadcrumbs Tracking: Theo dõi liên tục phân cấp (Tài liệu > Chương > Điều > Khoản > Tiểu mục).
2. Legal Reference Tagging: Nhận diện và trích xuất chuẩn xác [Điều X, Khoản Y] hoặc [Điều X].
3. Canonical Questions Extraction: Bắt các câu hỏi chuẩn từ FAQ (ví dụ: '### Câu 1: ...') phục vụ bước Reranking.
4. Breadcrumbs Prepended Content: Gắn tiền tố đường dẫn vào nội dung chunk để vector nhúng và BM25 bắt trọn vẹn ngữ cảnh.
"""

import re
from pathlib import Path

class BreadcrumbsChunker:
    def __init__(self, max_chunk_size=900, chunk_overlap=120):
        self.max_chunk_size = max_chunk_size
        self.chunk_overlap = chunk_overlap

    def _extract_document_identifier(self, breadcrumb_text: str, full_context: str, source_name: str = "") -> str:
        """
        Trích xuất tên hoặc số hiệu văn bản định danh một cách tổng quát:
        Tự động nhận diện từ Breadcrumb Root, Số hiệu văn bản pháp quy, hoặc Source Name.
        Tuyệt đối không hardcode tên văn bản.
        """
        # 1. Tìm số hiệu văn bản chính thức xuất hiện ở đầu văn bản (vd: Quyết định 2345/QĐ-NHNN, Thông tư 17/2024/TT-NHNN)
        doc_code_match = re.search(
            r"\b((?:Luật|Nghị quyết|Quyết định|Thông tư|Nghị định|Quy chế|Quy định)\s+(?:số\s+)?\d+(?:/[A-ZĐ0-9\-_]+)+)\b",
            breadcrumb_text or full_context[:200],
            re.IGNORECASE
        )
        if doc_code_match:
            return doc_code_match.group(1).strip()

        # 2. Lấy phần tử gốc đầu tiên của Breadcrumb (Root Node)
        if breadcrumb_text:
            root_part = breadcrumb_text.split(" > ")[0].strip()
            # Làm sạch các tiền tố số thứ tự hoặc tag phân loại dạng [PHÁP LÝ], vcb_md_01, v.v.
            clean_root = re.sub(r"^\[.*?\]\s*", "", root_part)
            clean_root = re.sub(r"^(?:vcb_)?(?:md|pdf|docx|excel)_\d+__?", "", clean_root, flags=re.IGNORECASE)
            clean_root = clean_root.replace("_", " ").strip()
            if len(clean_root) > 40:
                # Nếu tiêu đề quá dài, lấy vế chính trước dấu gạch ngang hoặc dấu hai chấm
                split_title = re.split(r"[-–—:]", clean_root)[0].strip()
                if len(split_title) >= 8:
                    clean_root = split_title
            if clean_root:
                return clean_root

        # 3. Fallback: Chuẩn hóa từ tên tệp tin nguồn
        if source_name:
            stem = Path(source_name).stem
            stem = re.sub(r"^(?:vcb_)?(?:md|pdf|docx|excel)_\d+__?", "", stem, flags=re.IGNORECASE)
            stem = stem.replace("_", " ").strip().title()
            if stem:
                return stem

        return ""

    def _extract_hierarchical_units(self, full_context: str, text_chunk: str) -> str:
        """
        Bóc tách tự động các đơn vị phân cấp chuẩn trong văn bản ngân hàng và quy phạm pháp luật:
        Chương > Mục > Điều > Khoản > Điểm > Câu hỏi FAQ.
        """
        units = []

        # 1. Phát hiện Điều (Article)
        dieu_match = re.search(r"\bĐiều\s+(\d+[a-zA-Z]?)\b", full_context, re.IGNORECASE)

        # 2. Phát hiện Khoản (Clause)
        khoan_match = re.search(r"\bKhoản\s+(\d+)\b", full_context, re.IGNORECASE)
        if not khoan_match and dieu_match:
            # Phát hiện khoản dạng số thứ tự đầu đoạn: "1. Khách hàng...", "2. Hạn mức..."
            khoan_num_match = re.search(r"(?:^|\n)\s*(\d{1,2})\.\s+[A-ZÀ-Ỹ]", text_chunk)
            if khoan_num_match:
                khoan_match = khoan_num_match

        # 3. Phát hiện Điểm (Point)
        diem_match = re.search(r"\bĐiểm\s+([a-zđ])\b", full_context, re.IGNORECASE)
        if not diem_match and khoan_match:
            # Phát hiện điểm dạng chữ cái đầu dòng: "a) Trường hợp...", "b) Đối với..."
            diem_char_match = re.search(r"(?:^|\n)\s*([a-zđ])\)\s+[A-ZÀ-Ỹa-zà-ỹ]", text_chunk)
            if diem_char_match:
                diem_match = diem_char_match

        # 4. Phát hiện Chương / Mục
        chuong_match = re.search(r"\b(?:Chương|Phần)\s+([IVXLCDM\d]+)\b", full_context, re.IGNORECASE)
        muc_match = re.search(r"\b(?:Tiểu\s+mục|Mục)\s+([IVXLCDM\d]+|\d+)\b", full_context, re.IGNORECASE)

        # 5. Phát hiện Câu hỏi (FAQ)
        faq_match = re.search(r"\b(?:Câu|Câu\s+hỏi)\s+(\d+)\b", full_context, re.IGNORECASE)

        # Tổng hợp các đơn vị theo thứ bậc
        if dieu_match:
            unit_str = f"Điều {dieu_match.group(1)}"
            if khoan_match:
                unit_str += f", Khoản {khoan_match.group(1)}"
                if diem_match:
                    unit_str += f", Điểm {diem_match.group(1)}"
            units.append(unit_str)
        elif faq_match:
            units.append(f"Câu {faq_match.group(1)}")
        elif muc_match:
            unit_str = f"Mục {muc_match.group(1)}"
            if khoan_match:
                unit_str += f", Khoản {khoan_match.group(1)}"
            units.append(unit_str)
        elif chuong_match:
            units.append(f"Chương {chuong_match.group(1)}")
        elif khoan_match:
            units.append(f"Khoản {khoan_match.group(1)}")

        return units[0] if units else ""

    def _extract_cross_statutory_citations(self, full_context: str, current_doc: str) -> List[str]:
        """
        Quét và trích xuất tự động mọi văn bản pháp quy được viện dẫn chéo trong nội dung
        (Quyết định NHNN, Thông tư, Nghị định, Luật các TCTD).
        """
        pattern = r"\b((?:Quyết định|Thông tư|Nghị định|Nghị quyết|Luật)\s+(?:số\s+)?\d+(?:/[A-ZĐ0-9\-_]+)+)\b"
        matches = re.findall(pattern, full_context, re.IGNORECASE)
        citations = []
        for m in matches:
            m_clean = m.strip()
            # Tránh trùng lặp và tránh trùng với tên của chính văn bản hiện tại
            if m_clean.lower() not in [c.lower() for c in citations] and m_clean.lower() not in current_doc.lower():
                citations.append(m_clean)
        return citations

    def _extract_legal_ref(self, breadcrumb_text: str, text_chunk: str, source_name: str = "") -> str:
        """
        Trích xuất tham chiếu pháp lý & cấu trúc chuẩn tắc TỔNG QUÁT (Generic Canonical Reference):
        Tự động nhận diện cho MỌI loại văn bản ngân hàng, biểu phí, quy chế, FAQ, nghị quyết:
        Ví dụ:
        - [Quy chế eKYC - Điều 12, Khoản 2; Quyết định 2345/QĐ-NHNN]
        - [Biểu phí thẻ quốc tế - Mục II, Khoản 1]
        - [Nghị quyết 728/NQ-HĐQT - Điều 1]
        - [FAQ Thẻ ngân hàng - Câu 5]
        """
        full_context = f"{breadcrumb_text} {text_chunk}"

        # 1. Bóc tách tên định danh văn bản
        doc_identifier = self._extract_document_identifier(breadcrumb_text, full_context, source_name)

        # 2. Bóc tách đơn vị phân cấp (Điều, Khoản, Điểm, Mục, Câu)
        hierarchical_unit = self._extract_hierarchical_units(full_context, text_chunk)

        # 3. Bóc tách các văn bản viện dẫn chéo
        cross_citations = self._extract_cross_statutory_citations(full_context, doc_identifier)

        parts = []
        # Ghép định danh văn bản với đơn vị phân cấp
        if doc_identifier and hierarchical_unit:
            parts.append(f"{doc_identifier} - {hierarchical_unit}")
        elif hierarchical_unit:
            parts.append(hierarchical_unit)
        elif doc_identifier:
            parts.append(doc_identifier)

        # Thêm các văn bản viện dẫn chéo
        for c in cross_citations:
            if c not in parts:
                parts.append(c)

        if parts:
            return f"[{'; '.join(parts)}]"
        return ""

    def chunk_text(self, text, source_name="document.md"):
        """
        Phân đoạn một văn bản markdown/text bảo toàn cấu trúc breadcrumbs.
        """
        lines = text.split("\n")
        chunks = []
        chunk_counter = 0

        # Thứ bậc tiêu đề: {1: "Tiêu đề H1", 2: "Tiêu đề H2", 3: "Điều 1...", 4: "Khoản 1..."}
        hierarchy = {}
        current_lines = []
        current_canonical_questions = []

        def get_breadcrumb_str():
            sorted_lvls = sorted(hierarchy.keys())
            titles = [hierarchy[lvl] for lvl in sorted_lvls if hierarchy[lvl]]
            return " > ".join(titles) if titles else Path(source_name).stem

        def flush_buffer():
            nonlocal chunk_counter, current_canonical_questions
            if not current_lines:
                return

            raw = "\n".join(current_lines).strip()
            current_lines.clear()

            if not raw or len(raw) < 15:
                return

            breadcrumb = get_breadcrumb_str()
            legal_ref = self._extract_legal_ref(breadcrumb, raw, source_name=source_name)

            # Xác định loại chunk
            chunk_type = "text"
            if legal_ref:
                chunk_type = "legal_clause"
            elif current_canonical_questions:
                chunk_type = "faq"
            elif "|" in raw and "-|-" in raw:
                chunk_type = "table"

            # Đóng gói nội dung có bảo toàn breadcrumbs
            header_prefix = f"[Breadcrumbs: {breadcrumb}]"
            if legal_ref:
                header_prefix += f"\n[Pháp lý: {legal_ref}]"
            full_content = f"{header_prefix}\n{raw}"

            # Nếu văn bản dài hơn max_chunk_size, cắt nhỏ nhưng vẫn giữ breadcrumbs
            if len(raw) <= self.max_chunk_size:
                chunks.append({
                    "chunk_id": f"{Path(source_name).stem}_bc_{chunk_counter}",
                    "source": source_name,
                    "breadcrumb": breadcrumb,
                    "legal_ref": legal_ref,
                    "canonical_questions": list(current_canonical_questions),
                    "chunk_type": chunk_type,
                    "content": full_content,
                    "raw_content": raw
                })
                chunk_counter += 1
            else:
                # Cắt theo đoạn
                paragraphs = raw.split("\n\n")
                sub_buf = []
                sub_len = 0
                for p in paragraphs:
                    if sub_len + len(p) > self.max_chunk_size and sub_buf:
                        sub_text = "\n\n".join(sub_buf).strip()
                        sub_legal = self._extract_legal_ref(breadcrumb, sub_text, source_name=source_name) or legal_ref
                        chunks.append({
                            "chunk_id": f"{Path(source_name).stem}_bc_{chunk_counter}",
                            "source": source_name,
                            "breadcrumb": breadcrumb,
                            "legal_ref": sub_legal,
                            "canonical_questions": list(current_canonical_questions),
                            "chunk_type": chunk_type,
                            "content": f"[Breadcrumbs: {breadcrumb}]{f'\\n[Pháp lý: {sub_legal}]' if sub_legal else ''}\n{sub_text}",
                            "raw_content": sub_text
                        })
                        chunk_counter += 1
                        sub_buf = [p]
                        sub_len = len(p)
                    else:
                        sub_buf.append(p)
                        sub_len += len(p)
                
                if sub_buf:
                    sub_text = "\n\n".join(sub_buf).strip()
                    sub_legal = self._extract_legal_ref(breadcrumb, sub_text) or legal_ref
                    chunks.append({
                        "chunk_id": f"{Path(source_name).stem}_bc_{chunk_counter}",
                        "source": source_name,
                        "breadcrumb": breadcrumb,
                        "legal_ref": sub_legal,
                        "canonical_questions": list(current_canonical_questions),
                        "chunk_type": chunk_type,
                        "content": f"[Breadcrumbs: {breadcrumb}]{f'\\n[Pháp lý: {sub_legal}]' if sub_legal else ''}\n{sub_text}",
                        "raw_content": sub_text
                    })
                    chunk_counter += 1

            current_canonical_questions.clear()

        i = 0
        total = len(lines)
        while i < total:
            line = lines[i]
            stripped = line.strip()

            # 1. Phát hiện Markdown Header (#, ##, ###, ####)
            header_match = re.match(r"^(#{1,6})\s+(.*)$", stripped)
            if header_match:
                flush_buffer()
                level = len(header_match.group(1))
                title = header_match.group(2).strip()

                # Kiểm tra xem tiêu đề có phải là câu hỏi chuẩn không (ví dụ: '### Câu 1: ...' hoặc '### Q: ...')
                faq_q_match = re.search(r"(?:Câu\s*\d+|Hỏi|Q\d*)\s*[:\.\-]\s*(.+)", title, re.IGNORECASE)
                if faq_q_match:
                    current_canonical_questions.append(faq_q_match.group(1).strip())

                hierarchy[level] = title
                # Xóa các cấp con sâu hơn
                for k in list(hierarchy.keys()):
                    if k > level:
                        del hierarchy[k]
                i += 1
                continue

            # 2. Phát hiện Điều khoản pháp lý (Điều X, Chương Y) ở dạng chữ hoa hoặc in đậm
            legal_title_match = re.match(r"^(\*{0,2}(?:Điều|Chương|Mục)\s+\d+[a-zA-Z]?[\.\:]?\s*[^\*\n]+(?:\*{0,2}))", stripped, re.IGNORECASE)
            if legal_title_match and len(stripped) < 120:
                flush_buffer()
                title = legal_title_match.group(1).replace("*", "").strip()
                hierarchy[4] = title
                for k in list(hierarchy.keys()):
                    if k > 4:
                        del hierarchy[k]
                current_lines.append(line)
                i += 1
                continue

            # 3. Phát hiện câu hỏi chuẩn trong thân bài: '**Hỏi:** ...' hoặc '**Câu hỏi:** ...'
            inline_q_match = re.search(r"\*\*(?:Hỏi|Câu hỏi|Q)\s*:\*\*\s*(.+)", stripped, re.IGNORECASE)
            if inline_q_match:
                current_canonical_questions.append(inline_q_match.group(1).strip())

            # 4. Phát hiện đường phân cách ngang
            if stripped in ["---", "***", "___"]:
                flush_buffer()
                i += 1
                continue

            current_lines.append(line)
            i += 1

        flush_buffer()
        return chunks

    def chunk_directory(self, dir_path, max_files=None, extensions=(".md", ".txt")):
        """Phân đoạn toàn bộ tài liệu trong thư mục"""
        target_dir = Path(dir_path)
        if not target_dir.exists():
            return []

        all_chunks = []
        files = [p for p in target_dir.glob("**/*") if p.is_file() and p.suffix.lower() in extensions]
        if max_files:
            files = files[:max_files]

        for file_path in files:
            try:
                text = file_path.read_text(encoding="utf-8", errors="ignore")
                chunks = self.chunk_text(text, source_name=file_path.name)
                all_chunks.extend(chunks)
            except Exception as e:
                print(f"[Breadcrumbs Chunker] Lỗi đọc file {file_path.name}: {e}")

        return all_chunks
