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

    def _extract_legal_ref(self, breadcrumb_text, text_chunk):
        """
        Trích xuất tham chiếu pháp lý chuẩn xác dạng:
        [Quy chế eKYC Vietcombank - Điều 12, Khoản 2; Quyết định 2345/QĐ-NHNN]
        """
        full_context = f"{breadcrumb_text} {text_chunk}"
        refs = []

        # Tên quy chế / văn bản
        doc_prefix = ""
        if "quy chế ekyc" in full_context.lower():
            doc_prefix = "Quy chế eKYC Vietcombank - "
        elif "điều kiện điều khoản" in full_context.lower():
            doc_prefix = "ĐKĐK VCB - "

        # Tìm Điều X và Khoản Y
        dieu_match = re.search(r"Điều\s+(\d+[a-zA-Z]?)", full_context, re.IGNORECASE)
        khoan_match = re.search(r"Khoản\s+(\d+)", full_context, re.IGNORECASE)
        if not khoan_match:
            khoan_num_match = re.search(r"(?:^|\n)\s*(\d{1,2})\.\s+[A-ZÀ-Ỹ]", text_chunk)
            if khoan_num_match and dieu_match:
                khoan_match = khoan_num_match

        if dieu_match and khoan_match:
            refs.append(f"{doc_prefix}Điều {dieu_match.group(1)}, Khoản {khoan_match.group(1)}")
        elif dieu_match:
            refs.append(f"{doc_prefix}Điều {dieu_match.group(1)}")
        elif khoan_match:
            refs.append(f"Khoản {khoan_match.group(1)}")

        # Tìm Quyết định 2345/QĐ-NHNN hoặc Thông tư
        qd_match = re.search(r"(Quyết định\s+\d+/[A-ZĐ\-]+)", full_context, re.IGNORECASE)
        if qd_match:
            qd_str = qd_match.group(1)
            if qd_str not in refs:
                refs.append(qd_str)

        if refs:
            return f"[{'; '.join(refs)}]"
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
            legal_ref = self._extract_legal_ref(breadcrumb, raw)

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
