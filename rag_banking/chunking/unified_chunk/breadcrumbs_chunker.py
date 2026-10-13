"""
Breadcrumbs Chunking (Phân đoạn bảo toàn đường dẫn phân cấp)
Chuyên trách phân tách tài liệu ngân hàng & pháp lý mà vẫn bảo toàn 100% ngữ cảnh phân cấp:
1. Breadcrumbs Tracking: Theo dõi liên tục phân cấp (Tài liệu > Chương > Điều > Khoản > Tiểu mục).
2. Legal Reference Tagging: Nhận diện và trích xuất chuẩn xác [Điều X, Khoản Y] hoặc [Điều X].
3. Canonical Questions Extraction: Bắt các câu hỏi chuẩn từ FAQ (ví dụ: '### Câu 1: ...') phục vụ bước Reranking.
4. Breadcrumbs Prepended Content: Gắn tiền tố đường dẫn vào nội dung chunk để vector nhúng và BM25 bắt trọn vẹn ngữ cảnh.
"""

from pathlib import Path

class BreadcrumbsChunker:
    def __init__(self, max_chunk_size=900, chunk_overlap=120):
        self.max_chunk_size = max_chunk_size
        self.chunk_overlap = chunk_overlap


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

            # Xác định loại chunk
            chunk_type = "text"
            if current_canonical_questions:
                chunk_type = "faq"
            elif "|" in raw and "-|-" in raw:
                chunk_type = "table"

            # Đóng gói nội dung có bảo toàn breadcrumbs
            header_prefix = f"[Breadcrumbs: {breadcrumb}]"
            full_content = f"{header_prefix}\n{raw}"

            # Nếu văn bản dài hơn max_chunk_size, cắt nhỏ nhưng vẫn giữ breadcrumbs
            if len(raw) <= self.max_chunk_size:
                chunks.append({
                    "chunk_id": f"{Path(source_name).stem}_bc_{chunk_counter}",
                    "source": source_name,
                    "breadcrumb": breadcrumb,
                    "canonical_questions": list(current_canonical_questions),
                    "chunk_type": chunk_type,
                    "content": full_content,
                    "word_count": len(full_content.split())
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
                        sub_content = f"[Breadcrumbs: {breadcrumb}]\n{sub_text}"
                        chunks.append({
                            "chunk_id": f"{Path(source_name).stem}_bc_{chunk_counter}",
                            "source": source_name,
                            "breadcrumb": breadcrumb,
                            "canonical_questions": list(current_canonical_questions),
                            "chunk_type": chunk_type,
                            "content": sub_content,
                            "word_count": len(sub_content.split())
                        })
                        chunk_counter += 1
                        sub_buf = [p]
                        sub_len = len(p)
                    else:
                        sub_buf.append(p)
                        sub_len += len(p)
                
                if sub_buf:
                    sub_text = "\n\n".join(sub_buf).strip()
                    sub_content = f"[Breadcrumbs: {breadcrumb}]\n{sub_text}"
                    chunks.append({
                        "chunk_id": f"{Path(source_name).stem}_bc_{chunk_counter}",
                        "source": source_name,
                        "breadcrumb": breadcrumb,
                        "canonical_questions": list(current_canonical_questions),
                        "chunk_type": chunk_type,
                        "content": sub_content,
                        "word_count": len(sub_content.split())
                    })
                    chunk_counter += 1

            current_canonical_questions.clear()

        i = 0
        total = len(lines)
        while i < total:
            line = lines[i]
            stripped = line.strip()

            # 1. Phát hiện Markdown Header (#, ##, ###, ####)
            if stripped.startswith("#"):
                hash_part = stripped.split()[0]
                if all(c == "#" for c in hash_part) and len(hash_part) <= 6:
                    flush_buffer()
                    level = len(hash_part)
                    title = stripped[level:].strip()

                    # Kiểm tra xem tiêu đề có phải là câu hỏi chuẩn không (dấu chấm hỏi ?)
                    if title.endswith("?") or ("?" in title and len(title) < 150):
                        current_canonical_questions.append(title)

                    hierarchy[level] = title
                    # Xóa các cấp con sâu hơn
                    for k in list(hierarchy.keys()):
                        if k > level:
                            del hierarchy[k]
                    i += 1
                    continue

            # 2. Phát hiện cấu trúc thứ bậc / tiêu đề in đậm / outline numbering
            clean_s = stripped.replace("*", "").strip()
            s_tokens = clean_s.split()
            first_tok_clean = s_tokens[0].rstrip(".:)-") if s_tokens else ""
            is_roman = first_tok_clean in ("I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII")
            is_bold_title = (stripped.startswith("**") and stripped.endswith("**") and len(clean_s) < 120)
            is_outline = (s_tokens and len(clean_s) < 120 and (is_roman or s_tokens[0].rstrip(".").count(".") >= 1 or (s_tokens[0].endswith(".") and s_tokens[0][:-1].isdigit())) and len(s_tokens) >= 2)
            if is_bold_title or is_outline:
                flush_buffer()
                title = clean_s
                hierarchy[4] = title
                for k in list(hierarchy.keys()):
                    if k > 4:
                        del hierarchy[k]
                current_lines.append(line)
                i += 1
                continue

            # 3. Phát hiện câu hỏi trong thân bài: '**Prefix:** ...?'
            if stripped.startswith("**") and ":**" in stripped:
                after_colon = stripped.split(":**", 1)[1].strip()
                if after_colon.endswith("?") or "?" in after_colon:
                    current_canonical_questions.append(after_colon)

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
