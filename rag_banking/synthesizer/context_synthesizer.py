"""
Context Synthesizer (Bước 4: Đóng gói Ngữ cảnh Markdown, Gắn Thẻ Pháp lý & Kiểm soát Giới hạn Token)
Nhiệm vụ:
1. Đóng gói danh sách Top-3 chunks thành định dạng Markdown tiêu chuẩn ngân hàng.
2. Tự động nhận diện và gắn thẻ nguồn tham chiếu pháp lý [Điều X, Khoản Y].
3. Kiểm soát ngân sách token (Token Budget Control), đảm bảo không vượt quá max_tokens và không tràn context window.
"""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Any

@dataclass
class SynthesizedContext:
    markdown_context: str
    total_tokens: int
    legal_citations: List[str]
    sources_used: List[str]
    chunks_included: int

class ContextSynthesizer:
    def __init__(self, max_context_tokens: int = 2000):
        self.max_context_tokens = max_context_tokens

    def estimate_tokens(self, text: str) -> int:
        """
        Ước tính số token cho văn bản tiếng Việt:
        Trung bình 1 token tương ứng ~3.5 ký tự hoặc ~1.2 từ tiếng Việt.
        """
        if not text:
            return 0
        char_based = len(text) / 3.5
        word_based = len(text.split()) * 1.25
        return int(max(char_based, word_based))

    def extract_legal_tags(self, text: str, default_ref: str = "") -> List[str]:
        """
        Trích xuất tất cả các thẻ tham chiếu pháp lý [Điều X, Khoản Y] hoặc Điều X, Khoản Y.
        """
        tags = set()
        if default_ref:
            tags.add(default_ref)

        # Regex tìm các dạng pháp lý: [Quy chế...], [Điều X, Khoản Y], Quyết định 2345/QĐ-NHNN
        patterns = [
            r"\[([^\]]*(?:Điều|Khoản|Quy chế|Quyết định|Thông tư)[^\]]*)\]",
            r"(?:^|\s)(Điều\s+\d+[a-zA-Z]?(?:[\,\.]\s*Khoản\s+\d+)?)",
            r"(Quyết định\s+\d+/[A-ZĐ\-]+)"
        ]

        for pat in patterns:
            matches = re.findall(pat, text, re.IGNORECASE)
            for m in matches:
                clean_m = m.strip()
                if not clean_m.startswith("["):
                    clean_m = f"[{clean_m}]"
                tags.add(clean_m)

        return sorted(list(tags))

    def synthesize(
        self,
        top_chunks: List[Dict[str, Any]],
        max_tokens: int = None
    ) -> SynthesizedContext:
        """
        Đóng gói ngữ cảnh Markdown từ danh sách Top-3 chunks kèm kiểm soát token.
        """
        budget = max_tokens or self.max_context_tokens
        formatted_blocks = []
        all_legal_citations = []
        sources = set()
        current_token_count = 0
        chunks_used = 0

        for idx, item in enumerate(top_chunks, start=1):
            chunk = item.get("chunk", item)
            source = chunk.get("source", "Tài liệu nghiệp vụ Vietcombank")
            breadcrumb = chunk.get("breadcrumb", "Quy định chung")
            raw_content = chunk.get("raw_content", chunk.get("content", ""))
            legal_ref = chunk.get("legal_ref", "")

            # Trích xuất toàn bộ các thẻ căn cứ pháp lý
            found_tags = self.extract_legal_tags(f"{breadcrumb}\n{raw_content}", default_ref=legal_ref)
            for tag in found_tags:
                if tag not in all_legal_citations:
                    all_legal_citations.append(tag)

            legal_display = ", ".join(found_tags) if found_tags else "[Quy chuẩn nghiệp vụ]"
            sources.add(source)

            # Ước tính token của block này
            block_header = (
                f"### [TRÍCH DẪN {idx}] Nguồn: {source}\n"
                f"- **Tham chiếu pháp lý:** `{legal_display}`\n"
                f"- **Đường dẫn phân cấp (Breadcrumbs):** `{breadcrumb}`\n"
                f"- **Nội dung xác thực:**\n"
            )
            header_tokens = self.estimate_tokens(block_header)
            content_tokens = self.estimate_tokens(raw_content)

            # Kiểm tra giới hạn token
            if current_token_count + header_tokens + content_tokens > budget:
                # Cắt tỉa nội dung an toàn để vừa khít ngân sách
                remaining_tokens = budget - current_token_count - header_tokens
                if remaining_tokens > 100:
                    allowed_chars = int(remaining_tokens * 3.5)
                    truncated_content = raw_content[:allowed_chars] + "\n... [Đoạn trích được rút gọn theo giới hạn token bảo đảm an toàn]."
                    full_block = f"{block_header}{truncated_content}\n"
                    formatted_blocks.append(full_block)
                    current_token_count += self.estimate_tokens(full_block)
                    chunks_used += 1
                break
            else:
                full_block = f"{block_header}{raw_content}\n"
                formatted_blocks.append(full_block)
                current_token_count += header_tokens + content_tokens
                chunks_used += 1

        markdown_context = "\n---\n".join(formatted_blocks)

        return SynthesizedContext(
            markdown_context=markdown_context,
            total_tokens=current_token_count,
            legal_citations=all_legal_citations,
            sources_used=list(sources),
            chunks_included=chunks_used
        )
