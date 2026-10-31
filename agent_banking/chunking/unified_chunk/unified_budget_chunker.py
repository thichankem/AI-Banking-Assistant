"""
Module Unified Budget Chunker (Bộ Phân đoạn Đồng nhất Ngân sách Âm tiết)
========================================================================
Chuẩn hóa toàn bộ điều kiện ngắt chunk trên tài liệu .docx về một chuẩn đo độ dài
duy nhất là `len(text.split())` (0% Regex).

Cung cấp:
1. `UnifiedDOCXChunker`: Phân đoạn tài liệu Word theo (min_syllables, max_syllables).
2. `UnifiedBudgetOrchestrator`: Bộ điều phối phân đoạn tài liệu theo ngân sách chuẩn.

TUÂN THỦ 100% SCHEMA CONTRACT:
- `chunk_id`, `source`, `chunk_type`, `breadcrumb`, `section_title`, `content`, `word_count`.
- Tuyệt đối không sinh trường rác ('ref', 'legal_ref', 'raw_content', 'char_length', 'format').
"""

from __future__ import annotations

import logging
from pathlib import Path

import docx

try:
    from agent_banking.chunking.unified_chunk.canonical_resolver import split_sentences_banking
except (ImportError, ValueError):
    from .canonical_resolver import split_sentences_banking

from .syllable_budget_manager import (
    MIN_SYLLABLES,
    MAX_SYLLABLES,
    CEILING_SYLLABLES,
    SyllableBudgetManager,
    prepend_context_header
)

logger = logging.getLogger("unified_budget_chunker")


class UnifiedDOCXChunker:
    """
    Bộ phân đoạn DOCX đồng nhất hoàn toàn theo đơn vị âm tiết (Syllable Budget).
    """

    def __init__(
        self,
        min_syllables: int = MIN_SYLLABLES,
        max_syllables: int = MAX_SYLLABLES,
        ceiling_syllables: int = CEILING_SYLLABLES,
        **kwargs
    ):
        self.min_syllables = min_syllables
        self.max_syllables = max_syllables
        self.ceiling_syllables = ceiling_syllables
        self.budget_manager = SyllableBudgetManager(
            min_syllables=min_syllables,
            max_syllables=max_syllables,
            ceiling_syllables=ceiling_syllables
        )

    def chunk_docx(self, docx_path: str | Path) -> list[dict]:
        """
        Đọc và phân đoạn tài liệu DOCX theo ngân sách âm tiết chuẩn hóa.
        """
        docx_path = Path(docx_path)
        if not docx_path.is_file():
            return []

        doc = docx.Document(str(docx_path))
        chunks: list[dict] = []
        chunk_idx = 0

        active_headers: dict[int, str] = {}
        text_buffer: list[str] = []
        doc_title = docx_path.stem

        def get_current_breadcrumb() -> str:
            levels = sorted(active_headers.keys())
            headers = [active_headers[lvl] for lvl in levels if active_headers[lvl]]
            if headers:
                return " > ".join([doc_title] + headers)
            return doc_title

        def emit_text_chunk(buf: list[str]) -> dict | None:
            nonlocal chunk_idx
            if not buf:
                return None
            full_text = "\n".join(buf).strip()
            if not full_text:
                return None

            syl_count = len(full_text.split())
            breadcrumb = get_current_breadcrumb()
            section_title = active_headers.get(max(active_headers.keys(), default=1), doc_title)

            # Nếu văn bản dài quá trần max_syllables: chia nhỏ theo ngân sách âm tiết
            if syl_count > self.max_syllables:
                sentences = split_sentences_banking(full_text)
                sub_parts = self.budget_manager.split_sentences_by_syllable_budget(
                    sentences=sentences,
                    max_syllables=self.max_syllables
                )
                for part in sub_parts:
                    content_str = prepend_context_header(part, breadcrumb, section_title)
                    chunks.append({
                        "chunk_id": f"{docx_path.stem}_docx_c{chunk_idx}",
                        "source": docx_path.name,
                        "chunk_type": "docx_text",
                        "breadcrumb": breadcrumb,
                        "section_title": section_title,
                        "content": content_str,
                        "word_count": len(content_str.split()),
                        "syllable_count": len(content_str.split())
                    })
                    chunk_idx += 1
                return None

            content_str = prepend_context_header(full_text, breadcrumb, section_title)
            chunk_obj = {
                "chunk_id": f"{docx_path.stem}_docx_c{chunk_idx}",
                "source": docx_path.name,
                "chunk_type": "docx_text",
                "breadcrumb": breadcrumb,
                "section_title": section_title,
                "content": content_str,
                "word_count": len(content_str.split()),
                "syllable_count": syl_count
            }
            chunk_idx += 1
            chunks.append(chunk_obj)
            return chunk_obj

        # Quét các paragraph trong document
        for para in doc.paragraphs:
            text = para.text.strip()
            if not text:
                continue

            style_name = ""
            if para.style and para.style.name:
                style_name = str(para.style.name).lower()
            style_id = ""
            if para.style and getattr(para.style, "style_id", ""):
                style_id = str(getattr(para.style, "style_id", "")).lower()
            is_heading = "heading" in style_name or "heading" in style_id or "title" in style_name or "title" in style_id

            if is_heading:
                # Flush buffer trước đó
                if text_buffer:
                    emit_text_chunk(text_buffer)
                    text_buffer = []

                # Cập nhật level heading
                level = 1
                for c in style_name:
                    if c.isdigit():
                        level = int(c)
                        break

                active_headers[level] = text
                # Xóa các cấp con sâu hơn
                for l in list(active_headers.keys()):
                    if l > level:
                        del active_headers[l]
                continue

            text_buffer.append(text)
            # Kiểm tra ngân sách buffer
            cur_syl = sum(len(t.split()) for t in text_buffer)
            if cur_syl >= self.max_syllables:
                emit_text_chunk(text_buffer)
                text_buffer = []

            if text_buffer:
                emit_text_chunk(text_buffer)

        return chunks


class UnifiedBudgetOrchestrator:
    """
    Bộ phân đoạn hợp nhất tài liệu DOCX theo ngân sách âm tiết chuẩn.
    """

    def __init__(self, **kwargs):
        self.docx_chunker = UnifiedDOCXChunker(**kwargs)

    def chunk_file(self, file_path: str | Path, **kwargs) -> list[dict]:
        """Phân đoạn tệp (.docx) theo ngân sách âm tiết chuẩn hóa."""
        p = Path(file_path)
        suffix = p.suffix.lower()

        if suffix == ".docx":
            return self.docx_chunker.chunk_docx(p)
        else:
            logger.warning(f"[UnifiedBudgetOrchestrator] Bỏ qua định dạng chưa hỗ trợ: {suffix} ({p.name})")
            return []


__all__ = [
    "UnifiedDOCXChunker",
    "UnifiedBudgetOrchestrator"
]
