"""
Module Hybrid PDF Chunker (Bộ Phân đoạn PDF Lai Bố cục Thống nhất)
===================================================================
Bộ phân đoạn tài liệu PDF thế hệ mới loại bỏ rẽ nhánh 50% scan ratio:
- Tích hợp UnifiedPageStreamProcessor.
- Xử lý bảng biểu cả trên trang scan (ScanTableDetector) lẫn trang digital (find_tables).
- Đóng gói chunks hợp nhất tuân thủ nghiêm ngặt Schema Contract.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pymupdf as fitz

try:
    from rag_banking.chunking.pdf_chunker.pdf_chunker import PDFChunker
    from rag_banking.chunking.unified_chunk.canonical_resolver import count_words
except (ImportError, ValueError):
    from .pdf_chunker import PDFChunker
    from ..unified_chunk.canonical_resolver import count_words

from .unified_page_stream_processor import UnifiedPageStreamProcessor

logger = logging.getLogger("hybrid_pdf_chunker")


class HybridPDFChunker:
    """
    Bộ phân đoạn PDF lai thế hệ mới (Hybrid PDF Chunker):
    Thống nhất xử lý tài liệu đa hình thái (trang đầu scan mộc đỏ, trang sau digital)
    thành một luồng dữ liệu phân đoạn xuyên suốt, không gãy vỡ phả hệ breadcrumbs.
    """

    def __init__(
        self,
        ocr_extractor: Any | None = None,
        max_chunk_size: int = 800,
        min_section_size: int = 120,
        chunk_overlap: int = 100,
        table_rows_per_chunk: int = 10
    ):
        self.max_chunk_size = max_chunk_size
        self.min_section_size = min_section_size
        self.chunk_overlap = chunk_overlap
        self.table_rows_per_chunk = table_rows_per_chunk

        self.processor = UnifiedPageStreamProcessor(
            ocr_extractor=ocr_extractor,
            max_chunk_size=max_chunk_size,
            min_section_size=min_section_size,
            chunk_overlap=chunk_overlap
        )

    def chunk_pdf(
        self,
        pdf_path: str | Path,
        max_pages: int | None = None
    ) -> list[dict[str, Any]]:
        """
        Phân đoạn tệp PDF lai theo luồng thống nhất không dùng ngưỡng 50%.
        """
        chunks, stats = self.processor.process_pdf_document(
            pdf_path=pdf_path,
            max_pages=max_pages
        )
        return chunks
