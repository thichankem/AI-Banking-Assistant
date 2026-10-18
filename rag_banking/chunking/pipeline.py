"""
Module Phase 1 Ingestion & Chunking Pipeline
=============================================
Pipeline điều phối quy trình phân đoạn tài liệu nhận thức bố cục DLA (Document Layout Analysis)
cho hệ thống AI Banking Assistant (Vietcombank).
Thiết kế tối giản, trực quan, dễ hiểu.
"""

from __future__ import annotations
from pathlib import Path

from .docx_chunker.docx_chunker import DOCXChunker
from .pdf_chunker.pdf_chunker import PDFChunker
from .unified_chunk.layout_aware_chunker import LayoutAwareChunker, DLAChunker


class ChunkingPipeline:
    """
    Pipeline cấp cao thực thi phân đoạn tài liệu ngân hàng (.docx, .pdf).
    """

    def __init__(
        self,
        max_chunk_size: int = 800,
        min_section_size: int = 120,
        chunk_overlap: int = 0,
        **kwargs
    ):
        self.max_chunk_size = max_chunk_size
        self.min_section_size = min_section_size
        self.chunk_overlap = chunk_overlap

        # Khởi tạo bộ phân đoạn trung tâm
        self.orchestrator = LayoutAwareChunker(
            max_chunk_size=max_chunk_size,
            min_section_size=min_section_size,
            chunk_overlap=chunk_overlap,
            **kwargs
        )
        self.docx_chunker = self.orchestrator.docx_chunker
        self.pdf_chunker = self.orchestrator.pdf_chunker

    def process_file(
        self,
        file_path: str | Path,
        max_pdf_pages: int | None = None,
        **kwargs
    ) -> list[dict]:
        """Phân đoạn một tệp (.docx, .pdf)."""
        return self.orchestrator.chunk_file(file_path, max_pdf_pages=max_pdf_pages, **kwargs)

    def process_directory(
        self,
        dir_path: str | Path,
        extensions: list[str] | None = None,
        max_files: int | None = None,
        max_pdf_pages: int | None = None
    ) -> list[dict]:
        """Phân đoạn toàn bộ tệp trong thư mục theo danh sách đuôi mở rộng."""
        return self.orchestrator.chunk_directory(
            dir_path=dir_path,
            extensions=extensions,
            max_files=max_files,
            max_pdf_pages=max_pdf_pages
        )

    def process_dataset(
        self,
        dataset_dir: str | Path,
        output_dir: str | Path | None = None,
        max_pdf_pages: int | None = 3
    ) -> tuple[list[dict], dict]:
        """Phân đoạn toàn bộ dataset tài liệu ngân hàng và xuất kết quả thống kê."""
        return self.orchestrator.chunk_dataset(
            dataset_dir=dataset_dir,
            output_dir=output_dir,
            max_pdf_pages=max_pdf_pages
        )


def run_chunking_pipeline(
    dataset_dir: str | Path,
    output_dir: str | Path | None = None,
    max_pdf_pages: int = 3
) -> tuple[list[dict], dict]:
    """Hàm tiện ích nhanh khởi chạy Phase 1 Chunking Pipeline."""
    pipeline = ChunkingPipeline()
    return pipeline.process_dataset(
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        max_pdf_pages=max_pdf_pages
    )


__all__ = ["ChunkingPipeline", "run_chunking_pipeline"]
