"""
Module Phase 1 Ingestion & Chunking Pipeline
=============================================
Pipeline điều phối quy trình phân đoạn tài liệu nhận thức bố cục DLA (Document Layout Analysis)
phục vụ hệ thống AI Banking Assistant (Vietcombank).

Cấu trúc phân hệ chuyên trách:
1. `DOCXChunker`: Phân tích AST Heading, Table Atomicity, Header 2 tầng, Section-Length-Aware.
2. `PDFChunker`: ScanDetector, RapidOCR ONNX, Lọc mộc đỏ HSV, OCRLayoutHierarchyPredictor (0% Regex),
   ScanStructureReconstructor (State Machine BreadcrumbTracker).
3. `ExcelChunker`: Bảng ma trận đa sheet, tuyến tính hóa "Cột: Giá trị", row batching.
4. `LayoutAwareChunker`: Bộ điều phối trung tâm tự động nhận diện phần mở rộng tệp.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union

from .docx_chunker import DOCXChunker
from .pdf_chunker import (
    PDFChunker,
    OCRExtractor,
    ScanStructureReconstructor,
    OCRLayoutHierarchyPredictor,
    LayoutHierarchyLevel
)
from .unified_chunk import (
    LayoutAwareChunker,
    DLAChunker,
    ExcelChunker,
    BreadcrumbsChunker,
    DocumentStructureChunker
)


class ChunkingPipeline:
    """
    Pipeline hợp nhất giai đoạn Phase 1 - Ingestion & Chunking cho hệ thống AI Banking Assistant.
    Cung cấp giao diện cấp cao để phân đoạn tệp đơn lẻ hoặc chạy hàng loạt toàn bộ dataset.
    """

    def __init__(
        self,
        max_chunk_size: int = 800,
        min_section_size: int = 120,
        chunk_overlap: int = 100,
        excel_rows_per_chunk: int = 3
    ):
        self.max_chunk_size = max_chunk_size
        self.min_section_size = min_section_size
        self.chunk_overlap = chunk_overlap

        # Khởi tạo bộ phân đoạn trung tâm (DLA Engine)
        self.orchestrator = LayoutAwareChunker(
            max_chunk_size=max_chunk_size,
            min_section_size=min_section_size,
            chunk_overlap=chunk_overlap,
            excel_rows_per_chunk=excel_rows_per_chunk
        )

        # Truy cập trực tiếp các sub-chunker chuyên biệt khi cần
        self.docx_chunker = self.orchestrator.docx_chunker
        self.pdf_chunker = self.orchestrator.pdf_chunker
        self.excel_chunker = self.orchestrator.excel_chunker
        self.structure_chunker = DocumentStructureChunker(
            chunk_size=max_chunk_size // 2,
            min_section_size=min_section_size,
            chunk_overlap=chunk_overlap // 2
        )

    def process_file(
        self,
        file_path: Union[str, Path],
        max_pdf_pages: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Xử lý phân đoạn tự động cho một tệp bất kỳ (.docx, .pdf, .xlsx, .md).
        """
        return self.orchestrator.chunk_file(file_path, max_pdf_pages=max_pdf_pages)

    def process_dataset(
        self,
        dataset_dir: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
        max_pdf_pages: Optional[int] = 3,
        include_excel: bool = False,
        include_md: bool = False
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Chạy toàn bộ dataset tài liệu ngân hàng theo lô, thu thập thống kê DLA và xuất artifacts.
        """
        return self.orchestrator.chunk_dataset(
            dataset_dir=dataset_dir,
            output_dir=output_dir,
            max_pdf_pages=max_pdf_pages,
            include_excel=include_excel,
            include_md=include_md
        )

    def process_directory(
        self,
        dir_path: Union[str, Path],
        extensions: Optional[List[str]] = None,
        max_files: Optional[int] = None,
        max_pdf_pages: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Xử lý phân đoạn toàn bộ tệp trong thư mục theo danh sách phần mở rộng chỉ định.
        """
        return self.orchestrator.chunk_directory(
            dir_path=dir_path,
            extensions=extensions,
            max_files=max_files,
            max_pdf_pages=max_pdf_pages
        )


def run_chunking_pipeline(
    dataset_dir: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    max_pdf_pages: int = 3
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Hàm tiện ích nhanh khởi chạy Phase 1 Chunking Pipeline."""
    pipeline = ChunkingPipeline()
    return pipeline.process_dataset(
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        max_pdf_pages=max_pdf_pages
    )
