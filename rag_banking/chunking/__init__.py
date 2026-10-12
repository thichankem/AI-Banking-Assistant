"""
Module Chunking: Chứa các giải thuật phân đoạn nhận thức bố cục (DLA - Document Layout Analysis)
và Breadcrumbs Chunking phục vụ RAG Banking.

Hỗ trợ 3 luồng định dạng dữ liệu chính:
- DOCX: DOCXChunker (Heading hierarchy H1-H4, Tables, Lists, Paragraphs)
- PDF:  PDFChunker (Văn bản số Digital + Tự động RapidOCR cho bản scan)
- Excel: ExcelChunker (Đa sheet, tự tìm Header, Linearized Table Rows)
Cùng bộ điều phối thống nhất LayoutAwareChunker (DLAChunker).
"""

from .docx_chunker import DOCXChunker
from .pdf_chunker import PDFChunker
from .excel_chunker import ExcelChunker
from .ocr_extractor import OCRExtractor
from .layout_aware_chunker import LayoutAwareChunker, DLAChunker
from .breadcrumbs_chunker import BreadcrumbsChunker
from .structure_chunker import DocumentStructureChunker

from .scan_structure_reconstructor import BreadcrumbTracker, ScanStructureReconstructor, TOCVerifier
from .ocr_layout_predictor import OCRLayoutHierarchyPredictor, LayoutHierarchyLevel

__all__ = [
    "LayoutAwareChunker",
    "DLAChunker",
    "DOCXChunker",
    "PDFChunker",
    "ExcelChunker",
    "OCRExtractor",
    "BreadcrumbsChunker",
    "DocumentStructureChunker",
    "BreadcrumbTracker",
    "ScanStructureReconstructor",
    "TOCVerifier",
    "OCRLayoutHierarchyPredictor",
    "LayoutHierarchyLevel"
]

