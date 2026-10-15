"""
PDF Chunker Module
Chuyên trách phân đoạn tệp PDF số (Digital) và scan mộc đỏ qua Tesseract OCR v5 & Phân cấp Hình học (0% Regex).
"""

from .pdf_chunker import PDFChunker
from .ocr_extractor import OCRExtractor
from .ocr_layout_predictor import OCRLayoutHierarchyPredictor, LayoutHierarchyLevel
from .scan_structure_reconstructor import ScanStructureReconstructor, BreadcrumbTracker, TOCVerifier
from .scan_detector import (
    detect_is_scanned_page,
    is_scanned_page,
    detect_pdf_file_scan_status,
    ScanDetector
)
from .vision_table_extractor import LocalVisionTableExtractor
from .scan_table_detector import ScanTableDetector, DetectedTableBox
from .paddle_vietocr_adapter import PaddleVietOCRAdapter
from .unified_page_stream_processor import UnifiedPageStreamProcessor, UnifiedGeometryLine
from .hybrid_pdf_chunker import HybridPDFChunker
from .table_formatter import TableFormatter, TableFormatType, NormalizedTable, TableCell

__all__ = [
    "PDFChunker",
    "OCRExtractor",
    "LocalVisionTableExtractor",
    "OCRLayoutHierarchyPredictor",
    "LayoutHierarchyLevel",
    "ScanStructureReconstructor",
    "BreadcrumbTracker",
    "TOCVerifier",
    "detect_is_scanned_page",
    "is_scanned_page",
    "detect_pdf_file_scan_status",
    "ScanDetector",
    "ScanTableDetector",
    "DetectedTableBox",
    "PaddleVietOCRAdapter",
    "UnifiedPageStreamProcessor",
    "UnifiedGeometryLine",
    "HybridPDFChunker",
    "TableFormatter",
    "TableFormatType",
    "NormalizedTable",
    "TableCell",
]
