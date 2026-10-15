"""
Module Chunking (Phase 1 Ingestion & Chunking):
Chứa các giải thuật phân đoạn nhận thức bố cục (DLA - Document Layout Analysis)
và Breadcrumbs Chunking phục vụ RAG Banking.

Cấu trúc phân hệ (3 sub-folders):
1. docx_chunker/   : Phân đoạn tệp Word (.docx) theo AST heading, table atomicity, header 2 tầng
2. pdf_chunker/    : Phân đoạn tệp PDF số & scan Tesseract OCR v5, phân cấp hình học OCR (0% Regex)
3. unified_chunk/  : Bộ điều phối LayoutAwareChunker, canonical_resolver, structure_chunker (chuyên biệt DOCX & PDF)

Cùng bộ điều phối Pipeline cấp cao:
- pipeline.py      : ChunkingPipeline & run_chunking_pipeline
"""

import sys
import importlib

# 1. Nạp các thành phần từ 3 phân hệ chính
from .docx_chunker import DOCXChunker
from .pdf_chunker import (
    PDFChunker,
    OCRExtractor,
    OCRLayoutHierarchyPredictor,
    LayoutHierarchyLevel,
    ScanStructureReconstructor,
    BreadcrumbTracker,
    TOCVerifier,
    ScanTableDetector,
    PaddleVietOCRAdapter,
    UnifiedPageStreamProcessor,
    HybridPDFChunker,
    TableFormatter,
    TableFormatType
)
from .unified_chunk import (
    LayoutAwareChunker,
    DLAChunker,
    BreadcrumbsChunker,
    DocumentStructureChunker,
    SyllableProfile,
    SyllableBudgetManager,
    PROFILE_A,
    PROFILE_B,
    PROFILE_C,
    STANDARD_PROFILES,
    UnifiedBudgetOrchestrator,
    count_words,
    count_syllables_vietnamese,
    strip_vietnamese_accents,
    extract_document_identifier,
    extract_hierarchical_units,
    split_sentences_banking,
    group_sentences_banking,
    clean_and_dehyphenate_text,
    is_valid_vietnamese_text,
    detect_and_remove_headers_footers
)
from .pipeline import ChunkingPipeline, run_chunking_pipeline

# 2. Hỗ trợ import tương thích ngược trong sys.modules (Backward compatibility without duplicate files)
_LEGACY_MODULE_MAP = {
    "docx_chunker": ".docx_chunker.docx_chunker",
    "pdf_chunker": ".pdf_chunker.pdf_chunker",
    "ocr_extractor": ".pdf_chunker.ocr_extractor",
    "ocr_layout_predictor": ".pdf_chunker.ocr_layout_predictor",
    "scan_detector": ".pdf_chunker.scan_detector",
    "scan_structure_reconstructor": ".pdf_chunker.scan_structure_reconstructor",
    "layout_aware_chunker": ".unified_chunk.layout_aware_chunker",
    "breadcrumbs_chunker": ".unified_chunk.breadcrumbs_chunker",
    "canonical_resolver": ".unified_chunk.canonical_resolver",
    "structure_chunker": ".unified_chunk.structure_chunker",
}

for _sub_name, _sub_rel_path in _LEGACY_MODULE_MAP.items():
    _full_sub_name = f"{__name__}.{_sub_name}"
    if _full_sub_name not in sys.modules:
        try:
            sys.modules[_full_sub_name] = importlib.import_module(_sub_rel_path, package=__name__)
        except Exception:
            pass


def __getattr__(name: str):
    """Fallback dynamic loader cho các script cũ import từ rag_banking.chunking.<module>."""
    if name in _LEGACY_MODULE_MAP:
        return sys.modules.get(f"{__name__}.{name}") or importlib.import_module(_LEGACY_MODULE_MAP[name], package=__name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Top-level Pipeline
    "ChunkingPipeline",
    "run_chunking_pipeline",
    # Unified & Core Orchestrator
    "LayoutAwareChunker",
    "DLAChunker",
    "BreadcrumbsChunker",
    "DocumentStructureChunker",
    # DOCX Subsystem
    "DOCXChunker",
    # PDF Subsystem
    "PDFChunker",
    "OCRExtractor",
    "OCRLayoutHierarchyPredictor",
    "LayoutHierarchyLevel",
    "ScanStructureReconstructor",
    "BreadcrumbTracker",
    "TOCVerifier",
    # Canonical Resolver Utilities
    "count_words",
    "count_syllables_vietnamese",
    "strip_vietnamese_accents",
    "extract_document_identifier",
    "extract_hierarchical_units",
    "split_sentences_banking",
    "group_sentences_banking",
    "clean_and_dehyphenate_text",
    "is_valid_vietnamese_text",
    "detect_and_remove_headers_footers",
]
