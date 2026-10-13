"""
Unified Chunk Module
Chứa bộ điều phối nhận thức bố cục LayoutAwareChunker (DLAChunker),
bộ phân giải thực thể pháp lý canonical_resolver, BreadcrumbsChunker,
DocumentStructureChunker (3 chiến lược), và ExcelChunker.
"""

from .layout_aware_chunker import LayoutAwareChunker, DLAChunker
from .canonical_resolver import (
    count_words,
    count_syllables_vietnamese,
    strip_vietnamese_accents,
    extract_document_identifier,
    extract_hierarchical_units,
    split_sentences_banking,
    group_sentences_banking,
    clean_and_dehyphenate_text,
    is_valid_vietnamese_text,
    detect_and_remove_headers_footers,
)
from .breadcrumbs_chunker import BreadcrumbsChunker
from .structure_chunker import DocumentStructureChunker
from .excel_chunker import ExcelChunker

__all__ = [
    "LayoutAwareChunker",
    "DLAChunker",
    "BreadcrumbsChunker",
    "DocumentStructureChunker",
    "ExcelChunker",
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
