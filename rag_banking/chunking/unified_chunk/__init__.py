"""
Unified Chunk Module
Chứa bộ điều phối nhận thức bố cục LayoutAwareChunker (DLAChunker),
bộ phân giải thực thể pháp lý canonical_resolver, BreadcrumbsChunker,
và DocumentStructureChunker (3 chiến lược).
Chuyên trách 2 định dạng cốt lõi: DOCX và PDF.
"""

from .layout_aware_chunker import LayoutAwareChunker, DLAChunker
from .canonical_resolver import (
    count_words,
    count_syllables_vietnamese,
    strip_vietnamese_accents,
    extract_document_identifier,
    extract_hierarchical_units,
    split_sentences_banking,
    split_sentences_generalized,
    group_sentences_banking,
    clean_and_dehyphenate_text,
    is_valid_vietnamese_text,
    is_valid_natural_text,
    detect_and_remove_headers_footers,
)
from .breadcrumbs_chunker import BreadcrumbsChunker
from .structure_chunker import DocumentStructureChunker
from .syllable_budget_manager import (
    SyllableProfile,
    SyllableBudgetManager,
    PROFILE_A,
    PROFILE_B,
    PROFILE_C,
    STANDARD_PROFILES
)
from .unified_budget_chunker import (
    UnifiedDOCXChunker,
    UnifiedStructureChunker,
    UnifiedBudgetOrchestrator
)

__all__ = [
    "LayoutAwareChunker",
    "DLAChunker",
    "BreadcrumbsChunker",
    "DocumentStructureChunker",
    "SyllableProfile",
    "SyllableBudgetManager",
    "PROFILE_A",
    "PROFILE_B",
    "PROFILE_C",
    "STANDARD_PROFILES",
    "UnifiedDOCXChunker",
    "UnifiedStructureChunker",
    "UnifiedBudgetOrchestrator",
    "count_words",
    "count_syllables_vietnamese",
    "strip_vietnamese_accents",
    "extract_document_identifier",
    "extract_hierarchical_units",
    "split_sentences_banking",
    "split_sentences_generalized",
    "group_sentences_banking",
    "clean_and_dehyphenate_text",
    "is_valid_vietnamese_text",
    "is_valid_natural_text",
    "detect_and_remove_headers_footers",
]
