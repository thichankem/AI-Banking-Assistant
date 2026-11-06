"""Legacy budget entrypoints delegate to the single verified DOCX pipeline.

Whitespace word limits remain an additional bound. Exact E5 tokens are always
enforced, and no independent paragraph-only/table-dropping path is retained.
"""
from .layout_aware_chunker import LayoutAwareChunker
from .syllable_budget_manager import MIN_SYLLABLES, MAX_SYLLABLES, CEILING_SYLLABLES


class UnifiedDOCXChunker(LayoutAwareChunker):
    def __init__(self, min_syllables=MIN_SYLLABLES, max_syllables=MAX_SYLLABLES,
                 ceiling_syllables=CEILING_SYLLABLES, **kwargs):
        self.min_syllables = min_syllables
        self.max_syllables = max_syllables
        self.ceiling_syllables = ceiling_syllables
        kwargs.setdefault("max_chunk_words", min(max_syllables, ceiling_syllables))
        super().__init__(**kwargs)


class UnifiedBudgetOrchestrator(UnifiedDOCXChunker):
    pass


__all__ = ["UnifiedDOCXChunker", "UnifiedBudgetOrchestrator"]
