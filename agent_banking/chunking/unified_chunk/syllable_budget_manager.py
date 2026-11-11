# unified_chunk/syllable_budget_manager.py
# Universal Domain-Agnostic Syllable Budget & Text Segmentation Engine
# Complies with GEMINI.md: 0% regex, no typing library, zero ternary, zero __init__.py

import time
import unicodedata
from pathlib import Path

try:
    from agent_banking.chunking.unified_chunk.canonical_resolver import (
        split_sentences,
        strip_vietnamese_accents
    )
except (ImportError, ValueError):
    try:
        from unified_chunk.canonical_resolver import (
            split_sentences,
            strip_vietnamese_accents
        )
    except (ImportError, ValueError):
        from .canonical_resolver import (
            split_sentences,
            strip_vietnamese_accents
        )

# Target Universal Schema Assembler
try:
    from agent_banking.chunking.universal.schema_assembler import (
        build_universal_chunk,
        validate_universal_chunk
    )
except (ImportError, ValueError):
    try:
        from universal.schema_assembler import (
            build_universal_chunk,
            validate_universal_chunk
        )
    except (ImportError, ValueError):
        build_universal_chunk = None
        validate_universal_chunk = None


# =============================================================================
# CONSTANTS: DEFAULT SYLLABLE BUDGETS
# =============================================================================
MIN_SYLLABLES = 100
MAX_SYLLABLES = 400
CEILING_SYLLABLES = 500


def prepend_context_header(content, breadcrumb, section_title):
    """
    Prepends contextual header: [Context: {breadcrumb} > {section_title}]
    to the first line of chunk content (0% Regex Law, Zero Ternary Law).
    """
    clean_bc = breadcrumb.strip()
    clean_st = section_title.strip()
    clean_content = content.strip()

    prefix = ""
    if clean_bc and clean_st and clean_st != clean_bc and not clean_bc.endswith(clean_st):
        prefix = "[Ngữ cảnh: " + clean_bc + " > " + clean_st + "]\n"
    else:
        if clean_bc:
            prefix = "[Ngữ cảnh: " + clean_bc + "]\n"
        else:
            if clean_st:
                prefix = "[Ngữ cảnh: " + clean_st + "]\n"

    if prefix and clean_content.startswith("[Ngữ cảnh:"):
        return clean_content

    return prefix + clean_content


def _clean_slug(segment):
    """
    Converts path segments into clean, lowercase snake_case slugs.
    Strips Vietnamese accents and leading numbers/indices (e.g. '1. ', '02 - ').
    Strictly 0% Regex Law and Zero Ternary Law.
    """
    if not segment:
        return ""

    s = strip_vietnamese_accents(segment.strip().lower())
    s = s.replace("\\", "/").strip("/")

    parts = s.split(" ", 1)
    if len(parts) == 2:
        prefix = parts[0].rstrip(".-_")
        if prefix.isdigit() or (prefix.startswith("c") and prefix[1:].isdigit()):
            s = parts[1]

    cleaned_chars = []
    for ch in s:
        if ch.isalnum():
            cleaned_chars.append(ch)
        else:
            if ch in (" ", "-", "_", "."):
                cleaned_chars.append("_")

    res = "".join(cleaned_chars).strip("_")
    while "__" in res:
        res = res.replace("__", "_")
    return res


def extract_universal_metadata_from_path(file_path):
    """
    100% Domain-Agnostic metadata extractor based on directory path and filename context.
    Eliminates all hardcoded banking categories, customer segments, and product keywords.
    Applicable universally to Healthcare, Legal, Engineering, Finance, Education, etc.

    Returns:
        dict: {
            "domain": str,
            "category": str,
            "sub_category": str,
            "doc_category": str,
            "document_name": str,
            "file_extension": str,
            "is_active": True
        }
    """
    if file_path is None:
        p_norm = ""
    else:
        p_norm = str(file_path).replace("\\\\", "/").replace("\\", "/")

    parts = []
    for seg in p_norm.split("/"):
        if seg and seg not in (".", "..", "input", "output"):
            parts.append(seg)

    if not parts:
        return {
            "domain": "general",
            "category": "general",
            "sub_category": "",
            "doc_category": "general",
            "document_name": "document",
            "file_extension": "",
            "is_active": True
        }

    filename = parts[-1]
    if "." in filename:
        doc_stem, ext = filename.rsplit(".", 1)
    else:
        doc_stem, ext = filename, ""

    clean_doc_name = _clean_slug(doc_stem)
    dir_segments = parts[:-1]

    if len(dir_segments) == 0:
        domain = "general"
        category = "general"
        sub_category = ""
    else:
        if len(dir_segments) == 1:
            domain = _clean_slug(dir_segments[0])
            category = domain
            sub_category = ""
        else:
            if len(dir_segments) == 2:
                domain = _clean_slug(dir_segments[0])
                category = _clean_slug(dir_segments[1])
                sub_category = ""
            else:
                domain = _clean_slug(dir_segments[0])
                category = _clean_slug(dir_segments[1])
                sub_category = _clean_slug("/".join(dir_segments[2:]))

    return {
        "domain": domain,
        "category": category,
        "sub_category": sub_category,
        "doc_category": category,
        "document_name": clean_doc_name,
        "file_extension": ext.lower(),
        "is_active": True
    }


# Backward-compatible alias for existing callers
extract_business_metadata_from_path = extract_universal_metadata_from_path


def build_7_field_chunk(
    chunk_id,
    source,
    chunk_type,
    breadcrumb,
    section_title,
    content,
    parent_id=None,
    parent_content=None,
    business_metadata=None,
    prev_chunk_id=None,
    next_chunk_id=None,
    table_data=None,
    images=None,
    **kwargs
):
    """
    Target Universal Schema Chunk Builder.
    Conforms strictly to 12 fields (13 dictionary keys).
    Eliminates all 13 legacy garbage fields (id, doc_name, stats, metadata, etc.).
    """
    final_content = prepend_context_header(content, breadcrumb, section_title)

    if business_metadata is not None:
        b_meta = business_metadata
    else:
        b_meta = extract_universal_metadata_from_path(source)

    if parent_content is not None:
        p_content = parent_content
    else:
        p_content = final_content

    p_chunk_id = prev_chunk_id
    if p_chunk_id is None and "prev_chunk_id" in kwargs:
        p_chunk_id = kwargs["prev_chunk_id"]

    n_chunk_id = next_chunk_id
    if n_chunk_id is None and "next_chunk_id" in kwargs:
        n_chunk_id = kwargs["next_chunk_id"]

    final_images = []
    if images is not None and len(images) > 0:
        final_images = images
    else:
        if "images" in kwargs and kwargs["images"]:
            final_images = kwargs["images"]

    clean_type = chunk_type.lower()
    if clean_type == "docx_text":
        clean_type = "text"

    final_table_data = table_data
    if final_table_data is None and "table_data" in kwargs:
        final_table_data = kwargs["table_data"]

    chunk = {
        "chunk_id": chunk_id,
        "parent_id": parent_id,
        "source": source,
        "chunk_type": clean_type,
        "breadcrumb": breadcrumb,
        "section_title": section_title,
        "content": final_content,
        "parent_content": p_content,
        "business_metadata": b_meta,
        "prev_chunk_id": p_chunk_id,
        "next_chunk_id": n_chunk_id,
        "table_data": final_table_data,
        "images": final_images
    }
    return chunk


build_banking_chunk = build_7_field_chunk


class SyllableBudgetManager:
    """
    Universal Domain-Agnostic Syllable Budget Manager (0% Regex, Zero Ternary).
    Supports dynamic budgeting and elastic sentence boundary preservation.
    """

    def __init__(
        self,
        min_syllables=MIN_SYLLABLES,
        max_syllables=MAX_SYLLABLES,
        ceiling_syllables=CEILING_SYLLABLES,
        overlap_syllables=50,
        overlap_sentences=1,
        **kwargs
    ):
        self.min_syllables = min_syllables
        self.max_syllables = max_syllables
        self.ceiling_syllables = ceiling_syllables
        self.overlap_syllables = overlap_syllables
        self.overlap_sentences = overlap_sentences

    def count_syllables(self, text):
        """Pure word / syllable count via standard string whitespace split."""
        if not text:
            return 0
        return len(text.split())

    def evaluate_budget_status(self, text):
        """Evaluates whether text is within standard budget golden zone."""
        syllables = self.count_syllables(text)
        if syllables < self.min_syllables:
            status = "under_budget"
        else:
            if syllables <= self.max_syllables:
                status = "golden_zone"
            else:
                status = "over_budget"

        return {
            "syllables": syllables,
            "status": status,
            "min_syllables": self.min_syllables,
            "max_syllables": self.max_syllables,
            "ceiling_syllables": self.ceiling_syllables
        }

    def split_text_elastic(
        self,
        text,
        window_start=250,
        window_end=420,
        ceiling=450,
        **kwargs
    ):
        """
        Elastic text splitting preserving sentence boundaries without regex.
        Uses universal sentence segmentation.
        """
        words = text.split()
        if len(words) <= ceiling:
            stripped_text = text.strip()
            if stripped_text:
                return [stripped_text]
            return []

        sentences = split_sentences(text)
        target = window_end
        if ceiling < target:
            target = ceiling

        if len(sentences) <= 1:
            chunks = []
            step = target
            if self.overlap_syllables > 0:
                step = target - self.overlap_syllables
            if step < 20:
                step = target
            for i in range(0, len(words), step):
                chunk = " ".join(words[i : i + target]).strip()
                if chunk:
                    chunks.append(chunk)
                if i + target >= len(words):
                    break
            return chunks

        return self.split_sentences_by_syllable_budget(
            sentences=sentences,
            max_syllables=target,
            **kwargs
        )

    def split_sentences_by_syllable_budget(
        self,
        sentences,
        max_syllables=None,
        **kwargs
    ):
        """
        Groups sentences into coherent chunks satisfying max syllable budget.
        Maintains controlled semantic overlap at chunk boundaries.
        """
        if not sentences:
            return []

        target_max = self.max_syllables
        if max_syllables is not None:
            target_max = max_syllables

        target_overlap = self.overlap_syllables
        if "overlap_syllables" in kwargs and kwargs["overlap_syllables"] is not None:
            target_overlap = kwargs["overlap_syllables"]

        target_overlap_sent = self.overlap_sentences
        if "overlap_sentences" in kwargs and kwargs["overlap_sentences"] is not None:
            target_overlap_sent = kwargs["overlap_sentences"]

        chunks = []
        current_chunk = []
        current_count = 0

        for s in sentences:
            s_clean = s.strip()
            if not s_clean:
                continue
            s_words = len(s_clean.split())

            if s_words > self.ceiling_syllables:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = []
                    current_count = 0
                sub_parts = self.split_text_elastic(
                    text=s_clean,
                    window_start=250,
                    window_end=target_max,
                    ceiling=self.ceiling_syllables
                )
                chunks.extend(sub_parts)
                continue

            if current_count + s_words > target_max and current_chunk:
                chunks.append(" ".join(current_chunk))

                overlap_items = []
                overlap_count = 0
                if target_overlap_sent > 0 and len(current_chunk) > 1:
                    max_take = len(current_chunk) - 1
                    if max_take > target_overlap_sent:
                        max_take = target_overlap_sent

                    for prev_s in reversed(current_chunk):
                        if len(overlap_items) >= max_take:
                            break
                        pw = len(prev_s.split())
                        if overlap_count + pw > target_overlap and overlap_items:
                            break
                        overlap_items.insert(0, prev_s)
                        overlap_count += pw

                current_chunk = list(overlap_items)
                current_count = overlap_count

            current_chunk.append(s_clean)
            current_count += s_words

        if current_chunk:
            chunk_str = " ".join(current_chunk)
            if not chunks:
                chunks.append(chunk_str)
            else:
                if chunk_str != chunks[-1]:
                    chunks.append(chunk_str)

        return chunks


__all__ = [
    "MIN_SYLLABLES",
    "MAX_SYLLABLES",
    "CEILING_SYLLABLES",
    "SyllableBudgetManager",
    "prepend_context_header",
    "extract_universal_metadata_from_path",
    "extract_business_metadata_from_path",
    "build_7_field_chunk",
    "build_banking_chunk"
]
