# universal/schema_assembler.py
# Universal Domain-Agnostic Schema Assembler
# Complies with GEMINI.md: 0% regex, no typing library, zero ternary, zero __init__.py

import os
import hashlib
from pathlib import Path

# Target Universal Schema 12 required fields (13 dictionary keys with bidirectional linking pair)
REQUIRED_UNIVERSAL_SCHEMA_KEYS = (
    "chunk_id",
    "parent_id",
    "source",
    "chunk_type",
    "breadcrumb",
    "section_title",
    "content",
    "parent_content",
    "business_metadata",
    "prev_chunk_id",
    "next_chunk_id",
    "table_data",
    "images",
)

# Standard allowed chunk types
VALID_CHUNK_TYPES = ("text", "table", "multimodal")

# Prohibited legacy hardcoded banking terms that must never appear in business_metadata keys or values
PROHIBITED_BANKING_TERMS = (
    "the_tin_dung",
    "tiet_kiem",
    "vay_von",
    "nghiep_vu_bank",
    "chuong_trinh_the",
    "van_ban_phap_ly",
    "huong_dan_app",
    "bao_mat_ekyc",
    "the_ghi_no",
    "phap_ly_tuan_thu",
)

# Directory names commonly used as generic container roots to be ignored in hierarchy extraction
GENERIC_ROOT_NAMES = (
    "input",
    "inputs",
    "output",
    "outputs",
    "data",
    "dataset",
    "datasets",
    "docs",
    "documents",
    "files",
    "source",
    "sources",
    "test_docs",
    "agent_banking",
    "chunking",
    "raw",
    "processed",
)


def normalize_chunk_type(raw_type):
    """
    Normalizes any chunk type string into one of the three standard types:
    'text', 'table', 'multimodal'.
    Strictly follows Zero Ternary Law and 0% Regex Law.
    """
    if raw_type is None:
        return "text"

    clean_type = str(raw_type).strip().lower()

    if clean_type in ("text", "docx_text", "paragraph", "heading", "sentence", "body"):
        return "text"
    elif clean_type in ("table", "data_table", "grid", "spreadsheet"):
        return "table"
    elif clean_type in ("multimodal", "image", "figure", "chart", "diagram", "picture", "caption"):
        return "multimodal"
    else:
        return "text"


def clean_path_segment(segment):
    """
    Normalizes a path segment string for metadata representation:
    replaces underscores and dashes with spaces, strips trailing and leading spaces.
    """
    if segment is None:
        return ""
    text = str(segment).strip()
    return text


def infer_business_metadata(file_path):
    """
    Dynamically infers domain-agnostic business metadata from file path, folder hierarchy,
    and document filename.
    Completely eliminates any hardcoded banking logic, banking product dictionaries,
    or fixed taxonomy tags.
    Works uniformly for Healthcare, Engineering, Legal, Education, Banking, or any other domain.

    Returns a dictionary:
    {
        "domain": str,
        "category": str,
        "folder": str,
        "document_name": str,
        "file_type": str,
        "path_hierarchy": list[str]
    }
    """
    if file_path is None:
        return {
            "domain": "general",
            "category": "general",
            "folder": "root",
            "document_name": "unknown",
            "file_type": "docx",
            "path_hierarchy": [],
        }

    raw_path_str = str(file_path).replace("\\", "/")
    segments = raw_path_str.split("/")

    # Filter out empty segments, drive letters like 'D:', and '.' / '..'
    clean_segments = []
    for seg in segments:
        s = seg.strip()
        if s:
            if s != ".":
                if s != "..":
                    if not s.endswith(":"):
                        clean_segments.append(s)

    if not clean_segments:
        return {
            "domain": "general",
            "category": "general",
            "folder": "root",
            "document_name": "unknown",
            "file_type": "docx",
            "path_hierarchy": [],
        }

    # Last segment is filename
    filename = clean_segments[-1]
    folder_segments = clean_segments[:-1]

    # Extract stem and extension
    if "." in filename:
        last_dot_idx = filename.rfind(".")
        doc_stem = filename[:last_dot_idx]
        file_ext = filename[last_dot_idx + 1:].lower()
    else:
        doc_stem = filename
        file_ext = "docx"

    # Check if any container anchor is in folder_segments
    last_anchor_idx = -1
    for idx, folder in enumerate(folder_segments):
        if folder.lower() in GENERIC_ROOT_NAMES:
            last_anchor_idx = idx

    meaningful_folders = []
    if last_anchor_idx != -1:
        # Take all folders after the last generic container root
        meaningful_folders = folder_segments[last_anchor_idx + 1:]
    else:
        # No anchor found, filter out any generic root names
        for folder in folder_segments:
            if folder.lower() not in GENERIC_ROOT_NAMES:
                meaningful_folders.append(folder)

    # Dynamic domain, category, and folder resolution (100% structural path hierarchy, zero keyword filtering)
    total_folders = len(meaningful_folders)
    if total_folders == 0:
        domain = "general"
        category = "general"
        folder = "root"
    elif total_folders == 1:
        domain = meaningful_folders[0]
        category = meaningful_folders[0]
        folder = meaningful_folders[0]
    elif total_folders == 2:
        domain = meaningful_folders[0]
        category = meaningful_folders[1]
        folder = meaningful_folders[1]
    else:
        domain = meaningful_folders[0]
        category = meaningful_folders[1]
        folder = meaningful_folders[-1]

    return {
        "domain": domain,
        "category": category,
        "folder": folder,
        "document_name": doc_stem,
        "file_type": file_ext,
        "path_hierarchy": meaningful_folders,
    }


def generate_chunk_id(source, index, content=None, prefix=None):
    """
    Generates a deterministic unique chunk ID based on source filename, chunk index,
    and optional content hash.
    Format:
    - Default: {stem}_c{index:04d}
    - With content hash: {stem}_c{index:04d}_{hash12}
    """
    clean_source = str(source).replace("\\", "/")
    filename = clean_source.split("/")[-1]
    if "." in filename:
        stem = filename.rsplit(".", 1)[0]
    else:
        stem = filename

    if not stem:
        stem = "doc"

    safe_stem = stem.replace(" ", "_").replace(".", "_")

    if prefix is not None:
        base_prefix = str(prefix).strip()
    else:
        source_hash = hashlib.sha256(clean_source.encode("utf-8")).hexdigest()[:12]
        base_prefix = safe_stem + "_" + source_hash

    try:
        idx_int = int(index)
    except Exception:
        idx_int = 0

    idx_formatted = f"{idx_int:04d}"

    if content is not None:
        content_str = str(content)
        seed = f"{base_prefix}::{idx_int}::{content_str}"
        hash_suffix = hashlib.sha256(seed.encode("utf-8")).hexdigest()[:8]
        return f"{base_prefix}_c{idx_formatted}_{hash_suffix}"
    else:
        return f"{base_prefix}_c{idx_formatted}"


def build_universal_chunk(
    chunk_id=None,
    parent_id=None,
    source="unknown.docx",
    chunk_type="text",
    breadcrumb="",
    section_title="",
    content="",
    parent_content=None,
    business_metadata=None,
    prev_chunk_id=None,
    next_chunk_id=None,
    table_data=None,
    images=None,
    **kwargs,
):
    """
    Assembles and returns a chunk dictionary strictly adhering to Target Universal Schema (R3).
    Guarantees that all 13 keys are present in every chunk dictionary:
    - chunk_id: str
    - parent_id: str ("" when root)
    - source: str
    - chunk_type: str ("text", "table", or "multimodal")
    - breadcrumb: str
    - section_title: str
    - content: str
    - parent_content: str ("" when root)
    - business_metadata: dict
    - prev_chunk_id: str ("" when first)
    - next_chunk_id: str ("" when last)
    - table_data: list, dict, or None
    - images: list

    Zero legacy fields (stats, metadata, element_ids, location, is_active, updated_time, etc.) are included.
    Follows GEMINI.md: 0% regex, no typing library, zero ternary.
    """
    # 1. Normalize source
    if source is not None:
        norm_source = str(source).strip()
    else:
        norm_source = "unknown.docx"

    # 2. Normalize chunk_id
    if chunk_id is not None:
        clean_chunk_id = str(chunk_id).strip()
    else:
        clean_chunk_id = generate_chunk_id(norm_source, 0, content=content)

    # 3. Normalize parent_id
    if parent_id is not None:
        clean_parent_id = str(parent_id).strip()
    else:
        clean_parent_id = ""

    # 4. Normalize chunk_type
    norm_chunk_type = normalize_chunk_type(chunk_type)

    # 5. Normalize breadcrumb
    if breadcrumb is not None:
        clean_breadcrumb = str(breadcrumb).strip()
    else:
        clean_breadcrumb = ""

    # 6. Normalize section_title
    if section_title is not None:
        clean_section_title = str(section_title).strip()
    else:
        clean_section_title = ""

    # 7. Normalize content
    if content is not None:
        clean_content = str(content)
    else:
        clean_content = ""

    # 8. Normalize parent_content
    if parent_content is not None:
        clean_parent_content = str(parent_content)
    else:
        clean_parent_content = ""

    # 9. Normalize business_metadata
    if business_metadata is not None:
        if isinstance(business_metadata, dict):
            final_metadata = dict(business_metadata)
        else:
            final_metadata = infer_business_metadata(norm_source)
    else:
        final_metadata = infer_business_metadata(norm_source)

    # 10. Normalize prev_chunk_id
    if prev_chunk_id is not None:
        clean_prev_chunk_id = str(prev_chunk_id).strip()
    else:
        clean_prev_chunk_id = ""

    # 11. Normalize next_chunk_id
    if next_chunk_id is not None:
        clean_next_chunk_id = str(next_chunk_id).strip()
    else:
        clean_next_chunk_id = ""

    # 12. Normalize table_data
    if table_data is not None:
        if isinstance(table_data, (list, dict)):
            final_table_data = table_data
        else:
            final_table_data = None
    else:
        final_table_data = None

    # 13. Normalize images
    if images is not None:
        if isinstance(images, list):
            final_images = list(images)
        else:
            final_images = []
    else:
        final_images = []

    # Assemble Target Universal Schema dictionary in standardized order
    chunk = {
        "chunk_id": clean_chunk_id,
        "parent_id": clean_parent_id,
        "source": norm_source,
        "chunk_type": norm_chunk_type,
        "breadcrumb": clean_breadcrumb,
        "section_title": clean_section_title,
        "content": clean_content,
        "parent_content": clean_parent_content,
        "business_metadata": final_metadata,
        "prev_chunk_id": clean_prev_chunk_id,
        "next_chunk_id": clean_next_chunk_id,
        "table_data": final_table_data,
        "images": final_images,
    }

    return chunk


def link_chunk_sequence(chunks):
    """
    Connects an ordered list of universal chunk dictionaries with bidirectional links:
    `prev_chunk_id` and `next_chunk_id`.
    Operates in-place and also returns the linked chunk list.

    Rules:
    - Empty list: returns []
    - Single chunk: prev_chunk_id = "", next_chunk_id = ""
    - Sequence [0 .. N-1]:
      chunk[0].prev_chunk_id = "", chunk[0].next_chunk_id = chunk[1].chunk_id
      chunk[i].prev_chunk_id = chunk[i-1].chunk_id, chunk[i].next_chunk_id = chunk[i+1].chunk_id
      chunk[N-1].prev_chunk_id = chunk[N-2].chunk_id, chunk[N-1].next_chunk_id = ""
    """
    if chunks is None:
        return []

    if not isinstance(chunks, list):
        return []

    total_chunks = len(chunks)
    if total_chunks == 0:
        return []

    if total_chunks == 1:
        chunks[0]["prev_chunk_id"] = ""
        chunks[0]["next_chunk_id"] = ""
        return chunks

    for i in range(total_chunks):
        curr_chunk = chunks[i]

        # Determine prev_chunk_id
        if i > 0:
            prev_chunk = chunks[i - 1]
            if isinstance(prev_chunk, dict):
                p_id = prev_chunk.get("chunk_id")
                if p_id is not None:
                    curr_chunk["prev_chunk_id"] = str(p_id)
                else:
                    curr_chunk["prev_chunk_id"] = ""
            else:
                curr_chunk["prev_chunk_id"] = ""
        else:
            curr_chunk["prev_chunk_id"] = ""

        # Determine next_chunk_id
        if i < total_chunks - 1:
            next_chunk = chunks[i + 1]
            if isinstance(next_chunk, dict):
                n_id = next_chunk.get("chunk_id")
                if n_id is not None:
                    curr_chunk["next_chunk_id"] = str(n_id)
                else:
                    curr_chunk["next_chunk_id"] = ""
            else:
                curr_chunk["next_chunk_id"] = ""
        else:
            curr_chunk["next_chunk_id"] = ""

    return chunks


def validate_universal_chunk(chunk, strict_domain_agnostic=True):
    """
    Validates that a dictionary strictly complies with the Target Universal Schema:
    1. Must be a dict.
    2. Must contain exactly all 13 required keys.
    3. Must not contain obsolete legacy keys.
    4. chunk_type must be one of ('text', 'table', 'multimodal').
    5. If strict_domain_agnostic is True, business_metadata must not contain prohibited banking labels.

    Returns:
    (is_valid: bool, issues: list[str])
    """
    issues = []

    if not isinstance(chunk, dict):
        return False, ["Chunk must be a dictionary"]

    # 1. Check all required keys exist
    for key in REQUIRED_UNIVERSAL_SCHEMA_KEYS:
        if key not in chunk:
            issues.append(f"Missing required field: '{key}'")

    # 2. Check no extraneous legacy keys exist
    chunk_keys = set(chunk.keys())
    required_keys_set = set(REQUIRED_UNIVERSAL_SCHEMA_KEYS)
    extra_keys = chunk_keys - required_keys_set
    if extra_keys:
        for ek in sorted(list(extra_keys)):
            issues.append(f"Disallowed extra legacy field present: '{ek}'")

    # 3. Validate chunk_type
    chunk_type = chunk.get("chunk_type")
    if chunk_type not in VALID_CHUNK_TYPES:
        issues.append(f"Invalid chunk_type '{chunk_type}', must be one of {VALID_CHUNK_TYPES}")

    # 4. Validate chunk_id
    chunk_id = chunk.get("chunk_id")
    if not chunk_id or not isinstance(chunk_id, str):
        issues.append("chunk_id must be a non-empty string")

    # 5. Validate business_metadata
    b_meta = chunk.get("business_metadata")
    if not isinstance(b_meta, dict):
        issues.append("business_metadata must be a dictionary")
    else:
        if strict_domain_agnostic:
            for b_key, b_val in b_meta.items():
                str_key = str(b_key).lower()
                str_val = str(b_val).lower()
                for term in PROHIBITED_BANKING_TERMS:
                    if term in str_key or term in str_val:
                        issues.append(f"Prohibited hardcoded banking term '{term}' detected in business_metadata ({b_key}={b_val})")

    if issues:
        return False, issues
    else:
        return True, []


def validate_chunk_sequence(chunks):
    """
    Validates a sequence of chunks for both individual schema compliance
    and bidirectional link integrity.

    Returns:
    (is_valid: bool, issues: list[str])
    """
    issues = []
    if not isinstance(chunks, list):
        return False, ["Chunks must be a list"]

    total = len(chunks)
    if total == 0:
        return True, []

    if any(not isinstance(chunk, dict) for chunk in chunks):
        return False, ["Every chunk must be a dictionary"]
    identifiers = [chunk.get("chunk_id") for chunk in chunks]
    string_ids = [identifier for identifier in identifiers if isinstance(identifier, str)]
    if len(string_ids) != len(set(string_ids)):
        issues.append("Duplicate chunk IDs in sequence")

    for i in range(total):
        c = chunks[i]
        c_valid, c_issues = validate_universal_chunk(c)
        if not c_valid:
            for ci in c_issues:
                issues.append(f"Chunk[{i}] ({c.get('chunk_id')}): {ci}")

        # Check bidirectional linking
        prev_id = c.get("prev_chunk_id")
        next_id = c.get("next_chunk_id")

        if i == 0:
            if prev_id not in (None, ""):
                issues.append(f"Chunk[0] has non-empty prev_chunk_id: {prev_id}")
        else:
            expected_prev = chunks[i - 1].get("chunk_id")
            if prev_id != expected_prev:
                issues.append(f"Chunk[{i}] prev_chunk_id '{prev_id}' does not match Chunk[{i-1}] chunk_id '{expected_prev}'")

        if i == total - 1:
            if next_id not in (None, ""):
                issues.append(f"Chunk[{total - 1}] has non-empty next_chunk_id: {next_id}")
        else:
            expected_next = chunks[i + 1].get("chunk_id")
            if next_id != expected_next:
                issues.append(f"Chunk[{i}] next_chunk_id '{next_id}' does not match Chunk[{i+1}] chunk_id '{expected_next}'")

    if issues:
        return False, issues
    else:
        return True, []
