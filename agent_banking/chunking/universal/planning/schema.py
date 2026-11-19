"""Construct and validate the 13-field chunk schema used downstream."""

import hashlib

from agent_banking.chunking.config.settings import get_rules

_SCHEMA_RULES = get_rules("schema")

REQUIRED_UNIVERSAL_SCHEMA_KEYS = tuple(_SCHEMA_RULES["required_universal_schema_keys"])

# Standard allowed chunk types
VALID_CHUNK_TYPES = tuple(_SCHEMA_RULES["valid_chunk_types"])

# Retired metadata field names. Source-derived values may contain these terms.
PROHIBITED_BANKING_TERMS = tuple(_SCHEMA_RULES["prohibited_banking_terms"])

# Directory names commonly used as generic container roots to be ignored in hierarchy extraction
GENERIC_ROOT_NAMES = tuple(_SCHEMA_RULES["generic_root_names"])

_TYPE_ALIASES = _SCHEMA_RULES["chunk_type_aliases"]
_TABLE_ALIASES = set(_TYPE_ALIASES["table"])
_MULTIMODAL_ALIASES = set(_TYPE_ALIASES["multimodal"])
_A_BANK_DOMAIN_RULES = _SCHEMA_RULES["a_bank_domain_rules"]
_DOC_TYPE_PREFIXES = tuple(tuple(pair) for pair in _SCHEMA_RULES["doc_type_prefixes"])
_DEFAULT_DOMAIN = _SCHEMA_RULES["default_domain"]
_DEFAULT_FOLDER = _SCHEMA_RULES["default_folder"]
_DEFAULT_FILE_TYPE = _SCHEMA_RULES["default_file_type"]
_DEFAULT_UNKNOWN_NAME = _SCHEMA_RULES["default_unknown_name"]
_DEFAULT_SOURCE_FILENAME = _SCHEMA_RULES["default_source_filename"]


def normalize_chunk_type(raw_type):
    """Map older aliases to their canonical chunk type."""
    clean_type = str(raw_type).strip().lower()
    if clean_type in _TABLE_ALIASES:
        return "table"
    if clean_type in _MULTIMODAL_ALIASES:
        return "multimodal"
    return "text"


def clean_path_segment(segment):
    """Return a display form of a path segment."""
    if segment is None:
        return ""
    return str(segment).strip()


def _infer_semantic_domain(folder_segments, default_domain):
    if len(folder_segments) >= 2 and folder_segments[0].casefold() == "a_bank":
        cat = folder_segments[1].casefold()
        for rule in _A_BANK_DOMAIN_RULES:
            if any(keyword in cat for keyword in rule["keywords"]):
                return rule["domain"]
        return folder_segments[1]
    return default_domain


def _infer_doc_type(stem, domain, extension):
    upper = stem.strip().upper()
    for prefix, label in _DOC_TYPE_PREFIXES:
        if upper.startswith(prefix):
            return label
    if domain and domain != _DEFAULT_DOMAIN:
        return domain
    return extension.lower()


def infer_business_metadata(file_path):
    """Derive category, semantic domain, doc_type and document title from meaningful path segments."""
    segments = []
    if file_path is not None:
        for segment in str(file_path).replace("\\", "/").split("/"):
            segment = clean_path_segment(segment)
            if segment and segment not in (".", "..") and not segment.endswith(":"):
                segments.append(segment)
    if segments:
        filename = segments[-1]
    else:
        filename = _DEFAULT_UNKNOWN_NAME
    folder_segments = segments[:-1]
    for index in range(len(folder_segments) - 1, -1, -1):
        if folder_segments[index].lower() in GENERIC_ROOT_NAMES:
            folder_segments = folder_segments[index + 1:]
            break
    stem, separator, extension = filename.rpartition(".")
    if not separator:
        stem, extension = filename, _DEFAULT_FILE_TYPE

    if folder_segments:
        raw_domain = folder_segments[0]
    else:
        raw_domain = _DEFAULT_DOMAIN
    domain = _infer_semantic_domain(folder_segments, raw_domain)
    if len(folder_segments) > 1:
        category = folder_segments[1]
    else:
        category = raw_domain
    if folder_segments:
        folder_name = folder_segments[-1]
    else:
        folder_name = _DEFAULT_FOLDER
    return {
        "domain": domain,
        "category": category,
        "doc_type": _infer_doc_type(stem, domain, extension),
        "folder": folder_name,
        "document_name": stem,
        "file_type": extension.lower(),
        "path_hierarchy": folder_segments,
    }


def generate_chunk_id(source, index, content=None, prefix=None):
    """Generate an ID from the source path, index and optional content."""
    clean_source = str(source).replace("\\", "/")
    filename = clean_source.rsplit("/", 1)[-1]
    stem = filename.rsplit(".", 1)[0] or "doc"
    if prefix is None:
        source_hash = hashlib.sha256(clean_source.encode("utf-8")).hexdigest()[:12]
        prefix = stem.replace(" ", "_").replace(".", "_") + "_" + source_hash
    else:
        prefix = str(prefix).strip()
    try:
        index = int(index)
    except (ValueError, TypeError, OverflowError):
        index = 0

    identifier = f"{prefix}_c{index:04d}"
    if content is not None:
        seed = f"{prefix}::{index}::{content}"
        identifier += "_" + hashlib.sha256(seed.encode("utf-8")).hexdigest()[:8]
    return identifier


def _as_text(value):
    if value is None:
        return ""
    return str(value)


def build_universal_chunk(
    chunk_id=None,
    parent_id=None,
    source=_DEFAULT_SOURCE_FILENAME,
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
    """Assemble the canonical chunk, preserving source names and content exactly."""
    source = _as_text(source)
    if not source.strip():
        source = _DEFAULT_SOURCE_FILENAME
    if chunk_id is None:
        chunk_id = generate_chunk_id(source, 0, content=content)
    if not isinstance(business_metadata, dict):
        business_metadata = infer_business_metadata(source)

    if isinstance(table_data, (list, dict)):
        clean_table_data = table_data
    else:
        clean_table_data = None
    if isinstance(images, list):
        clean_images = list(images)
    else:
        clean_images = []
    return {
        "chunk_id": str(chunk_id).strip(),
        "parent_id": _as_text(parent_id).strip(),
        "source": source,
        "chunk_type": normalize_chunk_type(chunk_type),
        "breadcrumb": _as_text(breadcrumb).strip(),
        "section_title": _as_text(section_title).strip(),
        "content": _as_text(content),
        "parent_content": _as_text(parent_content),
        "business_metadata": dict(business_metadata),
        "prev_chunk_id": _as_text(prev_chunk_id).strip(),
        "next_chunk_id": _as_text(next_chunk_id).strip(),
        "table_data": clean_table_data,
        "images": clean_images,
    }


def _document_key(chunk):
    metadata = chunk.get("business_metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    source = str(chunk.get("source") or "").replace("\\", "/")
    return source, metadata.get("document_id"), metadata.get("document_version")


def _neighbor_id(chunks, current, neighbor):
    if 0 <= neighbor < len(chunks):
        adjacent = chunks[neighbor]
        if isinstance(adjacent, dict) and _document_key(chunks[current]) == _document_key(adjacent):
            identifier = adjacent.get("chunk_id")
            if identifier is not None:
                return str(identifier)
    return ""


def link_chunk_sequence(chunks):
    """Link adjacent chunks in place, stopping at each document boundary."""
    if not isinstance(chunks, list):
        return []
    for index, chunk in enumerate(chunks):
        if isinstance(chunk, dict):
            chunk["prev_chunk_id"] = _neighbor_id(chunks, index, index - 1)
            chunk["next_chunk_id"] = _neighbor_id(chunks, index, index + 1)
    return chunks


def require_index_ready(chunks):
    """Reject explicitly unresolved canonical chunks at ingestion boundaries."""
    for chunk in chunks:
        metadata = chunk.get("business_metadata")
        if isinstance(metadata, dict):
            quality = metadata.get("quality")
        else:
            quality = None
        if isinstance(quality, dict) and quality.get("index_ready") is False:
            raise ValueError("Chunk is not ready for indexing: " + str(chunk.get("chunk_id") or "unknown"))


def payload_index_key(chunk):
    """Distinguish indexed payload revisions while preserving the source chunk ID."""
    metadata = chunk.get("business_metadata")
    if isinstance(metadata, dict) and metadata.get("publication_revision"):
        return metadata["publication_revision"]
    return chunk.get("chunk_id")


def validate_universal_chunk(chunk, strict_domain_agnostic=True):
    """Check the required fields, field types and retired metadata keys."""
    if not isinstance(chunk, dict):
        return False, ["Chunk must be a dictionary"]

    issues = [
        f"Missing required field: '{key}'"
        for key in REQUIRED_UNIVERSAL_SCHEMA_KEYS if key not in chunk
    ]
    extra_keys = set(chunk) - set(REQUIRED_UNIVERSAL_SCHEMA_KEYS)
    issues.extend(f"Disallowed extra legacy field present: '{key}'" for key in sorted(extra_keys, key=str))

    for field in ("chunk_id", "source"):
        value = chunk.get(field)
        if not isinstance(value, str) or not value.strip():
            issues.append(f"{field} must be a non-empty string")
    for field in ("parent_id", "breadcrumb", "section_title", "content", "parent_content",
                  "prev_chunk_id", "next_chunk_id"):
        if field in chunk and not isinstance(chunk[field], str):
            issues.append(f"{field} must be a string")
    chunk_type = chunk.get("chunk_type")
    if chunk_type not in VALID_CHUNK_TYPES:
        issues.append(f"Invalid chunk_type '{chunk_type}', must be one of {VALID_CHUNK_TYPES}")
    if chunk.get("table_data") is not None and not isinstance(chunk["table_data"], (list, dict)):
        issues.append("table_data must be a list, dictionary or null")
    if not isinstance(chunk.get("images"), list):
        issues.append("images must be a list")

    metadata = chunk.get("business_metadata")
    if not isinstance(metadata, dict):
        issues.append("business_metadata must be a dictionary")
    elif strict_domain_agnostic:
        for key in metadata:
            if isinstance(key, str) and key.casefold() in PROHIBITED_BANKING_TERMS:
                issues.append(f"Retired business_metadata field: '{key}'")
    return not issues, issues


def validate_chunk_sequence(chunks):
    """Check unique chunk IDs, the chunk schema and links within each document."""
    if not isinstance(chunks, list):
        return False, ["Chunks must be a list"]
    if any(not isinstance(chunk, dict) for chunk in chunks):
        return False, ["Every chunk must be a dictionary"]

    issues = []
    string_ids = [chunk["chunk_id"] for chunk in chunks if isinstance(chunk.get("chunk_id"), str)]
    if len(string_ids) != len(set(string_ids)):
        issues.append("Duplicate chunk IDs in sequence")
    for index, chunk in enumerate(chunks):
        _, chunk_issues = validate_universal_chunk(chunk)
        issues.extend(f"Chunk[{index}] ({chunk.get('chunk_id')}): {issue}" for issue in chunk_issues)
        for field, neighbor in (("prev_chunk_id", index - 1), ("next_chunk_id", index + 1)):
            expected = _neighbor_id(chunks, index, neighbor)
            if chunk.get(field) != expected:
                issues.append(f"Chunk[{index}] {field} must be '{expected}'")
    return not issues, issues
