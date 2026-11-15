"""Resolve PDF/DOCX inputs and directory-relative document identities."""

import hashlib
from pathlib import Path, PurePosixPath, PureWindowsPath
import unicodedata


SUPPORTED_EXTENSIONS = {".docx", ".pdf"}
UNSUPPORTED_DOCUMENT_EXTENSIONS = {
    ".azw", ".csv", ".djvu", ".doc", ".docm", ".epub", ".htm", ".html", ".json",
    ".jsonl", ".md", ".mobi", ".odf", ".odp", ".ods", ".odt", ".pages", ".ppt",
    ".pptm", ".pptx", ".rst", ".rtf", ".tex", ".tsv", ".txt", ".wps", ".xls",
    ".xlsb", ".xlsm", ".xlsx", ".xml", ".xps", ".bmp", ".gif", ".jpeg", ".jpg",
    ".png", ".tif", ".tiff", ".webp",
}


def document_source_name(value):
    return str(value).replace("\\", "/").rsplit("/", 1)[-1]


def source_key(value):
    """Normalize separators and case for comparing sources without discarding folders."""
    return unicodedata.normalize("NFC", str(value).replace("\\", "/")).casefold()


def preserved_source_keys(preserve_sources, current_sources):
    """Include a legacy basename for a failed nested source if it has no current owner."""
    keys = {source_key(source) for source in preserve_sources}
    current = {source_key(source) for source in current_sources}
    return keys | {document_source_name(key) for key in keys if document_source_name(key) not in current}


def discover_documents(directory, extensions=None, *, reject_unsupported=False):
    directory = Path(directory)
    if extensions is None:
        extensions = SUPPORTED_EXTENSIONS
    elif isinstance(extensions, str):
        extensions = (extensions,)
    allowed = {extension.lower() for extension in extensions}
    files = []
    unsupported = []
    for path in sorted(directory.rglob("*")):
        if path.name.startswith("~$") or not path.is_file():
            continue
        extension = path.suffix.lower()
        if extension in allowed:
            files.append(path)
        elif reject_unsupported and extension in UNSUPPORTED_DOCUMENT_EXTENSIONS:
            unsupported.append(path.relative_to(directory).as_posix())
    if unsupported:
        listing = "\n".join(unsupported[:20])
        if len(unsupported) > 20:
            listing += f"\n... and {len(unsupported) - 20} more"
        raise ValueError("Unsupported documents found. This input accepts PDF and DOCX:\n" + listing)
    groups = {}
    for path in files:
        stem = unicodedata.normalize("NFC", path.stem).casefold()
        while stem.endswith(")"):
            preceding, marker, suffix = stem[:-1].rpartition(" (")
            if not marker or not suffix.isascii() or not suffix.isdecimal() or suffix.startswith("0"):
                break
            stem = preceding
        parent = source_key(path.relative_to(directory).parent.as_posix())
        groups.setdefault((parent, stem, path.suffix.lower()), []).append(path)
    conflicts = [group for group in groups.values() if len(group) > 1]
    if conflicts:
        listing = "\n".join(
            " / ".join(f"({i}) {path.relative_to(directory).as_posix()}" for i, path in enumerate(group, 1))
            for group in conflicts
        )
        raise ValueError("Duplicate document names. Rename the source files before chunking:\n" + listing)
    return files


def _relative_document_name(value):
    raw = str(value).replace("\\", "/")
    parts = raw.split("/")
    if ("\0" in raw or PureWindowsPath(raw).drive or PurePosixPath(raw).is_absolute()
            or any(not part.strip() or part in (".", "..") for part in parts)):
        raise ValueError("source_name must be a relative document path")
    return raw


def resolve_document_source(source=None, source_bytes=None, source_name=None, expected_format=None):
    if expected_format is not None and expected_format not in ("docx", "pdf"):
        raise ValueError("expected_format must be docx or pdf")
    if isinstance(source, (bytes, bytearray)):
        if source_bytes is not None:
            raise ValueError("Provide a single document byte stream")
        source_bytes = source
    if source_bytes is not None:
        if not isinstance(source_bytes, (bytes, bytearray)):
            raise TypeError("source_bytes must be bytes or bytearray")
        if not source_bytes:
            raise ValueError("source_bytes is empty")
        name = None
        named_format = None
        if source_name is not None:
            name = document_source_name(source_name)
            if not name.strip() or name in (".", "..") or "\0" in name:
                raise ValueError("source_name must contain a document filename")
            extension = Path(name).suffix.lower()
            if extension in SUPPORTED_EXTENSIONS:
                named_format = extension[1:]
        if expected_format and named_format and expected_format != named_format:
            raise ValueError(expected_format.upper() + " adapter requires ." + expected_format + " input")
        fmt = expected_format or named_format
        if fmt is None:
            if source_bytes.startswith(b"%PDF-"):
                fmt = "pdf"
            else:
                fmt = "docx"
        if name is None:
            name = "in_memory_" + hashlib.sha256(source_bytes).hexdigest()[:20] + "." + fmt
        return bytes(source_bytes), name, fmt
    if not isinstance(source, (str, Path)):
        if source is None:
            raise ValueError("A document path or byte stream is required")
        raise TypeError("source must be a string or Path")
    if not str(source).strip():
        raise ValueError("source path is empty")
    path = Path(source)
    if not path.is_file():
        raise FileNotFoundError(str(path))
    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError("Unsupported document format: " + ext)
    fmt = ext[1:]
    if expected_format is not None and fmt != expected_format:
        raise ValueError(expected_format.upper() + " adapter requires ." + expected_format + " input")
    name = path.name if source_name is None else _relative_document_name(source_name)
    return str(path), name, fmt
