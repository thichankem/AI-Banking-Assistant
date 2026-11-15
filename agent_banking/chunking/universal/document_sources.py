"""Resolve PDF and DOCX inputs and their filename-based identities."""

import hashlib
from pathlib import Path
import unicodedata


def document_source_name(value):
    return str(value).replace("\\", "/").rsplit("/", 1)[-1]


def discover_documents(directory, extensions=None):
    if extensions is None:
        extensions = (".docx", ".pdf")
    elif isinstance(extensions, str):
        extensions = (extensions,)
    allowed = {extension.lower() for extension in extensions}
    files = sorted(path for path in Path(directory).rglob("*")
                   if path.is_file() and path.suffix.lower() in allowed and not path.name.startswith("~$"))
    groups = {}
    for path in files:
        stem = unicodedata.normalize("NFC", path.stem).casefold()
        while stem.endswith(")"):
            preceding, marker, suffix = stem[:-1].rpartition(" (")
            if not marker or not suffix.isascii() or not suffix.isdecimal() or suffix.startswith("0"):
                break
            stem = preceding
        groups.setdefault((stem, path.suffix.lower()), []).append(path)
    conflicts = [group for group in groups.values() if len(group) > 1]
    if conflicts:
        directory = Path(directory)
        listing = "\n".join(
            " / ".join(f"({i}) {path.relative_to(directory).as_posix()}" for i, path in enumerate(group, 1))
            for group in conflicts
        )
        raise ValueError("Duplicate document names. Rename the source files before chunking:\n" + listing)
    return files


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
            if extension in (".docx", ".pdf"):
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
    if ext not in (".docx", ".pdf"):
        raise ValueError("Unsupported document format: " + ext)
    fmt = ext[1:]
    if expected_format is not None and fmt != expected_format:
        raise ValueError(expected_format.upper() + " adapter requires ." + expected_format + " input")
    return str(path), path.name, fmt
