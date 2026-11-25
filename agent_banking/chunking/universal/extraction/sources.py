"""Resolve supported inputs and directory-relative document identities."""

import hashlib
from io import BytesIO
from pathlib import Path, PurePosixPath, PureWindowsPath
import unicodedata
from zipfile import BadZipFile, ZipFile

from agent_banking.chunking.config.settings import get_rules

_SRC_RULES = get_rules("sources")
SUPPORTED_EXTENSIONS = set(_SRC_RULES["supported_extensions"]) | {".xlsx"}
UNSUPPORTED_DOCUMENT_EXTENSIONS = set(_SRC_RULES["unsupported_document_extensions"]) - SUPPORTED_EXTENSIONS
_MAX_LISTED_UNSUPPORTED = _SRC_RULES["max_listed_unsupported"]
_IN_MEMORY_HASH_LEN = _SRC_RULES["in_memory_hash_len"]


def document_source_name(value):
    return str(value).replace("\\", "/").rsplit("/", 1)[-1]


def source_key(value):
    """Normalize separators and case for comparing sources without discarding folders."""
    return unicodedata.normalize("NFC", str(value).replace("\\", "/")).casefold()


def document_source_root(directory, canonical_root):
    directory = Path(directory).resolve()
    if canonical_root is not None:
        root = Path(canonical_root).resolve()
        if directory.is_relative_to(root):
            return root
    return directory


def legacy_source_resolver(current_sources):
    """Resolve exact paths or unique directory-relative aliases from older indexes."""
    current = {source_key(source) for source in current_sources}
    suffix_owners = {}
    for key in current:
        parts = key.split("/")
        for index in range(1, len(parts)):
            suffix_owners.setdefault("/".join(parts[index:]), set()).add(key)

    def resolve(source):
        if not isinstance(source, str) or not source.strip():
            return None
        key = source_key(source)
        if key in current:
            return key
        candidates = set(suffix_owners.get(key, ()))
        parts = key.split("/")
        for index in range(1, len(parts)):
            suffix = "/".join(parts[index:])
            if suffix in current:
                candidates.add(suffix)
        if len(candidates) == 1:
            return candidates.pop()
        return None

    return resolve


def discover_documents(directory, extensions=None, *, reject_unsupported=False, excluded_paths=()):
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
        stem_lower = unicodedata.normalize("NFC", path.stem).casefold()
        if "golden" in stem_lower and "evaluation" in stem_lower:
            continue
        if excluded_paths and path.resolve() in excluded_paths:
            continue
        extension = path.suffix.lower()
        if extension in allowed:
            files.append(path)
        elif reject_unsupported and extension in UNSUPPORTED_DOCUMENT_EXTENSIONS:
            unsupported.append(path.relative_to(directory).as_posix())
    if unsupported:
        listing = "\n".join(unsupported[:_MAX_LISTED_UNSUPPORTED])
        if len(unsupported) > _MAX_LISTED_UNSUPPORTED:
            listing += f"\n... and {len(unsupported) - _MAX_LISTED_UNSUPPORTED} more"
        raise ValueError("Unsupported documents found. This input accepts PDF, DOCX and XLSX:\n" + listing)
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


def _path_document_name(path, source_root):
    if source_root is not None:
        relative_path = path.parent.resolve() / path.name
        root = Path(source_root).resolve()
        if relative_path.is_relative_to(root):
            return relative_path.relative_to(root).as_posix()
    return path.name


def resolve_document_source(source=None, source_bytes=None, source_name=None, expected_format=None, *, source_root=None):
    if expected_format is not None and (not isinstance(expected_format, str)
                                        or "." + expected_format not in SUPPORTED_EXTENSIONS):
        raise ValueError("expected_format must be docx, pdf or xlsx")
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
            label = str(source_name)
            if not document_source_name(label).strip() or "\0" in label:
                raise ValueError("source_name must contain a document filename")
            if Path(label).is_absolute():
                label = _path_document_name(Path(label), source_root)
            elif PurePosixPath(label).is_absolute() or PureWindowsPath(label).is_absolute():
                label = document_source_name(label)
            name = _relative_document_name(label)
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
                if source_bytes.startswith(b"PK"):
                    try:
                        with ZipFile(BytesIO(source_bytes)) as archive:
                            types = archive.getinfo("[Content_Types].xml")
                            if (types.file_size <= 64 * 1024 and
                                    b"spreadsheetml.sheet.main+xml" in archive.read(types)):
                                fmt = "xlsx"
                    except (BadZipFile, KeyError):
                        pass
        if fmt == "xlsx" and name and Path(name).suffix.lower() in {".xls", ".xlsb", ".xlsm"}:
            raise ValueError("The XLSX adapter requires an .xlsx source name")
        if name is None:
            name = "in_memory_" + hashlib.sha256(source_bytes).hexdigest()[:_IN_MEMORY_HASH_LEN] + "." + fmt
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
    if source_name is None:
        name = _path_document_name(path, source_root)
    else:
        name = _relative_document_name(source_name)
    return str(path), name, fmt
