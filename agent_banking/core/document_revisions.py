"""Keep the exact source bytes behind an indexed document revision."""

from hashlib import file_digest
from functools import lru_cache
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

from agent_banking.chunking.publication_io import file_hash

DOCUMENT_EXTENSIONS = frozenset({".pdf", ".docx", ".xlsx"})


def is_document_version(value):
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def revision_path(root, source, version):
    suffix = Path(source).suffix.lower()
    if suffix not in DOCUMENT_EXTENSIONS or not is_document_version(version):
        raise ValueError("Invalid source revision")
    directory = Path(root).resolve()
    path = (directory / (version + suffix)).resolve()
    if path.parent != directory:
        raise ValueError("Invalid revision location")
    return path


@lru_cache(maxsize=256)
def _document_hash(path, modified, changed, size):
    return file_hash(path)


def document_matches_version(path, version):
    path = Path(path)
    stat = path.stat()
    return _document_hash(path, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_size) == version


def archive_document(root, path, version):
    """Publish a checked immutable copy using a file in the same directory."""
    path = Path(path)
    target = revision_path(root, path.name, version)
    if target.is_file() and file_hash(target) == version:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with path.open("rb") as source, NamedTemporaryFile(dir=target.parent, suffix=".tmp", delete=False) as dest:
            temporary = Path(dest.name)
            while block := source.read(1024 * 1024):
                dest.write(block)
        with temporary.open("rb") as stream:
            if file_digest(stream, "sha256").hexdigest() != version:
                raise ValueError("Source changed before its revision could be archived")
        os.replace(temporary, target)
        return target
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def copy_publication_revisions(publication_dir, index_dir, records):
    """Copy exact source bytes that are present in verified publications."""
    for record in records:
        if record["status"] == "FAILED":
            continue
        source, version = record["rel_path"], record.get("input_sha256")
        if Path(source).suffix.lower() not in DOCUMENT_EXTENSIONS or not is_document_version(version):
            continue
        path = revision_path(Path(publication_dir) / "revisions", source, version)
        if path.is_file():
            archive_document(Path(index_dir) / "revisions", path, version)
