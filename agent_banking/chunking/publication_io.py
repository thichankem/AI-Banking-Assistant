"""Atomic file operations, writer locking, and publication cleanup."""

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
import tempfile


@contextmanager
def publication_lock(output):
    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(directory / ".publication.lock", os.O_CREAT | os.O_RDWR, 0o600)
    locked = False
    try:
        if os.fstat(descriptor).st_size == 0:
            os.write(descriptor, b"0")
        os.lseek(descriptor, 0, os.SEEK_SET)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            locked = True
        except OSError as error:
            raise RuntimeError("Another chunking writer owns this output directory") from error
        yield
    finally:
        try:
            if locked:
                os.lseek(descriptor, 0, os.SEEK_SET)
                if os.name == "nt":
                    msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)

def _publication_cleanup_candidates(output_dir):
    """List prior publication files by publisher-owned names."""
    output = Path(output_dir)
    if _is_indirection(output.lstat()):
        raise ValueError("Refusing to clean an output directory link or junction")
    root = output.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("Output cleanup requires a directory")

    def matches_hash_name(path, extension, widths):
        segments = path.stem.split("_")
        return path.suffix == extension and tuple(map(len, segments)) == widths and all(
            character in "0123456789abcdef" for segment in segments for character in segment
        )

    targets = set()
    for name, extension, widths in (
        ("snapshots", ".json", (64,)),
        ("artifacts", ".json", (20, 16, 16, 16)),
        ("chunks_per_file", ".json", (20, 16, 16, 16)),
        ("assets", ".png", (64,)),
    ):
        directory = root / name
        try:
            info = directory.lstat()
        except FileNotFoundError:
            continue
        if _is_indirection(info) or not stat.S_ISDIR(info.st_mode):
            raise ValueError("Invalid publication directory: " + str(directory))
        files = {path for path in directory.iterdir() if matches_hash_name(path, extension, widths)}
        targets.update(files)
        if name == "artifacts":
            # A document removed in a previous non-clean run has no current
            # manifest record, but its revision artifacts still identify its alias.
            for artifact in files:
                alias = root / (artifact.name.split("_", 1)[0] + ".json")
                if alias.exists() or alias.is_symlink():
                    targets.add(alias)
    names = ("chunking_manifest.json", "chunking_last_run.json", ".state_tracker.json", "all_chunks.json")
    for name in names:
        path = root / name
        if path.exists() or path.is_symlink():
            _check_publication_file(root, path)
    manifest_path = root / "chunking_manifest.json"
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            manifest = None
        if (isinstance(manifest, dict) and manifest.get("module") == "chunking"
                and isinstance(manifest.get("records"), list)):
            for record in manifest["records"]:
                source = record.get("rel_path") if isinstance(record, dict) else None
                if isinstance(source, str) and source:
                    identifier = hashlib.sha256(source.replace("\\", "/").encode()).hexdigest()[:20]
                    path = root / (identifier + ".json")
                    if path.exists() or path.is_symlink():
                        targets.add(path)
    for path in sorted(targets):
        _check_publication_file(root, path)
    return root, targets


def _is_indirection(info):
    return stat.S_ISLNK(info.st_mode) or bool(
        getattr(info, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    )


def _check_publication_file(root, path):
    info = path.lstat()
    if (_is_indirection(info) or not stat.S_ISREG(info.st_mode)
            or path.resolve(strict=True) != path or not path.is_relative_to(root)):
        raise ValueError("Invalid publication file: " + str(path))


def _remove_obsolete_publication_files(root, previous_files, active_files):
    for path in sorted(previous_files - active_files):
        try:
            _check_publication_file(root, path)
            path.unlink()
        except FileNotFoundError:
            continue
        except (OSError, ValueError) as error:
            report("Could not remove obsolete publication file " + path.name + ": " + str(error))


def file_hash(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(65536), b""):
            hasher.update(block)
    return hasher.hexdigest()


def atomic_json(path, value):
    atomic_bytes(path, json.dumps(value, ensure_ascii=False, indent=2).encode("utf-8"))


def atomic_bytes(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def content_hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def read_verified_json(path, digest):
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != digest:
        raise ValueError("Publication checksum mismatch: " + path.name)
    return json.loads(data)


def report(message):
    try:
        print(message, flush=True)
    except UnicodeEncodeError:
        encoding = sys.stdout.encoding or "utf-8"
        print(message.encode(encoding, errors="backslashreplace").decode(encoding), flush=True)
