"""Exact embedding token budgeting. Downloads tokenizer data, never model weights."""
import hashlib
import os
from pathlib import Path
import tempfile
import threading
import urllib.request


def format_passage(chunk):
    metadata = chunk.get("business_metadata", {})
    return format_passage_fields(chunk.get("content", ""), metadata.get("retrieval", {}))


def format_passage_fields(content, retrieval):
    context = str(retrieval.get("context", "")).strip()
    derived = str(retrieval.get("situated_context", "")).strip()
    parts = ["passage:"]
    if context:
        parts.append("[" + context + "]")
    if derived:
        parts.append("Ngữ cảnh: " + derived)
    parts.append(str(content))
    return "\n".join(parts)

TOKENIZER_SHA256 = "62c24cdc13d4c9952d63718d6c9fa4c287974249e16b7ade6d5a85e7bbb75626"
TOKENIZER_URL = os.getenv("TOKENIZER_URL", os.getenv("EMBEDDING_TOKENIZER_URL", ""))
_TOKENIZER = None
_LOCK = threading.Lock()


def _verified_data(path):
    if path.stat().st_size > 50_000_000:
        return None
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() == TOKENIZER_SHA256:
        return data
    return None


def _install_tokenizer(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".e5-", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(data)
        try:
            temporary.replace(path)
        except OSError:
            # Another process can complete installation and open the target
            # between the existence check and Windows' atomic replace.
            if not path.is_file() or _verified_data(path) is None:
                raise
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def get_tokenizer(path=None):
    global _TOKENIZER
    with _LOCK:
        if path is None and _TOKENIZER is not None:
            return _TOKENIZER
        from tokenizers import Tokenizer
        implicit = path is None
        legacy_paths = []
        if path is None:
            configured = os.getenv("CHUNKING_TOKENIZER_PATH")
            if configured:
                path = Path(configured)
            else:
                module = Path(__file__).resolve().parent.parent
                path = module / "cache/tokenizers/tokenizer.json"
                legacy_paths = [module / "cache/tokenizers/e5-tokenizer.json",
                                module / "output/runtime/e5-tokenizer.json",
                                module / "output/runtime/tokenizer.json",
                                module / "output/audit_2026_11_06/e5-tokenizer.json"]
        path = Path(path)
        if not path.exists():
            data = None
            for candidate in legacy_paths:
                if candidate.is_file():
                    data = _verified_data(candidate)
                    if data is not None:
                        break
            if data is None:
                with urllib.request.urlopen(TOKENIZER_URL, timeout=60) as response:
                    data = response.read(50_000_001)
                if len(data) > 50_000_000 or hashlib.sha256(data).hexdigest() != TOKENIZER_SHA256:
                    raise RuntimeError("Tokenizer checksum mismatch; refusing unverified token budgets")
            _install_tokenizer(path, data)
        elif _verified_data(path) is None:
            raise RuntimeError(f"Configured tokenizer is not the audited {os.getenv('EMBEDDING_MODEL', 'embedding')} tokenizer")
        tokenizer = Tokenizer.from_file(str(path))
        tokenizer.no_truncation()
        tokenizer.no_padding()
        if implicit:
            _TOKENIZER = tokenizer
        return tokenizer


class TokenBudget:
    def __init__(self, target=400, maximum=512, tokenizer=None):
        if not 32 <= target <= maximum <= 512:
            raise ValueError("E5 budgets must satisfy 32 <= target <= maximum <= 512")
        self.target = target
        self.maximum = maximum
        self.tokenizer = tokenizer or get_tokenizer()

    def count(self, text):
        return len(self.tokenizer.encode(str(text)).ids)

    def passage_count(self, chunk):
        return self.count(format_passage(chunk))

    def shorten(self, text, limit):
        if self.count(text) <= limit:
            return text
        offsets = self.tokenizer.encode(text, add_special_tokens=False).offsets
        end = offsets[max(0, limit - 3)][1]
        candidate = text[:end].rstrip()
        while candidate and self.count(candidate) > limit:
            candidate = candidate[:-1]
        # Context labels must not turn a cut number/token into a different fact.
        if len(candidate) < len(text) and candidate and not text[len(candidate)].isspace():
            space = max(candidate.rfind(" "), candidate.rfind("\n"), candidate.rfind("\t"))
            if space < 0:
                return ""
            candidate = candidate[:space].rstrip()
        return candidate

    def split(self, text, fits, *, fits_range=None, align_words=False):
        """Return exact source substrings under a hard resource ceiling.

        A single oversized word falls back to Unicode character offsets; no text
        is rewritten, decoded from tokens or silently discarded.
        """
        if not text:
            return []
        if fits_range is None:
            def fits_range(start, end):
                return fits(text[start:end])
        pieces = []
        start = 0
        while start < len(text):
            end = start
            high = min(start + 64, len(text))
            while fits_range(start, high):
                end = high
                if high == len(text):
                    break
                high = min(start + 2 * (high - start), len(text))
            low = end + 1
            while low <= high:
                middle = (low + high) // 2
                if fits_range(start, middle):
                    end = middle
                    low = middle + 1
                else:
                    high = middle - 1
            if end == start:
                raise ValueError("Context consumes the embedding budget; no source character fits")
            if align_words and end < len(text) and not text[end - 1].isspace() and not text[end].isspace():
                boundary = end
                while boundary > start and not text[boundary - 1].isspace():
                    boundary -= 1
                if boundary > start and text[start:boundary].strip() and fits_range(start, boundary):
                    end = boundary
            pieces.append((start, end, text[start:end]))
            start = end
        if "".join(piece[2] for piece in pieces) != text:
            raise AssertionError("Token splitting changed the source text")
        return pieces
