"""Exact embedding token budgeting. Downloads tokenizer data, never model weights."""
import hashlib
from bisect import bisect_left, bisect_right
import os
from pathlib import Path
import tempfile
import threading
import urllib.request

from agent_banking.core.environment import load_environment

from .source_boundaries import protected_markup_spans as protected_markup_spans, safe_boundaries as safe_boundaries


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
    parts.append(str(retrieval.get("passage_content", retrieval.get("embedding_content", content))))
    return "\n".join(parts)



TOKENIZER_SHA256 = "62c24cdc13d4c9952d63718d6c9fa4c287974249e16b7ade6d5a85e7bbb75626"
DEFAULT_TOKENIZER_URL = (
    "https://huggingface.co/intfloat/multilingual-e5-large/resolve/"
    "81019eef13e0844a1caaeeaead846b79a96a4747/tokenizer.json"
)
_TOKENIZER = None
_TOKENIZER_PATH = None
_LOCK = threading.Lock()


def _verified_data(path):
    with path.open("rb") as stream:
        data = stream.read(50_000_001)
    if len(data) <= 50_000_000 and hashlib.sha256(data).hexdigest() == TOKENIZER_SHA256:
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
    global _TOKENIZER, _TOKENIZER_PATH
    with _LOCK:
        load_environment()
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
        path = Path(path).resolve()
        if implicit and _TOKENIZER is not None and path == _TOKENIZER_PATH:
            return _TOKENIZER
        if not path.exists():
            data = None
            for candidate in legacy_paths:
                if candidate.is_file():
                    data = _verified_data(candidate)
                    if data is not None:
                        break
            if data is None:
                url = os.getenv("TOKENIZER_URL") or os.getenv("EMBEDDING_TOKENIZER_URL") or DEFAULT_TOKENIZER_URL
                with urllib.request.urlopen(url, timeout=60) as response:
                    data = response.read(50_000_001)
                if len(data) > 50_000_000 or hashlib.sha256(data).hexdigest() != TOKENIZER_SHA256:
                    raise RuntimeError("Tokenizer checksum mismatch; refusing unverified token budgets")
            _install_tokenizer(path, data)
        else:
            data = _verified_data(path)
            if data is None:
                raise RuntimeError(f"Configured tokenizer is not the audited {os.getenv('EMBEDDING_MODEL', 'embedding')} tokenizer")
        tokenizer = Tokenizer.from_str(data.decode("utf-8"))
        tokenizer.no_truncation()
        tokenizer.no_padding()
        if implicit:
            _TOKENIZER = tokenizer
            _TOKENIZER_PATH = path
        return tokenizer


class TokenBudget:
    def __init__(self, target=512, maximum=1024, tokenizer=None):
        if type(target) is not int or type(maximum) is not int or not 32 <= target <= maximum <= 1024:
            raise ValueError("Passage budgets must satisfy 32 <= target <= maximum <= 1024")
        self.target = target
        self.maximum = maximum
        if tokenizer is None:
            self.tokenizer = get_tokenizer()
            self.tokenizer_sha256 = TOKENIZER_SHA256
        else:
            if getattr(tokenizer, "truncation", None) or getattr(tokenizer, "padding", None):
                raise ValueError("Injected tokenizer must have truncation and padding disabled")
            self.tokenizer = tokenizer
            self.tokenizer_sha256 = hashlib.sha256(tokenizer.to_str().encode("utf-8")).hexdigest()

    def count(self, text):
        return len(self.tokenizer.encode(str(text)).ids)

    def passage_count(self, chunk):
        return self.count(format_passage(chunk))

    def shorten(self, text, limit):
        if self.count(text) <= limit:
            return text
        offsets = self.tokenizer.encode(text, add_special_tokens=False).offsets
        if not offsets or limit < 1:
            return ""
        end = offsets[min(len(offsets) - 1, max(0, limit - 3))][1]
        boundaries = safe_boundaries(text)
        index = bisect_right(boundaries, end) - 1
        candidate = text[:boundaries[index]].rstrip()
        while candidate and self.count(candidate) > limit:
            index -= 1
            candidate = text[:boundaries[index]].rstrip()
        # Context labels must not turn a cut number/token into a different fact.
        if len(candidate) < len(text) and candidate and not text[len(candidate)].isspace():
            space = max(candidate.rfind(" "), candidate.rfind("\n"), candidate.rfind("\t"))
            if space < 0:
                return ""
            candidate = candidate[:space].rstrip()
        return candidate

    def split(self, text, fits, *, fits_range=None, align_words=False, boundaries=None,
              allow_oversized=False, max_pieces=None):
        """Return exact source substrings under a hard resource ceiling.

        A single oversized word falls back to safe Unicode offsets. Without
        an explicit piece limit, the result reconstructs the source exactly.
        """
        if not text:
            return []
        if fits_range is None:
            def fits_range(start, end):
                return fits(text[start:end])
        if boundaries is None:
            boundaries = safe_boundaries(text)
        pieces = []
        cursor = 0
        final = len(boundaries) - 1
        while cursor < final and (max_pieces is None or len(pieces) < max_pieces):
            start = boundaries[cursor]
            last_fit = cursor
            high = min(cursor + 64, final)
            while fits_range(start, boundaries[high]):
                last_fit = high
                if high == final:
                    break
                high = min(cursor + 2 * (high - cursor), final)
            low = last_fit + 1
            while low <= high:
                middle = (low + high) // 2
                if fits_range(start, boundaries[middle]):
                    last_fit = middle
                    low = middle + 1
                else:
                    high = middle - 1
            if last_fit == cursor:
                if not allow_oversized:
                    raise ValueError("Context consumes the embedding budget; no complete source element fits")
                last_fit = cursor + 1
            end = boundaries[last_fit]
            if align_words and end < len(text) and not text[end - 1].isspace() and not text[end].isspace():
                boundary = end
                while boundary > start and not text[boundary - 1].isspace():
                    boundary -= 1
                index = bisect_left(boundaries, boundary)
                if (boundary > start and index <= last_fit and boundaries[index] == boundary
                        and text[start:boundary].strip() and fits_range(start, boundary)):
                    end = boundary
                    last_fit = index
            pieces.append((start, end, text[start:end]))
            cursor = last_fit
        if max_pieces is None and "".join(piece[2] for piece in pieces) != text:
            raise AssertionError("Token splitting changed the source text")
        return pieces

    def embedding_excerpt(self, content, retrieval, preferred=None):
        """Budget a preview while the canonical chunk retains its complete content."""
        retrieval.pop("passage_content", None)
        retrieval.pop("passage_coverage", None)
        focus = retrieval.get("embedding_content", content)
        context = {key: value for key, value in retrieval.items() if key != "embedding_content"}

        def fits(text):
            return self.count(format_passage_fields(text, context)) <= self.maximum

        if fits(focus):
            return
        if preferred is None:
            preferred = focus
        if not fits(""):
            # An independently supplied breadcrumb may consume the entire
            # budget. Shorten the labels before selecting a content excerpt.
            for label in ("situated_context", "context"):
                if not fits("") and context.get(label):
                    value = self.shorten(context[label], self.maximum // 6)
                    context[label] = value
                    retrieval[label] = value
        if fits(preferred):
            excerpt = preferred
        else:
            excerpt = ""
            boundaries = safe_boundaries(preferred)
            for left, right in zip(boundaries, boundaries[1:]):
                first = preferred[left:right]
                if not first.strip() or not fits(first):
                    continue
                # Skip only leading whitespace or indivisible over-budget
                # markup. Keep subsequent source text searchable.
                pieces = self.split(preferred[left:], fits, align_words=True, max_pieces=1,
                                    boundaries=[point - left for point in boundaries[bisect_left(boundaries, left):]])
                excerpt = pieces[0][2]
                break
        retrieval["passage_content"] = excerpt
        retrieval["passage_coverage"] = "excerpt" if excerpt else "context_only"
