"""Exact embedding token budgeting. Downloads tokenizer data, never model weights."""
import hashlib
from bisect import bisect_left, bisect_right
import os
from pathlib import Path
import tempfile
import threading
import urllib.request

from agent_banking.chunking.config.settings import get_config_section
from agent_banking.core.environment import load_environment

from .boundaries import protected_markup_spans as protected_markup_spans, safe_boundaries as safe_boundaries

_TOK_CFG = get_config_section("tokenizer")
_PIPELINE_CFG = get_config_section("pipeline")
TOKENIZER_SHA256 = _TOK_CFG["sha256"]
DEFAULT_TOKENIZER_URL = _TOK_CFG["default_url"]
_DEFAULT_RELPATH = _TOK_CFG["default_relpath"]
_LEGACY_RELPATHS = tuple(_TOK_CFG["legacy_relpaths"])
_MAX_BYTES = _TOK_CFG["max_bytes"]
_DOWNLOAD_TIMEOUT = _TOK_CFG["download_timeout"]
_MIN_BUDGET_TOKENS = _TOK_CFG["min_budget_tokens"]
_MAX_BUDGET_TOKENS = _TOK_CFG["max_budget_tokens"]
_SHORTEN_MARGIN = _TOK_CFG["shorten_margin"]
_CONTEXT_SHORTEN_RATIO = _TOK_CFG["context_shorten_ratio"]
_PASSAGE_PREFIX = _TOK_CFG["passage_prefix"]
_SITUATED_CONTEXT_PREFIX = _TOK_CFG["situated_context_prefix"]
_EXCERPT_COVERAGE_LABEL = _TOK_CFG["excerpt_coverage_label"]
_CONTEXT_ONLY_COVERAGE_LABEL = _TOK_CFG["context_only_coverage_label"]

_TOKENIZER = None
_TOKENIZER_PATH = None
_TOKENIZER_DIGEST = None
_LOCK = threading.Lock()


def format_passage(chunk):
    metadata = chunk.get("business_metadata", {})
    return format_passage_fields(chunk.get("content", ""), metadata.get("retrieval", {}))


def format_passage_fields(content, retrieval):
    context = str(retrieval.get("context", "")).strip()
    derived = str(retrieval.get("situated_context", "")).strip()
    parts = [_PASSAGE_PREFIX]
    if context:
        parts.append("[" + context + "]")
    if derived:
        parts.append(_SITUATED_CONTEXT_PREFIX + derived)
    parts.append(str(retrieval.get("passage_content", retrieval.get("embedding_content", content))))
    return "\n".join(parts)


def _expected_tokenizer_sha256():
    configured = os.getenv("CHUNKING_TOKENIZER_SHA256", "").strip().lower()
    if configured:
        return configured
    return TOKENIZER_SHA256


def _verified_data(path):
    with path.open("rb") as stream:
        data = stream.read(_MAX_BYTES + 1)
    if len(data) > _MAX_BYTES:
        return None
    expected = _expected_tokenizer_sha256()
    if expected == "*":
        return data
    if hashlib.sha256(data).hexdigest() == expected:
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
    global _TOKENIZER, _TOKENIZER_PATH, _TOKENIZER_DIGEST
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
                module = Path(__file__).resolve().parents[2]
                path = module / _DEFAULT_RELPATH
                legacy_paths = [module / rel for rel in _LEGACY_RELPATHS]
        path = Path(path).resolve()
        expected = _expected_tokenizer_sha256()
        if implicit and _TOKENIZER is not None and path == _TOKENIZER_PATH and expected in ("*", _TOKENIZER_DIGEST):
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
                if not url:
                    raise RuntimeError("Tokenizer file missing and TOKENIZER_URL is not configured")
                with urllib.request.urlopen(url, timeout=_DOWNLOAD_TIMEOUT) as response:
                    data = response.read(_MAX_BYTES + 1)
                if len(data) > _MAX_BYTES or (expected != "*" and hashlib.sha256(data).hexdigest() != expected):
                    raise RuntimeError("Tokenizer checksum mismatch; refusing unverified token budgets")
            _install_tokenizer(path, data)
        else:
            data = _verified_data(path)
            if data is None:
                raise RuntimeError(f"Configured tokenizer is not the audited {os.getenv('EMBEDDING_MODEL', 'embedding')} tokenizer")
        digest = hashlib.sha256(data).hexdigest()
        tokenizer = Tokenizer.from_str(data.decode("utf-8"))
        tokenizer.no_truncation()
        tokenizer.no_padding()
        if implicit:
            _TOKENIZER = tokenizer
            _TOKENIZER_PATH = path
            _TOKENIZER_DIGEST = digest
        return tokenizer


class TokenBudget:
    def __init__(self, target=_PIPELINE_CFG["target_chunk_tokens"], maximum=_PIPELINE_CFG["max_chunk_tokens"], tokenizer=None):
        if type(target) is not int or type(maximum) is not int or not _MIN_BUDGET_TOKENS <= target <= maximum <= _MAX_BUDGET_TOKENS:
            raise ValueError(
                "Passage budgets must satisfy " + str(_MIN_BUDGET_TOKENS) + " <= target <= maximum <= " + str(_MAX_BUDGET_TOKENS)
            )
        self.target = target
        self.maximum = maximum
        if tokenizer is None:
            self.tokenizer = get_tokenizer()
            self.tokenizer_sha256 = _TOKENIZER_DIGEST or _expected_tokenizer_sha256()
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
        end = offsets[min(len(offsets) - 1, max(0, limit - _SHORTEN_MARGIN))][1]
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
                    value = self.shorten(context[label], self.maximum // _CONTEXT_SHORTEN_RATIO)
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
        if excerpt:
            retrieval["passage_coverage"] = _EXCERPT_COVERAGE_LABEL
        else:
            retrieval["passage_coverage"] = _CONTEXT_ONLY_COVERAGE_LABEL
