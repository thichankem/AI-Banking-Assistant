"""Exact E5 token budgeting. Downloads tokenizer data, never model weights."""
import hashlib
import os
from pathlib import Path
import threading
import urllib.request

from .passage import format_passage

TOKENIZER_SHA256 = "62c24cdc13d4c9952d63718d6c9fa4c287974249e16b7ade6d5a85e7bbb75626"
TOKENIZER_URL = "https://huggingface.co/universal-tokenizer/resolve/main/tokenizer.json"
_TOKENIZER = None
_LOCK = threading.Lock()


def get_tokenizer(path=None):
    global _TOKENIZER
    with _LOCK:
        if path is None and _TOKENIZER is not None:
            return _TOKENIZER
        from tokenizers import Tokenizer
        if path is None:
            configured = os.getenv("CHUNKING_TOKENIZER_PATH")
            if configured:
                path = Path(configured)
            else:
                module = Path(__file__).resolve().parent.parent
                path = module / "output/runtime/e5-tokenizer.json"
                audited = module / "output/audit_2026_11_06/e5-tokenizer.json"
                if not path.exists() and audited.exists():
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(audited.read_bytes())
        path = Path(path)
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            with urllib.request.urlopen(TOKENIZER_URL, timeout=60) as response:
                data = response.read(50_000_001)
            if len(data) > 50_000_000 or hashlib.sha256(data).hexdigest() != TOKENIZER_SHA256:
                raise RuntimeError("E5 tokenizer checksum mismatch; refusing unverified token budgets")
            temporary = path.with_suffix(".download")
            temporary.write_bytes(data)
            temporary.replace(path)
        if hashlib.sha256(path.read_bytes()).hexdigest() != TOKENIZER_SHA256:
            raise RuntimeError("Configured tokenizer is not the audited embedding-model tokenizer")
        tokenizer = Tokenizer.from_file(str(path))
        tokenizer.no_truncation()
        tokenizer.no_padding()
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

    def split(self, text, fits):
        """Return exact source substrings. Prefer paragraph/sentence/word ends.

        A single oversized word falls back to Unicode character offsets; no text
        is rewritten, decoded from tokens or silently discarded.
        """
        if not text:
            return []
        pieces = []
        start = 0
        while start < len(text):
            if fits(text[start:]):
                pieces.append((start, len(text), text[start:]))
                break
            low = start + 1
            high = len(text)
            end = start
            while low <= high:
                middle = (low + high) // 2
                if fits(text[start:middle]):
                    end = middle
                    low = middle + 1
                else:
                    high = middle - 1
            if end == start:
                raise ValueError("Context consumes the embedding budget; no source character fits")
            lower = start + max(1, (end - start) // 2)
            preferred = 0
            for boundary in ("\n\n", ". ", "; ", "\n", " "):
                index = text.rfind(boundary, lower, end)
                if index >= lower:
                    preferred = index + len(boundary)
                    break
            if preferred:
                end = preferred
            while end > start and not fits(text[start:end]):
                end -= 1
            pieces.append((start, end, text[start:end]))
            start = end
        if "".join(piece[2] for piece in pieces) != text:
            raise AssertionError("Token splitting changed the source text")
        return pieces
