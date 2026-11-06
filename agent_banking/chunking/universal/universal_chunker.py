"""Grounded DOCX ingestion: native structure, cloud enrichment, exact tokens."""
import hashlib
import json
import os
from importlib.metadata import version
from pathlib import Path

from .vision_cache import PageVisionCache
from .document_ir import DocxExtractor
from .chunk_planner import ChunkPlanner
from .model_enrichment import ModelEnricher, PROMPT_VERSION
from .image_context import IMAGE_POLICY, RESOLVED_IMAGE_STATUSES, mark_unassessed_images
from .schema_assembler import (
    build_universal_chunk, link_chunk_sequence, infer_business_metadata,
    generate_chunk_id, REQUIRED_UNIVERSAL_SCHEMA_KEYS, VALID_CHUNK_TYPES,
)

PIPELINE_VERSION = "grounded-docx-3.0-text-images"
MODULE = Path(__file__).resolve().parent.parent


def _parse_markdown_table_to_2d_list(text):
    """Compatibility parser; escaped pipes remain within their source cells."""
    rows = []
    for line in str(text or "").splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        cells = []
        value = []
        escaped = False
        for character in line[1:]:
            if escaped:
                value.append(character)
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == "|":
                cells.append("".join(value).strip())
                value = []
            else:
                value.append(character)
        if value:
            cells.append("".join(value).strip())
        if cells and not all(cell and all(character in "-: " for character in cell) for cell in cells):
            rows.append(cells)
    return rows or None


class UniversalDocxChunker:
    def __init__(self, cache_db_path=None, base_url=None, api_key=None, model=None,
                 dpi=200, max_chunk_words=350, cache=None, vision_client=None,
                 profile=None, target_chunk_tokens=400, max_chunk_tokens=512,
                 table_rows_per_chunk=4, asset_dir=None, workers=4,
                 use_llm=True, use_decision=True, tokenizer=None,
                 max_chunk_chars=None, min_chunk_words=30, min_section_chars=0, review_vision_client=None):
        self.profile = profile or os.getenv("CHUNKING_PROFILE", "accuracy")
        if self.profile not in ("native", "accuracy"):
            raise ValueError("Chunking profile must be native or accuracy")
        self.dpi = dpi
        self.max_chunk_words = max_chunk_words
        self.workers = workers
        self.use_llm = use_llm
        self.use_decision = use_decision
        if dpi <= 0:
            raise ValueError("DPI must be positive")
        self._owns_cache = cache is None
        self.cache = cache
        if self.cache is None:
            self.cache = PageVisionCache(str(cache_db_path or MODULE / "cache/grounded_models.db"))
        # Legacy vision arguments are accepted for call compatibility only.
        # The production route never instantiates a vision client or reads pixels.
        self.planner = ChunkPlanner(target_chunk_tokens, max_chunk_tokens, max_chunk_words,
                                    table_rows_per_chunk, tokenizer, max_chunk_chars, min_chunk_words, min_section_chars)
        self.extractor = DocxExtractor(asset_dir or MODULE / "output/assets", render_figures=False)
        self.last_artifact = None
        self._code_snapshot = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                               for path in sorted(Path(__file__).resolve().parent.glob("*.py"))}

    def fingerprint(self):
        configuration = {"pipeline": PIPELINE_VERSION, "profile": self.profile, "prompt": PROMPT_VERSION,
                         "image_policy": IMAGE_POLICY, "max_words": self.max_chunk_words,
                         "target_tokens": self.planner.budget.target, "max_tokens": self.planner.budget.maximum,
                         "table_rows": self.planner.table_rows, "llm_enabled": self.use_llm,
                         "max_chars": self.planner.max_chars, "min_words": self.planner.min_words, "min_chars": self.planner.min_chars,
                         "decision_enabled": self.use_decision, "llm_model": os.getenv("MODEL"),
                         "decision_model": os.getenv("JEV_MODEL"),
                         "llm_provider": os.getenv("BASE_URL"), "decision_provider": os.getenv("JEV_BASE_URL"),
                         "llm_max_tokens": os.getenv("CHUNKING_LLM_MAX_TOKENS", "65536"),
                         "dependencies": {package: version(package) for package in ("python-docx", "tokenizers")}}
        # Code changes invalidate document outputs even if the version constant
        # was accidentally left unchanged. Secrets are never part of manifests.
        from agent_banking.config.models import chain_fingerprint
        configuration["model_chains"] = {name: chain_fingerprint("chunking." + name)
                                          for name in ("text", "decision", "vision")}
        configuration["sources"] = self._code_snapshot
        return hashlib.sha256(json.dumps(configuration, sort_keys=True).encode("utf-8")).hexdigest()

    def _resolve_source(self, docx_path=None, docx_bytes=None, source_name=None):
        if isinstance(docx_path, (bytes, bytearray)):
            docx_bytes = bytes(docx_path)
            docx_path = None
        if docx_bytes is not None:
            if not isinstance(docx_bytes, (bytes, bytearray)):
                raise TypeError("docx_bytes must be bytes or bytearray")
            if not docx_bytes:
                raise ValueError("docx_bytes is empty")
            name = source_name
            if name is None:
                name = "in_memory_" + hashlib.sha256(docx_bytes).hexdigest()[:20] + ".docx"
            return bytes(docx_bytes), str(name)
        if not isinstance(docx_path, (str, Path)):
            if docx_path is None:
                raise ValueError("A DOCX path or byte stream is required")
            raise TypeError("docx_path must be a string or Path")
        if not str(docx_path).strip():
            raise ValueError("docx_path is empty")
        path = Path(docx_path)
        if not path.is_file():
            raise FileNotFoundError(str(path))
        if path.suffix.lower() != ".docx":
            raise ValueError("DOCX adapter requires .docx input")
        name = source_name or str(path.resolve())
        return str(path), str(name)

    def chunk_document(self, docx_path=None, docx_bytes=None, source_name=None):
        self.last_artifact = None
        source, name = self._resolve_source(docx_path, docx_bytes, source_name)
        ir = self.extractor.extract(source, name)
        ir["profile"] = self.profile
        if any(warning.get("requires_review") for warning in ir["warnings"]):
            raise RuntimeError("Unsupported or unresolved source content; publication refused: " + json.dumps(ir["warnings"]))
        if self.profile == "accuracy":
            enricher = ModelEnricher(self.cache, workers=self.workers, use_llm=self.use_llm, use_decision=self.use_decision,
                                     coverage_validator=lambda document: self.planner.plan(document)[1]["quality"])
            ir = enricher.enrich(ir)
        else:
            mark_unassessed_images(ir)
        chunks, artifact = self.planner.plan(ir)
        artifact["pipeline_version"] = PIPELINE_VERSION
        artifact["pipeline_fingerprint"] = self.fingerprint()
        resolved = all(block["image"].get("status") in RESOLVED_IMAGE_STATUSES for block in ir["blocks"] if block["type"] == "image")
        artifact["quality"]["index_ready"] = not ir["assets"] or self.profile == "accuracy" and resolved
        for chunk in chunks:
            chunk["business_metadata"]["pipeline_fingerprint"] = artifact["pipeline_fingerprint"]
            chunk["business_metadata"]["quality"]["index_ready"] = artifact["quality"]["index_ready"]
        self.last_artifact = artifact
        return chunks

    def chunk_docx(self, docx_path, **kwargs):
        return self.chunk_document(docx_path=docx_path, **kwargs)

    def close(self):
        if self._owns_cache:
            self.cache.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
