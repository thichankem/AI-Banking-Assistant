"""Fingerprint the code, prompts, models and effective planner settings."""

import hashlib
from importlib.metadata import version
import json
import os

from .enrichment import enricher as model_enrichment
from .planning import semantic_boundaries
from .enrichment.image_context import IMAGE_POLICY


def chunker_fingerprint(chunker, pipeline_version):
    from .extraction.pdf import extractor as pdf_extractor

    configuration = {"pipeline": pipeline_version,
                     "prompts": {"version": model_enrichment.PROMPT_VERSION,
                                 "structure": model_enrichment.STRUCTURE_SYSTEM_PROMPT,
                                 "pdf_version": pdf_extractor.PDF_PROMPT_VERSION,
                                 "pdf_schema": pdf_extractor.PDF_SCHEMA_VERSION,
                                 "page": pdf_extractor.SCAN_PAGE_PROMPT,
                                 "start": semantic_boundaries.START_PROMPT,
                                 "boundary": semantic_boundaries.BOUNDARY_PROMPT},
                     "semantic": chunker.semantic, "image_mode": chunker.image_mode,
                     "search_language": chunker.planner.search_language,
                     "dpi": chunker.dpi, "max_pdf_pages": chunker.pdf_extractor.max_pages,
                     "max_page_pixels": chunker.pdf_extractor.max_page_pixels,
                     "vision_client": getattr(chunker.pdf_extractor.vision_client, "model", None),
                     "review_client": getattr(chunker.pdf_extractor.review_client, "model", None),
                     "image_policy": IMAGE_POLICY, "max_words": chunker.max_chunk_words,
                     "target_tokens": chunker.planner.budget.target, "max_tokens": chunker.planner.budget.maximum,
                     "min_tokens": chunker.planner.min_tokens,
                     "tokenizer_sha256": chunker.planner.budget.tokenizer_sha256,
                     "table_rows": chunker.planner.table_rows, "llm_enabled": chunker.use_llm,
                     "max_chars": chunker.planner.max_chars, "min_words": chunker.planner.min_words, "min_chars": chunker.planner.min_chars,
                     "decision_enabled": chunker.use_decision, "llm_model": os.getenv("MODEL"),
                     "decision_model": os.getenv("DECISION_MODEL"),
                     "llm_provider": os.getenv("BASE_URL"), "decision_provider": os.getenv("DECISION_BASE_URL"),
                     "llm_max_tokens": os.getenv("CHUNKING_LLM_MAX_TOKENS", "65536"),
                     "models_disable_json": os.getenv("MODELS_DISABLE_JSON_MODE"),
                     "embedding_model": os.getenv("EMBEDDING_MODEL"),
                     "dependencies": {package: version(package) for package in ("python-docx", "tokenizers", "PyMuPDF", "Pillow", "regex")}}
    # Code changes invalidate document outputs even if the version constant
    # was accidentally left unchanged. Secrets are never part of manifests.
    from agent_banking.core.models import chain_fingerprint
    configuration["model_chains"] = {name: chain_fingerprint("chunking." + name)
                                      for name in ("text", "decision", "vision", "vision_review")}
    for role, client in (("extraction", chunker.pdf_extractor.vision_client), ("review", chunker.pdf_extractor.review_client)):
        if callable(getattr(client, "identity", None)):
            configuration[role + "_identity"] = client.identity()
    configuration["sources"] = chunker._code_snapshot
    return hashlib.sha256(json.dumps(configuration, sort_keys=True).encode("utf-8")).hexdigest()
