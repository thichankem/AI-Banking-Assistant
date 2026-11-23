"""Grounded document ingestion, cloud semantics and exact passage budgets."""
import hashlib
import json
import os
import threading
import time
from pathlib import Path

from agent_banking.chunking.config.settings import ChunkingConfig, get_config_section
from .models.response_cache import PageVisionCache
from .extraction.docx.extractor import DocxExtractor
from .extraction.xlsx.extractor import XlsxExtractor, validate_excel_metadata
from .planning.planner import ChunkPlanner
from .extraction.sources import (
    discover_documents as discover_documents,
    document_source_root,
    document_source_name as document_source_name,
    resolve_document_source,
)
from .fingerprint import chunker_fingerprint
from .enrichment.enricher import ModelEnricher
from .enrichment.image_context import RESOLVED_IMAGE_STATUSES
from .planning.schema import (
    build_universal_chunk, link_chunk_sequence, infer_business_metadata,
    generate_chunk_id, REQUIRED_UNIVERSAL_SCHEMA_KEYS, VALID_CHUNK_TYPES,
)

_PIPELINE = get_config_section("pipeline")
_ENRICH_IMAGE_CFG = get_config_section("enrichment")["image_context"]

PIPELINE_VERSION = _PIPELINE["version"]
MODULE = Path(__file__).resolve().parent.parent
_DOCUMENT_LOCK = threading.RLock()


def _parse_markdown_table_to_2d_list(text):
    """Compatibility parser; escaped pipes remain within their source cells."""
    rows = []
    had_separator = False
    for line in str(text or "").splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        cells = []
        value = []
        escaped = False
        for character in line[1:]:
            if escaped:
                if character not in ("\\", "|"):
                    value.append("\\")
                value.append(character)
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == "|":
                cells.append("".join(value).strip())
                value = []
            else:
                value.append(character)
        if escaped:
            value.append("\\")
        if value:
            cells.append("".join(value).strip())
        separator = bool(cells) and len(rows) == 1 and not had_separator and all(
            cell.strip(":") and set(cell.strip(":")) == {"-"} for cell in cells
        )
        if separator:
            had_separator = True
        elif cells:
            rows.append(cells)
    return rows or None


class UniversalDocxChunker:
    def __init__(self, cache_db_path=None, base_url=None, api_key=None, model=None,
                 dpi=_PIPELINE["dpi"], max_chunk_words=None, cache=None, vision_client=None,
                 target_chunk_tokens=_PIPELINE["target_chunk_tokens"],
                 max_chunk_tokens=_PIPELINE["max_chunk_tokens"],
                 table_rows_per_chunk=_PIPELINE["table_rows_per_chunk"], asset_dir=None,
                 workers=_PIPELINE["workers"],
                 use_llm=_PIPELINE["use_llm"], use_decision=_PIPELINE["use_decision"], tokenizer=None,
                 max_chunk_chars=None, min_chunk_words=_PIPELINE["min_chunk_words"],
                 min_section_chars=_PIPELINE["min_section_chars"], review_vision_client=None,
                 semantic=_PIPELINE["semantic"], image_mode=_PIPELINE["default_image_mode"],
                 max_pdf_pages=_PIPELINE["max_pdf_pages"],
                 max_page_pixels=_PIPELINE["max_page_pixels"],
                 max_chunk_size=None, min_section_size=None, config=None,
                 min_chunk_tokens=_PIPELINE["min_chunk_tokens"],
                 search_language=None, ocr_docx_images=_PIPELINE["ocr_docx_images"], excel_metadata=None,
                 review_incomplete_pdf=False, previous_partial_pdf_pages=0):
        if max_chunk_chars is None and max_chunk_size is not None:
            max_chunk_chars = max_chunk_size
        if min_section_size is not None and min_section_chars == 0:
            min_section_chars = min_section_size
        self.config = config
        if search_language is None:
            search_language = getattr(config, "CHUNK_SEARCH_LANGUAGE", ChunkingConfig.CHUNK_SEARCH_LANGUAGE)
        self.ocr_docx_images = ocr_docx_images
        self.excel_metadata = validate_excel_metadata(excel_metadata)
        self.dpi = dpi
        self.max_chunk_words = max_chunk_words
        if type(workers) is not int or workers < 1:
            raise ValueError("workers must be a positive integer")
        self.workers = min(workers, _PIPELINE["max_workers"])
        self.use_llm = use_llm
        self.use_decision = use_decision
        if image_mode not in ("vision", "text_context"):
            raise ValueError("image_mode must be vision or text_context")
        self.semantic = semantic
        self.image_mode = "ignore_standalone_images"
        from .extraction.pdf.extractor import PdfExtractor

        explicit_vision = vision_client is not None
        if vision_client is None and any(value is not None for value in (base_url, api_key, model)):
            from .extraction.pdf.page_review import OCR_MAX_OUTPUT_TOKENS
            from .models.vision_client import DocumentVisionClient
            vision_client = DocumentVisionClient(base_url=base_url, api_key=api_key, model=model,
                                                 max_tokens=OCR_MAX_OUTPUT_TOKENS)
        self._owns_cache = cache is None
        self.cache = cache
        if self.cache is None:
            self.cache = PageVisionCache(str(cache_db_path or MODULE / _PIPELINE["cache_db_relpath"]))
        if explicit_vision:
            ocr_workers = 1
        else:
            ocr_workers = max(1, int(os.getenv("CHUNKING_OCR_WORKERS", getattr(config, "OCR_WORKERS", ChunkingConfig.OCR_WORKERS))))
        try:
            self.pdf_extractor = PdfExtractor(
                asset_dir or MODULE / _PIPELINE["asset_dir_relpath"], self.cache, vision_client, review_vision_client,
                dpi, max_pdf_pages, max_page_pixels, use_decision,
                search_language=search_language, workers=ocr_workers, review_incomplete=review_incomplete_pdf,
                previous_partial_pages=previous_partial_pdf_pages,
            )
            self.planner = ChunkPlanner(
                target_chunk_tokens, max_chunk_tokens, max_chunk_words, table_rows_per_chunk, tokenizer,
                max_chunk_chars, min_chunk_words, min_section_chars, use_decision=use_decision,
                decision_cache=self.cache, min_tokens=min_chunk_tokens, search_language=search_language,
            )
            self.extractor = DocxExtractor(
                self.pdf_extractor.asset_dir, use_decision=use_decision, decision_cache=self.cache,
                search_language=search_language, ocr_images=ocr_docx_images,
                vision_client=vision_client, review_client=review_vision_client, max_image_pixels=max_page_pixels,
            )
            self.xlsx_extractor = XlsxExtractor(
                self.pdf_extractor.asset_dir, self.cache, vision_client, review_vision_client, max_page_pixels,
                use_llm=use_llm, use_decision=use_decision, workers=self.workers,
            )
            source_root = Path(__file__).resolve().parent
            excluded_parts = set(_PIPELINE["code_snapshot_excluded_parts"])
            excluded_files = set(_PIPELINE["code_snapshot_excluded_files"])
            self._code_snapshot = {
                path.relative_to(source_root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted(source_root.rglob("*.py"))
                if not excluded_parts.intersection(path.parts) and path.name not in excluded_files
            }
        except BaseException:
            self.close()
            raise
        self._result = threading.local()
        self._doc_lock = threading.RLock()

    def fingerprint(self):
        return chunker_fingerprint(self, PIPELINE_VERSION)

    _resolve_source = staticmethod(resolve_document_source)

    def _extract_ir(self, source, name, fmt):
        if fmt == "pdf":
            return self.pdf_extractor.extract(source, name)
        if fmt == "xlsx":
            return self.xlsx_extractor.extract(source, name, self.excel_metadata.get(name, {}))
        return self.extractor.extract(source, name)

    def chunk_document(self, docx_path=None, docx_bytes=None, source_name=None, expected_format=None):
        with self._doc_lock:
            self.last_artifact = None
            source_root = getattr(self.config, "CHUNKING_INPUT_DIR", MODULE / "input")
            source, name, fmt = self._resolve_source(
                docx_path, docx_bytes, source_name, expected_format, source_root=source_root,
            )
            from agent_banking.tracing.tracer import BankingTracer, LangSmithTraceRun

            trace_config = self.config
            if trace_config is None:
                from agent_banking.tracing.config.settings import TracingConfig
                trace_config = TracingConfig()
            doc_tracer = BankingTracer(config=trace_config)
            trace_dir = None
            if doc_tracer.should_export_local():
                trace_dir = doc_tracer.trace_dir
            with LangSmithTraceRun(trace_dir, "document", name, tracer=doc_tracer).capture() as run:
                run.details["format"] = fmt
                chunks = self._process_ir(self._extract_ir(source, name, fmt))
                run.details["chunks"] = len(chunks)
                return chunks

    @property
    def last_artifact(self):
        return getattr(self._result, "artifact", None)

    @last_artifact.setter
    def last_artifact(self, value):
        self._result.artifact = value

    def chunk_pdf(self, pdf_path=None, pdf_bytes=None, source_name=None):
        return self.chunk_document(docx_path=pdf_path, docx_bytes=pdf_bytes, source_name=source_name, expected_format="pdf")

    def chunk_xlsx(self, xlsx_path=None, xlsx_bytes=None, source_name=None):
        return self.chunk_document(docx_path=xlsx_path, docx_bytes=xlsx_bytes, source_name=source_name, expected_format="xlsx")

    def _process_ir(self, ir):
        from .text.language import classify_text_blocks

        ir["profile"] = _PIPELINE["profile"]
        ir["search_language"] = self.planner.search_language
        ir.setdefault("visual_policy", self.image_mode)
        classify_text_blocks(ir["blocks"], search_language=ir["search_language"])
        if any(warning.get("requires_review") for warning in ir["warnings"]):
            raise RuntimeError("Unsupported or unresolved source content; publication refused: " + json.dumps(ir["warnings"]))
        excel = ir.get("format") == "xlsx"
        if not excel:
            enricher = ModelEnricher(self.cache, workers=self.workers, use_llm=self.use_llm, use_decision=self.use_decision)
            ir = enricher.enrich(ir)
            if self.semantic and self.use_llm:
                from .planning.semantic_boundaries import SemanticBoundaryScorer
                SemanticBoundaryScorer(self.cache, self.planner.budget, self.workers).annotate(ir)
        chunks, artifact = self.planner.plan(ir)
        has_searchable_content = any(
            chunk["business_metadata"]["retrieval"].get("role") != "attachment_only" for chunk in chunks
        )
        if not excel and self.use_decision and has_searchable_content:
            ir["deterministic_coverage"] = artifact["quality"].copy()
            enricher.review(ir)
        decision_review = ir.get("decision_review", {})
        decision_route = decision_review.get("answers", {}).get("review_route", {}).get("choice")
        artifact["pipeline_version"] = PIPELINE_VERSION
        artifact["pipeline_fingerprint"] = self.fingerprint()
        ready = all(block["image"].get("status") in RESOLVED_IMAGE_STATUSES
                    for block in ir["blocks"] if block["type"] == "image")
        statistics = ir.get("statistics", {})
        ready = ready and (excel or self.use_llm and self.semantic)
        if excel:
            artifact["quality"].update(statistics)
            artifact["quality"]["excel_issues_detail"] = ir["excel_issues"]
        if decision_route == "REPAIR_EXTRACTION" and decision_review.get("status") != "advisory":
            ready = False
        artifact["quality"]["empty_searchable_content"] = not has_searchable_content
        artifact["quality"]["index_ready"] = bool(ready and not any(
            warning.get("requires_review") for warning in ir["warnings"]
        ))
        artifact["quality"]["segmentation_method"] = ir.get("segmentation_method", "token_lattice_diagnostic")
        artifact["quality"]["semantic_rejected_proposals"] = ir.get("semantic_rejected_proposals", 0)
        default_visual_policy = self.image_mode
        if ir.get("page_reviews"):
            default_visual_policy = "reviewed_pdf_pages"
        artifact["quality"]["visual_policy"] = ir.get("visual_policy", default_visual_policy)
        artifact["quality"]["reviewed_pages"] = statistics.get("ocr_pages_processed", 0)
        if "pages" in statistics:
            artifact["quality"]["source_pages"] = statistics["pages"]
            artifact["quality"]["pages_requiring_ocr"] = statistics.get("pages_requiring_ocr", 0)
            artifact["quality"]["ocr_pages_processed"] = statistics.get("ocr_pages_processed", 0)
            artifact["quality"]["ocr_partial_pages"] = statistics.get("ocr_partial_pages", 0)
            artifact["quality"]["ocr_failed_pages"] = statistics.get("ocr_failed_pages", 0)
            artifact["quality"]["ocr_failed_page_numbers"] = statistics.get("ocr_failed_page_numbers", [])
            artifact["quality"]["pdf_processing_mode"] = statistics.get("pdf_processing_mode")
        artifact["quality"]["model_warning_count"] = len(ir.get("model_warnings", []))
        artifact["quality"]["page_region_reviews"] = sum(len(review.get("region_audit", [])) for review in ir.get("page_reviews", []))
        artifact["quality"]["font_reconciled_lines"] = sum(len(review.get("native_reconciliation", [])) for review in ir.get("page_reviews", []))
        for chunk in chunks:
            chunk["business_metadata"]["pipeline_fingerprint"] = artifact["pipeline_fingerprint"]
            quality = chunk["business_metadata"]["quality"]
            quality["index_ready"] = artifact["quality"]["index_ready"]
            quality["decision_route"] = decision_route
            if "pages" in statistics:
                quality["ocr_partial_pages"] = artifact["quality"]["ocr_partial_pages"]
                quality["ocr_failed_page_numbers"] = artifact["quality"]["ocr_failed_page_numbers"]
        self.last_artifact = artifact
        return chunks

    def chunk_docx(self, docx_path=None, **kwargs):
        return self.chunk_document(docx_path=docx_path, expected_format="docx", **kwargs)

    def chunk_file(self, file_path, **kwargs):
        return self.chunk_document(docx_path=file_path, **kwargs)

    @property
    def docx_chunker(self):
        return self

    def chunk_directory(self, dir_path, extensions=None, max_files=None, *, source_root=None):
        target_dir = Path(dir_path)
        if not target_dir.exists():
            return []
        target_dir = target_dir.resolve()
        if source_root is None:
            source_root = document_source_root(target_dir, getattr(self.config, "CHUNKING_INPUT_DIR", MODULE / "input"))
        else:
            source_root = Path(source_root).resolve()
        if not target_dir.is_relative_to(source_root):
            raise ValueError("Document directory must be inside its source root")
        files = discover_documents(target_dir, extensions, reject_unsupported=extensions is None)
        if max_files is not None:
            if type(max_files) is not int or max_files < 0:
                raise ValueError("max_files must be a nonnegative integer")
            files = files[:max_files]
        all_chunks = []
        for path in files:
            chunks = self.chunk_file(path, source_name=path.relative_to(source_root).as_posix())
            all_chunks.extend(chunks)
        return all_chunks

    def chunk_dataset(self, dataset_dir, output_dir=None):
        start_t = time.time()
        dataset_path = Path(dataset_dir)
        all_chunks = self.chunk_directory(dataset_path)
        elapsed = round(time.time() - start_t, 2)
        stats = {
            "total_chunks": len(all_chunks),
            "elapsed_seconds": elapsed,
            "speed_chunks_per_sec": round(len(all_chunks) / max(elapsed, _PIPELINE["speed_min_elapsed"]), 1),
        }
        if output_dir:
            out_p = Path(output_dir)
            out_p.mkdir(parents=True, exist_ok=True)
            with open(out_p / "chunks.json", "w", encoding="utf-8") as f:
                json.dump(all_chunks, f, ensure_ascii=False, indent=2)
        return all_chunks, stats

    def close(self):
        if self._owns_cache:
            self.cache.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# Historical class names remain supported by all current callers.
UniversalDocumentChunker = UniversalDocxChunker
LayoutAwareChunker = UniversalDocxChunker
DLAChunker = UniversalDocxChunker
