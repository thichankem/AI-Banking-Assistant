"""Grounded PDF/DOCX ingestion, cloud semantics and exact passage budgets."""
import hashlib
import json
import threading
import time
from pathlib import Path

from .vision_cache import PageVisionCache
from .document_ir import DocxExtractor
from .chunk_planner import ChunkPlanner
from .document_sources import (
    discover_documents as discover_documents,
    document_source_name as document_source_name,
    resolve_document_source,
)
from .pipeline_fingerprint import chunker_fingerprint
from .model_enrichment import ModelEnricher
from .image_context import RESOLVED_IMAGE_STATUSES
from .schema_assembler import (
    build_universal_chunk, link_chunk_sequence, infer_business_metadata,
    generate_chunk_id, REQUIRED_UNIVERSAL_SCHEMA_KEYS, VALID_CHUNK_TYPES,
)

PIPELINE_VERSION = "universal-pdf-docx-6.1"
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
                 dpi=200, max_chunk_words=None, cache=None, vision_client=None,
                 target_chunk_tokens=512, max_chunk_tokens=1024,
                 table_rows_per_chunk=4, asset_dir=None, workers=2,
                 use_llm=True, use_decision=True, tokenizer=None,
                 max_chunk_chars=None, min_chunk_words=30, min_section_chars=0, review_vision_client=None,
                 semantic=True, image_mode="text_context", max_pdf_pages=2000, max_page_pixels=40_000_000,
                 max_chunk_size=None, min_section_size=None, config=None, min_chunk_tokens=128):
        if max_chunk_chars is None and max_chunk_size is not None:
            max_chunk_chars = max_chunk_size
        if min_section_size is not None and min_section_chars == 0:
            min_section_chars = min_section_size
        self.config = config
        self.dpi = dpi
        self.max_chunk_words = max_chunk_words
        if type(workers) is not int or workers < 1:
            raise ValueError("workers must be a positive integer")
        self.workers = min(workers, 2)
        self.use_llm = use_llm
        self.use_decision = use_decision
        if image_mode not in ("vision", "text_context"):
            raise ValueError("image_mode must be vision or text_context")
        self.semantic = semantic
        self.image_mode = "ignore_standalone_images"
        from .pdf_extractor import PdfExtractor

        if vision_client is None and any(value is not None for value in (base_url, api_key, model)):
            from .vision_client import DocumentVisionClient
            vision_client = DocumentVisionClient(base_url=base_url, api_key=api_key, model=model)
        self._owns_cache = cache is None
        self.cache = cache
        if self.cache is None:
            self.cache = PageVisionCache(str(cache_db_path or MODULE / "cache/grounded_models.db"))
        try:
            self.pdf_extractor = PdfExtractor(
                asset_dir or MODULE / "output/assets", self.cache, vision_client, review_vision_client,
                dpi, max_pdf_pages, max_page_pixels, use_decision,
            )
            self.planner = ChunkPlanner(
                target_chunk_tokens, max_chunk_tokens, max_chunk_words, table_rows_per_chunk, tokenizer,
                max_chunk_chars, min_chunk_words, min_section_chars, use_decision=use_decision,
                decision_cache=self.cache, min_tokens=min_chunk_tokens,
            )
            self.extractor = DocxExtractor(
                self.pdf_extractor.asset_dir, use_decision=use_decision, decision_cache=self.cache,
            )
            self._code_snapshot = {
                path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted(Path(__file__).resolve().parent.glob("*.py"))
            }
        except BaseException:
            self.close()
            raise
        self._result = threading.local()

    def fingerprint(self):
        return chunker_fingerprint(self, PIPELINE_VERSION)

    _resolve_source = staticmethod(resolve_document_source)

    def _extract_ir(self, source, name, fmt):
        if fmt == "pdf":
            return self.pdf_extractor.extract(source, name)
        return self.extractor.extract(source, name)

    def chunk_document(self, docx_path=None, docx_bytes=None, source_name=None, expected_format=None):
        with _DOCUMENT_LOCK:
            self.last_artifact = None
            source, name, fmt = self._resolve_source(docx_path, docx_bytes, source_name, expected_format)
            return self._process_ir(self._extract_ir(source, name, fmt))

    @property
    def last_artifact(self):
        return getattr(self._result, "artifact", None)

    @last_artifact.setter
    def last_artifact(self, value):
        self._result.artifact = value

    def chunk_pdf(self, pdf_path=None, pdf_bytes=None, source_name=None):
        return self.chunk_document(docx_path=pdf_path, docx_bytes=pdf_bytes, source_name=source_name, expected_format="pdf")

    def _process_ir(self, ir):
        from .language import classify_text_blocks

        ir["profile"] = "accuracy"
        ir.setdefault("visual_policy", self.image_mode)
        classify_text_blocks(ir["blocks"])
        if any(warning.get("requires_review") for warning in ir["warnings"]):
            raise RuntimeError("Unsupported or unresolved source content; publication refused: " + json.dumps(ir["warnings"]))
        enricher = ModelEnricher(self.cache, workers=self.workers, use_llm=self.use_llm, use_decision=self.use_decision)
        ir = enricher.enrich(ir)
        if self.semantic and self.use_llm:
            from .semantic_boundaries import SemanticBoundaryScorer
            SemanticBoundaryScorer(self.cache, self.planner.budget, self.workers).annotate(ir)
        chunks, artifact = self.planner.plan(ir)
        has_searchable_content = any(
            chunk["business_metadata"]["retrieval"].get("role") != "attachment_only" for chunk in chunks
        )
        if self.use_decision and has_searchable_content:
            ir["deterministic_coverage"] = artifact["quality"].copy()
            enricher.review(ir)
        decision_route = ir.get("decision_review", {}).get("answers", {}).get("review_route", {}).get("choice")
        artifact["pipeline_version"] = PIPELINE_VERSION
        artifact["pipeline_fingerprint"] = self.fingerprint()
        ready = all(block["image"].get("status") in RESOLVED_IMAGE_STATUSES
                    for block in ir["blocks"] if block["type"] == "image")
        if ir.get("extraction_method") == "pdf_cloud_vision_reviewed":
            ready = ready and len(ir["page_reviews"]) == ir["statistics"]["pages"]
        ready = ready and self.use_llm and self.semantic
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
        artifact["quality"]["reviewed_pages"] = len(ir.get("page_reviews", []))
        statistics = ir.get("statistics", {})
        if "pages" in statistics:
            artifact["quality"]["source_pages"] = statistics["pages"]
            artifact["quality"]["pages_without_native_text"] = statistics.get("pages_without_native_text", 0)
            artifact["quality"]["pages_requiring_ocr"] = statistics.get("pages_requiring_ocr", 0)
            artifact["quality"]["ocr_pages_processed"] = statistics.get("ocr_pages_processed", 0)
            artifact["quality"]["scan_page_numbers"] = statistics.get("scan_page_numbers", [])
            artifact["quality"]["pdf_processing_mode"] = statistics.get("pdf_processing_mode")
        artifact["quality"]["page_region_reviews"] = sum(len(review.get("region_audit", [])) for review in ir.get("page_reviews", []))
        artifact["quality"]["font_reconciled_lines"] = sum(len(review.get("native_reconciliation", [])) for review in ir.get("page_reviews", []))
        artifact["quality"]["page_level_locations"] = sum(block["provenance"].get("bbox_granularity") == "page" for block in ir["blocks"])
        for chunk in chunks:
            chunk["business_metadata"]["pipeline_fingerprint"] = artifact["pipeline_fingerprint"]
            chunk["business_metadata"]["quality"]["index_ready"] = artifact["quality"]["index_ready"]
            chunk["business_metadata"]["quality"]["decision_route"] = decision_route
        self.last_artifact = artifact
        return chunks

    def chunk_docx(self, docx_path=None, **kwargs):
        return self.chunk_document(docx_path=docx_path, expected_format="docx", **kwargs)

    def chunk_file(self, file_path, **kwargs):
        return self.chunk_document(docx_path=file_path, **kwargs)

    @property
    def docx_chunker(self):
        return self

    def chunk_directory(self, dir_path, extensions=None, max_files=None):
        target_dir = Path(dir_path)
        if not target_dir.exists():
            return []
        files = discover_documents(target_dir, extensions)
        if max_files is not None:
            if type(max_files) is not int or max_files < 0:
                raise ValueError("max_files must be a nonnegative integer")
            files = files[:max_files]
        all_chunks = []
        for f in files:
            chunks = self.chunk_file(f)
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
            "speed_chunks_per_sec": round(len(all_chunks) / max(elapsed, 0.001), 1),
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
