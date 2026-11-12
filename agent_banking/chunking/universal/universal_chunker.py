"""Grounded PDF/DOCX ingestion, cloud semantics and exact passage budgets."""
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

PIPELINE_VERSION = "universal-pdf-docx-5.0-benchmarked-vision"
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
                 max_chunk_chars=None, min_chunk_words=30, min_section_chars=0, review_vision_client=None,
                 semantic=True, image_mode="vision", max_pdf_pages=2000, max_page_pixels=40_000_000):
        self.profile = profile or os.getenv("CHUNKING_PROFILE", "accuracy")
        if self.profile not in ("native", "accuracy"):
            raise ValueError("Chunking profile must be native or accuracy")
        self.dpi = dpi
        self.max_chunk_words = max_chunk_words
        self.workers = workers
        self.use_llm = use_llm
        self.use_decision = use_decision
        if image_mode not in ("vision", "text_context"):
            raise ValueError("image_mode must be vision or text_context")
        self.semantic = semantic
        self.image_mode = image_mode
        if dpi <= 0:
            raise ValueError("DPI must be positive")
        self._owns_cache = cache is None
        self.cache = cache
        if self.cache is None:
            self.cache = PageVisionCache(str(cache_db_path or MODULE / "cache/grounded_models.db"))
        self.planner = ChunkPlanner(target_chunk_tokens, max_chunk_tokens, max_chunk_words,
                                    table_rows_per_chunk, tokenizer, max_chunk_chars, min_chunk_words, min_section_chars)
        self.extractor = DocxExtractor(asset_dir or MODULE / "output/assets",
                                       render_figures=self.profile == "accuracy" and image_mode == "vision")
        if vision_client is None and any(value is not None for value in (base_url, api_key, model)):
            from .vision_client import QwenVisionClient
            vision_client = QwenVisionClient(base_url=base_url, api_key=api_key, model=model)
        from .pdf_extractor import PdfExtractor
        self.pdf_extractor = PdfExtractor(self.extractor.asset_dir, self.cache, vision_client, review_vision_client, dpi, max_pdf_pages, max_page_pixels, use_decision)
        self.last_artifact = None
        self._code_snapshot = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                               for path in sorted(Path(__file__).resolve().parent.glob("*.py"))}

    def fingerprint(self):
        configuration = {"pipeline": PIPELINE_VERSION, "profile": self.profile, "prompt": PROMPT_VERSION,
                         "semantic": self.semantic, "image_mode": self.image_mode,
                         "dpi": self.dpi, "max_pdf_pages": self.pdf_extractor.max_pages,
                         "max_page_pixels": self.pdf_extractor.max_page_pixels,
                         "vision_client": getattr(self.pdf_extractor.vision_client, "model", None),
                         "review_client": getattr(self.pdf_extractor.review_client, "model", None),
                         "image_policy": IMAGE_POLICY, "max_words": self.max_chunk_words,
                         "target_tokens": self.planner.budget.target, "max_tokens": self.planner.budget.maximum,
                         "table_rows": self.planner.table_rows, "llm_enabled": self.use_llm,
                         "max_chars": self.planner.max_chars, "min_words": self.planner.min_words, "min_chars": self.planner.min_chars,
                         "decision_enabled": self.use_decision, "llm_model": os.getenv("MODEL"),
                         "decision_model": os.getenv("DECISION_MODEL"),
                         "llm_provider": os.getenv("BASE_URL"), "decision_provider": os.getenv("DECISION_BASE_URL"),
                         "llm_max_tokens": os.getenv("CHUNKING_LLM_MAX_TOKENS", "65536"),
                         "dependencies": {package: version(package) for package in ("python-docx", "tokenizers", "PyMuPDF", "Pillow")}}
        # Code changes invalidate document outputs even if the version constant
        # was accidentally left unchanged. Secrets are never part of manifests.
        from agent_banking.core.models import chain_fingerprint
        configuration["model_chains"] = {name: chain_fingerprint("chunking." + name)
                                          for name in ("text", "decision", "vision", "vision_review")}
        for role, client in (("extraction", self.pdf_extractor.vision_client), ("review", self.pdf_extractor.review_client)):
            if callable(getattr(client, "identity", None)):
                configuration[role + "_identity"] = client.identity()
        configuration["sources"] = self._code_snapshot
        return hashlib.sha256(json.dumps(configuration, sort_keys=True).encode("utf-8")).hexdigest()

    def _resolve_source(self, source=None, source_bytes=None, source_name=None, expected_format=None):
        if isinstance(source, (bytes, bytearray)):
            source_bytes = bytes(source)
            source = None
        if source_bytes is not None:
            if not isinstance(source_bytes, (bytes, bytearray)):
                raise TypeError("source_bytes must be bytes or bytearray")
            if not source_bytes:
                raise ValueError("source_bytes is empty")
            fmt = expected_format
            if fmt is None and source_name is not None:
                ext = Path(str(source_name)).suffix.lower()
                if ext in (".docx", ".pdf"):
                    fmt = ext[1:]
            if fmt is None:
                if bytes(source_bytes[:4]) == b"%PDF":
                    fmt = "pdf"
                else:
                    fmt = "docx"
            name = source_name
            if name is None:
                name = "in_memory_" + hashlib.sha256(source_bytes).hexdigest()[:20] + "." + fmt
            return bytes(source_bytes), str(name), fmt
        if not isinstance(source, (str, Path)):
            if source is None:
                raise ValueError("A document path or byte stream is required")
            raise TypeError("source must be a string or Path")
        if not str(source).strip():
            raise ValueError("source path is empty")
        path = Path(source)
        if not path.is_file():
            raise FileNotFoundError(str(path))
        ext = path.suffix.lower()
        if ext not in (".docx", ".pdf"):
            raise ValueError("Unsupported document format: " + ext)
        fmt = ext[1:]
        if expected_format is not None and fmt != expected_format:
            raise ValueError(expected_format.upper() + " adapter requires ." + expected_format + " input")
        if fmt == "docx" and source_name is None:
            name = str(path.resolve())
        else:
            name = source_name or str(path)
        return str(path), str(name), fmt

    def _extract_ir(self, source, name, fmt):
        if fmt == "pdf":
            return self.pdf_extractor.extract(source, name, self.profile)
        ir = self.extractor.extract(source, name)
        if self.profile == "accuracy" and self.image_mode == "vision":
            self._read_office_figures(ir)
        return ir

    def chunk_document(self, docx_path=None, docx_bytes=None, source_name=None, expected_format=None):
        self.last_artifact = None
        source, name, fmt = self._resolve_source(docx_path, docx_bytes, source_name, expected_format)
        return self._process_ir(self._extract_ir(source, name, fmt))

    def chunk_pdf(self, pdf_path=None, pdf_bytes=None, source_name=None):
        return self.chunk_document(docx_path=pdf_path, docx_bytes=pdf_bytes, source_name=source_name, expected_format="pdf")

    def _read_office_figures(self, ir):
        """Use the same reviewed visual adapter for embedded Office figures."""
        from PIL import Image
        import io
        from copy import deepcopy
        replacement = []
        for block in ir["blocks"]:
            if block["type"] != "image":
                replacement.append(block)
                continue
            info = block["image"]
            path = Path(info.get("visual_path", info["path"]))
            with Image.open(path) as original:
                buffer = io.BytesIO()
                original.convert("RGB").save(buffer, "PNG")
                width, height = original.size
            layout = self.pdf_extractor._read(buffer.getvalue(), [], width, height)
            block["type"] = "paragraph"
            block["text"] = info.get("native_text", "")
            block["visual_review"] = layout["visual_review"]
            replacement.append(block)
            for index, element in enumerate(layout["elements"]):
                temporary = {**ir, "blocks": [], "units": []}
                self.pdf_extractor._append(temporary, element, 0, info["sha256"], 0)
                child = temporary["blocks"][0]
                child["id"] = block["id"] + ":visual" + str(index)
                child["provenance"] = {**deepcopy(block["provenance"]), "method": "office_cloud_vision_reviewed",
                    "figure_asset_sha256": info["sha256"], "figure_bbox": element["bbox"],
                    "coordinate_system": "normalized_rendered_figure"}
                if child["type"] == "table":
                    child["table_data"]["table_id"] = child["id"]
                for unit in temporary["units"]:
                    unit["block_id"] = child["id"]
                    unit["id"] = child["id"] + ":u" + str(len(ir["units"]))
                    ir["units"].append(unit)
                replacement.append(child)
        ir["blocks"] = replacement
        ir["visual_policy"] = "reviewed_pixels_v1"

    def _process_ir(self, ir):
        ir["profile"] = self.profile
        if self.profile == "accuracy" and any(warning.get("requires_review") for warning in ir["warnings"]):
            raise RuntimeError("Unsupported or unresolved source content; publication refused: " + json.dumps(ir["warnings"]))
        if self.profile == "accuracy":
            enricher = ModelEnricher(self.cache, workers=self.workers, use_llm=self.use_llm, use_decision=self.use_decision,
                                     coverage_validator=lambda document: self.planner.plan(document)[1]["quality"])
            ir = enricher.enrich(ir)
            if self.semantic and self.use_llm:
                from .semantic_boundaries import SemanticBoundaryScorer
                SemanticBoundaryScorer(self.cache, self.planner.budget, self.workers).annotate(ir)
        else:
            mark_unassessed_images(ir)
        chunks, artifact = self.planner.plan(ir)
        artifact["pipeline_version"] = PIPELINE_VERSION
        artifact["pipeline_fingerprint"] = self.fingerprint()
        resolved = all(block["image"].get("status") in RESOLVED_IMAGE_STATUSES for block in ir["blocks"] if block["type"] == "image")
        ready = not ir["assets"] or self.profile == "accuracy" and resolved
        if ir.get("extraction_method") == "pdf_cloud_vision_reviewed":
            ready = self.profile == "accuracy" and len(ir["page_reviews"]) == ir["statistics"]["pages"]
        if self.profile == "accuracy":
            ready = ready and self.use_llm and self.semantic
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
            pages_without_text = statistics.get("pages_without_native_text", 0)
            artifact["quality"]["pages_without_native_text"] = pages_without_text
            if self.profile == "native":
                artifact["quality"]["pages_requiring_ocr"] = pages_without_text
            else:
                artifact["quality"]["pages_requiring_ocr"] = 0
        artifact["quality"]["page_region_reviews"] = sum(len(review.get("region_audit", [])) for review in ir.get("page_reviews", []))
        artifact["quality"]["font_reconciled_lines"] = sum(len(review.get("native_reconciliation", [])) for review in ir.get("page_reviews", []))
        artifact["quality"]["page_level_locations"] = sum(block["provenance"].get("bbox_granularity") == "page" for block in ir["blocks"])
        for chunk in chunks:
            chunk["business_metadata"]["pipeline_fingerprint"] = artifact["pipeline_fingerprint"]
            chunk["business_metadata"]["quality"]["index_ready"] = artifact["quality"]["index_ready"]
        self.last_artifact = artifact
        return chunks

    def chunk_docx(self, docx_path, **kwargs):
        return self.chunk_document(docx_path=docx_path, expected_format="docx", **kwargs)

    def chunk_file(self, file_path, **kwargs):
        return self.chunk_document(docx_path=file_path, **kwargs)

    def close(self):
        if self._owns_cache:
            self.cache.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# Historical class name remains supported by all current callers.
UniversalDocumentChunker = UniversalDocxChunker
