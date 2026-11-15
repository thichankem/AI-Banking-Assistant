"""Extract PDF documents into the chunking intermediate representation."""
from copy import deepcopy
import hashlib
import math
from pathlib import Path

from .layout import has_meaningful_layout
from .page_review import CloudOCRUnavailable, PdfVisualReviewer


class PdfExtractor:
    def __init__(self, asset_dir, cache, vision_client=None, review_client=None, dpi=200, max_pages=2000,
                 max_page_pixels=40_000_000, use_decision=True, *, search_language="vi"):
        if isinstance(dpi, bool) or not isinstance(dpi, int) or not 72 <= dpi <= 600 or isinstance(max_pages, bool) or not isinstance(max_pages, int) or max_pages < 1:
            raise ValueError("PDF limits require 72..600 DPI and positive max_pages")
        self.asset_dir = Path(asset_dir)
        self.cache = cache
        self.vision_client = vision_client
        self.review_client = review_client
        self.dpi = dpi
        self.max_pages = max_pages
        if isinstance(max_page_pixels, bool) or not isinstance(max_page_pixels, int) or max_page_pixels < 1:
            raise ValueError("PDF pixel limit must be positive")
        self.max_page_pixels = max_page_pixels
        self.use_decision = use_decision
        self.search_language = search_language

    def extract(self, source, source_name):
        import pymupdf
        data = source
        if not isinstance(data, (bytes, bytearray)):
            data = Path(source).read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        identity = str(source_name).replace("\\", "/")
        document_id = hashlib.sha256(identity.encode()).hexdigest()[:20]
        ir = {"version": "2.0", "document_id": document_id, "document_version": digest,
              "source": identity, "blocks": [], "units": [], "assets": {}, "warnings": [], "relations": [],
              "profile": "accuracy", "page_reviews": [], "search_language": self.search_language}
        with pymupdf.open(stream=data, filetype="pdf") as document:
            if document.needs_pass:
                raise ValueError("Encrypted PDF requires decryption before ingestion")
            if len(document) > self.max_pages:
                raise ValueError("PDF exceeds configured page limit; no partial ingestion")
            ir["extraction_method"] = "pdf_cloud_ocr"
            ir["visual_policy"] = "pdf_cloud_ocr"
            failed_pages = []
            for page_index, page in enumerate(document):
                try:
                    elements, sha = self._scan_page(ir, page, page_index)
                except CloudOCRUnavailable as error:
                    error.pdf_page_number = page_index + 1
                    failed_pages.append(error)
                    ir["page_reviews"].append({
                        "page_index": page_index, "method": "pdf_cloud_ocr",
                        "content": "unavailable", "error": str(error),
                        "render_sha256": getattr(error, "pdf_render_sha256", None),
                    })
                    continue
                except (OSError, RuntimeError, ValueError) as error:
                    error.pdf_page_number = page_index + 1
                    raise
                for element in elements:
                    self._append(ir, element, page_index, sha, page.rotation)
            ocr_statistics = {
                "source_pages": len(document),
                "pages_requiring_ocr": len(document),
                "ocr_pages_processed": len(document) - len(failed_pages),
                "ocr_partial_pages": sum(review.get("incomplete") is True for review in ir["page_reviews"]),
                "ocr_failed_pages": len(failed_pages),
                "ocr_failed_page_numbers": [error.pdf_page_number for error in failed_pages],
            }
            if failed_pages and not ir["blocks"]:
                failed_pages[0].pdf_ocr_statistics = ocr_statistics
                raise failed_pages[0]
            if self.use_decision:
                from ...enrichment.decisions.engine import decide_continued_tables
                from ..docx.extractor import link_visual_captions
                decide_continued_tables(ir, cache=self.cache)
                link_visual_captions(ir, cache=self.cache)
            ir["statistics"] = {
                "pages": len(document), "blocks": len(ir["blocks"]), "units": len(ir["units"]),
                **ocr_statistics,
                "pdf_processing_mode": ir["visual_policy"],
                "tables": sum(block["type"] == "table" for block in ir["blocks"]),
            }
        return ir

    def _scan_page(self, ir, page, page_index):
        import pymupdf
        expected_pixels = math.ceil(page.rect.width * self.dpi / 72) * math.ceil(page.rect.height * self.dpi / 72)
        if expected_pixels > self.max_page_pixels:
            raise ValueError("PDF page exceeds configured render pixel limit; no partial ingestion")
        pixmap = page.get_pixmap(dpi=self.dpi, colorspace=pymupdf.csRGB, alpha=False, annots=False)
        pixels = pixmap.tobytes("png")
        sha = hashlib.sha256(pixels).hexdigest()
        path = self.asset_dir / (sha + ".png")
        if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != sha:
            from agent_banking.chunking.publication_io import atomic_bytes
            atomic_bytes(path, pixels)
        asset = ir["assets"].setdefault(sha, {
            "path": str(path.resolve()), "sha256": sha, "kind": "page", "page_index": page_index,
            "page_indices": [], "width": pixmap.width, "height": pixmap.height,
        })
        asset["page_indices"].append(page_index)
        try:
            reader = PdfVisualReviewer(self.asset_dir, self.cache, self.vision_client, self.review_client, self.use_decision)
            layout = reader.read(pixels, [], pixmap.width, pixmap.height)
        except (OSError, RuntimeError, ValueError) as error:
            error.pdf_render_sha256 = sha
            raise
        content = "text"
        if not has_meaningful_layout(layout):
            content = "non_text_only" if layout["review"].get("non_text_only") is True else "blank"
        ir["page_reviews"].append({"page_index": page_index, **layout["visual_review"], "method": "pdf_cloud_ocr",
                                   "content": content, "render_sha256": sha,
                                   "region_audit": layout.get("region_audit", [])})
        return layout["elements"], sha

    def _append(self, ir, element, page, sha, rotation):
        identifier = ir["document_id"] + ":v" + ir["document_version"][:16] + ":b" + str(len(ir["blocks"]))
        kind = element["type"]
        block = {"id": identifier, "type": kind, "text": element.get("text", ""), "order": len(ir["blocks"]),
                 "provenance": {"page_index": page, "page_number": page + 1, "region": "body",
                                "bbox": element["bbox"], "coordinate_system": "normalized_rendered_page",
                                "bbox_granularity": element.get("bbox_granularity", "region"),
                                "order_evidence": element.get("order_evidence", "visual_reading_order"),
                                "rotation": rotation,
                                "method": "pdf_cloud_ocr", "page_asset_sha256": sha}}
        if element.get("typography"):
            block["typography"] = deepcopy(element["typography"])
        if element.get("representation"):
            block["provenance"]["representation"] = element["representation"]
            block["provenance"]["region_pixel_box"] = element["region_pixel_box"]
            block["provenance"]["region_pixel_path"] = element.get("region_pixel_path", [element["region_pixel_box"]])
        if kind == "heading":
            block["level"] = element["level"]
        if kind == "table":
            table = deepcopy(element["table_data"])
            table["table_id"] = identifier
            table["grid"] = table["header_grid"] + table["rows"]
            table["header_detection"] = "visual_model_review"
            block["table_data"] = table
            if element.get("table_schema_repair"):
                block["table_schema_repair"] = deepcopy(element["table_schema_repair"])
            for row_index, row in enumerate(table["grid"]):
                for column, value in enumerate(row):
                    self._unit(ir, block, value, {"row": row_index, "column": column})
        else:
            self._unit(ir, block, block["text"], {})
        ir["blocks"].append(block)

    def _unit(self, ir, block, text, location):
        if text.strip():
            ir["units"].append({"id": block["id"] + ":u" + str(len(ir["units"])),
                                "block_id": block["id"], "text": text, **location})
