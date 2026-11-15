"""Extract PDF documents into the chunking intermediate representation."""
from collections import Counter
from copy import deepcopy
import hashlib
import math
from pathlib import Path

from .layout import (
    compact as compact,
    has_meaningful_layout as has_meaningful_layout,
    inside_box as inside_box,
    is_inline_list_marker as is_inline_list_marker,
    layout_text as layout_text,
    merge_region_observation as merge_region_observation,
    normalized_box as normalized_box,
    reading_order as reading_order,
    region_overlap as region_overlap,
    region_signature as region_signature,
    validate_page as validate_page,
    validate_scan_pixels as validate_scan_pixels,
)
from .native_text import (
    furniture_key as furniture_key,
    furniture_signature as furniture_signature,
    is_furniture as is_furniture,
    native_page,
    native_page_number as native_page_number,
    native_table_elements as native_table_elements,
    native_text_elements as native_text_elements,
    normalized_page_box as normalized_page_box,
    page_requires_scan as page_requires_scan,
)
from .table_review import numeric_signature as numeric_signature
from .tables import table_from_html as table_from_html, validate_table as validate_table
from .page_review import (
    PDF_PROMPT_VERSION as PDF_PROMPT_VERSION,
    PDF_SCHEMA_VERSION as PDF_SCHEMA_VERSION,
    SCAN_PAGE_PROMPT as SCAN_PAGE_PROMPT,
    PdfVisualReviewer,
)


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
            scan_pages = set()
            pages_without_native_text = 0
            margin_counts = Counter()
            for page_index, page in enumerate(document):
                native = native_text_elements(page)
                pages_without_native_text += not native
                if page_requires_scan(page, native):
                    scan_pages.add(page_index)
                margin_counts.update({
                    furniture_signature(element, key) for element in native if (key := furniture_key(element)) is not None
                })
            repeated_margins = {key for key, count in margin_counts.items() if count >= 2}
            ir["extraction_method"] = "pdf_native_text"
            ir["visual_policy"] = "native_pdf_no_image_ocr"
            if scan_pages:
                ir["extraction_method"] = "pdf_cloud_vision_reviewed"
                ir["visual_policy"] = "pdf_scan_all_pages"
            for page_index in range(len(document)):
                page = document[page_index]
                elements = native_page(page, repeated_margins)
                sha = None
                if scan_pages:
                    # Native glyphs provide witnesses even on a mixed page.
                    # Lines already known to have a broken character map must
                    # be read from pixels, rather than forced into OCR output.
                    witnesses = [line for line in layout_text({"elements": elements}).splitlines() if "\ufffd" not in line]
                    elements, sha = self._scan_page(
                        ir, page, page_index, witnesses, allow_ocr_additions=page_index in scan_pages,
                    )
                for element in elements:
                    self._append(ir, element, page_index, sha, page.rotation)
            if self.use_decision:
                from ...enrichment.decisions.engine import decide_continued_tables
                decide_continued_tables(ir, cache=self.cache)
            self._infer_pdf_visual_typography(ir)
            ir["statistics"] = {"pages": len(document), "blocks": len(ir["blocks"]), "units": len(ir["units"]),
                                  "pages_without_native_text": pages_without_native_text,
                                  "pages_requiring_ocr": len(scan_pages), "ocr_pages_processed": len(ir["page_reviews"]),
                                  "scan_page_numbers": [index + 1 for index in sorted(scan_pages)],
                                  "pdf_processing_mode": ir["visual_policy"],
                                  "tables": sum(block["type"] == "table" for block in ir["blocks"])}
        return ir

    def _scan_page(self, ir, page, page_index, native, *, allow_ocr_additions=False):
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
        reviewer = PdfVisualReviewer(self.asset_dir, self.cache, self.vision_client, self.review_client, self.use_decision)
        layout = reviewer.read(pixels, native, pixmap.width, pixmap.height, allow_ocr_additions=allow_ocr_additions)
        content = "text"
        if not has_meaningful_layout(layout):
            content = "non_text_only" if layout["review"].get("non_text_only") is True else "blank"
        ir["page_reviews"].append({"page_index": page_index, **layout["visual_review"],
                                   "content": content, "native_lines_checked": len(native),
                                   "region_audit": layout.get("region_audit", [])})
        return layout["elements"], sha

    def _infer_pdf_visual_typography(self, ir):
        """Populate missing block typography from bbox line-height and run shared hierarchy & caption inference."""
        from ..docx.extractor import apply_typography_hierarchy, build_typography_record, link_visual_captions
        for block in ir["blocks"]:
            if block["type"] == "table" or block.get("typography") is not None:
                continue
            text = block.get("text", "").strip()
            if not text:
                continue
            prov = block.get("provenance", {})
            if prov.get("bbox_granularity") == "page":
                continue
            bbox = prov.get("bbox", [0, 0, 1, 0.02])
            lines = [ln for ln in text.splitlines() if ln.strip()]
            line_count = len(lines)
            if line_count < 1:
                line_count = 1
            raw_lh = ((bbox[3] - bbox[1]) / line_count) * 1000.0
            bucket = round(raw_lh * 2.0) / 2.0
            ch_len = len(text)
            center_x = (bbox[0] + bbox[2]) / 2.0
            centered = 0.40 <= center_x <= 0.60 and (bbox[2] - bbox[0]) < 0.85
            letters = [ch for ch in text if ch.isalpha()]
            is_upper = False
            if letters:
                upper_count = sum(1 for ch in letters if ch.isupper())
                if upper_count * 5 >= len(letters) * 4:
                    is_upper = True
            all_bold = block["type"] == "heading" or is_upper
            starts_bold = ":" in text[:35] and not is_upper
            block["typography"] = build_typography_record(
                text, bucket, ch_len, {bucket: ch_len}, all_bold, False, starts_bold, centered
            )
        if self.use_decision:
            apply_typography_hierarchy(ir, cache=self.cache)
            link_visual_captions(ir, cache=self.cache)

    def _append(self, ir, element, page, sha, rotation):
        identifier = ir["document_id"] + ":v" + ir["document_version"][:16] + ":b" + str(len(ir["blocks"]))
        kind = element["type"]
        block = {"id": identifier, "type": kind, "text": element.get("text", ""), "order": len(ir["blocks"]),
                 "provenance": {"page_index": page, "page_number": page + 1, "region": "body",
                                "bbox": element["bbox"], "coordinate_system": "normalized_rendered_page",
                                "bbox_granularity": element.get("bbox_granularity", "region"),
                                "order_evidence": element.get("order_evidence", "visual_reading_order"),
                                "rotation": rotation,
                                "method": ir.get("extraction_method", "pdf_cloud_vision_reviewed")}}
        if sha:
            block["provenance"]["page_asset_sha256"] = sha
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
            table["header_detection"] = "native_grid"
            if sha:
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
