"""Extract the DOCX body into ordered blocks, source units and provenance."""
import hashlib
import io
from pathlib import Path

import docx
from docx.enum.style import WD_STYLE_TYPE
from docx.oxml import parse_xml
from docx.oxml.ns import qn

from agent_banking.chunking.config.settings import get_config_section
from ...enrichment.image_context import IMAGE_POLICY
from .typography import (
    MAX_HEADING_LEVEL,
    _authored_language,
    build_typography_record,
    extract_paragraph_typography,
    heading_level,
    hidden_run_checker,
    resolve_run_font_pt,
)
from .inline_content import paragraph_parts
from .tables import read_table
from .openxml import (
    EMBEDDED_CONTENT,
    MC,
    NONCONTENT,
    content_children,
    content_nodes,
    local_name,
    native_math,
)

_PIPELINE = get_config_section("pipeline")
_DOCX_CFG = get_config_section("extraction")["docx"]
_DEFAULT_SEARCH_LANG = get_config_section(None)["search_language"]


def apply_typography_hierarchy(ir, cache=None):
    """Shared decision-model hierarchy and caption detector for both DOCX and PDF IRs."""
    from ...enrichment.decisions.engine import decide_breadcrumb_hierarchy
    decide_breadcrumb_hierarchy(ir, cache=cache)


def link_visual_captions(ir, cache=None):
    """Link caption blocks to adjacent image or table blocks across both DOCX and PDF IRs."""
    from ...enrichment.decisions.engine import decide_caption_attachments
    decide_caption_attachments(ir, cache=cache)


class DocxExtractor:
    def __init__(self, asset_dir, render_figures=False, *, use_decision=True, decision_cache=None,
                 search_language=_DEFAULT_SEARCH_LANG, ocr_images=False, vision_client=None, review_client=None,
                 max_image_pixels=_DOCX_CFG["max_image_pixels"]):
        self.asset_dir = Path(asset_dir)
        self.use_decision = use_decision
        self.decision_cache = decision_cache
        self.search_language = search_language
        self.image_ocr = None
        if ocr_images:
            from .image_ocr import DocxImageOCR
            self.image_ocr = DocxImageOCR(self.asset_dir, decision_cache, vision_client, review_client, max_image_pixels)

    def extract(self, source, source_name):
        if isinstance(source, (bytes, bytearray)):
            data = bytes(source)
        else:
            data = Path(source).read_bytes()
        ir = _DocxReader(data, source_name, self.image_ocr).extract()
        if not ir["units"] and any(review.get("issues") for review in ir.get("docx_image_reviews", [])):
            from ..pdf.page_review import CloudOCRUnavailable
            raise CloudOCRUnavailable("Image-only DOCX has no usable cloud OCR content")
        ir["search_language"] = self.search_language
        if self.use_decision:
            apply_typography_hierarchy(ir, cache=self.decision_cache)
            link_visual_captions(ir, cache=self.decision_cache)
        ir["statistics"].update({
            "blocks": len(ir["blocks"]),
            "units": len(ir["units"]),
            "image_occurrences": sum(block["type"] == "image" for block in ir["blocks"]),
            "tables": sum(block["type"] == "table" for block in ir["blocks"]),
        })
        return ir


class _DocxReader:
    def __init__(self, data, source_name, image_ocr):
        self.image_ocr = image_ocr
        digest = hashlib.sha256(data).hexdigest()
        identity = str(source_name).replace("\\", "/")
        document_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:_DOCX_CFG["document_id_hex_len"]]
        self.document = docx.Document(io.BytesIO(data))
        self._hidden_run = hidden_run_checker(self.document)
        self._script_styles = {}
        hops_limit = _DOCX_CFG["style_inheritance_hops"]
        for style in self.document.styles:
            if style.type != WD_STYLE_TYPE.CHARACTER:
                continue
            inherited = style
            for _ in range(hops_limit):
                if inherited is None:
                    break
                script = inherited.element.find("./" + qn("w:rPr") + "/" + qn("w:vertAlign"))
                if script is not None:
                    kind = {"superscript": "sup", "subscript": "sub"}.get(script.get(qn("w:val")))
                    if kind:
                        self._script_styles[style.style_id] = kind
                    break
                inherited = getattr(inherited, "base_style", None)
        self.ir = {"version": _PIPELINE["schema_version"], "document_id": document_id,
                   "document_version": digest, "source": identity, "blocks": [],
                   "units": [], "assets": {}, "warnings": [], "relations": []}
        self._unit_ids = set()
        self._part_name = "word/document.xml"

    def extract(self):
        body, part = self.document.element.body, self.document.part
        self._walk_part(body, part, "body")
        self._walk_notes(body, part)
        embedded_image_rels = sum(
            1
            for doc_part in self.document.part.package.iter_parts()
            for rel in getattr(doc_part, "rels", {}).values()
            if not rel.is_external and rel.reltype.endswith("/image")
        )
        self.ir["statistics"] = {"embedded_image_relationships": embedded_image_rels}
        return self.ir

    def _walk_notes(self, body, part):
        references = {"footnote": {}, "endnote": {}}
        for node in content_nodes(body, hidden_run=self._hidden_run):
            for kind, by_id in references.items():
                if node.tag == qn("w:" + kind + "Reference"):
                    by_id.setdefault(node.get(qn("w:id")), []).append(node.getroottree().getpath(node))
        for kind, by_id in references.items():
            if not by_id:
                continue
            relationship = next((rel for rel in part.rels.values()
                                 if not rel.is_external and rel.reltype.endswith("/" + kind + "s")), None)
            if relationship is None:
                self.ir["warnings"].append({"code": "DOCX_NOTE_PART_MISSING", "note_type": kind})
                continue
            note_part = relationship.target_part
            note_nodes = {node.get(qn("w:id")): node for node in parse_xml(note_part.blob)
                          if node.tag == qn("w:" + kind)}
            self._part_name = str(note_part.partname).lstrip("/")
            try:
                for identifier, anchors in by_id.items():
                    note = note_nodes.get(identifier)
                    if note is None:
                        self.ir["warnings"].append({"code": "DOCX_NOTE_MISSING", "note_type": kind,
                                                    "note_id": identifier})
                        continue
                    start = len(self.ir["blocks"])
                    self._walk_part(note, note_part, kind + ":" + identifier)
                    for block in self.ir["blocks"][start:]:
                        block["provenance"]["note_references"] = anchors
            finally:
                self._part_name = "word/document.xml"

    def _append(self, kind, text, node, region, **attributes):
        index = len(self.ir["blocks"])
        block_id = self.ir["document_id"] + ":v" + self.ir["document_version"][:_DOCX_CFG["version_prefix_len"]] + ":b" + str(index)
        block = {"id": block_id, "type": kind, "text": text, "order": index,
                 "provenance": {"part": self._part_name,
                                "xpath": node.getroottree().getpath(node), "region": region}}
        block.update(attributes)
        self.ir["blocks"].append(block)
        return block

    def _unit(self, text, node, region, block_id):
        if not text.strip():
            return None
        key = self.ir["document_id"] + self.ir["document_version"] + region + ":" + node.getroottree().getpath(node)
        unit_id = hashlib.sha256(key.encode("utf-8")).hexdigest()[:_DOCX_CFG["document_id_hex_len"]]
        if unit_id in self._unit_ids:
            return unit_id
        self._unit_ids.add(unit_id)
        self.ir["units"].append({"id": unit_id, "block_id": block_id, "text": text,
                                  "region": region, "xpath": node.getroottree().getpath(node)})
        return unit_id

    def _walk_part(self, container, part, region):
        for node in content_nodes(container, hidden_run=self._hidden_run):
            code = None
            if local_name(node) in EMBEDDED_CONTENT:
                code = "UNSUPPORTED_EMBEDDED_CONTENT"
            elif node.tag == MC + "AlternateContent":
                code = "UNSUPPORTED_ALTERNATE_CONTENT"
            if code:
                self.ir["warnings"].append({"code": code, "region": region,
                                            "xpath": node.getroottree().getpath(node), "requires_review": True})
        self._walk(container, part, region)

    def _walk(self, container, part, region):
        for node in content_children(container):
            name = local_name(node)
            if name in NONCONTENT or name in EMBEDDED_CONTENT or node.tag == MC + "AlternateContent":
                continue
            if name == "p":
                self._paragraph(node, part, region)
            elif name == "tbl":
                self._table(node, part, region)
            elif name in ("oMath", "oMathPara"):
                text = native_math(node)
                if text.strip():
                    block = self._append("paragraph", text, node, region)
                    block["unit_ids"] = [self._unit(text, node, region, block["id"])]
            else:
                self._walk(node, part, region)

    def _paragraph(self, node, part, region):
        paragraph = docx.text.paragraph.Paragraph(node, self.document)
        level = heading_level(paragraph, True)
        language = _authored_language(paragraph, self._hidden_run)
        segment = 0
        for part_kind, value, hyperlinks in paragraph_parts(
            node, part, self._script_styles, self._hidden_run, include_images=self.image_ocr is not None,
        ):
            if part_kind == "image":
                self._append_image_text(value, part, region)
                continue
            if part_kind == "textbox":
                self._walk_textboxes(value, part, region)
                continue
            if not value.strip():
                continue
            kind = "paragraph"
            if level and segment == 0:
                kind = "heading"
            elif paragraph.style is not None and paragraph.style.style_id == "Caption":
                kind = "caption"
            block = self._append(kind, value, node, region, inline_segment=segment, hyperlinks=hyperlinks)
            if language:
                block["source_language"] = language
            if kind == "heading":
                block.update(level=min(level, MAX_HEADING_LEVEL), authored_level=level,
                             heading_format="openxml_outline")
            if segment == 0 or kind != "paragraph":
                typo = extract_paragraph_typography(paragraph, value, self._hidden_run)
                if typo is not None:
                    block["typography"] = typo
                    if kind != "heading" and typo["all_bold"] and len(typo["size_chars"]) == 1:
                        block["heading_format"] = {
                            "bold": True,
                            "font_pt": typo["font_pt"],
                            "centered": typo["centered"],
                        }
            unit_id = self._unit(value, node, region + ":inline" + str(segment), block["id"])
            block["unit_ids"] = [unit_id]
            segment += 1

    def _append_image_text(self, node, part, region):
        relation_id = node.get(qn("r:embed")) or node.get(qn("r:id"))
        texts, digest = self.image_ocr.read(self.ir, part.rels.get(relation_id), node.getroottree().getpath(node))
        cleaned = [item.strip() for item in texts if item and item.strip()]
        if not cleaned:
            return
        combined = "\n\n".join(cleaned)
        block = self._append("image", combined, node, region)
        block["image"] = {
            "sha256": digest,
            "status": "ocr_transcribed",
            "text_context_assessed": True,
            "image_policy": IMAGE_POLICY,
            "ocr_text": combined,
            "description": combined,
            "context_evidence": [{
                "source_block_id": block["id"],
                "quote": combined,
                "source_span": [0, len(combined)],
            }],
        }
        block["provenance"].update(method="docx_cloud_image_ocr", image_asset_sha256=digest)
        block["unit_ids"] = [self._unit(combined, node, region + ":image:0", block["id"])]

    def _walk_textboxes(self, node, part, region):
        for textbox in content_nodes(node, stop_at=("txbxContent",), hidden_run=self._hidden_run):
            if textbox.tag == qn("w:txbxContent"):
                self._walk(textbox, part, region + ":textbox")

    def _table(self, node, part, region, parent_cell=None):
        xpath = node.getroottree().getpath(node)
        table_key = hashlib.sha256(xpath.encode("utf-8")).hexdigest()[:_DOCX_CFG["document_id_hex_len"]]
        prefix = self.ir["document_id"] + ":v" + self.ir["document_version"][:_DOCX_CFG["version_prefix_len"]] + ":cell:" + table_key + ":"
        table, units, attachments = read_table(
            node, self.document, part, self._script_styles, self._hidden_run, prefix, parent_cell,
            include_images=self.image_ocr is not None,
        )
        if not table["grid"]:
            return
        block = self._append("table", "", node, region, table_data=table, unit_ids=[])
        if parent_cell:
            block["provenance"]["parent_cell"] = parent_cell
        table["table_id"] = block["id"]
        for text, paragraph in units:
            unit_id = self._unit(text, paragraph, region, block["id"])
            if unit_id:
                block["unit_ids"].append(unit_id)
        for cell in table["cells"]:
            self.ir["relations"].append({"from": block["id"], "to": cell["id"], "type": "contains_cell"})
        for child, cell, position in attachments:
            cell_id = cell["id"]
            if position["type"] == "table":
                nested = self._table(child, part, region + ":nested", cell_id)
                if nested is None:
                    continue
                position["table_id"] = nested["id"]
            else:
                start = len(self.ir["blocks"])
                if position["type"] == "image":
                    self._append_image_text(child, part, region + ":cell")
                else:
                    self._walk_textboxes(child, part, region + ":cell")
                blocks = self.ir["blocks"][start:]
                if not blocks:
                    continue
                for item in blocks:
                    item["provenance"].setdefault("parent_cell", cell_id)
                    if item["type"] == "table" and item["table_data"]["parent_cell"] is None:
                        item["table_data"]["parent_cell"] = cell_id
                position["block_ids"] = [item["id"] for item in blocks]
            cell.setdefault("attachments", []).append(position)
        return block
