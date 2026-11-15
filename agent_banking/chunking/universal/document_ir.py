"""Extract the DOCX body into ordered blocks, source units and provenance."""
import hashlib
import io
from pathlib import Path

import docx
from docx.enum.style import WD_STYLE_TYPE
from docx.oxml.ns import qn

from .document_typography import (
    MAX_HEADING_LEVEL,
    _authored_language,
    build_typography_record,
    extract_paragraph_typography,
    heading_level,
    resolve_run_font_pt,
)
from .docx_tables import read_table
from .openxml_content import (
    EMBEDDED_CONTENT,
    MC,
    NONCONTENT,
    _run_script,
    _script_text,
    content_children,
    content_nodes,
    local_name,
    native_math,
    native_text,
    table_items,
)


def apply_typography_hierarchy(ir, cache=None):
    """Shared decision-model hierarchy and caption detector for both DOCX and PDF IRs."""
    from .decision_engine import decide_breadcrumb_hierarchy
    decide_breadcrumb_hierarchy(ir, cache=cache)


def link_visual_captions(ir, cache=None):
    """Link caption blocks to adjacent image or table blocks across both DOCX and PDF IRs."""
    from .decision_engine import decide_caption_attachments
    decide_caption_attachments(ir, cache=cache)


class DocxExtractor:
    def __init__(self, asset_dir, render_figures=False, *, use_decision=True, decision_cache=None):
        self.asset_dir = Path(asset_dir)
        self.use_decision = use_decision
        self.decision_cache = decision_cache

    def extract(self, source, source_name):
        if isinstance(source, (bytes, bytearray)):
            data = bytes(source)
        else:
            data = Path(source).read_bytes()
        ir = _DocxReader(data, source_name).extract()
        if self.use_decision:
            apply_typography_hierarchy(ir, cache=self.decision_cache)
            link_visual_captions(ir, cache=self.decision_cache)
        ir["statistics"] = {"blocks": len(ir["blocks"]), "units": len(ir["units"]),
                            "image_occurrences": sum(block["type"] == "image" for block in ir["blocks"]),
                            "tables": sum(block["type"] == "table" for block in ir["blocks"])}
        return ir


class _DocxReader:
    def __init__(self, data, source_name):
        digest = hashlib.sha256(data).hexdigest()
        identity = str(source_name).replace("\\", "/")
        document_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]
        self.document = docx.Document(io.BytesIO(data))
        self._script_styles = {}
        for style in self.document.styles:
            if style.type != WD_STYLE_TYPE.CHARACTER:
                continue
            inherited = style
            for _ in range(10):
                if inherited is None:
                    break
                script = inherited.element.find("./" + qn("w:rPr") + "/" + qn("w:vertAlign"))
                if script is not None:
                    kind = {"superscript": "sup", "subscript": "sub"}.get(script.get(qn("w:val")))
                    if kind:
                        self._script_styles[style.style_id] = kind
                    break
                inherited = getattr(inherited, "base_style", None)
        self.ir = {"version": "2.0", "document_id": document_id,
                   "document_version": digest, "source": identity, "blocks": [],
                   "units": [], "assets": {}, "warnings": [], "relations": []}
        self._unit_ids = set()

    def extract(self):
        self._walk_part(self.document.element.body, self.document.part, "body")
        return self.ir

    def _append(self, kind, text, node, region, **attributes):
        index = len(self.ir["blocks"])
        block_id = self.ir["document_id"] + ":v" + self.ir["document_version"][:16] + ":b" + str(index)
        block = {"id": block_id, "type": kind, "text": text, "order": index,
                 "provenance": {"part": "word/document.xml",
                                "xpath": node.getroottree().getpath(node), "region": region}}
        block.update(attributes)
        self.ir["blocks"].append(block)
        return block

    def _unit(self, text, node, region, block_id):
        if not text.strip():
            return None
        key = self.ir["document_id"] + self.ir["document_version"] + region + ":" + node.getroottree().getpath(node)
        unit_id = hashlib.sha256(key.encode("utf-8")).hexdigest()[:20]
        if unit_id in self._unit_ids:
            return unit_id
        self._unit_ids.add(unit_id)
        self.ir["units"].append({"id": unit_id, "block_id": block_id, "text": text,
                                  "region": region, "xpath": node.getroottree().getpath(node)})
        return unit_id

    def _walk_part(self, container, part, region):
        for node in content_nodes(container):
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
        language = _authored_language(paragraph)
        buffer = []
        link_fragments = {}
        segment = 0

        def emit():
            nonlocal segment
            value = "".join(buffer)
            if not value.strip():
                return
            kind = "paragraph"
            if level and segment == 0:
                kind = "heading"
            elif paragraph.style is not None and paragraph.style.style_id == "Caption":
                kind = "caption"
            block = self._append(kind, value, node, region, inline_segment=segment)
            if language:
                block["source_language"] = language
            if kind == "heading":
                block.update(level=min(level, MAX_HEADING_LEVEL), authored_level=level,
                             heading_format="openxml_outline")
            if segment == 0:
                typo = extract_paragraph_typography(paragraph, value)
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
            hyperlinks = []
            for link, fragments in link_fragments.items():
                rel_id = link.get(qn("r:id"))
                target = None
                if rel_id and rel_id in part.rels:
                    target = part.rels[rel_id].target_ref
                elif not rel_id and link.get(qn("w:anchor")):
                    target = "#" + link.get(qn("w:anchor"))
                if target:
                    hyperlinks.append({"text": "".join(fragments), "target": target})
            block["hyperlinks"] = hyperlinks

        for kind, value, link in self._inline_events(node):
            if kind == "text":
                buffer.append(value)
                if link is not None:
                    link_fragments.setdefault(link, []).append(value)
            else:
                emit()
                buffer.clear()
                link_fragments.clear()
                self._walk_textboxes(value, part, region)
        emit()

    def _walk_textboxes(self, node, part, region):
        for textbox in content_nodes(node, stop_at=("txbxContent",)):
            if textbox.tag == qn("w:txbxContent"):
                self._walk(textbox, part, region + ":textbox")

    def _inline_events(self, node, link=None):
        for child in content_children(node):
            name = local_name(child)
            if name in NONCONTENT or name in EMBEDDED_CONTENT or child.tag == MC + "AlternateContent":
                continue
            if name in ("drawing", "pict", "txbxContent"):
                if any(node.tag == qn("w:txbxContent") for node in content_nodes(child, stop_at=("txbxContent",))):
                    yield "visual", child, None
            elif name in ("t", "tab", "br", "cr", "noBreakHyphen", "softHyphen", "sym", "oMath", "oMathPara"):
                yield "text", native_text(child), link
            elif name == "r":
                script = _run_script(child, self._script_styles)
                if not script:
                    yield from self._inline_events(child, link)
                    continue
                text = []
                for kind, value, event_link in self._inline_events(child, link):
                    if kind == "text":
                        text.append(value)
                        continue
                    if text:
                        yield "text", _script_text("".join(text), script), link
                        text.clear()
                    yield kind, value, event_link
                if text:
                    yield "text", _script_text("".join(text), script), link
            else:
                yield from self._inline_events(child, child if name == "hyperlink" else link)

    def _table(self, node, part, region, parent_cell=None):
        xpath = node.getroottree().getpath(node)
        table_key = hashlib.sha256(xpath.encode("utf-8")).hexdigest()[:20]
        prefix = self.ir["document_id"] + ":v" + self.ir["document_version"][:16] + ":cell:" + table_key + ":"
        table, units, nested, textbox_paragraphs = read_table(
            node, self.document, self._script_styles, prefix, parent_cell
        )
        if not table["grid"]:
            return
        for paragraph in textbox_paragraphs:
            self._walk_textboxes(paragraph, part, region + ":cell")
        block = self._append("table", "", node, region, table_data=table, unit_ids=[])
        table["table_id"] = block["id"]
        for text, paragraph in units:
            unit_id = self._unit(text, paragraph, region, block["id"])
            if unit_id:
                block["unit_ids"].append(unit_id)
        for cell in table["cells"]:
            self.ir["relations"].append({"from": block["id"], "to": cell["id"], "type": "contains_cell"})
        for child, cell_id in nested:
            self._table(child, part, region + ":nested", cell_id)
