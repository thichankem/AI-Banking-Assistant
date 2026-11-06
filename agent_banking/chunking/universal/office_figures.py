"""Render a native Office drawing in a DOCX fragment, preserving its parts."""
from copy import deepcopy
import io
import zipfile

from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from lxml import etree

from .renderer import render_docx_pages


def drawing_preview(document_bytes, document_element, reference, allow_blank=False):
    drawing = reference
    while drawing.getparent() is not None and drawing.tag not in (qn("w:drawing"), qn("w:pict")):
        drawing = drawing.getparent()
    if drawing.tag not in (qn("w:drawing"), qn("w:pict")):
        raise ValueError("Office figure has no drawing anchor")
    root = deepcopy(document_element)
    body = root.find(qn("w:body"))
    for child in list(body):
        if child.tag != qn("w:sectPr"):
            body.remove(child)
        else:
            for reference_node in list(child):
                if reference_node.tag in (qn("w:headerReference"), qn("w:footerReference")):
                    child.remove(reference_node)
    paragraph = OxmlElement("w:p")
    run = OxmlElement("w:r")
    run.append(deepcopy(drawing))
    paragraph.append(run)
    body.insert(0, paragraph)
    replacement = etree.tostring(root, encoding="UTF-8", xml_declaration=True, standalone=True)
    output = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(document_bytes)) as original, zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as fragment:
        for item in original.infolist():
            data = original.read(item.filename)
            if item.filename == "word/document.xml":
                data = replacement
            fragment.writestr(item, data)
    from PIL import Image, ImageStat
    def rank(pages):
        values = []
        for page in pages:
            with Image.open(io.BytesIO(page["image_bytes"])) as image:
                values.append((sum(ImageStat.Stat(image.convert("RGB")).var), page["image_bytes"]))
        values.sort(key=lambda item: item[0], reverse=True)
        return values
    try:
        ranked = rank(render_docx_pages(output.getvalue(), dpi=200))
    except RuntimeError:
        ranked = []
    if not ranked or (ranked[0][0] < 0.01 and not allow_blank):
        from .office_renderer import render_with_word
        ranked = rank(render_with_word(output.getvalue(), dpi=200))
    if not ranked:
        raise RuntimeError("Office renderer did not render the native chart/diagram")
    if ranked[0][0] < 0.01 and not allow_blank:
        raise RuntimeError("Native Office figure rendered blank; visual review required")
    return ranked[0][1]
