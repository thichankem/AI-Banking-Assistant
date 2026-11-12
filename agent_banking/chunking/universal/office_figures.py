"""Render a native Office drawing in a DOCX fragment, preserving its parts."""
from copy import deepcopy
import hashlib
import io
import os
from pathlib import Path
import subprocess
import tempfile
import zipfile

from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from lxml import etree

try:
    import pymupdf
    fitz = pymupdf
except ImportError:
    import fitz


def render_docx_pages(docx_source, dpi=150):
    if dpi <= 0:
        raise ValueError("dpi must be greater than 0")
    if docx_source is None:
        raise ValueError("docx_source cannot be None")
    doc = None
    if isinstance(docx_source, str):
        if not os.path.exists(docx_source):
            raise FileNotFoundError("DOCX file not found: " + docx_source)
        doc = fitz.open(docx_source)
    elif isinstance(docx_source, (bytes, bytearray)):
        if len(docx_source) == 0:
            raise ValueError("docx_source stream is empty")
        doc = fitz.open(stream=docx_source, filetype="docx")
    elif hasattr(docx_source, "__fspath__"):
        path_str = str(docx_source)
        if not os.path.exists(path_str):
            raise FileNotFoundError("DOCX file not found: " + path_str)
        doc = fitz.open(path_str)
    else:
        raise TypeError("docx_source must be a str, Path, or bytes stream")
    pages = []
    try:
        for page_index in range(len(doc)):
            page = doc[page_index]
            pix = page.get_pixmap(dpi=dpi)
            image_bytes = pix.tobytes("png")
            pages.append({
                "page_index": page_index,
                "image_bytes": image_bytes,
                "sha256": hashlib.sha256(image_bytes).hexdigest(),
                "width": pix.width,
                "height": pix.height,
            })
    finally:
        if doc is not None:
            doc.close()
    return pages


def render_with_word(document_bytes, dpi=200):
    if os.name != "nt":
        raise RuntimeError("Native Word rendering requires Windows and Microsoft Word")
    script = r'''
$ErrorActionPreference = 'Stop'
$word = $null
$document = $null
$discardChanges = 0
try {
    $word = New-Object -ComObject Word.Application
    $word.Visible = $false
    $word.DisplayAlerts = 0
    $word.AutomationSecurity = 3
    $word.Options.UpdateLinksAtOpen = $false
    $document = $word.Documents.Open($env:CHUNKING_WORD_INPUT, $false, $true, $false)
    $document.ExportAsFixedFormat($env:CHUNKING_WORD_OUTPUT, 17)
} finally {
    if ($null -ne $document) {
        $document.Close([ref]$discardChanges)
        [void][Runtime.InteropServices.Marshal]::FinalReleaseComObject($document)
    }
    if ($null -ne $word) {
        $word.Quit([ref]$discardChanges)
        [void][Runtime.InteropServices.Marshal]::FinalReleaseComObject($word)
    }
}
'''
    with tempfile.TemporaryDirectory(prefix="chunking_word_") as directory:
        incoming = Path(directory) / "drawing.docx"
        outgoing = Path(directory) / "drawing.pdf"
        incoming.write_bytes(document_bytes)
        environment = os.environ.copy()
        environment["CHUNKING_WORD_INPUT"] = str(incoming)
        environment["CHUNKING_WORD_OUTPUT"] = str(outgoing)
        result = subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script],
                                env=environment, capture_output=True, timeout=120,
                                creationflags=subprocess.CREATE_NO_WINDOW)
        if result.returncode or not outgoing.is_file():
            raise RuntimeError("Microsoft Word could not render the native Office figure")
        pages = []
        with fitz.open(str(outgoing)) as document:
            for index, page in enumerate(document):
                pixmap = page.get_pixmap(dpi=dpi, alpha=False)
                image = pixmap.tobytes("png")
                pages.append({"page_index": index, "image_bytes": image,
                              "sha256": hashlib.sha256(image).hexdigest(),
                              "width": pixmap.width, "height": pixmap.height})
        return pages


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
        ranked = rank(render_with_word(output.getvalue(), dpi=200))
    if not ranked:
        raise RuntimeError("Office renderer did not render the native chart/diagram")
    if ranked[0][0] < 0.01 and not allow_blank:
        raise RuntimeError("Native Office figure rendered blank; visual review required")
    return ranked[0][1]
