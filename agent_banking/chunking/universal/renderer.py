# universal/renderer.py
# Universal Domain-Agnostic DOCX In-RAM Rasterizer
# Complies with GEMINI.md: 0% regex, zero ternary, no typing library, zero __init__.py

import os
import hashlib
from pathlib import Path

try:
    import pymupdf
    fitz = pymupdf
except ImportError:
    import fitz


def get_docx_page_count(docx_source):
    """
    Returns the total number of pages in a DOCX document without full rasterization.

    Parameters:
        docx_source: str (file path), pathlib.Path, or bytes/bytearray (file stream).

    Returns:
        int: Number of pages.
    """
    if docx_source is None:
        raise ValueError("docx_source cannot be None")

    doc = None
    if isinstance(docx_source, str):
        if not os.path.exists(docx_source):
            raise FileNotFoundError("DOCX file not found: " + docx_source)
        doc = fitz.open(docx_source)
    else:
        if isinstance(docx_source, (bytes, bytearray)):
            if len(docx_source) == 0:
                raise ValueError("docx_source stream is empty")
            doc = fitz.open(stream=docx_source, filetype="docx")
        else:
            if hasattr(docx_source, "__fspath__"):
                path_str = str(docx_source)
                if not os.path.exists(path_str):
                    raise FileNotFoundError("DOCX file not found: " + path_str)
                doc = fitz.open(path_str)
            else:
                raise TypeError("docx_source must be a str, Path, or bytes stream")

    try:
        count = len(doc)
    finally:
        if doc is not None:
            doc.close()

    return count


def render_docx_page(docx_source, page_index=0, dpi=150):
    """
    Renders a single specified page of a DOCX document to PNG bytes.

    Parameters:
        docx_source: str, Path, or bytes/bytearray stream.
        page_index: int, 0-indexed page number to render.
        dpi: int or float, rendering resolution (default 150).

    Returns:
        dict with keys: "page_index", "image_bytes", "sha256", "width", "height".
    """
    if dpi <= 0:
        raise ValueError("dpi must be greater than 0")

    if docx_source is None:
        raise ValueError("docx_source cannot be None")

    doc = None
    if isinstance(docx_source, str):
        if not os.path.exists(docx_source):
            raise FileNotFoundError("DOCX file not found: " + docx_source)
        doc = fitz.open(docx_source)
    else:
        if isinstance(docx_source, (bytes, bytearray)):
            if len(docx_source) == 0:
                raise ValueError("docx_source stream is empty")
            doc = fitz.open(stream=docx_source, filetype="docx")
        else:
            if hasattr(docx_source, "__fspath__"):
                path_str = str(docx_source)
                if not os.path.exists(path_str):
                    raise FileNotFoundError("DOCX file not found: " + path_str)
                doc = fitz.open(path_str)
            else:
                raise TypeError("docx_source must be a str, Path, or bytes stream")

    try:
        total_pages = len(doc)
        if page_index < 0:
            raise IndexError("page_index out of range (negative)")
        if page_index >= total_pages:
            raise IndexError("page_index out of range (max: " + str(total_pages - 1) + ")")

        page = doc[page_index]
        pix = page.get_pixmap(dpi=dpi)
        image_bytes = pix.tobytes("png")
        sha256_hash = hashlib.sha256(image_bytes).hexdigest()
        width = pix.width
        height = pix.height

        result = {
            "page_index": page_index,
            "image_bytes": image_bytes,
            "sha256": sha256_hash,
            "width": width,
            "height": height
        }

        pix = None
        page = None
    finally:
        if doc is not None:
            doc.close()

    return result


def render_docx_pages(docx_source, dpi=150):
    """
    Renders all pages of a DOCX document into PNG byte streams with deterministic SHA-256 hashes.

    Parameters:
        docx_source: str (file path), pathlib.Path, or bytes/bytearray (file stream).
        dpi: int or float, rendering resolution in dots per inch (default 150).

    Returns:
        list of dicts, each with keys:
            - "page_index": int (0-indexed page number)
            - "image_bytes": bytes (PNG format binary data)
            - "sha256": str (64-char lowercase hex digest)
            - "width": int (pixel width at specified DPI)
            - "height": int (pixel height at specified DPI)
    """
    if dpi <= 0:
        raise ValueError("dpi must be greater than 0")

    if docx_source is None:
        raise ValueError("docx_source cannot be None")

    doc = None
    if isinstance(docx_source, str):
        if not os.path.exists(docx_source):
            raise FileNotFoundError("DOCX file not found: " + docx_source)
        doc = fitz.open(docx_source)
    else:
        if isinstance(docx_source, (bytes, bytearray)):
            if len(docx_source) == 0:
                raise ValueError("docx_source stream is empty")
            doc = fitz.open(stream=docx_source, filetype="docx")
        else:
            if hasattr(docx_source, "__fspath__"):
                path_str = str(docx_source)
                if not os.path.exists(path_str):
                    raise FileNotFoundError("DOCX file not found: " + path_str)
                doc = fitz.open(path_str)
            else:
                raise TypeError("docx_source must be a str, Path, or bytes stream")

    pages = []
    try:
        total_pages = len(doc)
        for page_index in range(total_pages):
            page = doc[page_index]
            pix = page.get_pixmap(dpi=dpi)
            image_bytes = pix.tobytes("png")
            sha256_hash = hashlib.sha256(image_bytes).hexdigest()
            width = pix.width
            height = pix.height

            pages.append({
                "page_index": page_index,
                "image_bytes": image_bytes,
                "sha256": sha256_hash,
                "width": width,
                "height": height
            })

            pix = None
            page = None
    finally:
        if doc is not None:
            doc.close()

    return pages
