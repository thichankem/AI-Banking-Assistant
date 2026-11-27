#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Full-Document Preview & Passage Localization Service (PDF, DOCX, XLSX)
======================================================================
Provides safe document resolution, rich structured previews, full-PDF
page rendering, and Level-2 passage localization (XPath for DOCX,
Sheet/Cell range for XLSX, and OCR/native bounding boxes for PDF).

MANDATORY RULES:
- 0% REGEX LAW: Pure string methods only (no import re).
- ZERO TERNARY: Explicit if/else statements only.
- NO __init__.py: Direct module imports.
"""

import hashlib
import io
import json
import os
import shutil
import sqlite3
import threading
import time
import unicodedata
from pathlib import Path
from urllib.parse import quote
from zipfile import BadZipFile, ZipFile

import openpyxl
import pymupdf
import pytesseract
from lxml import etree
from PIL import Image


_W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
_R_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
_A_NS = "http://schemas.openxmlformats.org/drawingml/2006/main"
_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"

_DOC_EXTENSIONS = (".pdf", ".docx", ".xlsx")
_DOC_INDEX_CACHE = {}
_DOC_INDEX_LOCK = threading.Lock()
_OCR_DB_LOCK = threading.Lock()
_CHUNK_INDEX_CACHE = {}
_CHUNK_INDEX_LOCK = threading.Lock()

_TESSERACT_CANDIDATES = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
)


def _w_tag(local: str) -> str:
    return "{" + _W_NS + "}" + local


def _r_tag(local: str) -> str:
    return "{" + _R_NS + "}" + local


def _a_tag(local: str) -> str:
    return "{" + _A_NS + "}" + local


def _resolve_tesseract_cmd() -> str | None:
    found = shutil.which("tesseract")
    if found:
        return found
    for candidate in _TESSERACT_CANDIDATES:
        if os.path.isfile(candidate):
            return candidate
    return None


def _configure_tesseract() -> str | None:
    cmd = _resolve_tesseract_cmd()
    if cmd:
        pytesseract.pytesseract.tesseract_cmd = cmd
    return cmd


def fold_search_text(value: str) -> str:
    """Normalize Vietnamese text by removing diacritics and punctuation without regex."""
    raw = unicodedata.normalize("NFKD", str(value or "").lower())
    chars = []
    for ch in raw:
        if unicodedata.combining(ch):
            continue
        if ch == "đ":
            chars.append("d")
        elif ch.isalnum():
            chars.append(ch)
        else:
            chars.append(" ")
    return " ".join("".join(chars).split())


def escape_html(value: str) -> str:
    """Escape HTML special characters safely."""
    text = str(value or "")
    text = text.replace("&", "&amp;")
    text = text.replace("<", "&lt;")
    text = text.replace(">", "&gt;")
    text = text.replace('"', "&quot;")
    text = text.replace("'", "&#39;")
    return text


def normalize_doc_source(source: str, allowed_exts: tuple[str, ...] = _DOC_EXTENSIONS) -> str:
    """Validate and normalize a relative document path without regex."""
    if not isinstance(source, str) or "\x00" in source:
        raise ValueError("Invalid document source parameter")
    clean = source.replace("\\", "/").strip()
    while clean.startswith("/"):
        clean = clean[1:]
    if not clean:
        raise ValueError("Empty document source parameter")
    if ":" in clean:
        raise PermissionError("Invalid document path")
    for seg in clean.split("/"):
        if not seg:
            continue
        if seg == "." or seg == "..":
            raise PermissionError("Invalid document path")
    lower = clean.lower()
    matched_ext = False
    for ext in allowed_exts:
        if lower.endswith(ext):
            matched_ext = True
            break
    if not matched_ext:
        raise ValueError("Unsupported document extension")
    return clean


def is_within_allowed_dirs(resolved: Path, allowed_dirs: list[Path]) -> bool:
    for allowed_dir in allowed_dirs:
        if resolved.is_relative_to(allowed_dir):
            return True
    return False


def find_document_path(
    source: str,
    allowed_dirs: list[Path],
    allowed_exts: tuple[str, ...] = _DOC_EXTENSIONS,
) -> Path | None:
    """Locate a document inside allowed directories, supporting relative paths and bare filenames."""
    clean = normalize_doc_source(source, allowed_exts=allowed_exts)
    parts = [seg for seg in clean.split("/") if seg]
    for base_dir in allowed_dirs:
        candidate = base_dir
        for seg in parts:
            candidate = candidate / seg
        resolved = candidate.resolve()
        if not is_within_allowed_dirs(resolved, allowed_dirs):
            raise PermissionError("Invalid document path")
        if resolved.is_file():
            return resolved

    if "/" not in clean:
        target_lower = clean.lower()
        with _DOC_INDEX_LOCK:
            cached = _DOC_INDEX_CACHE.get(target_lower)
        if cached is not None and cached.is_file() and is_within_allowed_dirs(cached, allowed_dirs):
            return cached
        for base_dir in allowed_dirs:
            for item in base_dir.rglob("*"):
                if not item.is_file():
                    continue
                if item.name == clean or item.name.lower() == target_lower:
                    resolved = item.resolve()
                    if is_within_allowed_dirs(resolved, allowed_dirs):
                        with _DOC_INDEX_LOCK:
                            _DOC_INDEX_CACHE[target_lower] = resolved
                        return resolved
    return None


def get_document_type(source: str) -> str:
    lower = str(source or "").strip().lower()
    if lower.endswith(".pdf"):
        return "pdf"
    if lower.endswith(".docx"):
        return "docx"
    if lower.endswith(".xlsx"):
        return "xlsx"
    return "unknown"


# -----------------------------------------------------------------------------
# Chunk Fallback Lookup (supports legacy saved sessions lacking provenance)
# -----------------------------------------------------------------------------
def _load_knowledge_chunk(chunk_id: str, knowledge_root: Path) -> dict | None:
    if not chunk_id or not knowledge_root.is_dir():
        return None
    doc_id = ""
    if ":" in chunk_id:
        doc_id = chunk_id.split(":", 1)[0].strip()
    if not doc_id:
        return None
    all_chunks_file = knowledge_root / doc_id / "all_chunks.json"
    if not all_chunks_file.is_file():
        return None
    try:
        mtime_ns = int(all_chunks_file.stat().st_mtime_ns)
    except OSError:
        return None
    with _CHUNK_INDEX_LOCK:
        cached_entry = _CHUNK_INDEX_CACHE.get(doc_id)
    if cached_entry is not None and cached_entry[0] == mtime_ns:
        cached_map = cached_entry[1]
    else:
        try:
            raw_list = json.loads(all_chunks_file.read_text(encoding="utf-8"))
        except Exception:
            return None
        if not isinstance(raw_list, list):
            return None
        built = {}
        for item in raw_list:
            if isinstance(item, dict):
                cid = str(item.get("chunk_id") or "")
                if cid:
                    built[cid] = item
        with _CHUNK_INDEX_LOCK:
            if len(_CHUNK_INDEX_CACHE) >= 256 and doc_id not in _CHUNK_INDEX_CACHE:
                oldest_key = next(iter(_CHUNK_INDEX_CACHE))
                _CHUNK_INDEX_CACHE.pop(oldest_key, None)
            _CHUNK_INDEX_CACHE[doc_id] = (mtime_ns, built)
            cached_map = built
    return cached_map.get(chunk_id)


def load_document_chunks_for_source(source: str, knowledge_root: Path) -> list[dict]:
    """Load published chunks for a PDF source to power fast in-PDF keyword search."""
    if not source or not knowledge_root.is_dir():
        return []
    clean = str(source or "").replace("\\", "/").strip().lower()
    fname = clean.rsplit("/", 1)[-1]
    vdb_root = knowledge_root.parent.parent.parent / "vector_db"
    inv_candidates = []
    env_qdrant = os.environ.get("ABANK_QDRANT_STORAGE_PATH", "").strip()
    if env_qdrant:
        inv_candidates.append(Path(env_qdrant) / "knowledge_inventory.json")
    inv_candidates.append(vdb_root / "qdrant_db" / "knowledge_inventory.json")
    inv_candidates.append(vdb_root / "qdrant_full_141" / "knowledge_inventory.json")
    inv_file = inv_candidates[0]
    for cand_inv in inv_candidates:
        if cand_inv.is_file():
            inv_file = cand_inv
            break
    candidate_pubs = []
    if inv_file.is_file():
        try:
            inv = json.loads(inv_file.read_text(encoding="utf-8"))
            for entry in inv.get("publications") or []:
                for src in entry.get("sources") or []:
                    norm_src = str(src or "").replace("\\", "/").strip().lower()
                    if norm_src == clean or norm_src.rsplit("/", 1)[-1] == fname:
                        pub_name = Path(str(entry.get("publication") or "")).name
                        if pub_name:
                            candidate_pubs.append(knowledge_root / pub_name)
        except Exception:
            pass
    if not candidate_pubs:
        for d in knowledge_root.iterdir():
            if d.is_dir() and (d / "all_chunks.json").is_file():
                candidate_pubs.append(d)
    for pub_dir in candidate_pubs:
        ac = pub_dir / "all_chunks.json"
        if not ac.is_file():
            continue
        try:
            items = json.loads(ac.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(items, list) or not items:
            continue
        first_src = str(items[0].get("source") or "").replace("\\", "/").strip().lower()
        if first_src == clean or first_src.rsplit("/", 1)[-1] == fname:
            return items
    return []


# -----------------------------------------------------------------------------
# DOCX Structured HTML Preview (1:1 XPath Mapping + Embedded Media + Tables)
# -----------------------------------------------------------------------------
def _parse_docx_relationships(zf: ZipFile) -> dict[str, dict]:
    rels = {}
    rels_name = "word/_rels/document.xml.rels"
    if rels_name not in zf.namelist():
        return rels
    try:
        root = etree.fromstring(zf.read(rels_name))
    except Exception:
        return rels
    for rel in root:
        if etree.QName(rel).localname != "Relationship":
            continue
        rid = rel.get("Id") or ""
        target = rel.get("Target") or ""
        rel_type = rel.get("Type") or ""
        target_mode = rel.get("TargetMode") or ""
        if rid:
            rels[rid] = {
                "target": target,
                "type": rel_type,
                "external": target_mode.lower() == "external",
            }
    return rels


def _parse_docx_styles(zf: ZipFile) -> dict[str, dict]:
    styles = {}
    if "word/styles.xml" not in zf.namelist():
        return styles
    try:
        root = etree.fromstring(zf.read("word/styles.xml"))
    except Exception:
        return styles
    for st in root.findall(_w_tag("style")):
        sid = st.get(_w_tag("styleId")) or ""
        if not sid:
            continue
        name_el = st.find(_w_tag("name"))
        name_val = ""
        if name_el is not None:
            name_val = str(name_el.get(_w_tag("val")) or "")
        based_el = st.find(_w_tag("basedOn"))
        based_val = ""
        if based_el is not None:
            based_val = str(based_el.get(_w_tag("val")) or "")
        outline_lvl = None
        ppr = st.find(_w_tag("pPr"))
        if ppr is not None:
            ol = ppr.find(_w_tag("outlineLvl"))
            if ol is not None:
                raw_lvl = str(ol.get(_w_tag("val")) or "").strip()
                if raw_lvl.isdigit():
                    outline_lvl = int(raw_lvl) + 1
        styles[sid] = {
            "name": name_val,
            "based_on": based_val,
            "outline_level": outline_lvl,
        }
    return styles


def _resolve_heading_level(style_id: str, styles_map: dict[str, dict]) -> int:
    visited = set()
    curr = style_id
    while curr and curr not in visited:
        visited.add(curr)
        info = styles_map.get(curr)
        if info:
            if info.get("outline_level"):
                return min(6, max(1, int(info["outline_level"])))
            lower_name = info.get("name", "").strip().lower()
            if lower_name == "title":
                return 1
            if lower_name == "subtitle":
                return 2
            if lower_name.startswith("heading"):
                tail = lower_name[len("heading"):].strip()
                if tail.isdigit():
                    return min(6, max(1, int(tail)))
            curr = info.get("based_on", "")
        else:
            lower_id = curr.strip().lower()
            if lower_id == "title":
                return 1
            if lower_id.startswith("heading"):
                tail = lower_id[len("heading"):].strip()
                if tail.isdigit():
                    return min(6, max(1, int(tail)))
            break
    return 0


def _resolve_docx_media_part(target: str) -> str:
    clean = str(target or "").replace("\\", "/").strip()
    while clean.startswith("/"):
        clean = clean[1:]
    if not clean.startswith("word/"):
        clean = "word/" + clean
    parts = []
    for seg in clean.split("/"):
        if not seg or seg == ".":
            continue
        if seg == "..":
            if parts:
                parts.pop()
            continue
        parts.append(seg)
    return "/".join(parts)


def _render_docx_run(run_el, rels: dict[str, dict], source: str) -> str:
    rpr = run_el.find(_w_tag("rPr"))
    is_bold = False
    is_italic = False
    is_underline = False
    color_hex = ""
    if rpr is not None:
        b_el = rpr.find(_w_tag("b"))
        if b_el is not None and b_el.get(_w_tag("val")) not in ("0", "false"):
            is_bold = True
        i_el = rpr.find(_w_tag("i"))
        if i_el is not None and i_el.get(_w_tag("val")) not in ("0", "false"):
            is_italic = True
        u_el = rpr.find(_w_tag("u"))
        if u_el is not None and u_el.get(_w_tag("val")) not in ("none", "0", "false"):
            is_underline = True
        c_el = rpr.find(_w_tag("color"))
        if c_el is not None:
            raw_c = str(c_el.get(_w_tag("val")) or "").strip()
            if len(raw_c) == 6 and raw_c.lower() != "auto":
                valid_hex = True
                for ch in raw_c.lower():
                    if ch not in "0123456789abcdef":
                        valid_hex = False
                        break
                if valid_hex:
                    color_hex = "#" + raw_c

    chunks = []
    for child in run_el:
        lname = etree.QName(child).localname
        if lname == "t":
            chunks.append(escape_html(child.text or ""))
        elif lname == "tab":
            chunks.append("&emsp;")
        elif lname == "br" or lname == "cr":
            chunks.append("<br>")
        elif lname == "drawing" or lname == "pict":
            for blip in child.iter(_a_tag("blip")):
                rid = blip.get(_r_tag("embed")) or blip.get(_r_tag("link")) or ""
                rel = rels.get(rid)
                if rel and not rel.get("external") and rel.get("target"):
                    part_path = _resolve_docx_media_part(rel["target"])
                    blip_xpath = blip.getroottree().getpath(blip)
                    media_url = (
                        "/api/documents/docx/media?source="
                        + quote(source, safe="")
                        + "&part="
                        + quote(part_path, safe="")
                    )
                    chunks.append(
                        '<img class="docx-inline-img" loading="lazy" data-xpath="'
                        + escape_html(blip_xpath)
                        + '" alt="Hình minh họa trong tài liệu" src="'
                        + escape_html(media_url)
                        + '">'
                    )
            for node in child.iter():
                if etree.QName(node).localname == "txbxContent":
                    for sub_node in node:
                        if etree.QName(sub_node).localname == "p":
                            tb_txt = "".join(t_el.text or "" for t_el in sub_node.findall(".//" + _w_tag("t"))).strip()
                            if tb_txt:
                                chunks.append('<span class="docx-textbox">' + escape_html(tb_txt) + "</span>")

    inner = "".join(chunks)
    if not inner:
        return ""
    if is_bold:
        inner = "<strong>" + inner + "</strong>"
    if is_italic:
        inner = "<em>" + inner + "</em>"
    if is_underline:
        inner = "<u>" + inner + "</u>"
    if color_hex and color_hex.lower() not in ("#000000", "#0f172a"):
        inner = '<span style="color:' + color_hex + '">' + inner + "</span>"
    return inner


def _infer_visual_heading_level(p_el, plain_text: str) -> int:
    """Detect headings styled with explicit font size + bold when pStyle is absent."""
    if not plain_text or len(plain_text) > 140:
        return 0
    runs = p_el.findall(_w_tag("r"))
    if not runs:
        return 0
    max_sz = 0
    has_text_run = False
    all_bold = True
    for r_el in runs:
        t_txt = "".join(t.text or "" for t in r_el.findall(_w_tag("t"))).strip()
        if not t_txt:
            continue
        has_text_run = True
        rpr = r_el.find(_w_tag("rPr"))
        run_bold = False
        if rpr is not None:
            b_el = rpr.find(_w_tag("b"))
            if b_el is not None and b_el.get(_w_tag("val")) not in ("0", "false"):
                run_bold = True
            sz_el = rpr.find(_w_tag("sz"))
            if sz_el is not None:
                raw_sz = str(sz_el.get(_w_tag("val")) or "").strip()
                if raw_sz.isdigit():
                    max_sz = max(max_sz, int(raw_sz))
        if not run_bold:
            all_bold = False
    if not has_text_run or not all_bold:
        return 0
    if max_sz >= 30:
        return 1
    if max_sz >= 24:
        return 2
    if max_sz >= 22:
        return 3
    upper_start = plain_text.upper()
    if upper_start.startswith(("CHƯƠNG ", "PHẦN ", "MỤC ", "ĐIỀU ")):
        return 3
    return 0


def _collect_docx_inline_nodes(container, rels: dict[str, dict], source: str, outline: list[dict], parts: list[str], plain_parts: list[str]) -> None:
    for child in container:
        lname = etree.QName(child).localname
        if lname == "r":
            rendered = _render_docx_run(child, rels, source)
            if rendered:
                parts.append(rendered)
            for t_el in child.findall(_w_tag("t")):
                if t_el.text:
                    plain_parts.append(t_el.text)
            for node in child.iter():
                if etree.QName(node).localname == "txbxContent":
                    for sub_node in node:
                        if etree.QName(sub_node).localname == "p":
                            tb_txt = "".join(t_el.text or "" for t_el in sub_node.findall(".//" + _w_tag("t"))).strip()
                            if tb_txt:
                                plain_parts.append(" " + tb_txt + " ")
        elif lname == "hyperlink":
            rid = child.get(_r_tag("id")) or ""
            rel = rels.get(rid)
            link_inner = []
            link_plain = []
            _collect_docx_inline_nodes(child, rels, source, outline, link_inner, link_plain)
            plain_parts.extend(link_plain)
            if link_inner:
                joined = "".join(link_inner)
                if rel and rel.get("external") and rel.get("target"):
                    href = str(rel["target"]).strip()
                    if href.startswith("http://") or href.startswith("https://"):
                        parts.append(
                            '<a href="' + escape_html(href) + '" target="_blank" rel="noopener noreferrer">'
                            + joined + "</a>"
                        )
                    else:
                        parts.append(joined)
                else:
                    parts.append(joined)
        elif lname in ("ins", "smartTag"):
            _collect_docx_inline_nodes(child, rels, source, outline, parts, plain_parts)
        elif lname == "sdt":
            sdt_content = child.find(_w_tag("sdtContent"))
            if sdt_content is not None:
                _collect_docx_inline_nodes(sdt_content, rels, source, outline, parts, plain_parts)
        elif lname == "AlternateContent":
            chosen = None
            fallback = None
            for branch in child:
                bname = etree.QName(branch).localname
                if bname == "Choice" and chosen is None:
                    chosen = branch
                elif bname == "Fallback" and fallback is None:
                    fallback = branch
            target_branch = chosen
            if target_branch is None:
                target_branch = fallback
            if target_branch is not None:
                _collect_docx_inline_nodes(target_branch, rels, source, outline, parts, plain_parts)


def _render_docx_paragraph(
    p_el,
    rels: dict[str, dict],
    styles: dict[str, dict],
    source: str,
    outline: list[dict],
    in_table: bool = False,
) -> str:
    xpath = p_el.getroottree().getpath(p_el)
    ppr = p_el.find(_w_tag("pPr"))
    style_id = ""
    jc = ""
    is_list = False
    list_ilvl = 0
    if ppr is not None:
        ps = ppr.find(_w_tag("pStyle"))
        if ps is not None:
            style_id = str(ps.get(_w_tag("val")) or "")
        jc_el = ppr.find(_w_tag("jc"))
        if jc_el is not None:
            jc = str(jc_el.get(_w_tag("val")) or "").lower()
        num_pr = ppr.find(_w_tag("numPr"))
        if num_pr is not None:
            is_list = True
            ilvl_el = num_pr.find(_w_tag("ilvl"))
            if ilvl_el is not None:
                raw_ilvl = str(ilvl_el.get(_w_tag("val")) or "").strip()
                if raw_ilvl.isdigit():
                    list_ilvl = min(4, int(raw_ilvl))

    parts = []
    plain_parts = []
    _collect_docx_inline_nodes(p_el, rels, source, outline, parts, plain_parts)

    content_html = "".join(parts)
    plain_text = " ".join("".join(plain_parts).split())
    if not content_html.strip():
        return '<div class="docx-spacer" data-xpath="' + escape_html(xpath) + '"></div>'

    heading_lvl = _resolve_heading_level(style_id, styles)
    if heading_lvl == 0 and not in_table:
        heading_lvl = _infer_visual_heading_level(p_el, plain_text)
    classes = ["docx-block"]
    if jc in ("center", "right", "both", "justify"):
        if jc == "both":
            classes.append("docx-align-justify")
        else:
            classes.append("docx-align-" + jc)
    if is_list:
        classes.append("docx-list-item")
        classes.append("docx-list-lvl-" + str(list_ilvl))

    if heading_lvl > 0:
        tag = "h" + str(min(6, max(1, heading_lvl)))
        classes.append("docx-heading-" + str(heading_lvl))
        if plain_text:
            outline.append({"level": heading_lvl, "text": plain_text[:120], "xpath": xpath})
    else:
        tag = "p"
        classes.append("docx-para")

    return (
        "<" + tag + ' class="' + " ".join(classes) + '" data-xpath="' + escape_html(xpath) + '">'
        + content_html + "</" + tag + ">"
    )


def _iter_docx_tr_cells(tr_el):
    for child in tr_el:
        lname = etree.QName(child).localname
        if lname == "tc":
            yield child
        elif lname in ("ins", "smartTag"):
            for sub in _iter_docx_tr_cells(child):
                yield sub
        elif lname == "sdt":
            sdt_content = child.find(_w_tag("sdtContent"))
            if sdt_content is not None:
                for sub in _iter_docx_tr_cells(sdt_content):
                    yield sub


def _render_docx_table(tbl_el, rels: dict[str, dict], styles: dict[str, dict], source: str, outline: list[dict]) -> str:
    tbl_xpath = tbl_el.getroottree().getpath(tbl_el)
    tr_nodes = tbl_el.findall(_w_tag("tr"))
    if not tr_nodes:
        return ""

    rows_meta = []
    for tr_idx, tr in enumerate(tr_nodes):
        tr_xpath = tr.getroottree().getpath(tr)
        tr_pr = tr.find(_w_tag("trPr"))
        is_header_row = False
        grid_before = 0
        if tr_pr is not None:
            if tr_pr.find(_w_tag("tblHeader")) is not None:
                is_header_row = True
            gb_el = tr_pr.find(_w_tag("gridBefore"))
            if gb_el is not None:
                raw_gb = str(gb_el.get(_w_tag("val")) or "").strip()
                if raw_gb.isdigit() and int(raw_gb) > 0:
                    grid_before = int(raw_gb)
        elif tr_idx == 0:
            is_header_row = True

        cells_meta = []
        col_cursor = 0
        if grid_before > 0:
            cells_meta.append({
                "col_start": 0,
                "colspan": grid_before,
                "rowspan": 1,
                "vmerge": "",
                "skip": False,
                "bg": "",
                "html": "",
            })
            col_cursor = grid_before
        for tc in _iter_docx_tr_cells(tr):
            tc_pr = tc.find(_w_tag("tcPr"))
            colspan = 1
            vmerge = ""
            bg_hex = ""
            if tc_pr is not None:
                gs = tc_pr.find(_w_tag("gridSpan"))
                if gs is not None:
                    raw_gs = str(gs.get(_w_tag("val")) or "").strip()
                    if raw_gs.isdigit() and int(raw_gs) > 1:
                        colspan = int(raw_gs)
                vm = tc_pr.find(_w_tag("vMerge"))
                if vm is not None:
                    vmerge = str(vm.get(_w_tag("val")) or "continue").lower()
                shd = tc_pr.find(_w_tag("shd"))
                if shd is not None:
                    fill = str(shd.get(_w_tag("fill")) or "").strip()
                    if len(fill) == 6 and fill.lower() != "auto":
                        valid_hex = True
                        for ch in fill.lower():
                            if ch not in "0123456789abcdef":
                                valid_hex = False
                                break
                        if valid_hex:
                            bg_hex = "#" + fill

            inner_parts = []
            for child in tc:
                lname = etree.QName(child).localname
                if lname == "p":
                    inner_parts.append(_render_docx_paragraph(child, rels, styles, source, outline, in_table=True))
                elif lname == "tbl":
                    inner_parts.append(_render_docx_table(child, rels, styles, source, outline))
            cells_meta.append({
                "col_start": col_cursor,
                "colspan": colspan,
                "rowspan": 1,
                "vmerge": vmerge,
                "skip": False,
                "bg": bg_hex,
                "html": "".join(inner_parts),
            })
            col_cursor += colspan
        rows_meta.append({"xpath": tr_xpath, "is_header": is_header_row, "cells": cells_meta})

    # Compute rowspan for vMerge cells across rows
    active_anchors = {}
    for row in rows_meta:
        for cell in row["cells"]:
            col = cell["col_start"]
            vm = cell["vmerge"]
            if vm == "restart":
                active_anchors[col] = cell
            elif vm and col in active_anchors:
                active_anchors[col]["rowspan"] += 1
                cell["skip"] = True
            else:
                if col in active_anchors:
                    del active_anchors[col]

    head_rows = []
    body_rows = []
    in_header = True
    for row in rows_meta:
        if not row["is_header"]:
            in_header = False
        tag = "td"
        if in_header:
            tag = "th"
        cell_htmls = []
        for cell in row["cells"]:
            if cell["skip"]:
                continue
            attrs = []
            if cell["colspan"] > 1:
                attrs.append('colspan="' + str(cell["colspan"]) + '"')
            if cell["rowspan"] > 1:
                attrs.append('rowspan="' + str(cell["rowspan"]) + '"')
            if cell["bg"]:
                attrs.append('style="background:' + cell["bg"] + '; color:#0f172a;"')
            attr_str = ""
            if attrs:
                attr_str = " " + " ".join(attrs)
            cell_htmls.append("<" + tag + attr_str + ">" + cell["html"] + "</" + tag + ">")
        row_html = '<tr data-xpath="' + escape_html(row["xpath"]) + '">' + "".join(cell_htmls) + "</tr>"
        if in_header:
            head_rows.append(row_html)
        else:
            body_rows.append(row_html)

    table_inner = ""
    if head_rows:
        table_inner += "<thead>" + "".join(head_rows) + "</thead>"
    if body_rows:
        table_inner += "<tbody>" + "".join(body_rows) + "</tbody>"
    return (
        '<div class="docx-table-wrap" tabindex="0" role="region" aria-label="Bảng dữ liệu trong tài liệu Word">'
        + '<table class="docx-table" data-xpath="' + escape_html(tbl_xpath) + '">'
        + table_inner
        + "</table></div>"
    )


def _walk_docx_container(container, rels: dict[str, dict], styles: dict[str, dict], source: str, outline: list[dict]) -> list[str]:
    html_parts = []
    for child in container:
        lname = etree.QName(child).localname
        if lname == "p":
            html_parts.append(_render_docx_paragraph(child, rels, styles, source, outline))
        elif lname == "tbl":
            tbl_html = _render_docx_table(child, rels, styles, source, outline)
            if tbl_html:
                html_parts.append(tbl_html)
        elif lname == "sdt":
            sdt_content = child.find(_w_tag("sdtContent"))
            if sdt_content is not None:
                html_parts.extend(_walk_docx_container(sdt_content, rels, styles, source, outline))
    return html_parts


def render_docx_preview(doc_path: Path, source: str, chunk_id: str = "", knowledge_root: Path | None = None,
                        *, indexed_chunk=None, document_version="") -> dict:
    """Render a DOCX file into structured HTML preserving 1:1 XML XPath anchors."""
    try:
        with ZipFile(str(doc_path), "r") as zf:
            if "word/document.xml" not in zf.namelist():
                raise ValueError("DOCX missing word/document.xml")
            rels = _parse_docx_relationships(zf)
            styles = _parse_docx_styles(zf)
            doc_root = etree.fromstring(zf.read("word/document.xml"))
    except BadZipFile as exc:
        raise ValueError("Invalid DOCX archive") from exc

    body = doc_root.find(_w_tag("body"))
    outline = []
    blocks_html = []
    if body is not None:
        blocks_html = _walk_docx_container(body, rels, styles, source, outline)
        if document_version:
            media = "/api/documents/docx/media?source=" + quote(source, safe="")
            encoded_media = escape_html(media)
            blocks_html = [block.replace(encoded_media + "&amp;part=", encoded_media + "&amp;version=" +
                                         quote(document_version, safe="") + "&amp;part=") for block in blocks_html]

    fallback_xpaths = []
    chunk_rec = indexed_chunk
    if chunk_rec is None and chunk_id and knowledge_root is not None:
        chunk_rec = _load_knowledge_chunk(chunk_id, knowledge_root)
    if chunk_rec:
        locs = ((chunk_rec.get("business_metadata") or {}).get("provenance") or {}).get("locations") or []
        for loc in locs:
            if isinstance(loc, dict) and loc.get("xpath"):
                fallback_xpaths.append(str(loc["xpath"]))

    return {
        "file_type": "docx",
        "file_name": Path(source).name or doc_path.name,
        "source": source,
        "html": '<article class="docx-paper">' + "".join(blocks_html) + "</article>",
        "outline": outline,
        "fallback_xpaths": fallback_xpaths,
    }


def read_docx_media(doc_path: Path, part_path: str) -> tuple[bytes, str]:
    """Read an embedded media file inside a DOCX package safely."""
    clean = str(part_path or "").replace("\\", "/").strip()
    while clean.startswith("/"):
        clean = clean[1:]
    if not clean.startswith("word/media/") or ".." in clean.split("/") or "\x00" in clean:
        raise PermissionError("Invalid DOCX media path")
    with ZipFile(str(doc_path), "r") as zf:
        if clean not in zf.namelist():
            raise FileNotFoundError("DOCX media part not found")
        raw = zf.read(clean)
    lower = clean.lower()
    mime = "application/octet-stream"
    if lower.endswith(".png"):
        mime = "image/png"
    elif lower.endswith(".jpg") or lower.endswith(".jpeg"):
        mime = "image/jpeg"
    elif lower.endswith(".gif"):
        mime = "image/gif"
    elif lower.endswith(".webp"):
        mime = "image/webp"
    elif lower.endswith(".svg"):
        mime = "image/svg+xml"
    return raw, mime


# -----------------------------------------------------------------------------
# XLSX Multi-Sheet Preview + A1 Coordinate Parsing
# -----------------------------------------------------------------------------
def _col_letters_to_index(letters: str) -> int:
    total = 0
    for ch in letters.upper():
        if "A" <= ch <= "Z":
            total = (total * 26) + (ord(ch) - ord("A") + 1)
    return total


def _col_index_to_letters(index: int) -> str:
    if index < 1:
        return ""
    chars = []
    num = index
    while num > 0:
        rem = (num - 1) % 26
        chars.append(chr(ord("A") + rem))
        num = (num - 1) // 26
    chars.reverse()
    return "".join(chars)


def _split_a1_cell(token: str) -> tuple[int, int] | None:
    clean = str(token or "").replace("$", "").strip().upper()
    if not clean:
        return None
    letters = []
    digits = []
    seen_digit = False
    for ch in clean:
        if "A" <= ch <= "Z" and not seen_digit:
            letters.append(ch)
        elif ch.isdigit():
            seen_digit = True
            digits.append(ch)
        else:
            return None
    if not letters or not digits:
        return None
    col = _col_letters_to_index("".join(letters))
    row = int("".join(digits))
    if row <= 0 or col <= 0:
        return None
    return (row, col)


def parse_a1_range(range_text: str) -> dict | None:
    """Parse A1 or A1:D10 (optionally prefixed with Sheet1!) into 1-based bounds without regex."""
    raw = str(range_text or "").strip()
    if not raw:
        return None
    sheet_hint = ""
    if "!" in raw:
        sheet_part, cell_part = raw.rsplit("!", 1)
        sheet_hint = sheet_part.strip()
        if sheet_hint.startswith("'") and sheet_hint.endswith("'") and len(sheet_hint) >= 2:
            sheet_hint = sheet_hint[1:-1].replace("''", "'")
        raw = cell_part.strip()
    if ":" in raw:
        left, right = raw.split(":", 1)
        start = _split_a1_cell(left)
        end = _split_a1_cell(right)
        if start is None or end is None:
            return None
        return {
            "sheet_name": sheet_hint,
            "min_row": min(start[0], end[0]),
            "min_col": min(start[1], end[1]),
            "max_row": max(start[0], end[0]),
            "max_col": max(start[1], end[1]),
        }
    single = _split_a1_cell(raw)
    if single is None:
        return None
    return {
        "sheet_name": sheet_hint,
        "min_row": single[0],
        "min_col": single[1],
        "max_row": single[0],
        "max_col": single[1],
    }


def _format_excel_cell_value(val) -> str:
    if val is None:
        return ""
    if isinstance(val, bool):
        if val:
            return "TRUE"
        return "FALSE"
    if isinstance(val, float):
        if val.is_integer():
            return str(int(val))
        return str(round(val, 6)).rstrip("0").rstrip(".")
    return str(val)


def render_xlsx_preview(
    doc_path: Path,
    source: str,
    target_sheet: str = "",
    target_range: str = "",
    chunk_id: str = "",
    knowledge_root: Path | None = None,
    *,
    indexed_chunk=None,
) -> dict:
    """Extract all sheets of an XLSX workbook with cell grid, merged regions, and target highlight bounds."""
    wb = openpyxl.load_workbook(str(doc_path), data_only=True, read_only=False)
    try:
        resolved_sheet = str(target_sheet or "").strip()
        resolved_range = str(target_range or "").strip()
        extra_range_texts = []
        if "," in resolved_range:
            parts = [seg.strip() for seg in resolved_range.split(",") if seg.strip()]
            if parts:
                resolved_range = parts[0]
                extra_range_texts.extend(parts)
        elif resolved_range:
            extra_range_texts.append(resolved_range)
        if not resolved_sheet or not resolved_range:
            chunk_rec = indexed_chunk
            if chunk_rec is None and chunk_id and knowledge_root is not None:
                chunk_rec = _load_knowledge_chunk(chunk_id, knowledge_root)
            if chunk_rec:
                bm_rec = chunk_rec.get("business_metadata") or {}
                locs = (bm_rec.get("provenance") or {}).get("locations") or []
                for loc in locs:
                    if not isinstance(loc, dict):
                        continue
                    if not resolved_sheet and loc.get("sheet_name"):
                        resolved_sheet = str(loc["sheet_name"])
                    if loc.get("cell_range"):
                        cr = str(loc["cell_range"]).strip()
                        if not resolved_range:
                            resolved_range = cr
                        if cr and cr not in extra_range_texts:
                            extra_range_texts.append(cr)
                cit_rec = bm_rec.get("citation") or {}
                if isinstance(cit_rec, dict):
                    if not resolved_sheet and cit_rec.get("sheet_name"):
                        resolved_sheet = str(cit_rec["sheet_name"])
                    for cr in cit_rec.get("cell_ranges") or ():
                        cr_txt = str(cr or "").strip()
                        if not resolved_range and cr_txt:
                            resolved_range = cr_txt
                        if cr_txt and cr_txt not in extra_range_texts:
                            extra_range_texts.append(cr_txt)

        parsed_target = parse_a1_range(resolved_range)
        if parsed_target and parsed_target.get("sheet_name") and not resolved_sheet:
            resolved_sheet = parsed_target["sheet_name"]
        parsed_targets = []
        for rt in extra_range_texts:
            pt = parse_a1_range(rt)
            if pt is not None:
                parsed_targets.append(pt)

        sheets_out = []
        max_rows_cap = 800
        max_cols_cap = 45
        for ws in wb.worksheets:
            raw_max_row = max(1, int(ws.max_row or 1))
            raw_max_col = max(1, int(ws.max_column or 1))

            row_start = 1
            row_end = min(raw_max_row, max_rows_cap)
            col_start = 1
            col_end = min(raw_max_col, max_cols_cap)

            if parsed_target and (not resolved_sheet or ws.title == resolved_sheet):
                if parsed_target["max_row"] > row_end:
                    row_start = max(1, parsed_target["min_row"] - 20)
                    row_end = min(raw_max_row, row_start + max_rows_cap - 1)
                if parsed_target["max_col"] > col_end:
                    col_start = max(1, parsed_target["min_col"] - 5)
                    col_end = min(raw_max_col, col_start + max_cols_cap - 1)

            merges = []
            for m_range in ws.merged_cells.ranges:
                if (
                    m_range.max_row >= row_start
                    and m_range.min_row <= row_end
                    and m_range.max_col >= col_start
                    and m_range.min_col <= col_end
                ):
                    merges.append({
                        "min_row": int(m_range.min_row),
                        "min_col": int(m_range.min_col),
                        "max_row": int(m_range.max_row),
                        "max_col": int(m_range.max_col),
                    })

            col_headers = []
            for c_idx in range(col_start, col_end + 1):
                col_headers.append({"col": c_idx, "label": _col_index_to_letters(c_idx)})

            rows_data = []
            for r_idx in range(row_start, row_end + 1):
                cells_row = []
                for c_idx in range(col_start, col_end + 1):
                    cell = ws.cell(row=r_idx, column=c_idx)
                    text = _format_excel_cell_value(cell.value)
                    is_bold = False
                    if cell.font is not None and bool(cell.font.bold):
                        is_bold = True
                    align = ""
                    if cell.alignment is not None and cell.alignment.horizontal:
                        align = str(cell.alignment.horizontal).lower()
                    cells_row.append({
                        "r": r_idx,
                        "c": c_idx,
                        "a1": _col_index_to_letters(c_idx) + str(r_idx),
                        "v": text,
                        "b": is_bold,
                        "al": align,
                    })
                rows_data.append({"row_num": r_idx, "cells": cells_row})

            frozen_rows_data = []
            frozen_row_count = 0
            if ws.freeze_panes:
                fp_cell = _split_a1_cell(str(ws.freeze_panes))
                if fp_cell and fp_cell[0] > 1:
                    frozen_row_count = min(4, fp_cell[0] - 1)
            if frozen_row_count == 0 and row_start > 1:
                frozen_row_count = min(4, row_start - 1)
            if frozen_row_count > 0 and row_start > 1:
                for fr_idx in range(1, frozen_row_count + 1):
                    f_cells = []
                    for c_idx in range(col_start, col_end + 1):
                        f_cell = ws.cell(row=fr_idx, column=c_idx)
                        f_cells.append({
                            "r": fr_idx,
                            "c": c_idx,
                            "a1": _col_index_to_letters(c_idx) + str(fr_idx),
                            "v": _format_excel_cell_value(f_cell.value),
                            "b": True,
                            "al": "",
                        })
                    frozen_rows_data.append({"row_num": fr_idx, "cells": f_cells})

            sheets_out.append({
                "name": ws.title,
                "state": str(ws.sheet_state or "visible"),
                "total_rows": raw_max_row,
                "total_cols": raw_max_col,
                "row_start": row_start,
                "row_end": row_end,
                "col_start": col_start,
                "col_end": col_end,
                "truncated": raw_max_row > row_end or raw_max_col > col_end,
                "col_headers": col_headers,
                "merges": merges,
                "frozen_row_count": frozen_row_count,
                "frozen_rows": frozen_rows_data,
                "rows": rows_data,
            })

        return {
            "file_type": "xlsx",
            "file_name": Path(source).name or doc_path.name,
            "source": source,
            "sheets": sheets_out,
            "target_sheet": resolved_sheet,
            "target_range": parsed_target,
            "target_ranges": parsed_targets,
        }
    finally:
        wb.close()


# -----------------------------------------------------------------------------
# PDF Full Reader Manifest, Page Rendering & Level-2 OCR Passage Locator
# -----------------------------------------------------------------------------
def _ocr_cache_connection(cache_db_path: Path) -> sqlite3.Connection:
    cache_db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(cache_db_path), timeout=30.0, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(
        "CREATE TABLE IF NOT EXISTS page_ocr_lines ("
        "cache_key TEXT PRIMARY KEY, "
        "page_width INTEGER NOT NULL, "
        "page_height INTEGER NOT NULL, "
        "ocr_source TEXT NOT NULL, "
        "lines_json TEXT NOT NULL, "
        "created_at REAL NOT NULL"
        ")"
    )
    conn.commit()
    return conn


def get_pdf_manifest(doc_path: Path, source: str) -> dict:
    """Return page count and normalized page sizes for the full-PDF reader."""
    doc = pymupdf.open(str(doc_path))
    try:
        pages = []
        for idx in range(len(doc)):
            rect = doc[idx].rect
            pages.append({
                "page_number": idx + 1,
                "width": round(float(rect.width), 2),
                "height": round(float(rect.height), 2),
            })
        return {
            "file_type": "pdf",
            "file_name": Path(source).name or doc_path.name,
            "source": source,
            "total_pages": len(doc),
            "pages": pages,
        }
    finally:
        doc.close()


def render_pdf_page_png(doc_path: Path, page_number: int, dpi: int = 144) -> bytes:
    """Render a single PDF page to PNG bytes."""
    bounded_dpi = max(72, min(int(dpi), 220))
    doc = pymupdf.open(str(doc_path))
    try:
        if page_number < 1 or page_number > len(doc):
            raise IndexError("Page number out of range")
        pix = doc[page_number - 1].get_pixmap(dpi=bounded_dpi)
        return pix.tobytes("png")
    finally:
        doc.close()


def render_pdf_region_png(doc_path: Path, page_number: int, bbox: tuple[float, ...]) -> bytes:
    """Render normalized display coordinates at up to 480 DPI, bounded to 2400 px per side."""
    if len(bbox) != 4 or not (0 <= bbox[0] < bbox[2] <= 1 and 0 <= bbox[1] < bbox[3] <= 1):
        raise ValueError("Invalid PDF region")
    with pymupdf.open(str(doc_path)) as doc:
        if page_number < 1 or page_number > len(doc):
            raise IndexError("Page number out of range")
        page = doc[page_number - 1]
        rect = page.rect
        clip = pymupdf.Rect(
            rect.x0 + bbox[0] * rect.width, rect.y0 + bbox[1] * rect.height,
            rect.x0 + bbox[2] * rect.width, rect.y0 + bbox[3] * rect.height,
        )
        if clip.width < 1 or clip.height < 1:
            raise ValueError("PDF region is too small")
        # Leave room for MuPDF's outward rounding at both edges.
        scale = min(480 / 72, 2398 / max(clip.width, clip.height))
        pixmap = page.get_pixmap(
            matrix=pymupdf.Matrix(scale, scale), clip=clip,
            colorspace=pymupdf.csRGB, alpha=False, annots=False,
        )
        return pixmap.tobytes("png")


def _extract_native_pdf_lines(page) -> list[dict]:
    words = page.get_text("words")
    if not words or len(words) < 4:
        return []
    rect = page.rect
    pw = max(1.0, float(rect.width))
    ph = max(1.0, float(rect.height))
    grouped = {}
    for w_item in words:
        x0, y0, x1, y1, word_text, block_no, line_no, _ = w_item
        clean = str(word_text or "").strip()
        if not clean:
            continue
        key = (int(block_no), int(line_no))
        if key not in grouped:
            grouped[key] = {"words": [clean], "x0": x0, "y0": y0, "x1": x1, "y1": y1}
        else:
            ln = grouped[key]
            ln["words"].append(clean)
            ln["x0"] = min(ln["x0"], x0)
            ln["y0"] = min(ln["y0"], y0)
            ln["x1"] = max(ln["x1"], x1)
            ln["y1"] = max(ln["y1"], y1)
    out = []
    for ln in grouped.values():
        joined = " ".join(ln["words"])
        f = fold_search_text(joined)
        if not f:
            continue
        out.append({
            "text": joined,
            "fold": f,
            "box": [
                round(max(0.0, ln["x0"] / pw), 4),
                round(max(0.0, ln["y0"] / ph), 4),
                round(min(1.0, ln["x1"] / pw), 4),
                round(min(1.0, ln["y1"] / ph), 4),
            ],
        })
    out.sort(key=lambda r: (round(r["box"][1], 2), r["box"][0]))
    return out


def _extract_ocr_pdf_lines(page) -> list[dict]:
    cmd = _configure_tesseract()
    if not cmd:
        return []
    rect = page.rect
    dpi = 125
    if float(rect.width) > float(rect.height):
        dpi = 145
    pix = page.get_pixmap(dpi=dpi, colorspace=pymupdf.csGRAY)
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    w, h = img.size
    if w <= 0 or h <= 0:
        return []
    data = pytesseract.image_to_data(img, lang="vie", output_type=pytesseract.Output.DICT)
    grouped = {}
    total = len(data.get("text") or [])
    for idx in range(total):
        txt = str(data["text"][idx] or "").strip()
        if not txt:
            continue
        key = (
            int(data["block_num"][idx]),
            int(data["par_num"][idx]),
            int(data["line_num"][idx]),
        )
        x = int(data["left"][idx])
        y = int(data["top"][idx])
        bw = int(data["width"][idx])
        bh = int(data["height"][idx])
        if key not in grouped:
            grouped[key] = {"words": [txt], "x0": x, "y0": y, "x1": x + bw, "y1": y + bh}
        else:
            ln = grouped[key]
            ln["words"].append(txt)
            ln["x0"] = min(ln["x0"], x)
            ln["y0"] = min(ln["y0"], y)
            ln["x1"] = max(ln["x1"], x + bw)
            ln["y1"] = max(ln["y1"], y + bh)

    out = []
    for ln in grouped.values():
        joined = " ".join(ln["words"])
        f = fold_search_text(joined)
        if not f:
            continue
        out.append({
            "text": joined,
            "fold": f,
            "box": [
                round(max(0.0, ln["x0"] / w), 4),
                round(max(0.0, ln["y0"] / h), 4),
                round(min(1.0, ln["x1"] / w), 4),
                round(min(1.0, ln["y1"] / h), 4),
            ],
        })
    out.sort(key=lambda r: (round(r["box"][1], 2), r["box"][0]))
    return out


def get_pdf_page_lines(
    doc_path: Path,
    page_number: int,
    cache_db_path: Path,
    run_ocr_if_missing: bool = True,
) -> tuple[list[dict], str]:
    """Return cached or freshly extracted line bounding boxes for one PDF page."""
    stat = doc_path.stat()
    key_raw = f"{str(doc_path.resolve()).lower()}|{stat.st_size}|{int(stat.st_mtime_ns)}|{page_number}"
    cache_key = hashlib.sha256(key_raw.encode("utf-8")).hexdigest()

    with _OCR_DB_LOCK:
        conn = _ocr_cache_connection(cache_db_path)
        try:
            row = conn.execute(
                "SELECT ocr_source, lines_json FROM page_ocr_lines WHERE cache_key = ?",
                (cache_key,),
            ).fetchone()
        finally:
            conn.close()
    if row is not None:
        try:
            cached_lines = json.loads(row[1])
            if cached_lines:
                return cached_lines, str(row[0])
        except Exception:
            pass

    doc = pymupdf.open(str(doc_path))
    try:
        if page_number < 1 or page_number > len(doc):
            return [], "out_of_range"
        page = doc[page_number - 1]
        lines = _extract_native_pdf_lines(page)
        ocr_source = "native_text"
        if not lines:
            if not run_ocr_if_missing:
                return [], "uncached"
            lines = _extract_ocr_pdf_lines(page)
            ocr_source = "tesseract_vie"
        pw = int(round(float(page.rect.width)))
        ph = int(round(float(page.rect.height)))
    finally:
        doc.close()

    if lines:
        with _OCR_DB_LOCK:
            conn = _ocr_cache_connection(cache_db_path)
            try:
                conn.execute(
                    "INSERT OR REPLACE INTO page_ocr_lines "
                    "(cache_key, page_width, page_height, ocr_source, lines_json, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (cache_key, pw, ph, ocr_source, json.dumps(lines, ensure_ascii=False), time.time()),
                )
                conn.commit()
            finally:
                conn.close()
    return lines, ocr_source


def _build_target_text_for_chunk(
    excerpt: str,
    section_title: str,
    chunk_rec: dict | None,
) -> str:
    parts = []
    if section_title:
        parts.append(section_title)
    if excerpt:
        parts.append(excerpt)
    if chunk_rec:
        if chunk_rec.get("section_title"):
            parts.append(str(chunk_rec["section_title"]))
        if chunk_rec.get("content"):
            parts.append(str(chunk_rec["content"]))
        tbl = chunk_rec.get("table_data")
        if isinstance(tbl, dict):
            grid = tbl.get("grid")
            if isinstance(grid, list):
                for row in grid:
                    if isinstance(row, list):
                        parts.append(" ".join(str(cell) for cell in row))
            header_grid = tbl.get("header_grid")
            if isinstance(header_grid, list):
                for h_row in header_grid:
                    if isinstance(h_row, list):
                        parts.append(" ".join(str(cell) for cell in h_row if cell))
            headers = tbl.get("headers") or tbl.get("header")
            if isinstance(headers, list) and headers:
                parts.append(" ".join(str(h) for h in headers if h))
            rows = tbl.get("rows") or tbl.get("cells")
            if isinstance(rows, list):
                for row in rows:
                    if isinstance(row, list):
                        parts.append(" ".join(str(cell) for cell in row))
                    elif isinstance(row, dict):
                        parts.append(" ".join(str(val) for val in row.values()))
            ex = tbl.get("excel")
            if isinstance(ex, dict):
                ex_cols = ex.get("columns")
                if isinstance(ex_cols, dict) and ex_cols:
                    parts.append(" ".join(str(v) for v in ex_cols.values() if v))
                for ex_row in ex.get("rows") or ():
                    if isinstance(ex_row, dict):
                        vals = ex_row.get("values") or {}
                        row_txts = []
                        for cell_obj in vals.values():
                            if isinstance(cell_obj, dict):
                                row_txts.append(str(cell_obj.get("text") or cell_obj.get("value") or ""))
                            else:
                                row_txts.append(str(cell_obj or ""))
                        if any(row_txts):
                            parts.append(" ".join(t for t in row_txts if t))
    return "\n".join(parts)


def _match_ocr_lines_to_target(
    page_lines: list[dict],
    target_text: str,
    region_boxes: list[list[float]] | None = None,
) -> list[dict]:
    """Match OCR lines against chunk text and keep the primary contiguous cluster(s)."""
    target_fold = fold_search_text(target_text)
    if not target_fold or not page_lines:
        return []
    target_tokens = set(target_fold.split())
    raw_target_lines = [fold_search_text(ln) for ln in str(target_text).splitlines()]
    target_line_tokens = [set(ln.split()) for ln in raw_target_lines if len(ln) >= 6]

    candidates = []
    for idx, line in enumerate(page_lines):
        lf = str(line.get("fold") or "")
        if len(lf) < 5:
            continue
        l_tokens = set(lf.split())
        if not l_tokens:
            continue

        score = 0.0
        strong_anchor = False
        if len(lf) >= 12 and lf in target_fold:
            score = 1.0
            strong_anchor = True
        else:
            inter_all = len(l_tokens & target_tokens)
            global_cov = inter_all / len(l_tokens)
            best_line_cov = 0.0
            for tl_tok in target_line_tokens:
                if not tl_tok:
                    continue
                ov = len(l_tokens & tl_tok)
                cov = max(ov / len(l_tokens), ov / len(tl_tok))
                if cov > best_line_cov:
                    best_line_cov = cov
            score = max(global_cov, best_line_cov)
            if score >= 0.85 and len(l_tokens) >= 4:
                strong_anchor = True

        # Boost lines sitting inside or immediately under a cited visual region bbox
        if region_boxes and score >= 0.45:
            bx = line["box"]
            cx = (bx[0] + bx[2]) * 0.5
            cy = (bx[1] + bx[3]) * 0.5
            for rb in region_boxes:
                if (rb[0] - 0.04) <= cx <= (rb[2] + 0.04) and (rb[1] - 0.06) <= cy <= (rb[3] + 0.22):
                    score = min(1.0, score + 0.2)
                    strong_anchor = True
                    break

        if score >= 0.68 and (len(l_tokens) >= 3 or len(lf) >= 12):
            candidates.append({
                "idx": idx,
                "score": score,
                "strong": strong_anchor,
                "mass": len(lf) * score,
                "box": list(line["box"]),
                "text": str(line.get("text") or ""),
            })

    if not candidates:
        return []

    # Group spatially close candidates into vertical clusters to drop stray isolated hits
    candidates.sort(key=lambda c: (c["box"][1], c["box"][0]))
    clusters = []
    current = [candidates[0]]
    max_y = candidates[0]["box"][3]
    for cand in candidates[1:]:
        if cand["box"][1] - max_y <= 0.085:
            current.append(cand)
            max_y = max(max_y, cand["box"][3])
        else:
            clusters.append(current)
            current = [cand]
            max_y = cand["box"][3]
    clusters.append(current)

    scored_clusters = []
    for cluster in clusters:
        mass = sum(item["mass"] for item in cluster)
        has_strong = any(item["strong"] for item in cluster)
        if has_strong:
            mass = mass * 1.25
        scored_clusters.append((mass, cluster))
    scored_clusters.sort(key=lambda item: item[0], reverse=True)

    best_mass = scored_clusters[0][0]
    kept = []
    for mass, cluster in scored_clusters:
        if mass == best_mass or (mass >= best_mass * 0.45 and len(cluster) >= 2):
            kept.extend(cluster)

    kept.sort(key=lambda c: (c["box"][1], c["box"][0]))
    return kept


def _merge_line_boxes(matched_lines: list[dict]) -> tuple[list[dict], list[float] | None]:
    """Build padded per-line/block highlight boxes and overall focus bounding box."""
    if not matched_lines:
        return [], None
    boxes = []
    min_x0 = 1.0
    min_y0 = 1.0
    max_x1 = 0.0
    max_y1 = 0.0

    for item in matched_lines:
        x0, y0, x1, y1 = item["box"]
        px0 = round(max(0.0, x0 - 0.006), 4)
        py0 = round(max(0.0, y0 - 0.003), 4)
        px1 = round(min(1.0, x1 + 0.006), 4)
        py1 = round(min(1.0, y1 + 0.003), 4)
        min_x0 = min(min_x0, px0)
        min_y0 = min(min_y0, py0)
        max_x1 = max(max_x1, px1)
        max_y1 = max(max_y1, py1)

        if boxes:
            prev = boxes[-1]["bbox"]
            same_band = abs(py0 - prev[1]) <= 0.014 or (py0 <= prev[3] + 0.010 and abs(px0 - prev[0]) <= 0.12)
            if same_band and abs(py1 - prev[3]) <= 0.025:
                prev[0] = round(min(prev[0], px0), 4)
                prev[1] = round(min(prev[1], py0), 4)
                prev[2] = round(max(prev[2], px1), 4)
                prev[3] = round(max(prev[3], py1), 4)
                continue

        boxes.append({
            "kind": "text",
            "bbox": [px0, py0, px1, py1],
            "label": "",
        })

    focus_box = [round(min_x0, 4), round(min_y0, 4), round(max_x1, 4), round(max_y1, 4)]
    return boxes, focus_box


def locate_pdf_passages(
    doc_path: Path,
    cache_db_path: Path,
    page_numbers: list[int],
    excerpt: str = "",
    section_title: str = "",
    chunk_id: str = "",
    images: list[dict] | None = None,
    knowledge_root: Path | None = None,
    *,
    indexed_chunk=None,
) -> dict:
    """Locate Level-2 highlight boxes on target PDF pages for a citation."""
    chunk_rec = indexed_chunk
    if chunk_rec is None and chunk_id and knowledge_root is not None:
        chunk_rec = _load_knowledge_chunk(chunk_id, knowledge_root)

    effective_pages = []
    seen = set()
    for p in page_numbers or []:
        if type(p) is int and p > 0 and p not in seen:
            seen.add(p)
            effective_pages.append(p)
    if not effective_pages and chunk_rec:
        bm_chunk = chunk_rec.get("business_metadata") or {}
        raw_p = (bm_chunk.get("citation") or {}).get("page_numbers") or []
        for p in raw_p:
            if type(p) is int and p > 0 and p not in seen:
                seen.add(p)
                effective_pages.append(p)
        if not effective_pages:
            locs = (bm_chunk.get("provenance") or {}).get("locations") or []
            for loc in locs:
                if isinstance(loc, dict):
                    p = loc.get("page_number")
                    if type(p) is not int and type(loc.get("page_index")) is int and loc["page_index"] >= 0:
                        p = loc["page_index"] + 1
                    if type(p) is int and p > 0 and p not in seen:
                        seen.add(p)
                        effective_pages.append(p)
    if not effective_pages:
        effective_pages = [1]

    region_by_page = {}
    candidate_imgs = list(images or [])
    has_bbox = False
    passed_asset_ids = set()
    for c_img in candidate_imgs:
        if not isinstance(c_img, dict):
            continue
        if isinstance(c_img.get("bbox"), list) and len(c_img["bbox"]) == 4:
            has_bbox = True
        raw_aid = str(c_img.get("asset_id") or c_img.get("sha256") or "").strip().lower()
        if not raw_aid and c_img.get("url"):
            u = str(c_img["url"]).strip()
            if "/api/assets/" in u:
                raw_aid = u.rsplit("/", 1)[-1].split("?", 1)[0].strip().lower()
        if raw_aid:
            passed_asset_ids.add(raw_aid)

    if not has_bbox and chunk_rec:
        rec_imgs = list(chunk_rec.get("images") or [])
        if passed_asset_ids:
            matched_rec_imgs = [
                ri for ri in rec_imgs
                if isinstance(ri, dict)
                and str(ri.get("asset_id") or ri.get("sha256") or "").strip().lower() in passed_asset_ids
            ]
            if matched_rec_imgs:
                candidate_imgs = matched_rec_imgs
            else:
                candidate_imgs = rec_imgs
        else:
            candidate_imgs = rec_imgs

    for img in candidate_imgs:
        if not isinstance(img, dict):
            continue
        p_num = img.get("page_number")
        bbox = img.get("bbox")
        if type(p_num) is int and p_num > 0 and isinstance(bbox, list) and len(bbox) == 4:
            clean_box = [round(float(v), 4) for v in bbox]
            region_by_page.setdefault(p_num, []).append({
                "kind": "region",
                "bbox": clean_box,
                "label": str(img.get("caption") or img.get("display_caption") or ""),
            })

    target_text = _build_target_text_for_chunk(excerpt, section_title, chunk_rec)
    pages_result = {}
    for p_num in effective_pages[:6]:
        page_lines, ocr_source = get_pdf_page_lines(doc_path, p_num, cache_db_path, run_ocr_if_missing=True)
        regions = region_by_page.get(p_num, [])
        region_bboxes = [r["bbox"] for r in regions]
        matched = _match_ocr_lines_to_target(page_lines, target_text, region_boxes=region_bboxes)
        text_boxes, focus_box = _merge_line_boxes(matched)

        combined_boxes = list(regions) + list(text_boxes)
        if focus_box is None and region_bboxes:
            focus_box = [
                min(b[0] for b in region_bboxes),
                min(b[1] for b in region_bboxes),
                max(b[2] for b in region_bboxes),
                max(b[3] for b in region_bboxes),
            ]
        elif focus_box is not None and region_bboxes:
            all_b = [focus_box] + region_bboxes
            focus_box = [
                round(min(b[0] for b in all_b), 4),
                round(min(b[1] for b in all_b), 4),
                round(max(b[2] for b in all_b), 4),
                round(max(b[3] for b in all_b), 4),
            ]

        pages_result[str(p_num)] = {
            "page_number": p_num,
            "ocr_source": ocr_source,
            "boxes": combined_boxes,
            "focus_box": focus_box,
        }

    return {"pages": pages_result}


def search_pdf_document(
    doc_path: Path,
    source: str,
    query: str,
    cache_db_path: Path,
    knowledge_root: Path | None = None,
    max_pages: int = 5,
    *,
    indexed_chunks=None,
) -> dict:
    """Search inside a PDF using published chunks, native text, and OCR cache."""
    clean_query = fold_search_text(query)
    if len(clean_query) < 2:
        return {"query": query, "matches": []}

    q_tokens = set(clean_query.split())
    candidate_pages = []
    seen_pages = set()

    # 1. Check published chunks for instant page discovery even on 143-page scanned PDFs
    if indexed_chunks is not None or knowledge_root is not None:
        if indexed_chunks is None:
            indexed_chunks = load_document_chunks_for_source(source, knowledge_root)
        for ch in indexed_chunks:
            hay = fold_search_text(
                str(ch.get("section_title") or "") + " " + str(ch.get("content") or "")
            )
            if not hay:
                continue
            matched = clean_query in hay
            if not matched and len(q_tokens) >= 2:
                if q_tokens.issubset(set(hay.split())):
                    matched = True
            if matched:
                p_list = ((ch.get("business_metadata") or {}).get("citation") or {}).get("page_numbers") or []
                for p in p_list:
                    if type(p) is int and p > 0 and p not in seen_pages:
                        seen_pages.add(p)
                        candidate_pages.append(p)

    # 2. If native text PDF or small PDF (< 15 pages), also scan remaining pages directly
    doc = pymupdf.open(str(doc_path))
    total_pages = len(doc)
    doc.close()

    if not candidate_pages:
        scan_limit = min(total_pages, 15)
        for p in range(1, scan_limit + 1):
            candidate_pages.append(p)
            seen_pages.add(p)

    matches = []
    for p_num in candidate_pages:
        if len(matches) >= max_pages:
            break
        lines, ocr_source = get_pdf_page_lines(doc_path, p_num, cache_db_path, run_ocr_if_missing=True)
        hit_boxes = []
        snippets = []
        for ln in lines:
            lf = str(ln.get("fold") or "")
            if not lf:
                continue
            hit = clean_query in lf
            if not hit and len(q_tokens) >= 2:
                if q_tokens.issubset(set(lf.split())):
                    hit = True
            if hit:
                hit_boxes.append({"kind": "search", "bbox": ln["box"], "label": ""})
                snippets.append(str(ln.get("text") or ""))
        if hit_boxes:
            matches.append({
                "page_number": p_num,
                "ocr_source": ocr_source,
                "boxes": hit_boxes,
                "focus_box": hit_boxes[0]["bbox"],
                "snippet": snippets[0][:140],
            })
        elif p_num in seen_pages and ocr_source != "uncached":
            # Published chunk matched this page even if OCR line broke across two lines
            matches.append({
                "page_number": p_num,
                "ocr_source": ocr_source,
                "boxes": [],
                "focus_box": None,
                "snippet": "Tìm thấy trong nội dung trang " + str(p_num),
            })

    return {"query": query, "matches": matches}


def prewarm_pdf_citations(
    citations: list[dict],
    allowed_dirs: list[Path],
    cache_db_path: Path,
) -> int:
    """Warm OCR line cache in the background for cited PDF pages."""
    if not isinstance(citations, list) or not citations:
        return 0
    warmed = 0
    seen_targets = set()
    for item in citations:
        if not isinstance(item, dict):
            continue
        src = str(item.get("source") or "").strip()
        if not src.lower().endswith(".pdf"):
            continue
        pages = item.get("page_numbers") or []
        if not pages:
            pages = [1]
        try:
            doc_path = find_document_path(src, allowed_dirs, allowed_exts=(".pdf",))
        except Exception:
            continue
        if doc_path is None:
            continue
        for p in pages[:2]:
            if type(p) is not int or p <= 0:
                continue
            key = (str(doc_path), p)
            if key in seen_targets:
                continue
            seen_targets.add(key)
            try:
                get_pdf_page_lines(doc_path, p, cache_db_path, run_ocr_if_missing=True)
                warmed += 1
            except Exception:
                continue
    return warmed
