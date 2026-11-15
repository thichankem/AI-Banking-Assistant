"""Authored DOCX typography, outline levels and language annotations."""
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.text.run import Run

from .openxml_content import content_nodes


MAX_HEADING_LEVEL = 7


def resolve_run_font_pt(run, paragraph):
    if run.font is not None and run.font.size is not None:
        return round(run.font.size.pt, 1)
    run_style = getattr(run, "style", None)
    hops = 0
    while run_style is not None and hops < 10:
        font = getattr(run_style, "font", None)
        if font is not None and font.size is not None:
            return round(font.size.pt, 1)
        run_style = getattr(run_style, "base_style", None)
        hops += 1
    para_style = getattr(paragraph, "style", None)
    hops = 0
    while para_style is not None and hops < 10:
        font = getattr(para_style, "font", None)
        if font is not None and font.size is not None:
            return round(font.size.pt, 1)
        para_style = getattr(para_style, "base_style", None)
        hops += 1
    return 10.5


def build_typography_record(value, mode_font_pt, total_chars, size_chars, all_bold, all_italic, starts_bold_label, centered):
    clean_val = value.strip()
    letters = [ch for ch in clean_val if ch.isalpha()]
    is_upper = False
    if letters:
        upper_count = sum(1 for ch in letters if ch.isupper())
        if upper_count * 5 >= len(letters) * 4:
            is_upper = True
    single_line = "\n" not in clean_val and len(clean_val.split()) <= 45
    return {
        "font_pt": mode_font_pt,
        "char_count": total_chars,
        "size_chars": size_chars,
        "all_bold": bool(all_bold),
        "all_italic": bool(all_italic),
        "starts_bold_label": bool(starts_bold_label),
        "is_upper": is_upper,
        "centered": bool(centered),
        "single_line": single_line,
    }


def extract_paragraph_typography(paragraph, value):
    runs = visible_text_runs(paragraph)
    if not runs:
        return None
    size_chars = {}
    total_chars = 0
    for run in runs:
        sz = resolve_run_font_pt(run, paragraph)
        ch_len = len(run.text.strip())
        size_chars[sz] = size_chars.get(sz, 0) + ch_len
        total_chars += ch_len
    mode_font_pt = 10.5
    best_chars = -1
    for sz, ch_len in size_chars.items():
        if ch_len > best_chars:
            best_chars = ch_len
            mode_font_pt = sz
    all_bold = all(run.bold is True for run in runs)
    all_italic = all(run.italic is True for run in runs)
    starts_bold_label = False
    if not all_bold and runs[0].bold is True:
        first_txt = runs[0].text.strip()
        if first_txt.endswith(":") and len(first_txt.split()) <= 6:
            starts_bold_label = True
    alignment = paragraph.alignment
    style = paragraph.style
    hops = 0
    while alignment is None and style is not None and hops < 10:
        alignment = getattr(getattr(style, "paragraph_format", None), "alignment", None)
        style = getattr(style, "base_style", None)
        hops += 1
    centered = alignment == WD_ALIGN_PARAGRAPH.CENTER
    return build_typography_record(
        value, mode_font_pt, total_chars, size_chars, all_bold, all_italic, starts_bold_label, centered
    )


def visible_text_runs(paragraph):
    runs = (Run(node, paragraph) for node in content_nodes(
        paragraph._p, stop_at=("drawing", "pict", "txbxContent")
    ) if node.tag == qn("w:r"))
    return [run for run in runs if run.text.strip()]


def heading_level(paragraph, text):
    if not text:
        return 0
    outline_path = "./" + qn("w:pPr") + "/" + qn("w:outlineLvl")
    outline = paragraph._p.find(outline_path)
    style = paragraph.style
    hops = 0
    while outline is None and style is not None and hops < 10:
        if getattr(style, "element", None) is not None:
            outline = style.element.find(outline_path)
        style = getattr(style, "base_style", None)
        hops += 1
    if outline is not None:
        try:
            value = int(outline.get(qn("w:val"), "9"))
        except ValueError:
            return 0
        if 0 <= value < 9:
            return value + 1
    # Only authored OpenXML outline metadata is structural evidence. Content,
    # capitalisation, font size and language-specific words are not rules.
    return 0


def _authored_language(paragraph):
    default = paragraph._p.find("./" + qn("w:pPr") + "/" + qn("w:rPr") + "/" + qn("w:lang"))
    default_tag = default.get(qn("w:val"), "") if default is not None else ""
    tags = set()
    for node in content_nodes(paragraph._p, stop_at=("drawing", "pict", "txbxContent")):
        if node.tag != qn("w:r") or not any(
                ch.isalpha() for item in content_nodes(node) if item.tag == qn("w:t") for ch in (item.text or "")):
            continue
        language = node.find("./" + qn("w:rPr") + "/" + qn("w:lang"))
        tag = language.get(qn("w:val"), default_tag) if language is not None else default_tag
        tags.add(tag.casefold())
    if len(tags) == 1:
        return tags.pop()
    return ""
