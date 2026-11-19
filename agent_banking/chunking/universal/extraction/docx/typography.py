"""Authored DOCX typography, outline levels and language annotations."""
from functools import lru_cache

from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.text.paragraph import Paragraph
from docx.text.run import Run

from agent_banking.chunking.config.settings import get_config_section
from .openxml import content_nodes

_DOCX_CFG = get_config_section("extraction")["docx"]
_TYPO_CFG = _DOCX_CFG["typography"]

MAX_HEADING_LEVEL = _DOCX_CFG["max_heading_level"]
_DEFAULT_FONT_PT = _DOCX_CFG["default_font_pt"]
_STYLE_INHERITANCE_HOPS = _DOCX_CFG["style_inheritance_hops"]
_UPPER_RATIO_NUM = _TYPO_CFG["upper_ratio_num"]
_UPPER_RATIO_DEN = _TYPO_CFG["upper_ratio_den"]
_SINGLE_LINE_MAX_WORDS = _TYPO_CFG["single_line_max_words"]
_BOLD_LABEL_MAX_WORDS = _TYPO_CFG["bold_label_max_words"]
_MAX_OUTLINE_LEVEL = _TYPO_CFG["max_outline_level"]
_PARAGRAPH_CACHE_MAXSIZE = _TYPO_CFG["paragraph_cache_maxsize"]
VANISH_PATH = "./" + qn("w:rPr") + "/" + qn("w:vanish")


def hidden_run_checker(document):
    """Resolve Word's style toggles and the run's absolute hidden override."""
    default_vanish = document.styles.element.find(
        "./" + qn("w:docDefaults") + "/" + qn("w:rPrDefault") + VANISH_PATH[1:]
    )
    doc_hidden = _vanish_on(default_vanish)
    styles = {style.style_id: style for style in document.styles}

    @lru_cache(maxsize=None)
    def style_toggle(style_id):
        toggle = False
        style = styles.get(style_id)
        seen = set()
        while style is not None and style.style_id not in seen:
            seen.add(style.style_id)
            if _vanish_on(style.element.find(VANISH_PATH)):
                toggle = not toggle
            style = style.base_style
        return toggle

    @lru_cache(maxsize=_PARAGRAPH_CACHE_MAXSIZE)
    def paragraph_toggle(node):
        style = Paragraph(node, document).style
        return style is not None and style_toggle(style.style_id)

    def hidden_run(node):
        if node.tag != qn("w:r"):
            return False
        direct = node.find(VANISH_PATH)
        if direct is not None:
            return _vanish_on(direct)
        paragraph = next(node.iterancestors(qn("w:p")), None)
        hidden = doc_hidden
        if paragraph is not None:
            hidden ^= paragraph_toggle(paragraph)
        style = node.find("./" + qn("w:rPr") + "/" + qn("w:rStyle"))
        if style is not None:
            hidden ^= style_toggle(style.get(qn("w:val")))
        return hidden

    return hidden_run


def _vanish_on(node):
    return node is not None and node.get(qn("w:val"), "true").casefold() not in ("0", "false", "off")


def resolve_run_font_pt(run, paragraph):
    if run.font is not None and run.font.size is not None:
        return round(run.font.size.pt, 1)
    run_style = getattr(run, "style", None)
    hops = 0
    while run_style is not None and hops < _STYLE_INHERITANCE_HOPS:
        font = getattr(run_style, "font", None)
        if font is not None and font.size is not None:
            return round(font.size.pt, 1)
        run_style = getattr(run_style, "base_style", None)
        hops += 1
    para_style = getattr(paragraph, "style", None)
    hops = 0
    while para_style is not None and hops < _STYLE_INHERITANCE_HOPS:
        font = getattr(para_style, "font", None)
        if font is not None and font.size is not None:
            return round(font.size.pt, 1)
        para_style = getattr(para_style, "base_style", None)
        hops += 1
    return _DEFAULT_FONT_PT


def build_typography_record(value, mode_font_pt, total_chars, size_chars, all_bold, all_italic, starts_bold_label, centered):
    clean_val = value.strip()
    letters = [ch for ch in clean_val if ch.isalpha()]
    is_upper = False
    if letters:
        upper_count = sum(1 for ch in letters if ch.isupper())
        if upper_count * _UPPER_RATIO_NUM >= len(letters) * _UPPER_RATIO_DEN:
            is_upper = True
    single_line = "\n" not in clean_val and len(clean_val.split()) <= _SINGLE_LINE_MAX_WORDS
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


def extract_paragraph_typography(paragraph, value, hidden_run=None):
    runs = visible_text_runs(paragraph, hidden_run)
    if not runs:
        return None
    size_chars = {}
    total_chars = 0
    for run in runs:
        sz = resolve_run_font_pt(run, paragraph)
        ch_len = len(run.text.strip())
        size_chars[sz] = size_chars.get(sz, 0) + ch_len
        total_chars += ch_len
    mode_font_pt = _DEFAULT_FONT_PT
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
        if first_txt.endswith(":") and len(first_txt.split()) <= _BOLD_LABEL_MAX_WORDS:
            starts_bold_label = True
    alignment = paragraph.alignment
    style = paragraph.style
    hops = 0
    while alignment is None and style is not None and hops < _STYLE_INHERITANCE_HOPS:
        alignment = getattr(getattr(style, "paragraph_format", None), "alignment", None)
        style = getattr(style, "base_style", None)
        hops += 1
    centered = alignment == WD_ALIGN_PARAGRAPH.CENTER
    return build_typography_record(
        value, mode_font_pt, total_chars, size_chars, all_bold, all_italic, starts_bold_label, centered
    )


def visible_text_runs(paragraph, hidden_run=None):
    runs = (Run(node, paragraph) for node in content_nodes(
        paragraph._p, stop_at=("drawing", "pict", "txbxContent"), hidden_run=hidden_run
    ) if node.tag == qn("w:r"))
    return [run for run in runs if run.text.strip()]


def heading_level(paragraph, text):
    if not text:
        return 0
    outline_path = "./" + qn("w:pPr") + "/" + qn("w:outlineLvl")
    outline = paragraph._p.find(outline_path)
    style = paragraph.style
    hops = 0
    while outline is None and style is not None and hops < _STYLE_INHERITANCE_HOPS:
        if getattr(style, "element", None) is not None:
            outline = style.element.find(outline_path)
        style = getattr(style, "base_style", None)
        hops += 1
    if outline is not None:
        try:
            value = int(outline.get(qn("w:val"), str(_MAX_OUTLINE_LEVEL)))
        except ValueError:
            return 0
        if 0 <= value < _MAX_OUTLINE_LEVEL:
            return value + 1
    # Only authored OpenXML outline metadata is structural evidence. Content,
    # capitalisation, font size and language-specific words are not rules.
    return 0


def _authored_language(paragraph, hidden_run=None):
    default = paragraph._p.find("./" + qn("w:pPr") + "/" + qn("w:rPr") + "/" + qn("w:lang"))
    if default is not None:
        default_tag = default.get(qn("w:val"), "")
    else:
        default_tag = ""
    tags = set()
    for node in content_nodes(paragraph._p, stop_at=("drawing", "pict", "txbxContent"), hidden_run=hidden_run):
        if node.tag != qn("w:r") or not any(
                ch.isalpha() for item in content_nodes(node) if item.tag == qn("w:t") for ch in (item.text or "")):
            continue
        language = node.find("./" + qn("w:rPr") + "/" + qn("w:lang"))
        if language is not None:
            tag = language.get(qn("w:val"), default_tag)
        else:
            tag = default_tag
        tags.add(tag.casefold())
    if len(tags) == 1:
        return tags.pop()
    return ""
