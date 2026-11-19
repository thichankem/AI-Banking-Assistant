"""Shared formatting for agent retrieval context."""

from agent_banking.agent.config.response_profiles import get_messages
from agent_banking.agent.config.settings import get_int_setting, get_setting
from agent_banking.core.search_text import is_procedure_query, search_key


_GUIDE_SCREEN_FIELDS = tuple(get_setting("guide_screen_fields", ()))
_MAX_TABLES_PER_CHUNK = get_int_setting("max_tables_per_chunk")
_MAX_TABLE_ROWS = get_int_setting("max_table_rows_per_table")
_MAX_IMAGES_PER_CHUNK = get_int_setting("max_images_per_chunk")
_TABLE_INTRO_MAX_CHARS = get_int_setting("table_intro_max_chars")
_TABLE_INTRO_BUDGET_DIVISOR = get_int_setting("table_intro_budget_divisor")
_TABLE_BUDGET_MARGIN_CHARS = get_int_setting("table_budget_margin_chars")
_GUIDE_SCAN_LINES = get_int_setting("guide_scan_lines")
_GUIDE_FRAGMENT_LINES = get_int_setting("guide_fragment_lines")
_GUIDE_MIN_INSTRUCTION_WORDS = get_int_setting("guide_min_instruction_words")
_GUIDE_TIME_PREFIX = str(get_setting("guide_time_prefix", ""))
_GUIDE_BRAND_MARKERS = tuple(get_setting("guide_brand_markers", ()))
_GUIDE_FOOTER_PREFIX = str(get_setting("guide_footer_prefix", ""))
_GUIDE_FOOTER_TOKEN = str(get_setting("guide_footer_token", ""))
_DOC_MESSAGES = get_messages("document_context")


def _unwrap_doc_dict(doc) -> dict:
    """Return canonical chunk dict whether wrapped in {'chunk': ...} or already flat."""
    if isinstance(doc, dict) and isinstance(doc.get("chunk"), dict):
        return doc["chunk"]
    if isinstance(doc, dict):
        return doc
    return {}


def _render_table_data_text(table_data, query="", budget=None) -> str:
    """Render canonical chunk table_data (str, dict, or list) into concise Markdown."""
    if not table_data:
        return ""
    if isinstance(table_data, str):
        return table_data.strip()

    tables = []
    if isinstance(table_data, dict):
        tables = [table_data]
    elif isinstance(table_data, list):
        if table_data and isinstance(table_data[0], dict):
            tables = table_data
        else:
            row_lines = []
            for row in table_data[:_MAX_TABLE_ROWS]:
                if isinstance(row, list):
                    cells = [str(c).strip() for c in row]
                    row_lines.append("| " + " | ".join(cells) + " |")
                elif row:
                    row_lines.append(str(row).strip())
            return "\n".join(row_lines)

    rendered = []
    query_terms = {term for term in search_key(query).split() if len(term) >= 3 or term.isdigit()}
    for tbl in tables[:_MAX_TABLES_PER_CHUNK]:
        if not isinstance(tbl, dict):
            continue
        headers = tbl.get("headers") or tbl.get("header") or []
        rows = tbl.get("rows") or tbl.get("cells") or []
        lines = []
        for key in ("title", "caption", "units", "unit", "conditions", "notes", "footnotes"):
            value = tbl.get(key)
            if isinstance(value, str) and value.strip():
                if value.strip() not in lines:
                    lines.append(value.strip())
            elif isinstance(value, list):
                lines.extend(note.strip() for note in value if isinstance(note, str) and note.strip())
        if isinstance(headers, list) and headers:
            h_cells = [str(h).strip() for h in headers]
            lines.append("| " + " | ".join(h_cells) + " |")
            lines.append("| " + " | ".join(["---"] * len(h_cells)) + " |")
        if isinstance(rows, list):
            row_lines = []
            for row in rows:
                if isinstance(row, list):
                    r_cells = [str(c).strip() for c in row]
                    row_lines.append("| " + " | ".join(r_cells) + " |")
                elif isinstance(row, dict):
                    r_vals = [str(v).strip() for v in row.values()]
                    row_lines.append("| " + " | ".join(r_vals) + " |")
            ranked = sorted(range(len(row_lines)), key=lambda i: -len(query_terms.intersection(search_key(row_lines[i]).split())))
            if budget is None:
                space = None
            else:
                space = budget - sum(len(line) + 1 for line in lines)
            selected = []
            for i in ranked:
                length = len(row_lines[i]) + 1
                if space is None or length <= space:
                    selected.append(i)
                    if space is not None:
                        space -= length
                if len(selected) == _MAX_TABLE_ROWS:
                    break
            if row_lines and not selected:
                continue
            lines.extend(row_lines[i] for i in sorted(selected))
        if lines:
            rendered.append("\n".join(lines))
        else:
            fallback_str = str(tbl.get("markdown") or tbl.get("summary") or tbl.get("caption") or "").strip()
            if fallback_str:
                rendered.append(fallback_str)
    return "\n\n".join(rendered)


def _render_images_text(images) -> str:
    """Extract readable image captions, descriptions, or OCR text from multimodal chunks."""
    if not isinstance(images, list) or not images:
        return ""
    parts = []
    for img in images[:_MAX_IMAGES_PER_CHUNK]:
        if isinstance(img, str):
            txt = img.strip()
            if txt:
                parts.append(txt)
            continue
        if not isinstance(img, dict):
            continue
        labels = []
        for key in ("caption", "description", "native_text", "ocr_text", "alt_text", "title"):
            val = str(img.get(key) or "").strip()
            if val and val not in labels:
                labels.append(val)
        if labels:
            parts.append(" - ".join(labels))
    return "; ".join(parts)


def _label_guide_screen(content, source, section, query):
    name = source.replace("\\", "/").rsplit("/", 1)[-1]
    if not search_key(name).startswith("hdsd ") or not search_key(section).startswith("buoc "):
        return content
    lines = content.splitlines()
    for index, line in enumerate(lines[:_GUIDE_SCAN_LINES]):
        if index and line.strip().startswith(_GUIDE_TIME_PREFIX):
            fragment = search_key(" ".join(lines[index:index + _GUIDE_FRAGMENT_LINES]))
            if not any(marker in fragment for marker in _GUIDE_BRAND_MARKERS):
                continue
            instructions = "\n".join(lines[:index]).strip()
            if len(instructions.split()) >= _GUIDE_MIN_INSTRUCTION_WORDS:
                screen = "\n".join(lines[index:])
                label = _DOC_MESSAGES.get("offscreen_instructions_header", "") + instructions
                screen_key = search_key(screen)
                if is_procedure_query(query) and all(field in screen_key for field in _GUIDE_SCREEN_FIELDS):
                    # The filled-in transaction mockup repeats across unrelated workflows.
                    footer = next(
                        (
                            line for line in lines[index:]
                            if search_key(line).startswith(_GUIDE_FOOTER_PREFIX)
                            and _GUIDE_FOOTER_TOKEN in search_key(line)
                        ),
                        "",
                    )
                    if footer:
                        label += _DOC_MESSAGES.get("footer_contact_header", "") + footer
                    return label
                return label + _DOC_MESSAGES.get("illustrative_screen_header", "") + screen
    return content


def _format_retrieved_doc_context(doc, max_chars=None, query="") -> str:
    """Format a 13-key universal chunk (or wrapper dict) for grading or grounded generation."""
    d = _unwrap_doc_dict(doc)
    if not d:
        return ""

    source_name = str(d.get("source") or "")
    breadcrumb = str(d.get("breadcrumb") or "")
    section_title = str(d.get("section_title") or "")
    chunk_type = str(d.get("chunk_type") or "text")
    meta = d.get("business_metadata")
    if not isinstance(meta, dict):
        meta = {}

    doc_name = str(meta.get("document_name") or "")
    domain = str(meta.get("domain") or "")
    category = str(meta.get("category") or "")
    retrieval_meta = meta.get("retrieval")
    if not isinstance(retrieval_meta, dict):
        retrieval_meta = {}
    situated = str(retrieval_meta.get("situated_context") or "").strip()

    display_source = doc_name
    if not display_source:
        display_source = source_name

    header_parts = []
    if display_source:
        header_parts.append(display_source)
    if breadcrumb and breadcrumb != display_source:
        header_parts.append(breadcrumb)
    if section_title and section_title not in breadcrumb:
        header_parts.append(section_title)
    prov = meta.get("provenance")
    if isinstance(prov, dict):
        locs = prov.get("locations")
        if isinstance(locs, list):
            pages = []
            for loc in locs:
                if isinstance(loc, dict):
                    p_num = loc.get("page_number")
                    if type(p_num) is int and p_num > 0 and p_num not in pages:
                        pages.append(p_num)
            if pages:
                pages.sort()
                header_parts.append("Trang " + ", ".join(str(p) for p in pages))

    meta_tags = []
    if chunk_type and chunk_type != "text":
        meta_tags.append(chunk_type)
    if domain:
        meta_tags.append(domain)
    if category:
        meta_tags.append(category)

    body_parts = []
    original = str(d.get("content") or "").strip()
    parent_raw = str(d.get("parent_content") or "").strip()
    parent = parent_raw
    if original and original in parent:
        parent = parent.replace(original, "", 1).strip()
    table_prose = []
    if d.get("table_data"):
        for part in (original, parent):
            for line in part.splitlines():
                line = line.strip()
                if line and not line.startswith("|") and line not in table_prose:
                    table_prose.append(line)
    table_intro = ""
    if table_prose:
        table_intro = _DOC_MESSAGES.get("section_text_header", "") + "\n".join(table_prose)
        if max_chars is not None:
            intro_budget = min(_TABLE_INTRO_MAX_CHARS, max_chars // _TABLE_INTRO_BUDGET_DIVISOR)
            if len(table_intro) > intro_budget:
                table_intro = table_intro[:intro_budget].rsplit(maxsplit=1)[0] + "…"
    table_budget = None
    if max_chars is not None:
        table_budget = max_chars - len(table_intro) - _TABLE_BUDGET_MARGIN_CHARS
    table_text = _render_table_data_text(d.get("table_data"), query, table_budget)
    if not table_text and table_intro and table_budget is not None:
        table_text = _render_table_data_text(d.get("table_data"), query, max_chars - _TABLE_BUDGET_MARGIN_CHARS)
        if table_text:
            table_intro = ""
    if table_text:
        if table_intro:
            body_parts.append(table_intro)
        body_parts.append(_DOC_MESSAGES.get("table_data_header", "") + table_text)
        if parent_raw and "|" in parent_raw:
            parent_table_lines = []
            for p_line in parent_raw.splitlines():
                s_line = p_line.strip()
                if s_line and s_line not in table_text and s_line not in table_intro and s_line not in parent_table_lines:
                    parent_table_lines.append(s_line)
            if parent_table_lines:
                extra_parent_table = "\n".join(parent_table_lines)
                current_len = sum(len(bp) + 1 for bp in body_parts)
                if max_chars is None or current_len + len(extra_parent_table) <= max_chars - 64:
                    body_parts.append("[Ngữ cảnh bảng đầy đủ]:\n" + extra_parent_table)
                elif max_chars is not None and max_chars - current_len > 240:
                    rem_budget = max_chars - current_len - 40
                    body_parts.append("[Ngữ cảnh bảng đầy đủ]:\n" + extra_parent_table[:rem_budget])
    if situated:
        body_parts.append(_DOC_MESSAGES.get("situated_context_prefix", "") + situated)

    if original and not table_text:
        text_source = original
        if parent_raw and len(parent_raw) > len(original) + 60 and len(original) < 1100:
            if max_chars is None or len(parent_raw) <= max_chars - 100:
                text_source = parent_raw
            else:
                text_source = original + "\n[Ngữ cảnh mục cha]:\n" + parent
        body_parts.append(_label_guide_screen(text_source, source_name, section_title, query))

    image_text = _render_images_text(d.get("images"))
    if image_text and image_text not in original:
        body_parts.append(_DOC_MESSAGES.get("image_context_prefix", "") + image_text)

    body_text = "\n".join(body_parts).strip()
    if max_chars is not None and len(body_text) > max_chars:
        body_text = body_text[:max_chars]

    header_line = ""
    if header_parts:
        header_line = _DOC_MESSAGES.get("source_header_prefix", "") + " | ".join(header_parts)
        if meta_tags:
            header_line = header_line + " (" + ", ".join(meta_tags) + ")"
    quality = meta.get("quality")
    if not isinstance(quality, dict):
        quality = {}
    uncertain_pages = quality.get("ocr_uncertain_pages", [])
    if not isinstance(uncertain_pages, list):
        uncertain_pages = []
    uncertain_pages = [page for page in uncertain_pages if type(page) is int and page > 0]
    if uncertain_pages:
        pages_text = ", ".join(str(page) for page in uncertain_pages)
        warning = _DOC_MESSAGES.get("ocr_uncertain_pages_template", "").replace("{pages}", pages_text)
        header_line = header_line + "\n" + warning
    if quality.get("ocr_uncertain_images") is True:
        header_line += _DOC_MESSAGES.get("ocr_uncertain_images_warning", "")
    if d.get("document_contact") is True:
        header_line += _DOC_MESSAGES.get("document_contact_flag", "")

    if header_line and body_text:
        return f"{header_line}\n{body_text}"
    if header_line:
        return header_line
    return body_text
