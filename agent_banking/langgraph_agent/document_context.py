"""Shared formatting for agent retrieval context."""

def _unwrap_doc_dict(doc) -> dict:
    """Return canonical chunk dict whether wrapped in {'chunk': ...} or already flat."""
    if isinstance(doc, dict) and isinstance(doc.get("chunk"), dict):
        return doc["chunk"]
    if isinstance(doc, dict):
        return doc
    return {}


def _render_table_data_text(table_data) -> str:
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
            for row in table_data[:20]:
                if isinstance(row, list):
                    cells = [str(c).strip() for c in row]
                    row_lines.append("| " + " | ".join(cells) + " |")
                elif row:
                    row_lines.append(str(row).strip())
            return "\n".join(row_lines)

    rendered = []
    for tbl in tables[:3]:
        if not isinstance(tbl, dict):
            continue
        headers = tbl.get("headers") or tbl.get("header") or []
        rows = tbl.get("rows") or tbl.get("cells") or []
        lines = []
        if isinstance(headers, list) and headers:
            h_cells = [str(h).strip() for h in headers]
            lines.append("| " + " | ".join(h_cells) + " |")
            lines.append("| " + " | ".join(["---"] * len(h_cells)) + " |")
        if isinstance(rows, list):
            for row in rows[:20]:
                if isinstance(row, list):
                    r_cells = [str(c).strip() for c in row]
                    lines.append("| " + " | ".join(r_cells) + " |")
                elif isinstance(row, dict):
                    r_vals = [str(v).strip() for v in row.values()]
                    lines.append("| " + " | ".join(r_vals) + " |")
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
    for img in images[:5]:
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


def _format_retrieved_doc_context(doc, max_chars=None) -> str:
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

    meta_tags = []
    if chunk_type and chunk_type != "text":
        meta_tags.append(chunk_type)
    if domain:
        meta_tags.append(domain)
    if category:
        meta_tags.append(category)

    body_parts = []
    if situated:
        body_parts.append(f"[Ngữ cảnh]: {situated}")

    content = str(d.get("content") or "").strip()
    if content:
        body_parts.append(content)

    table_text = _render_table_data_text(d.get("table_data"))
    if table_text and table_text not in content:
        body_parts.append(f"[Bảng dữ liệu]:\n{table_text}")

    image_text = _render_images_text(d.get("images"))
    if image_text and image_text not in content:
        body_parts.append(f"[Hình ảnh/Minh họa]: {image_text}")

    body_text = "\n".join(body_parts).strip()
    if max_chars is not None and len(body_text) > max_chars:
        body_text = body_text[:max_chars]

    header_line = ""
    if header_parts:
        header_line = "### Nguồn: " + " | ".join(header_parts)
        if meta_tags:
            header_line = header_line + " (" + ", ".join(meta_tags) + ")"

    if header_line and body_text:
        return f"{header_line}\n{body_text}"
    if header_line:
        return header_line
    return body_text
