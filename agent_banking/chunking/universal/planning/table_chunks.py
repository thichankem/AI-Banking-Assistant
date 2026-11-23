"""Build table chunks and searchable views of the preserved rows."""
from copy import deepcopy

from agent_banking.chunking.config.settings import get_config_section
from ..text.language import project_search_text, source_foreign_spans, without_foreign_spans
from .source_spans import build_lead_in, has_visible_text
from .text_chunks import flush_text, split_excel_text
from .table_rendering import markdown_row, table_fragment_data, table_markdown
from ..text.token_budget import format_passage_fields

_PLAN_CFG = get_config_section("planning")
_CTX_CFG = _PLAN_CFG["context_limits"]
_TBL_CFG = _PLAN_CFG["table_chunks"]


def _emit_excel_fragments(planner, chunk, parent_id):
    """Preserve one typed row on the first chunk and index every source cell segment."""
    metadata = chunk["business_metadata"]
    data = chunk["table_data"]
    excel = data["excel"]
    rows = excel["rows"]
    if len(rows) != 1:
        planner._emit(chunk)
        return
    row = rows[0]
    labels = excel["columns"]
    values = row["values"]
    question_column = excel["question_column"]
    key_column = question_column or next((key for key in labels if values.get(key, {}).get("text")), None)
    key = values.get(key_column, {}).get("text", "")
    retrieval = {name: value for name, value in metadata["retrieval"].items()
                 if name not in {"passage_content", "passage_coverage", "embedding_content", "part_coverage"}}
    prefix_budget = max(1, planner.budget.maximum // 4)
    full_key = ""
    if key:
        full_key = labels[key_column] + ": " + key
    key_prefix = full_key
    if not question_column or planner._measure(key_prefix + "\n\nAnswer: X", retrieval) is None:
        key_prefix = planner.budget.shorten(full_key, prefix_budget)
    if key_prefix and planner._measure(key_prefix + "\n\nX", retrieval) is None:
        key_prefix = ""
    groups = []
    pending = []

    def render(items, prefix=key_prefix):
        body = "\n".join(label + value for label, value, _ in items)
        if prefix:
            return prefix + "\n\n" + body
        return body

    for column, label in labels.items():
        cell = values.get(column, {})
        value = cell.get("text", "")
        if not value or column == key_column and key_prefix == full_key:
            continue
        label = planner.budget.shorten(label, max(1, prefix_budget // 2)) + ": "
        prefix = key_prefix
        if column == key_column:
            prefix = ""
        if planner._measure(prefix + "\n\n" + label + "X", retrieval) is None:
            label = ""
        if planner._measure(prefix + "\n\nX", retrieval) is None:
            prefix = ""
        pieces = split_excel_text(
            planner.budget, value,
            lambda part: planner._measure(render([(label, part, None)], prefix), retrieval) is not None,
        )
        for start, end, part in pieces:
            item = (label, part, {"address": cell.get("origin", cell["address"]), "start": start, "end": end})
            if pending and (prefix != key_prefix or planner._measure(render([*pending, item]), retrieval, target=True) is None):
                groups.append((key_prefix, pending))
                pending = []
            if prefix != key_prefix:
                groups.append((prefix, [item]))
            else:
                pending.append(item)
    if pending:
        groups.append((key_prefix, pending))
    if not groups:
        planner._emit(chunk)
        return
    for part_number, (prefix, items) in enumerate(groups, 1):
        content = render(items, prefix)
        if part_number == 1:
            child = chunk
            child["content"] = content
        else:
            child = planner._prototype("text", content, parent_id, metadata["provenance"]["block_ids"], retrieval=retrieval)
            child["business_metadata"]["provenance"]["locations"] = deepcopy(metadata["provenance"]["locations"])
        child_meta = child["business_metadata"]
        child_meta["excel_table_id"] = excel["id"]
        child_meta["excel_row_part"] = {"row": row["row"], "part": part_number, "total": len(groups),
                                        "cells": [location for _, _, location in items]}
        child_meta["retrieval"] = deepcopy(retrieval)
        if planner.search_language == "vi":
            projection = project_search_text(content, search_language="vi")
            child_meta["retrieval"]["embedding_content"] = projection
            if not has_visible_text(projection):
                child_meta["retrieval"]["role"] = "attachment_only"
        planner._bound_passage(child)
        planner._emit(child)


def plan_table(planner, block, parent_id, lead_in_blocks=None):
    table = block["table_data"]
    headers = table["headers"]
    rows = table["rows"]
    pending = []
    header_count = table["header_row_count"]
    header_grid = table["grid"][:header_count]
    row_provenance = table.get("row_provenance", [])
    foreign_cells = {}
    projected_cells = {}
    if planner.search_language == "vi":
        for row_index, values in enumerate(table["grid"]):
            for column, value in enumerate(values):
                spans = source_foreign_spans(value)
                if spans:
                    foreign_cells[row_index, column] = spans
                    projected_cells[row_index, column] = without_foreign_spans(value, spans)

    def projected(values, row_index):
        return [projected_cells.get((row_index, column), value) for column, value in enumerate(values)]

    projected_grid = [projected(row, index) for index, row in enumerate(header_grid)]
    projected_headers = headers
    has_foreign_headers = any(row_index < header_count for row_index, _ in foreign_cells)
    if has_foreign_headers:
        projected_headers = []
        for column in range(len(headers)):
            labels = dict.fromkeys(row[column].strip() for row in projected_grid if row[column].strip())
            projected_headers.append(" / ".join(labels))
    searchable_header = any(has_visible_text(value) for row in projected_grid for value in row)
    searchable_body = any(has_visible_text(value) for index, row in enumerate(rows, start=header_count)
                          for value in projected(row, index))
    start_row = header_count
    ids = [block["id"]]
    table_title = table.get("title", "") or block.get("caption", "")
    retrieval = planner._retrieval(parent_id, block.get("context_quotes", []), "table")
    if table_title and not retrieval.get("situated_context"):
        title_projection = project_search_text(table_title, search_language=planner.search_language).strip()
        if title_projection:
            retrieval["situated_context"] = planner.budget.shorten(
                title_projection, min(_CTX_CFG["situated_max"], planner.budget.maximum // _CTX_CFG["situated_ratio"])
            )
    lead_text, lead_spans, lead_ids = build_lead_in(lead_in_blocks)
    lead_attached = not bool(lead_text)
    if not lead_text and lead_in_blocks is not None:
        lead_in_blocks.clear()

    def fits(content, target=False):
        return planner._measure(content, retrieval, target) is not None

    def with_lead(text):
        if not lead_attached and lead_text:
            return lead_text + "\n\n" + text
        return text

    def attach_lead_metadata(chunk):
        nonlocal lead_attached
        if not lead_attached and lead_text:
            prov = chunk["business_metadata"]["provenance"]
            prov["source_spans"] = deepcopy(lead_spans)
            lead_attached = True
            if lead_in_blocks is not None:
                lead_in_blocks.clear()
        return chunk

    def table_chunk(content, data, block_ids):
        source_rows = data.get("row_provenance", [])
        header_sources = row_provenance[:header_count]
        for occurrence in table.get("header_occurrences", []):
            header_sources = [*header_sources, *occurrence["row_provenance"]]
        chunk = planner._prototype("table", content, parent_id, block_ids, table_data=data, retrieval=retrieval)
        if "excel" in data:
            chunk["business_metadata"]["excel_table_id"] = data["excel"]["id"]
        if source_rows or header_sources:
            locations = []
            for row in [*header_sources, *source_rows]:
                location = {key: value for key, value in row.items() if key not in ("source_row", "source_block_id")}
                if location not in locations:
                    locations.append(location)
            chunk["business_metadata"]["provenance"]["locations"][-1:] = locations
        return chunk

    visible_headers = {value for value in headers if value}
    needs_header_geometry = any(value and not any(value in label for label in visible_headers)
                                for row in header_grid for value in row)
    header_has_formula = any("\n$$" in value or "$$\n" in value for row in header_grid for value in row)

    def render(selected, row_start, data=None):
        if (needs_header_geometry or header_has_formula
                or any("\n$$" in value or "$$\n" in value for row in selected for value in row)):
            if data is None:
                data = table_fragment_data(block, selected, row_start)
            return data["html"]
        return table_markdown(headers, selected)

    def project_rows(selected, row_start):
        return [projected(row, index) for index, row in enumerate(selected, start=row_start)]

    def make(selected, row_start):
        data = table_fragment_data(block, selected, row_start)
        chunk_ids = ids
        if not lead_attached and lead_text:
            chunk_ids = lead_ids + ids
        chunk = table_chunk(with_lead(render(selected, row_start, data)), data, chunk_ids)
        filtered_rows = project_rows(selected, row_start)
        elements = []
        for row_index, values in [*enumerate(header_grid), *enumerate(selected, row_start)]:
            for column, value in enumerate(values):
                for left, right, language in foreign_cells.get((row_index, column), ()):
                    location = block["provenance"]
                    if row_index < len(row_provenance):
                        location = row_provenance[row_index]
                    elements.append({"type": "foreign_language", "language": language, "original_type": "table_cell",
                                     "block_id": block["id"], "source_span": [left, right], "text": value[left:right],
                                     "location": {**location, "row": row_index, "column": column}})
        if elements:
            chunk["business_metadata"]["elements"] = elements
            chunk["business_metadata"]["retrieval"]["embedding_content"] = with_lead(
                table_markdown(projected_headers, filtered_rows))
        # When only the authored header remains searchable, give it one
        # carrier chunk. Searchable data rows already include their header.
        carries_header = searchable_header and not searchable_body and row_start == header_count
        has_searchable_data = any(has_visible_text(value) for row in filtered_rows for value in row)
        if lead_attached and not (carries_header or has_searchable_data):
            chunk["business_metadata"]["retrieval"]["role"] = "attachment_only"
        if selected:
            labels = [
                planner.budget.shorten(header, min(_TBL_CFG["header_shorten_max"], planner.budget.maximum // _TBL_CFG["header_shorten_ratio"]))
                for header in projected_headers
            ]
            preferred = with_lead(table_markdown(labels, filtered_rows))
            if not fits(preferred):
                preferred = "\n".join(markdown_row(row) for row in filtered_rows)
            planner._bound_passage(chunk, preferred)
        else:
            planner._bound_passage(chunk)
        return attach_lead_metadata(chunk)

    header_overflow = not fits(table_markdown(projected_headers, []))

    def emit(selected, row_start):
        chunk = make(selected, row_start)
        if "excel" in table and len(selected) == 1 and not fits(chunk["content"]):
            _emit_excel_fragments(planner, chunk, parent_id)
        else:
            planner._emit(chunk)

    def group_passage(selected, row_start):
        filtered_rows = project_rows(selected, row_start)
        if header_overflow:
            # When the header alone cannot fit, cap the independently
            # searchable rows without splitting the authored header.
            return with_lead("\n".join(markdown_row(row) for row in filtered_rows)), retrieval
        rendered = with_lead(render(selected, row_start))
        if has_foreign_headers or filtered_rows != selected:
            embedding_content = with_lead(table_markdown(projected_headers, filtered_rows))
            return rendered, {**retrieval, "embedding_content": embedding_content}
        return rendered, retrieval

    for index, row in enumerate(rows, start=header_count):
        if pending:
            final_row = index == header_count + len(rows) - 1
            pending_text, pending_retrieval = group_passage(pending, start_row)
            under_minimum = planner.budget.count(format_passage_fields(pending_text, pending_retrieval)) < planner.min_tokens
            use_target = not (final_row or under_minimum)
            if len(pending) < planner.table_rows:
                candidate, candidate_retrieval = group_passage(pending + [row], start_row)
                if planner._measure(candidate, candidate_retrieval, target=use_target) is not None:
                    pending.append(row)
                    continue
            emit(pending, start_row)
            pending = []
        start_row = index
        if not lead_attached and lead_text:
            candidate, candidate_retrieval = group_passage([row], index)
            if planner._measure(candidate, candidate_retrieval) is None:
                flush_text(planner, lead_in_blocks, parent_id)
                lead_attached = True
        if "excel" in table and not fits(render([row], index)):
            emit([row], index)
        else:
            pending.append(row)
    if pending:
        emit(pending, start_row)
    if not rows:
        if not lead_attached and lead_text and lead_in_blocks:
            flush_text(planner, lead_in_blocks, parent_id)
            lead_attached = True
        planner._emit(make([], header_count))
