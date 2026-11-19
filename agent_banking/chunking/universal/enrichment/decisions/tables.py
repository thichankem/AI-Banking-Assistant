"""Decision-model table headings, titles, and cross-page continuation."""
from copy import deepcopy

from agent_banking.chunking.config.prompt_config import (
    TABLE_CONTINUATION_PROMPT,
    TABLE_STRUCTURE_TITLE_PROMPT,
    render_prompt,
)
from agent_banking.chunking.config.settings import get_config_section
from .images import adjacent_source
from ..image_context import region_key
from ...planning.source_spans import is_furniture_block
from ...text.language import project_vietnamese

_TABLE_DECISION_CFG = get_config_section("enrichment")["decisions"]["tables"]


def resolve_table_structures_and_titles(ir, ask_decisions, cache=None):
    """Use decision model to resolve unmarked table header counts and assign descriptive table titles."""
    blocks = ir["blocks"]
    table_entries = []
    questions = {}
    table_map = {}
    headings = {}
    by_id = {block["id"]: block for block in blocks}
    caption_sources = {}
    caption_targets = {}
    for relation in ir.get("relations", []):
        if relation["type"] == "caption_of" and relation["from"] in by_id:
            caption_sources.setdefault(relation["to"], []).append(by_id[relation["from"]])
            caption_targets.setdefault(relation["from"], set()).add(relation["to"])

    def caption_conflicts(source_id, table_id):
        targets = caption_targets.get(source_id)
        return bool(targets and table_id not in targets)

    for idx, block in enumerate(blocks):
        region = region_key(block)
        if block["type"] == "heading":
            heading = block.get("text", "")
            if ir.get("search_language") != "all":
                heading = project_vietnamese(heading)
            heading = heading.strip()
            if heading:
                headings[region] = (heading, block["id"])
            continue
        if block["type"] != "table":
            continue
        table = block["table_data"]
        grid = table.get("grid", [])
        if not grid:
            continue
        tid = "t" + str(len(table_entries))
        width = len(grid[0])
        max_cols = _TABLE_DECISION_CFG["max_cols_preview"]
        cell_chars = _TABLE_DECISION_CFG["cell_preview_chars"]
        top_rows = [[text[:cell_chars] for text in row[:max_cols]] for row in grid[:_TABLE_DECISION_CFG["top_rows_count"]]]
        row_bold = []
        for r_idx in range(min(_TABLE_DECISION_CFG["bold_check_rows"], len(grid))):
            r_cells = [c for c in table.get("cells", []) if c["row"] == r_idx and c.get("text", "").strip()]
            is_b = bool(r_cells) and all(c.get("all_text_bold", False) for c in r_cells)
            row_bold.append(is_b)
        banner = next((cell["text"].strip() for cell in table.get("cells", [])
                       if cell["row"] == 0 and cell["column"] == 0 and width > 1
                       and cell.get("col_span", cell.get("column_span", 1)) == width), "")
        preceding = adjacent_source(blocks, idx, -1, {"paragraph", "caption", "heading"})
        following = adjacent_source(blocks, idx, 1, {"caption", "paragraph"})
        title_spec = deepcopy(TABLE_STRUCTURE_TITLE_PROMPT["title"])
        title_criteria = title_spec["criteria"]
        if preceding and caption_conflicts(preceding["id"], block["id"]):
            preceding = None
            del title_criteria["PREV_BLOCK"]
        if following and caption_conflicts(following["id"], block["id"]):
            following = None
            del title_criteria["NEXT_BLOCK"]
        prev_text = ""
        prev_type = "none"
        if preceding:
            prev_text = preceding.get("text", "").strip()
            prev_type = preceding["type"]
        next_text = ""
        next_type = "none"
        if following:
            next_text = following.get("text", "").strip()
            next_type = following["type"]
        heading, heading_id = headings.get(region, ("", None))
        if caption_conflicts(heading_id, block["id"]):
            heading, heading_id = "", None
        entry = {
            "id": tid,
            "width": width,
            "total_rows": len(grid),
            "top_rows": top_rows,
            "row_bold": row_bold,
            "row_0_banner": bool(banner),
            "existing_header_count": table.get("header_row_count", 0),
            "prev_type": prev_type,
            "prev_text": prev_text[:_TABLE_DECISION_CFG["prev_text_chars"]],
            "next_type": next_type,
            "next_text": next_text[:_TABLE_DECISION_CFG["next_text_chars"]],
            "section_heading": heading[:_TABLE_DECISION_CFG["section_heading_chars"]],
        }
        table_entries.append(entry)
        if preceding is not None:
            prev_pair = (prev_text, preceding["id"])
        else:
            prev_pair = (prev_text, None)
        if following is not None:
            next_pair = (next_text, following["id"])
        else:
            next_pair = (next_text, None)
        table_map[tid] = (block, entry, banner, (heading, heading_id), prev_pair, next_pair)
        needs_header_check = (
            table.get("header_detection") == "unmarked"
            or (table.get("header_detection") == "visual_model_review" and not table.get("header_row_count"))
        )
        if needs_header_check and len(grid) >= 2 and "row_provenance" not in table:
            hdr_spec = deepcopy(TABLE_STRUCTURE_TITLE_PROMPT["header"])
            if len(grid) <= 2:
                hdr_spec["criteria"].pop("HEADERS_2", None)
            hdr_spec["instructions"] = render_prompt(hdr_spec["instructions"], tid=tid)
            questions[tid + "_hdr"] = hdr_spec
        title_spec["instructions"] = render_prompt(title_spec["instructions"], tid=tid)
        questions[tid + "_title"] = title_spec
    if not table_entries:
        return
    answers = ask_decisions("table_structure_and_title", {"tables": table_entries}, questions, cache=cache)
    for tid, (block, entry, banner, section_title, previous_title, next_title) in table_map.items():
        table = block["table_data"]
        grid = table["grid"]
        width = entry["width"]
        hdr_key = tid + "_hdr"
        if hdr_key in answers:
            hdr_choice = answers[hdr_key].get("choice", "HEADERS_1")
            count = {"HEADERS_0": 0, "HEADERS_1": 1, "HEADERS_2": 2}[hdr_choice]
            if 0 <= count < len(grid):
                table["header_row_count"] = count
                table["header_grid"] = grid[:count]
                table["rows"] = grid[count:]
                table["header_detection"] = "decision_model_headers"
        title_choice = answers.get(tid + "_title", {}).get("choice")
        resolved_title, title_source = "", None
        if title_choice:
            title_options = {"ROW_0_BANNER": (banner, None), "PREV_BLOCK": previous_title,
                             "NEXT_BLOCK": next_title, "SECTION_HEADING": section_title}
            resolved_title, title_source = title_options[title_choice]
            caption = block.get("caption", "")
            attached_captions = caption_sources.get(block["id"], [])
            caption_source = next((item["id"] for item in attached_captions if item["text"] == caption), None)
            if caption_source is None and caption == table.get("title") and caption_conflicts(table.get("title_source_block_id"), block["id"]):
                # A previous decision may have copied its title into the block's
                # caption fallback before the explicit caption link was known.
                block.pop("caption", None)
                caption = ""
            if not caption:
                own_caption = next((item for item in attached_captions if item["text"].strip()), None)
                if own_caption:
                    caption, caption_source = own_caption["text"], own_caption["id"]
            if not resolved_title:
                resolved_title, title_source = next((candidate for candidate in (
                    (caption, caption_source), section_title, previous_title) if candidate[0]), ("", None))
        header_count = table["header_row_count"]
        if header_count and (hdr_key in answers or title_choice):
            start_hdr_row = 0
            if banner and header_count >= 2:
                start_hdr_row = 1
                if not resolved_title:
                    resolved_title = banner
            headers = []
            for column in range(width):
                labels = []
                for row in grid[start_hdr_row:header_count]:
                    if row[column] and row[column] not in labels:
                        labels.append(row[column])
                headers.append(" / ".join(labels))
            if "row_provenance" not in table:
                table["headers"] = headers
        elif not header_count and "row_provenance" not in table and table.get("headers") and all(
            not str(label).strip() for label in table["headers"]
        ):
            table["headers"] = ["#" + str(column + 1) for column in range(width)]
        if not title_choice:
            continue
        table["title"] = resolved_title
        table.pop("title_source_block_id", None)
        if title_source:
            table["title_source_block_id"] = title_source
        if resolved_title and not block.get("caption"):
            block["caption"] = resolved_title


def table_header_cells(table):
    return [cell for cell in table.get("cells", []) if cell["row"] < table["header_row_count"]]


def _normalized_header_grid(grid):
    return [[" ".join(str(cell).split()) for cell in row] for row in grid]


def _header_cell_signatures(table):
    signatures = []
    for cell in table_header_cells(table):
        col_span = cell.get("column_span")
        if col_span is None:
            col_span = cell.get("col_span", 1)
        signatures.append((
            cell["row"],
            cell["column"],
            cell["row_span"],
            col_span,
            " ".join(str(cell["text"]).split()),
        ))
    return sorted(signatures)


def compatible_table_headers(first, continuation):
    count = continuation["header_row_count"]
    if not count:
        return True
    if (_normalized_header_grid(first["header_grid"]) != _normalized_header_grid(continuation["header_grid"])
            or [" ".join(str(h).split()) for h in first["headers"]] != [" ".join(str(h).split()) for h in continuation["headers"]]):
        return False
    cells = table_header_cells(continuation)
    if any(cell["row"] + cell["row_span"] > count for cell in cells):
        return False
    return _header_cell_signatures(first) == _header_cell_signatures(continuation)


def table_row_provenance(block):
    table = block["table_data"]
    return [{**deepcopy(block["provenance"]), "source_block_id": block["id"], "source_row": row}
            for row in range(table["header_row_count"] + len(table["rows"]))]


def record_header_occurrence(target, source, row_provenance):
    count = source["header_row_count"]
    if count:
        target.setdefault("header_occurrences", []).append({
            "row_provenance": row_provenance[:count], "grid": deepcopy(source["header_grid"]),
            "cells": deepcopy(table_header_cells(source)),
        })


def merge_continued_tables(ir, ask_decisions, cache=None):
    """Use decision model to determine whether consecutive multi-page PDF tables should merge."""
    blocks = ir["blocks"]
    if len(blocks) < 2:
        return
    pairs = []
    questions = {}
    pair_indices = {}
    last_table_idx = None
    pair_spans = {}
    for idx in range(len(blocks)):
        curr = blocks[idx]
        if curr["type"] != "table":
            continue
        if last_table_idx is not None:
            intervening = blocks[last_table_idx + 1:idx]
            prev = blocks[last_table_idx]
            prev_page = prev.get("provenance", {}).get("page_index", -2)
            curr_page = curr.get("provenance", {}).get("page_index", -1)
            t1 = prev["table_data"]
            t2 = curr["table_data"]
            if t1.get("grid"):
                width1 = len(t1["grid"][0])
            else:
                width1 = len(t1.get("headers", []))
            if t2.get("grid"):
                width2 = len(t2["grid"][0])
            else:
                width2 = len(t2.get("headers", []))
            max_intervening = _TABLE_DECISION_CFG["max_intervening_blocks"]
            max_intervening_words = _TABLE_DECISION_CFG["max_intervening_words"]
            label_chars = _TABLE_DECISION_CFG["intervening_label_chars"]
            row_cols = _TABLE_DECISION_CFG["row_preview_cols"]
            cell_chars = _TABLE_DECISION_CFG["cell_chars"]
            hdr_cols = _TABLE_DECISION_CFG["header_preview_cols"]
            non_furniture = [item for item in intervening if not is_furniture_block(item)]
            bridgeable = (
                len(non_furniture) <= max_intervening
                and all(
                    item["type"] in ("caption", "paragraph", "heading")
                    and len(item.get("text", "").split()) <= max_intervening_words
                    and item.get("provenance", {}).get("page_index") in (prev_page, curr_page)
                    for item in non_furniture
                )
            )
            if prev_page >= 0 and curr_page == prev_page + 1 and width1 > 0 and width1 == width2 and bridgeable:
                pid = "tbl_" + str(len(pairs))
                prev_tail = []
                if t1["rows"]:
                    prev_tail = [value[:cell_chars] for value in t1["rows"][-1][:row_cols]]
                curr_head = []
                if t2["rows"]:
                    curr_head = [value[:cell_chars] for value in t2["rows"][0][:row_cols]]
                pair_info = {
                    "id": pid,
                    "prev_page": prev_page,
                    "curr_page": curr_page,
                    "prev_bbox": prev.get("provenance", {}).get("bbox", []),
                    "curr_bbox": curr.get("provenance", {}).get("bbox", []),
                    "prev_headers": [value[:cell_chars] for value in t1["headers"][:hdr_cols]],
                    "curr_headers": [value[:cell_chars] for value in t2["headers"][:hdr_cols]],
                    "prev_last_row": prev_tail,
                    "curr_first_row": curr_head,
                }
                if non_furniture:
                    pair_info["intervening_labels"] = [item.get("text", "")[:label_chars] for item in non_furniture]
                pairs.append(pair_info)
                pair_indices[pid] = idx
                pair_spans[idx] = last_table_idx
                cont_spec = deepcopy(TABLE_CONTINUATION_PROMPT)
                cont_spec["instructions"] = render_prompt(cont_spec["instructions"], pid=pid)
                questions[pid] = cont_spec
        last_table_idx = idx
    if not questions:
        return
    answers = ask_decisions("pdf_table_continuation", {"table_pairs": pairs}, questions, cache=cache)
    merge_indices = {
        idx for pid, idx in pair_indices.items()
        if answers.get(pid, {}).get("choice") == "MERGE_CONTINUED"
    }
    if not merge_indices:
        return
    merged_blocks = []
    merged_sources = {}
    target_for_merged = {}
    for idx, block in enumerate(blocks):
        if idx in merge_indices:
            prev_idx = pair_spans[idx]
            prev = target_for_merged.get(prev_idx, blocks[prev_idx])
            if prev["type"] == "table":
                t1 = prev["table_data"]
                t2 = block["table_data"]
                if not compatible_table_headers(t1, t2):
                    # Both headers are source data. Keep them in their own tables
                    # while recording the model-confirmed continuation.
                    ir["relations"].append({"from": block["id"], "to": prev["id"], "type": "continues"})
                    merged_blocks.append(block)
                    continue
                row_offset = len(t1["header_grid"]) + len(t1["rows"])
                t2_header_count = t2["header_row_count"]
                if "row_provenance" not in t1:
                    t1["row_provenance"] = table_row_provenance(prev)
                    record_header_occurrence(t1, t1, t1["row_provenance"])
                next_rows = table_row_provenance(block)
                record_header_occurrence(t1, t2, next_rows)
                t1["row_provenance"].extend(next_rows[t2_header_count:])
                for cell in t2.get("cells", []):
                    if cell["row"] < t2_header_count:
                        continue
                    shifted = deepcopy(cell)
                    shifted["row"] = row_offset + (cell["row"] - t2_header_count)
                    t1.setdefault("cells", []).append(shifted)
                t1["rows"].extend(t2["rows"])
                t1["grid"] = t1["header_grid"] + t1["rows"]
                merged_sources[block["id"]] = (prev["id"], row_offset, t2_header_count)
                target_for_merged[idx] = prev
                continue
        merged_blocks.append(block)
    if merged_sources:
        for relation in ir.get("relations", []):
            for endpoint_key in ("from", "to"):
                if relation.get(endpoint_key) in merged_sources:
                    relation[endpoint_key] = merged_sources[relation[endpoint_key]][0]
        for index, block in enumerate(merged_blocks):
            block["order"] = index
        for unit in ir["units"]:
            source = unit["block_id"]
            if source in merged_sources:
                target, offset, header_count = merged_sources[source]
                unit["block_id"] = target
                unit["source_block_id"] = source
                unit["source_row"] = unit["row"]
                if unit["row"] >= header_count:
                    unit["row"] += offset - header_count
        ir["blocks"] = merged_blocks
