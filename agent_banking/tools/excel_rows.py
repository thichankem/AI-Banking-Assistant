"""Evaluate structured Excel lookups against rows in the published corpus."""

from collections import defaultdict
from datetime import date, datetime, time, timezone, timedelta
from decimal import Decimal, InvalidOperation
from operator import eq, ge, gt, le, lt, ne
from pathlib import PurePosixPath
import unicodedata

from agent_banking.chunking.universal.extraction.sources import source_key
from agent_banking.core.search_text import search_key


_OPERATORS = {"eq": eq, "ne": ne, "gt": gt, "gte": ge, "lt": lt, "lte": le}
_NUMERIC = {"number", "currency", "percent", "duration"}
_TEMPORAL = {"date": date, "datetime": datetime, "time": time}
_SAIGON = timezone(timedelta(hours=7))
_ADMIN_PREFIXES = ("thanh pho ", "tp ", "tinh ")
_EXCEL_TEXT_ALIASES = {
    "hcm": "ho chi minh",
    "tphcm": "ho chi minh",
    "sai gon": "ho chi minh",
    "hn": "ha noi",
    "thua thien hue": "hue",
    "pgd": "phong gd ngan hang",
    "phong giao dich": "phong gd ngan hang",
    "phong giao dich ngan hang": "phong gd ngan hang",
    "cn": "chi nhanh",
    "chi nhanh ngan hang": "chi nhanh",
    "cay atm": "atm",
    "may atm": "atm",
    "may cdm": "cdm",
    "may nop tien cdm": "cdm",
}


def canonical_excel_text(value: str) -> str:
    """Fold administrative prefixes and branch-type aliases for robust Excel text matching."""
    folded = search_key(unicodedata.normalize("NFC", str(value or "")))
    if not folded:
        return ""
    if folded in _EXCEL_TEXT_ALIASES:
        return _EXCEL_TEXT_ALIASES[folded]
    for prefix in _ADMIN_PREFIXES:
        if folded.startswith(prefix):
            rest = folded[len(prefix):].strip()
            if rest:
                folded = rest
                break
    if folded in _EXCEL_TEXT_ALIASES:
        return _EXCEL_TEXT_ALIASES[folded]
    return folded


def _result(status, **details):
    return {"status": status, "documents": [], "sources": [], "citations": [], **details}


def _version(chunks_by_version, as_of):
    if not chunks_by_version:
        return None, "excel_table_not_found"
    if not as_of and len(chunks_by_version) == 1:
        (version,) = chunks_by_version
        if version[2] is None or version[2] <= datetime.now(_SAIGON).date().isoformat():
            if version[4] != "conflicting":
                return version, None
    if any(not version[2] for version in chunks_by_version):
        return None, "excel_version_unknown"
    as_of = as_of or datetime.now(_SAIGON).date().isoformat()
    eligible = [version for version in chunks_by_version if version[2] <= as_of]
    if not eligible:
        return None, "excel_no_effective_version"
    latest = max(version[2] for version in eligible)
    versions = [version for version in eligible if version[2] == latest]
    if len(versions) != 1:
        return None, "excel_version_ambiguous"
    return versions[0], None


def _columns_for_table(chunks):
    columns = defaultdict(set)
    for _, data in chunks:
        for column, title in data["columns"].items():
            columns[column].add(title)
    return dict(columns)


def _typed_value(cell, query_value):
    kind = cell.get("type")
    source = cell.get("value")
    if source is None:
        return None
    if kind in _NUMERIC:
        if type(query_value) not in (str, int, float):
            return None
        value = str(query_value).strip()
        try:
            actual = Decimal(source)
            if kind == "percent":
                clean_expected = value.removesuffix("%")
            else:
                clean_expected = value
            expected = Decimal(clean_expected)
            if not (actual.is_finite() and expected.is_finite()):
                return None
            if kind == "percent":
                actual *= 100
            return actual, expected
        except (InvalidOperation, ValueError):
            return None
    if kind in _TEMPORAL and isinstance(query_value, str):
        parser = _TEMPORAL[kind].fromisoformat
        try:
            return parser(source), parser(query_value)
        except ValueError:
            return None
    if kind == "boolean" and type(query_value) is bool:
        return source, query_value
    if kind == "text" and isinstance(query_value, str):
        norm_source = unicodedata.normalize("NFC", source.strip()).casefold()
        norm_expected = unicodedata.normalize("NFC", query_value.strip()).casefold()
        return norm_source, norm_expected
    return None


def _test(cell, condition):
    operation = condition["op"]
    if operation == "is_blank":
        return cell is None or cell.get("type") == "text" and cell.get("value") == ""
    if not cell or condition.get("unit") and condition["unit"].casefold() != cell.get("unit", "").casefold():
        return False
    if operation == "contains":
        if cell.get("type") != "text" or not isinstance(condition.get("value"), str):
            return False
        raw_src = unicodedata.normalize("NFC", str(cell.get("value") or "")).strip()
        raw_q = unicodedata.normalize("NFC", condition["value"]).strip()
        if not raw_q:
            return False
        q_key = search_key(raw_q)
        if q_key:
            if (" " + q_key + " ") in (" " + search_key(raw_src) + " ") or raw_q.casefold() in raw_src.casefold():
                return True
            canon_q = canonical_excel_text(raw_q)
            canon_src = canonical_excel_text(raw_src)
            if canon_q and (" " + canon_q + " ") in (" " + canon_src + " "):
                return True
            return False
        return raw_q.casefold() in raw_src.casefold()
    pair = _typed_value(cell, condition["value"])
    if pair is None:
        return False
    try:
        matched = _OPERATORS[operation](*pair)
        if cell.get("type") == "text" and isinstance(condition.get("value"), str):
            if operation == "eq" and not matched:
                return canonical_excel_text(cell.get("value") or "") == canonical_excel_text(condition["value"])
            if operation == "ne" and matched:
                if canonical_excel_text(cell.get("value") or "") == canonical_excel_text(condition["value"]):
                    return False
        return matched
    except TypeError:
        return False


def _citation(chunk, data, row, index):
    meta = chunk["business_metadata"]
    citation_meta = meta.get("citation", {})
    areas = list(dict.fromkeys([row["range"], *(cell["origin"] for cell in row["values"].values()
                                               if cell.get("origin"))]))
    return {"index": index, "label": f"[{citation_meta.get('file_name', chunk['source'])} · {data['sheet']}!{row['range']}]",
            "source": chunk["source"], "chunk_id": f"{chunk['chunk_id']}:row:{row['row']}", "chunk_type": "table",
            "document_name": meta.get("document_name", ""), "sheet_name": data["sheet"], "cell_ranges": areas}


def _document(chunk, data, row, citation):
    display = [f"{data['columns'][key]}: {cell['text']}" for key, cell in row["values"].items() if cell.get("text")]
    meta = dict(chunk["business_metadata"])
    meta["citation"] = {**meta.get("citation", {}), "sheet_name": data["sheet"], "cell_ranges": citation["cell_ranges"]}
    return {"chunk_id": citation["chunk_id"], "source_chunk_id": chunk["chunk_id"],
            "source": chunk["source"], "chunk_type": "table",
            "content": "\n".join(display), "section_title": chunk.get("section_title", ""),
            "breadcrumb": chunk.get("breadcrumb", ""), "business_metadata": meta}


def _summarize_matches(columns, matches, requested_group_by=None):
    if not matches:
        return {"total_matches": 0, "group_counts": {}}
    selected_cols = []
    if isinstance(requested_group_by, (list, tuple)):
        for col_req in requested_group_by[:4]:
            if not isinstance(col_req, str) or not col_req.strip():
                continue
            norm_req = search_key(col_req)
            for key, labels in columns.items():
                if col_req.upper() == key or any(search_key(lbl) == norm_req for lbl in labels):
                    if key not in selected_cols:
                        selected_cols.append(key)
                    break
    if not selected_cols:
        priority_tokens = ("loại", "nhóm", "tỉnh", "thành phố", "quận", "huyện")
        ranked_cols = []
        for key, labels in columns.items():
            label_text = " / ".join(sorted(labels))
            low_label = label_text.casefold()
            prio = 0
            for idx_p, tok in enumerate(priority_tokens):
                if tok in low_label:
                    prio = len(priority_tokens) - idx_p
                    break
            distinct_vals = set()
            for _, _, row in matches:
                cell = row["values"].get(key)
                if cell and cell.get("text"):
                    distinct_vals.add(str(cell["text"]).strip())
            if 1 <= len(distinct_vals) <= 60 and prio > 0:
                ranked_cols.append((-prio, len(distinct_vals), key))
        ranked_cols.sort()
        selected_cols = [item[2] for item in ranked_cols[:4]]

    group_counts = {}
    for key in selected_cols:
        labels = columns.get(key) or {key}
        col_label = sorted(labels)[0]
        counts = defaultdict(int)
        for _, _, row in matches:
            cell = row["values"].get(key)
            if cell and cell.get("text"):
                val = " ".join(str(cell["text"]).split())
                if val:
                    counts[val] += 1
        if counts:
            ordered_counts = dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))
            group_counts[col_label] = ordered_counts

    summary = {"total_matches": len(matches), "group_counts": group_counts}
    if len(matches) <= 35:
        name_keys = []
        for key, labels in columns.items():
            lbl_low = " ".join(sorted(labels)).casefold()
            if any(tok in lbl_low for tok in ("tên", "khoản mục", "mã phí", "địa chỉ", "loại")):
                name_keys.append(key)
        if not name_keys:
            name_keys = list(columns.keys())[:3]
        items = []
        for _, _, row in matches:
            parts = []
            for key in name_keys[:4]:
                cell = row["values"].get(key)
                if cell and cell.get("text"):
                    val = " ".join(str(cell["text"]).split())
                    if val:
                        parts.append(val)
            if parts:
                items.append(" | ".join(parts)[:140])
        if items:
            summary["matched_items"] = items
    return summary


def _summary_doc_and_citation(first_chunk, first_data, summary, index):
    meta = dict(first_chunk["business_metadata"])
    citation_meta = meta.get("citation", {})
    sheet_name = first_data.get("sheet", "")
    total = summary.get("total_matches", 0)
    summary_range = f"Tổng hợp {total} dòng"
    meta["citation"] = {**citation_meta, "sheet_name": sheet_name, "cell_ranges": [summary_range]}
    file_label = citation_meta.get("file_name", first_chunk["source"])
    citation = {
        "index": index,
        "label": f"[{file_label} · {sheet_name}!{summary_range}]",
        "source": first_chunk["source"],
        "chunk_id": f"{first_chunk['chunk_id']}:summary",
        "chunk_type": "table",
        "document_name": meta.get("document_name", ""),
        "sheet_name": sheet_name,
        "cell_ranges": [summary_range],
    }
    lines = [f"Tổng số dòng khớp trong bảng: {total}"]
    for col_label, counts in (summary.get("group_counts") or {}).items():
        parts = [f"{k}: {v}" for k, v in list(counts.items())[:40]]
        lines.append(f"Phân bổ theo {col_label}: " + "; ".join(parts))
    items = summary.get("matched_items") or []
    if items:
        lines.append(f"Danh sách đầy đủ ({len(items)} dòng):")
        for item in items:
            lines.append("- " + item)
    doc = {
        "chunk_id": citation["chunk_id"],
        "source_chunk_id": first_chunk["chunk_id"],
        "source": first_chunk["source"],
        "chunk_type": "table",
        "content": "\n".join(lines),
        "section_title": first_chunk.get("section_title", ""),
        "breadcrumb": first_chunk.get("breadcrumb", ""),
        "business_metadata": meta,
    }
    return doc, citation


def lookup_excel_rows(corpus, excel, *, scope="", source="", limit=3, offset=0):
    requested_source = source_key(source)
    table_id = excel["table_id"]
    as_of = excel.get("as_of")
    requested_group_by = excel.get("group_by")
    if as_of:
        try:
            if date.fromisoformat(as_of).isoformat() != as_of:
                raise ValueError
        except (ValueError, TypeError):
            return _result("excel_invalid_date")
    all_versions = defaultdict(list)
    for chunk in corpus:
        metadata = chunk.get("business_metadata", {})
        if metadata.get("excel_table_id") != table_id:
            continue
        if scope and scope not in metadata.get("publication_scopes", ()):
            continue
        actual_source = source_key(chunk["source"])
        if requested_source:
            actual_name = PurePosixPath(actual_source).name
            actual_stem = actual_name.rsplit(".", 1)[0]
            if (
                requested_source not in {actual_source, actual_name, actual_stem}
                and not actual_name.startswith(requested_source + " ")
            ):
                continue
        data = (chunk.get("table_data") or {}).get("excel")
        if data is None:
            continue
        revision = metadata.get("excel_revision", {})
        key = (chunk["source"], metadata.get("document_version"), revision.get("effective_from"),
               revision.get("business_version"), revision.get("effective_date_status"))
        all_versions[key].append((chunk, data))
    selected, error = _version(all_versions, as_of)
    if error:
        return _result(error)
    chunks = all_versions[selected]
    columns = _columns_for_table(chunks)
    by_row = {}
    for chunk, data in chunks:
        for row in data["rows"]:
            by_row.setdefault(row["row"], (chunk, data, row))
    units = defaultdict(set)
    for _, _, row in by_row.values():
        for key, cell in row["values"].items():
            if cell.get("unit"):
                units[key].add(cell["unit"])
    conditions = []
    for condition in excel.get("filters", []):
        key = condition["column"]
        normalized = search_key(key)
        matches = [column for column, labels in columns.items()
                   if any(search_key(label) == normalized for label in labels)]
        if key.upper() in columns:
            matches = [key.upper()]
        if len(matches) != 1:
            if matches:
                col_status = "excel_column_ambiguous"
            else:
                col_status = "excel_column_not_found"
            return _result(col_status, excel_columns=_columns_to_json(columns))
        column = matches[0]
        if len(units[column]) > 1 and not condition.get("unit") and condition["op"] != "is_blank":
            return _result("excel_unit_ambiguous", excel_units=sorted(units[column]))
        conditions.append((column, condition))
    matches = [(chunk, data, row) for _, (chunk, data, row) in sorted(by_row.items())
               if all(_test(row["values"].get(column), condition) for column, condition in conditions)]
    page = matches[offset:offset + limit]
    documents = []
    citations = []
    for index, (chunk, data, row) in enumerate(page, 1):
        citation = _citation(chunk, data, row, index)
        citations.append(citation)
        documents.append(_document(chunk, data, row, citation))
    excel_summary = _summarize_matches(columns, matches, requested_group_by=requested_group_by)
    if documents and (len(matches) > len(page) or requested_group_by):
        first_chunk, first_data, _ = matches[0]
        sum_doc, sum_cit = _summary_doc_and_citation(first_chunk, first_data, excel_summary, len(documents) + 1)
        documents.append(sum_doc)
        citations.append(sum_cit)
    if documents:
        final_status = "success"
        final_sources = [selected[0]]
    else:
        final_status = "no_results"
        final_sources = []
    result = _result(final_status, documents=documents, citations=citations,
                     sources=final_sources, excel_match_count=len(matches),
                     excel_summary=excel_summary,
                     excel_columns=_columns_to_json(columns))
    if not documents and chunks:
        result["excel_checked_source"] = selected[0]
        result["excel_checked_sheet"] = str(chunks[0][1].get("sheet") or "")
        result["excel_total_rows"] = len(by_row)
    if offset + len(page) < len(matches):
        result["next_offset"] = offset + len(page)
    return result


def _columns_to_json(columns):
    return {column: sorted(labels) for column, labels in columns.items()}

