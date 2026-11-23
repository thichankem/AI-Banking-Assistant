"""Evaluate structured Excel lookups against rows in the published corpus."""

from collections import defaultdict
from datetime import date, datetime, time, timezone, timedelta
from decimal import Decimal, InvalidOperation
from operator import eq, ge, gt, le, lt, ne
from pathlib import PurePosixPath

from agent_banking.chunking.universal.extraction.sources import source_key
from agent_banking.core.search_text import search_key


_OPERATORS = {"eq": eq, "ne": ne, "gt": gt, "gte": ge, "lt": lt, "lte": le}
_NUMERIC = {"number", "currency", "percent", "duration"}
_TEMPORAL = {"date": date, "datetime": datetime, "time": time}
_SAIGON = timezone(timedelta(hours=7))


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
            expected = Decimal(value.removesuffix("%") if kind == "percent" else value)
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
        return source.strip().casefold(), query_value.strip().casefold()
    return None


def _test(cell, condition):
    operation = condition["op"]
    if operation == "is_blank":
        return cell is None or cell.get("type") == "text" and cell.get("value") == ""
    if not cell or condition.get("unit") and condition["unit"].casefold() != cell.get("unit", "").casefold():
        return False
    pair = _typed_value(cell, condition["value"])
    if pair is None:
        return False
    try:
        return _OPERATORS[operation](*pair)
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


def lookup_excel_rows(corpus, excel, *, scope="", source="", limit=3, offset=0):
    requested_source = source_key(source)
    table_id = excel["table_id"]
    as_of = excel.get("as_of")
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
        if requested_source and requested_source not in {actual_source, PurePosixPath(actual_source).name}:
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
            return _result("excel_column_ambiguous" if matches else "excel_column_not_found",
                           excel_columns=_columns_to_json(columns))
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
    result = _result("success" if documents else "no_results", documents=documents, citations=citations,
                     sources=[selected[0]] if documents else [], excel_match_count=len(matches),
                     excel_columns=_columns_to_json(columns))
    if offset + len(page) < len(matches):
        result["next_offset"] = offset + len(page)
    return result


def _columns_to_json(columns):
    return {column: sorted(labels) for column, labels in columns.items()}

