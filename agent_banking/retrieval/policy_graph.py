"""Import verified policy links and use them for one-hop retrieval."""

import argparse
from contextlib import closing
import json
import logging
from pathlib import Path, PurePosixPath
import sqlite3

from agent_banking.core.search_text import search_key

GRAPH_FILENAME = "policy_graph.sqlite3"
_RELATIONS = {"amends", "replaces", "references"}
_FIELDS = ("from_source", "to_source", "relation", "evidence_source", "evidence_quote",
           "evidence_page", "evidence_section", "from_section", "to_section")
_MAX_ANCHORS = 3
_MAX_LINKS = 4
_LOGGER = logging.getLogger(__name__)


def _source(value):
    if not isinstance(value, str) or not value.strip() or "\\" in value or ":" in value:
        raise ValueError("Graph sources must be exact relative paths from the index")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != value:
        raise ValueError("Graph sources must be exact relative paths from the index")
    return value


def _validate_link(record):
    if not isinstance(record, dict) or record.keys() - set(_FIELDS):
        raise ValueError("Invalid policy link fields")
    link = {key: record.get(key) for key in _FIELDS}
    start, end, evidence = (_source(link[key]) for key in ("from_source", "to_source", "evidence_source"))
    if start == end or evidence not in {start, end} or link["relation"] not in _RELATIONS:
        raise ValueError("A policy link requires two distinct endpoints, a supported relation and evidence in one endpoint")
    for field in ("evidence_quote", "evidence_section", "from_section", "to_section"):
        value = link[field]
        if value is not None and (not isinstance(value, str) or not value.strip() or len(value) > 1000):
            raise ValueError(f"Invalid policy link field {field}")
        if value is not None:
            link[field] = value.strip()
    if len(link["evidence_quote"] or "") < 12:
        raise ValueError("A policy link requires a supporting quotation")
    page = link["evidence_page"]
    if page is not None and (type(page) is not int or page < 1):
        raise ValueError("evidence_page must be a positive PDF page number")
    if page is None and not link["evidence_section"]:
        raise ValueError("A policy link requires an evidence page or section")
    return link


def replace_links(db_path, records):
    """Publish a complete validated set of reviewed links in one transaction."""
    if not isinstance(records, list):
        raise ValueError("Policy links must be a JSON array")
    links = list(map(_validate_link, records))
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(db_path)) as connection, connection:
        connection.execute("BEGIN IMMEDIATE")
        connection.execute("""CREATE TABLE IF NOT EXISTS policy_links (
            from_source TEXT NOT NULL, to_source TEXT NOT NULL, relation TEXT NOT NULL,
            evidence_source TEXT NOT NULL, evidence_quote TEXT NOT NULL, evidence_page INTEGER,
            evidence_section TEXT, from_section TEXT, to_section TEXT
        )""")
        connection.execute("CREATE INDEX IF NOT EXISTS policy_links_source ON policy_links(from_source)")
        connection.execute("CREATE INDEX IF NOT EXISTS policy_links_target ON policy_links(to_source)")
        connection.execute("DELETE FROM policy_links")
        unique = dict.fromkeys(tuple(link[key] for key in _FIELDS) for link in links)
        connection.executemany("INSERT INTO policy_links VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", unique)
    return len(unique)


def _read_links(path, anchors):
    if not path or not anchors or not Path(path).is_file():
        return []
    placeholders = ",".join("?" for _ in anchors)
    try:
        with closing(sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True, timeout=0.2)) as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(
                f"SELECT * FROM policy_links WHERE from_source IN ({placeholders}) "
                f"OR (to_source IN ({placeholders}) AND relation IN ('amends', 'replaces')) "
                "ORDER BY CASE relation WHEN 'amends' THEN 0 WHEN 'replaces' THEN 1 ELSE 2 END, rowid",
                (*anchors, *anchors),
            )
            return [dict(row) for row in rows]
    except sqlite3.Error as error:
        _LOGGER.warning("Policy graph unavailable: %s", error)
        return []


def _section_matches(chunk, section):
    if not section:
        return True
    query = search_key(section)
    return bool(query) and f" {query} " in " " + search_key(
        f"{chunk.get('section_title', '')} {chunk.get('breadcrumb', '')}"
    ) + " "


def _quote_key(text):
    return " ".join(text.casefold().split())


def _matching_quote(chunk, link):
    pages = chunk.get("business_metadata", {}).get("citation", {}).get("page_numbers", ())
    if not pages:
        locations = chunk.get("business_metadata", {}).get("provenance", {}).get("locations", ())
        pages = [location["page_number"] for location in locations if type(location.get("page_number")) is int]
        pages.extend(location["page_index"] + 1 for location in locations if type(location.get("page_index")) is int)
    return (_section_matches(chunk, link["evidence_section"])
            and (not link["evidence_page"] or link["evidence_page"] in pages)
            and _quote_key(link["evidence_quote"]) in _quote_key(chunk.get("content", "")))


def verified_replacements(path, catalog):
    """Yield older source editions limited by an active, explicitly documented replacement."""
    if not path or not Path(path).is_file():
        return {}
    try:
        with closing(sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True, timeout=0.2)) as connection:
            connection.row_factory = sqlite3.Row
            links = list(connection.execute(
                "SELECT * FROM policy_links WHERE relation = 'replaces' AND from_source = evidence_source"))
    except sqlite3.Error as error:
        _LOGGER.warning("Policy replacements unavailable: %s", error)
        return {}
    retired = {}
    for link in links:
        for replacement in catalog.chunks_by_source.get(link["from_source"], ()):
            revision = catalog.revision(replacement)
            start = revision.get("effective_from")
            if (not start or start > catalog.day
                    or revision.get("effective_date_status") == "conflict" or not _matching_quote(replacement, link)):
                continue
            for previous in catalog.chunks_by_source.get(link["to_source"], ()):
                previous_start = catalog.revision(previous).get("effective_from")
                if not previous_start or previous_start < start:
                    retired[catalog.edition(previous)] = start
    return retired


def expand_links(path, candidates, catalog, labels, query, *, max_hops=1):
    """Expand search seeds (default 1-hop; opt-in multi-hop) and verify stored quotes in the indexed edition."""
    anchors = {}
    for item in candidates:
        chunk = item.get("chunk", item)
        source = chunk.get("source")
        if not catalog.allows_graph(chunk, labels):
            continue
        if source in anchors:
            anchors[source].append(chunk)
        elif source and len(anchors) < _MAX_ANCHORS:
            anchors[source] = [chunk]
    if not anchors:
        return []
    terms = set(search_key(query).split())
    expansions = []
    seen_pairs = set()
    hop_limit = max(1, int(max_hops or 1))
    frontier = dict(anchors)
    for _hop in range(hop_limit):
        if not frontier or len(expansions) >= _MAX_LINKS:
            break
        links = _read_links(path, list(frontier))
        if not links:
            break
        names = {link[key] for link in links for key in ("from_source", "to_source")}
        passages = {
            name: [chunk for chunk in catalog.chunks_by_source.get(name, ()) if catalog.allows_graph(chunk, labels)]
            for name in names
        }
        next_frontier = {}
        for link in links:
            start, end = link["from_source"], link["to_source"]
            forward = start in frontier
            if forward:
                anchor_name, target_name = start, end
                anchor_section, target_section = link["from_section"], link["to_section"]
            else:
                anchor_name, target_name = end, start
                anchor_section, target_section = link["to_section"], link["from_section"]
            anchor = next((chunk for chunk in frontier[anchor_name] if _section_matches(chunk, anchor_section)), None)
            if anchor is None:
                continue
            targets = [chunk for chunk in passages[target_name] if _section_matches(chunk, target_section)]
            if not targets:
                continue
            evidence = next((chunk for chunk in passages[link["evidence_source"]] if _matching_quote(chunk, link)), None)
            if evidence is None:
                continue
            target = max(
                targets,
                key=lambda chunk: (len(terms & set(search_key(chunk.get("content", "")).split())), chunk is evidence),
            )
            pair_key = (anchor.get("chunk_id"), target.get("chunk_id"), link.get("relation"))
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)
            details = {key: value for key, value in link.items() if key != "evidence_quote" and value is not None}
            details["evidence_chunk_id"] = evidence["chunk_id"]
            expansions.append((anchor, target, evidence, details))
            if target_name not in anchors and len(anchors) + len(next_frontier) < _MAX_ANCHORS * 2:
                next_frontier.setdefault(target_name, []).append(target)
            if len(expansions) >= _MAX_LINKS:
                break
        frontier = next_frontier
    return expansions


def main():
    from agent_banking.core.settings import PathConfig

    parser = argparse.ArgumentParser(description="Import reviewed policy relationships for GraphRAG")
    parser.add_argument("links", type=Path, help="JSON array of reviewed links")
    parser.add_argument("--db", type=Path, default=PathConfig.STORAGE_DIR / GRAPH_FILENAME)
    parser.add_argument("--replace", action="store_true", help="Replace the links in an existing graph database")
    args = parser.parse_args()
    if args.db.exists() and not args.replace:
        parser.error("Graph database already exists. Use --replace to publish a new link set.")
    count = replace_links(args.db, json.loads(args.links.read_text(encoding="utf-8-sig")))
    print(f"Published {count} reviewed policy link(s).")


if __name__ == "__main__":
    main()
