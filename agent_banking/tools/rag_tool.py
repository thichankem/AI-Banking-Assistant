# -*- coding: utf-8 -*-
"""
Banking RAG Tool for LangGraph / Agentic Workflows
======================================================================
Encapsulates RAG pipeline (query preparation, hybrid retrieval and reranking)
into a standard tool for AI Agents:
- Reuses classes and logic from pipeline.py.
- Independent of LangChain Runnable or BaseRetriever.
- Returns normalized payload for Context Grounding and Citation.
"""

from pathlib import Path
import time

from agent_banking.chunking.universal.planning.schema import payload_index_key
from agent_banking.core.policy_dates import policy_query_date
from agent_banking.core.numeric_tokens import extract_canonical_numeric_tokens
from agent_banking.core.search_text import is_procedure_query, search_key
from agent_banking.retrieval.hybrid_retriever import HybridRetriever, RetrievalUnavailable
from agent_banking.retrieval.policy_graph import GRAPH_FILENAME, expand_links
from agent_banking.retrieval.policy_versions import PolicyVersions
from agent_banking.tools.config.prompts import RAG_TOOL_DESCRIPTION, TOOL_TEMPLATES
from agent_banking.tools.config.settings import get_tool_runtime_config
from agent_banking.tools.excel_rows import canonical_excel_text, lookup_excel_rows
from agent_banking.tools.extensions.banking_query import decompose_subqueries
from agent_banking.tracing.tracer import trace_span
from agent_banking.vector_db.qdrant_store import QdrantBankingStore, compute_corpus_table_paths

_RAG_CFG = get_tool_runtime_config()["rag_tool"]
_HEADER_CFG = _RAG_CFG["guide_header"]
_LOGO_CFG = _RAG_CFG["guide_logo"]
_NEIGHBOR_CFG = _RAG_CFG["neighbors"]
_SOURCE_CFG = _RAG_CFG["source_matching"]

_PIN_STOPWORDS = frozenset({
    "cua", "cho", "trong", "theo", "quy", "dinh", "ngan", "hang", "abank",
    "khach", "va", "hay", "hoac", "nhu", "the", "nao", "bao", "nhieu",
    "cac", "nhung", "mot", "duoc", "khong", "khi", "sau", "truoc", "tai",
    "ve", "tu", "den", "la", "gi", "co", "voi", "tren", "duoi", "nay", "do",
})
_FEE_TABLE_CUES = (
    " bieu phi ", " bang phi ", " muc phi ", " phi chuyen doi ",
    " phi thuong nien ", " phu luc ", " ma phi ",
)
_DIAGRAM_CUES = (
    " hinh 1 ", " hinh 2 ", " hinh 3 ", " so do ",
    " luu do ", " chu trinh ", " lo trinh ", " thap ",
)
_DIAGRAM_CHUNK_MARKERS = (
    " hinh 1 ", " hinh 2 ", " hinh 3 ", " so do ",
    " quy trinh chuan ", " ma tran ",
)
_WORKFLOW_DISCRIMINATORS = ("lan dau", "dinh ky", "sau khi kich hoat", "truoc han", "toan bo", "mot phan")
_DIGITAL_RESERVE_CUES = (
    " truc tuyen ", " online ", " a pay ", " apay ", " napas 247 ", " 24 7 ",
    " a smart ", " virtual card ", " chung khoan ", " a digi ", " lan dau ",
    " kich hoat tam thoi ",
)
_STRUCTURAL_UNITS = ("dieu", "buoc", "hinh", "bang")


def _parse_source_code(text: str):
    """Parse leading document code such as 'hdsd 38', 'hdsd 03', or 'hdsd38' without regex."""
    if not text:
        return None
    parts = text.split()
    if not parts:
        return None
    first = parts[0]
    alpha_len = 0
    while alpha_len < len(first) and "a" <= first[alpha_len] <= "z":
        alpha_len += 1
    if not 2 <= alpha_len <= 5:
        return None
    prefix = first[:alpha_len]
    raw_digits = ""
    if alpha_len == len(first):
        if len(parts) < 2:
            return None
        second = parts[1]
        if not second.isascii() or not second.isdecimal():
            return None
        raw_digits = second
    else:
        tail = first[alpha_len:]
        if not tail.isascii() or not tail.isdecimal():
            return None
        raw_digits = tail
    trimmed = raw_digits[:-1].lstrip("0") + raw_digits[-1]
    if not 1 <= len(trimmed) <= 3:
        return None
    return (prefix, trimmed)


def _is_structural_number_token(token: str) -> bool:
    if not token or not token.isascii():
        return False
    digit_len = 0
    while digit_len < len(token) and token[digit_len].isdigit():
        digit_len += 1
    if not 1 <= digit_len <= 3:
        return False
    rest = token[digit_len:]
    if not rest:
        return True
    return len(rest) == 1 and "a" <= rest <= "z"


def _extract_structural_targets(folded_key: str) -> set:
    """Extract targets like 'dieu 6', 'buoc 2', 'phu luc 1a' from search_key text without regex."""
    targets = set()
    if not folded_key:
        return targets
    tokens = folded_key.split()
    count = len(tokens)
    for idx in range(count):
        tok = tokens[idx]
        if tok in _STRUCTURAL_UNITS and idx + 1 < count:
            nxt = tokens[idx + 1]
            if _is_structural_number_token(nxt):
                targets.add(f"{tok} {nxt}")
        elif tok == "phu" and idx + 2 < count and tokens[idx + 1] == "luc":
            nxt = tokens[idx + 2]
            if _is_structural_number_token(nxt):
                targets.add(f"phu luc {nxt}")
    return targets


def _find_query_page_spans(text: str) -> list:
    """Locate 'trang [số] N' or 'page N' spans (1 <= N <= 9999) without regex."""
    if not text:
        return []
    tokens = []
    n_chars = len(text)
    cursor = 0
    while cursor < n_chars:
        while cursor < n_chars and text[cursor].isspace():
            cursor += 1
        if cursor >= n_chars:
            break
        start = cursor
        while cursor < n_chars and not text[cursor].isspace():
            cursor += 1
        tokens.append((start, cursor, text[start:cursor]))

    spans = []
    idx = 0
    n_toks = len(tokens)
    punct = ",.;:!?()[]{}\"'"
    while idx < n_toks:
        s0, e0, w0 = tokens[idx]
        lead0 = len(w0) - len(w0.lstrip(punct))
        clean0 = w0.strip(punct).lower()
        if clean0 in ("trang", "page") and idx + 1 < n_toks:
            num_tok_idx = idx + 1
            _, _, w1 = tokens[idx + 1]
            clean1 = w1.strip(punct).lower()
            if clean1 in ("số", "so") and idx + 2 < n_toks:
                num_tok_idx = idx + 2
            s_num, _, w_num = tokens[num_tok_idx]
            trail_trimmed = w_num.rstrip(punct)
            lead_trimmed = trail_trimmed.lstrip(punct)
            if (
                lead_trimmed
                and lead_trimmed == trail_trimmed
                and 1 <= len(lead_trimmed) <= 4
                and lead_trimmed.isascii()
                and lead_trimmed.isdecimal()
                and lead_trimmed[0] != "0"
            ):
                span_start = s0 + lead0
                span_end = s_num + len(trail_trimmed)
                spans.append((span_start, span_end, lead_trimmed))
                idx = num_tok_idx + 1
                continue
        idx += 1
    return spans


def _extract_query_pages(text: str) -> set:
    pages = set()
    for _, _, page_str in _find_query_page_spans(text):
        pages.add(page_str)
    return pages


def _strip_query_page_mentions(text: str) -> str:
    spans = _find_query_page_spans(text)
    if not spans:
        return text
    parts = []
    last = 0
    for s_idx, e_idx, _ in spans:
        parts.append(text[last:s_idx])
        parts.append(" ")
        last = e_idx
    parts.append(text[last:])
    return "".join(parts)


class AmbiguousSource(ValueError):
    def __init__(self, sources):
        super().__init__("Source basename is ambiguous; use the complete source label")
        self.sources = sorted(sources)


def _source_stem(value):
    name = value.replace("\\", "/").rsplit("/", 1)[-1].casefold()
    for extension in (".pdf", ".docx", ".xlsx"):
        if name.endswith(extension):
            return name.removesuffix(extension)
    return name


def _merge_subquery_candidates(primary_items, sub_batches, limit, secondary_weight: float = 0.70):
    """Fuse multi-subquery hybrid pools using weighted RRF without extra rerank calls."""
    if not sub_batches:
        return primary_items
    merged = {}
    for rank, item in enumerate(primary_items, start=1):
        raw = item.get("chunk", item)
        if not isinstance(raw, dict):
            continue
        key = (str(raw.get("source") or ""), payload_index_key(raw))
        entry = dict(item)
        base_rrf = entry.get("rrf_score")
        if not isinstance(base_rrf, (int, float)):
            base_rrf = 1.0 / (60.0 + rank)
        entry["_fusion_score"] = float(base_rrf) + (0.015 / (60.0 + rank))
        merged[key] = entry
    for batch_info in sub_batches:
        if isinstance(batch_info, tuple) and len(batch_info) == 2:
            batch, b_weight = batch_info
        else:
            batch, b_weight = batch_info, secondary_weight
        for rank, item in enumerate(batch, start=1):
            raw = item.get("chunk", item)
            if not isinstance(raw, dict):
                continue
            key = (str(raw.get("source") or ""), payload_index_key(raw))
            add_score = b_weight / (60.0 + rank)
            if key in merged:
                merged[key]["_fusion_score"] = merged[key]["_fusion_score"] + add_score
            else:
                entry = dict(item)
                entry["_fusion_score"] = add_score
                merged[key] = entry
    ranked_pool = sorted(merged.values(), key=lambda x: x.get("_fusion_score", 0.0), reverse=True)
    for item in ranked_pool:
        item.pop("_fusion_score", None)
    return ranked_pool[:limit]


def _reserve_query_targets(candidates, ranked, seen, query):
    """Keep bounded, structurally aligned and digital-channel evidence that the reranker missed."""
    query_key = search_key(query)
    padded_q = f" {query_key} "
    targets = _extract_structural_targets(query_key)
    workflow = {term for term in _WORKFLOW_DISCRIMINATORS if f" {term} " in padded_q}
    wants_digital_app = any(cue in padded_q for cue in _DIGITAL_RESERVE_CUES)
    has_ranked_app = any(_app_guide(ch) for ch in ranked)
    need_digital_slot = wants_digital_app and not has_ranked_app
    if not targets and not workflow and not need_digital_slot:
        return []
    q_words = {tok for tok in query_key.split() if (len(tok) >= 3 or tok.isdigit()) and tok not in _PIN_STOPWORDS}
    ranked_sources = {chunk.get("source") for chunk in ranked}
    chosen = []
    digital_added = False
    for item in candidates[:18]:
        if not isinstance(item, dict):
            continue
        chunk = item.get("chunk", item)
        source = chunk.get("source")
        identity = source, payload_index_key(chunk)
        if identity in seen or not source:
            continue
        heading = search_key(str(chunk.get("section_title") or "") + " " + str(chunk.get("content") or "")[:180])
        title = search_key(_source_stem(source))
        is_target = bool(targets.intersection(_extract_structural_targets(heading)))
        is_workflow = source not in ranked_sources and any(f" {term} " in f" {title} " for term in workflow)
        is_digital_reserve = False
        if need_digital_slot and not digital_added and _app_guide(chunk):
            cand_words = set((title + " " + search_key(str(chunk.get("content") or "")[:320])).split())
            if len(q_words.intersection(cand_words)) >= 2:
                is_digital_reserve = True
        if is_target or is_workflow or is_digital_reserve:
            chosen.append(item)
            seen.add(identity)
            ranked_sources.add(source)
            if is_digital_reserve:
                digital_added = True
            if len(chosen) == 2:
                break
    return chosen


def _exact_excel_fee_evidence(candidates, query_key, policy, labels):
    """Resolve service phrases to priced workbook rows in the retrieved sources."""
    if " phi " not in query_key or not any(
        cue in query_key for cue in (*_FEE_TABLE_CUES, " bao nhieu ", " mien phi ", " tinh phi ",
                                     " mat phi ", " chiu phi ", " so sanh ")
    ):
        return []
    fee_query = query_key.replace(" chuyen khoan ", " chuyen tien ").replace(
        " chuyen tien lien ngan hang nhanh ", " chuyen tien nhanh lien ngan hang "
    )
    if " citad " in fee_query:
        fee_query += "kenh thanh toan dien tu lien ngan hang "
    operations = {operation for operation in (
        "mo tai khoan", "dong tai khoan", "chuyen tien", "nop tien", "rut tien", "tra no",
    ) if f" {operation} " in fee_query}
    words = fee_query.split()
    phrases = {}
    for size in range(7, 2, -1):
        for index in range(len(words) - size + 1):
            terms = words[index:index + size]
            if terms[0] not in _PIN_STOPWORDS and sum(term not in _PIN_STOPWORDS for term in terms) >= 2:
                phrases[" ".join(terms)] = size
    if not phrases:
        return []
    inspected = set()
    matches = []
    for item in candidates:
        seed = item.get("chunk", item)
        source = seed.get("source")
        table_id = seed.get("business_metadata", {}).get("excel_table_id")
        key = (source, table_id)
        if not source or not table_id or key in inspected:
            continue
        inspected.add(key)
        chunks = [chunk for chunk in policy.chunks_by_source.get(source, ())
                  if chunk.get("business_metadata", {}).get("excel_table_id") == table_id
                  and policy.allows(chunk, labels)]
        seen_rows = set()
        for chunk in chunks:
            data = (chunk.get("table_data") or {}).get("excel") or {}
            columns = data.get("columns", {})
            names = [col for col, title in columns.items() if any(
                cue in f" {search_key(title)} " for cue in (" khoan muc ", " dich vu ", " loai phi ", " ten phi ")
            )]
            prices = [col for col, title in columns.items()
                      if col not in names and " phi " in f" {search_key(title)} "
                      and " ma phi " not in f" {search_key(title)} "]
            if not names or not prices:
                continue
            for row in data.get("rows", ()):
                row_id = row["row"]
                if row_id in seen_rows:
                    continue
                seen_rows.add(row_id)
                values = row["values"]
                priced = any(
                    cell.get("type") in {"currency", "number", "percent"}
                    or search_key(cell.get("text", "")).startswith("mien phi")
                    or extract_canonical_numeric_tokens(cell.get("text", ""), max_tokens=1)
                    for col in prices if (cell := values.get(col))
                )
                if not priced:
                    continue
                for column in names:
                    value = values.get(column, {}).get("text", "")
                    label = (" " + search_key(value.partition("\n")[0]) + " ").replace(" chuyen khoan ", " chuyen tien ")
                    service = label.strip().removeprefix("phi ") + " "
                    if operations and not any(f" {operation} " in (" " + service) for operation in operations):
                        continue
                    match = max(((size, int(phrase == label.strip())) for phrase, size in phrases.items()
                                 if f" {phrase} " in label), default=(0, 0))
                    if match[0]:
                        matches.append((match, label, chunks, source, table_id, column, value))
        if len(inspected) == 3:
            break
    related = []
    included = set()
    compare = " so sanh " in query_key or " so voi " in query_key
    for _, label, chunks, source, table_id, column, value in sorted(matches, key=lambda match: match[0], reverse=True)[:6]:
        if label in included:
            continue
        spec = {"table_id": table_id, "filters": [{"column": column, "op": "eq", "value": value}]}
        if policy.dated:
            spec["as_of"] = policy.day
        result = lookup_excel_rows(chunks, spec, scope=policy.scope, source=source, limit=1)
        if result.get("status") == "success" and result["excel_match_count"] == 1:
            related.append({"chunk": result["documents"][0]})
            included.add(label)
            if not compare or len(related) >= 2:
                break
    return related


def matches_exact(item: dict, exact: str) -> bool:
    raw = item.get("chunk", item)
    meta = raw.get("business_metadata") or {}
    fields = [raw.get(field) for field in ("source", "section_title", "breadcrumb", "content")]
    fields += [meta.get(field) for field in ("document_name", "product_name", "service_code")]
    needle = " " + search_key(exact) + " "
    return any(needle in " " + search_key(value) + " " for value in fields if isinstance(value, str))


def _relevant_result(item):
    method = item.get("rerank_method")
    if "relevance_score" not in item:
        return True
    score = float(item.get("relevance_score") or 0.0)
    dense = float(item.get("dense_score") or 0.0)
    if method == "local_lexical":
        return score > 0.0 or dense >= float(_RAG_CFG.get("min_dense_evidence_score", 0.42))
    if method == "cloud_cross_encoder":
        return (score >= float(_RAG_CFG.get("min_rerank_evidence_score", 0.02))
                or dense >= float(_RAG_CFG.get("min_dense_evidence_score", 0.42)))
    return True


def _extract_item_pages(raw: dict) -> list:
    meta = raw.get("business_metadata")
    if not isinstance(meta, dict):
        return []
    cit = meta.get("citation")
    if isinstance(cit, dict) and isinstance(cit.get("page_numbers"), list) and cit["page_numbers"]:
        pages = []
        for p in cit["page_numbers"]:
            if isinstance(p, int) and p > 0 and p not in pages:
                pages.append(p)
        return sorted(pages)
    prov = meta.get("provenance")
    if not isinstance(prov, dict):
        return []
    locs = prov.get("locations")
    if not isinstance(locs, list):
        return []
    pages = []
    for loc in locs:
        if not isinstance(loc, dict):
            continue
        pnum = loc.get("page_number")
        if not isinstance(pnum, int) and isinstance(loc.get("page_index"), int):
            pnum = loc["page_index"] + 1
        if isinstance(pnum, int) and pnum > 0 and pnum not in pages:
            pages.append(pnum)
    return sorted(pages)


def _summarize_rag_items(items, limit=15):
    """Build compact score, full bounded content, and snippet previews for tracing."""
    if not isinstance(items, list):
        return []
    summary = []
    for idx, item in enumerate(items[:limit]):
        if not isinstance(item, dict):
            continue
        raw = item.get("chunk", item)
        if not isinstance(raw, dict):
            raw = {}
        full_text = str(raw.get("content") or raw.get("original_text") or raw.get("text") or "").strip()
        snippet = full_text
        if len(snippet) > 600:
            snippet = snippet[:597] + "..."
        summary.append({
            "rank": idx + 1,
            "chunk_id": str(raw.get("chunk_id") or item.get("chunk_id") or ""),
            "source": str(raw.get("source") or ""),
            "section_title": str(raw.get("section_title") or ""),
            "breadcrumb": str(raw.get("breadcrumb") or ""),
            "chunk_type": str(raw.get("chunk_type") or "text"),
            "rrf_score": item.get("rrf_score"),
            "dense_score": item.get("dense_score"),
            "bm25_score": item.get("bm25_score"),
            "dense_rank": item.get("dense_rank"),
            "bm25_rank": item.get("bm25_rank"),
            "rerank_score": item.get("rerank_score"),
            "rerank_method": item.get("rerank_method"),
            "page_numbers": _extract_item_pages(raw),
            "char_count": len(full_text),
            "snippet": snippet,
            "content": full_text[:4000],
        })
    return summary


def _context_section(chunk, procedure):
    breadcrumb = str(chunk.get("breadcrumb") or "")
    title_key = search_key(str(chunk.get("section_title") or ""))
    step_prefixes = tuple(_RAG_CFG["step_prefixes"])
    separator_token = str(_RAG_CFG["breadcrumb_separator"])
    if procedure and any(title_key.startswith(prefix) for prefix in step_prefixes):
        parent, separator, _ = breadcrumb.rpartition(separator_token)
        if separator:
            return parent
    return breadcrumb or str(chunk.get("section_title") or "")


def _app_guide(chunk):
    name = str(chunk.get("source") or "").replace("\\", "/").rsplit("/", 1)[-1]
    return search_key(name).startswith(str(_RAG_CFG["app_guide_prefix"]))


def _guide_header(chunk):
    body = str(chunk.get("content") or "")
    key = search_key(body)
    max_len = int(_HEADER_CFG["max_length"])
    req_terms = _HEADER_CFG["required_terms"]
    exc_terms = _HEADER_CFG["excluded_terms"]
    return (
        len(body) <= max_len
        and all(term in key for term in req_terms)
        and all(term not in body for term in exc_terms)
    )


def _guide_logo(chunk):
    title = search_key(str(chunk.get("section_title") or ""))
    body = str(chunk.get("content") or "")
    titles = tuple(_LOGO_CFG["titles"])
    max_len = int(_LOGO_CFG["max_length"])
    body_term = str(_LOGO_CFG["body_term"])
    return title in titles and len(body) <= max_len and body_term in search_key(body)


def _source_neighbors(anchor, ordered, positions, procedure):
    position = positions.get(payload_index_key(anchor))
    section = _context_section(anchor, procedure)
    if position is None or not section:
        return []
    anchor_scopes = set(anchor.get("business_metadata", {}).get("publication_scopes", []))
    found = []
    app_steps = procedure and _app_guide(anchor)
    hop_limit = int(_NEIGHBOR_CFG["default_hop_limit"])
    step_limit = int(_NEIGHBOR_CFG["default_step_limit"])
    if app_steps:
        hop_limit = int(_NEIGHBOR_CFG["app_guide_hop_limit"])
        step_limit = int(_NEIGHBOR_CFG["app_guide_step_limit"])
    separator = str(_RAG_CFG["breadcrumb_separator"])
    section_prefix = section + separator
    step_prefixes = tuple(_RAG_CFG["step_prefixes"])
    seen_texts = set()
    anchor_norm = " ".join(str(anchor.get("content") or "").split()).casefold()
    if anchor_norm:
        seen_texts.add(anchor_norm)
    for direction in (-1, 1):
        previous = anchor
        scopes = anchor_scopes
        logical_distance = 0
        for hop in range(1, hop_limit + 1):
            index = position + direction * hop
            if not 0 <= index < len(ordered):
                break
            chunk = ordered[index]
            left, right = previous, chunk
            if direction == -1:
                left, right = chunk, previous
            if (left.get("next_chunk_id") != right.get("chunk_id")
                    or right.get("prev_chunk_id") != left.get("chunk_id")):
                break
            neighbor_scopes = set(chunk.get("business_metadata", {}).get("publication_scopes", []))
            if (scopes or neighbor_scopes) and not scopes.intersection(neighbor_scopes):
                break
            if scopes:
                scopes = scopes.intersection(neighbor_scopes)
            previous = chunk
            current_section = _context_section(chunk, procedure)
            if app_steps:
                child_heading = current_section.removeprefix(section_prefix).split(separator, 1)[0]
                nested_in_step = current_section.startswith(section_prefix) and search_key(child_heading).startswith(
                    step_prefixes
                )
                # An illustrative child under a step should not end the workflow.
                if nested_in_step or _guide_logo(chunk) or (current_section == section and _guide_header(chunk)):
                    continue
            if current_section != section:
                break
            chunk_norm = " ".join(str(chunk.get("content") or "").split()).casefold()
            if chunk_norm and chunk_norm in seen_texts:
                continue
            if chunk_norm:
                seen_texts.add(chunk_norm)
            logical_distance += 1
            found.append((logical_distance, index, chunk))
            if logical_distance == step_limit:
                break
    return sorted(found, key=lambda item: (item[0], item[1]))


def _attach_table_hierarchy_view(items, table_paths, table_notes):
    """Attach transient inherited table path and note metadata on shallow chunk copies."""
    if not items or (not table_paths and not table_notes):
        return items
    enriched = []
    for item in items:
        if not isinstance(item, dict):
            enriched.append(item)
            continue
        raw = item.get("chunk", item)
        if not isinstance(raw, dict):
            enriched.append(item)
            continue
        key = payload_index_key(raw)
        path_val = ""
        if table_paths:
            path_val = str(table_paths.get(key) or "").strip()
        note_val = ""
        if table_notes:
            note_val = str(table_notes.get(key) or "").strip()
        if not path_val and not note_val:
            enriched.append(item)
            continue
        meta = raw.get("business_metadata")
        if isinstance(meta, dict):
            meta_copy = dict(meta)
        else:
            meta_copy = {}
        chunk_copy = dict(raw)
        if path_val:
            chunk_copy["_inherited_table_path"] = path_val
            if not meta_copy.get("table_group_path"):
                meta_copy["table_group_path"] = path_val
        if note_val and not meta_copy.get("table_notes"):
            meta_copy["table_notes"] = note_val
        chunk_copy["business_metadata"] = meta_copy
        if "chunk" in item:
            item_copy = dict(item)
            item_copy["chunk"] = chunk_copy
            enriched.append(item_copy)
        else:
            enriched.append(chunk_copy)
    return enriched


def _maybe_auto_excel_summary(clean_query, ranked, ordered, corpus, scope, as_of=None):
    """Compute an exact row-aggregation summary chunk for Excel count/list queries."""
    if not ranked or not clean_query:
        return None
    q_search = " " + search_key(clean_query) + " "
    q_search = q_search.replace(" tp hcm ", " ho chi minh ").replace(" tphcm ", " ho chi minh ").replace(" hcm ", " ho chi minh ")
    agg_cues = (
        " bao nhieu ", " tong cong ", " tong so ", " so luong ", " danh sach ",
        " liet ke ", " nhung tinh ", " tinh thanh ", " hien dien ", " toan quoc ",
        " phan bo ", " co may ", " may diem ", " gom nhung ",
    )
    is_agg_query = any(cue in q_search for cue in agg_cues)
    if not is_agg_query:
        return None

    anchor = None
    table_id = ""
    src = ""
    for cand in ranked[:2]:
        meta = cand.get("business_metadata")
        if isinstance(meta, dict) and meta.get("excel_table_id"):
            anchor = cand
            table_id = str(meta["excel_table_id"])
            src = str(cand.get("source") or "")
            break
    if anchor is None or not table_id or not src:
        return None

    table_chunks = []
    if src in ordered:
        for ch in ordered[src][0]:
            if (ch.get("business_metadata") or {}).get("excel_table_id") == table_id:
                table_chunks.append(ch)
    elif corpus:
        for ch in corpus:
            meta = ch.get("business_metadata") or {}
            if ch.get("source") == src and meta.get("excel_table_id") == table_id:
                if not scope or scope in meta.get("publication_scopes", ()):
                    table_chunks.append(ch)
    if not table_chunks:
        return None

    col_labels = {}
    distinct_by_col = {}
    for ch in table_chunks:
        excel_data = (ch.get("table_data") or {}).get("excel")
        if not isinstance(excel_data, dict):
            continue
        for c_key, c_lbl in (excel_data.get("columns") or {}).items():
            if c_key not in col_labels and c_lbl:
                col_labels[c_key] = str(c_lbl)
        for row in excel_data.get("rows") or ():
            vals = row.get("values") or {}
            for c_key in col_labels:
                cell = vals.get(c_key) or {}
                txt = str(cell.get("text") or "").strip()
                if txt:
                    bucket = distinct_by_col.setdefault(c_key, set())
                    if len(bucket) < 120:
                        bucket.add(txt)

    filters = []
    group_by = []
    tinh_col = ""
    quan_col = ""
    loai_col = ""
    for c_key, c_lbl in col_labels.items():
        lbl_key = search_key(c_lbl)
        if not lbl_key.startswith("ma ") and "tinh" in lbl_key and "thanh pho" in lbl_key:
            tinh_col = c_lbl
        elif not lbl_key.startswith("ma ") and "quan" in lbl_key and "huyen" in lbl_key:
            quan_col = c_lbl
        elif lbl_key == "loai":
            loai_col = c_lbl

    matched_quan = False
    if tinh_col:
        matched_location = False
        for c_key, c_lbl in col_labels.items():
            if c_lbl == tinh_col:
                matched_vals = []
                for val in distinct_by_col.get(c_key, ()):
                    c_val = canonical_excel_text(val)
                    if c_val and len(c_val) >= 3 and (" " + c_val + " ") in q_search:
                        matched_vals.append((len(c_val), val))
                if matched_vals:
                    location_values = [val for _, val in sorted(matched_vals, reverse=True)]
                    filters.append({"column": tinh_col, "op": "in", "value": location_values})
                    matched_location = True
                break
        if quan_col:
            for c_key, c_lbl in col_labels.items():
                if c_lbl == quan_col:
                    matched_q_vals = []
                    for val in distinct_by_col.get(c_key, ()):
                        c_val = canonical_excel_text(val)
                        if c_val and len(c_val) >= 3 and (" " + c_val + " ") in q_search:
                            matched_q_vals.append((len(c_val), val))
                    if matched_q_vals:
                        q_values = [val for _, val in sorted(matched_q_vals, reverse=True)]
                        filters.append({"column": quan_col, "op": "in", "value": q_values})
                        matched_quan = True
                    break
        if not matched_location and not matched_quan and any(phrase in q_search for phrase in (" tai ", " o ", " tinh ", " thanh pho ")):
            return None

    if quan_col and not matched_quan and (matched_quan or any(phrase in q_search for phrase in (" quan ", " huyen ", " theo quan ", " tung quan "))):
        group_by.append(quan_col)

    if loai_col:
        want_cn = " chi nhanh " in q_search
        want_pgd = " phong giao dich " in q_search or " phong gd " in q_search or " pgd " in q_search
        want_atm = " atm " in q_search or " cay atm " in q_search
        want_cdm = " cdm " in q_search or " may nop tien tu dong " in q_search
        mentioned = []
        if want_cn:
            mentioned.append("Chi nhánh")
        if want_pgd:
            mentioned.append("Phòng giao dịch")
        if want_atm:
            mentioned.append("ATM")
        if want_cdm:
            mentioned.append("CDM")
        if mentioned:
            filters.append({"column": loai_col, "op": "in", "value": mentioned})
        group_by.append(loai_col)
    if tinh_col and not filters:
        group_by.append(tinh_col)

    src_key = search_key(src)
    if not group_by and not tinh_col and not loai_col and ("bp " in (" " + src_key + " ") or "bieu phi" in src_key):
        preferred_fee_col = ""
        fallback_fee_col = ""
        for c_key, c_lbl in col_labels.items():
            lbl_key = search_key(c_lbl)
            if not lbl_key or lbl_key == "stt" or lbl_key.startswith(("stt ", "ma ", "ghi chu", "don vi", "muc phi", "ty le")):
                continue
            if not fallback_fee_col:
                fallback_fee_col = c_lbl
            if any(tok in (" " + lbl_key + " ") for tok in (" nhom ", " dich vu ", " loai ", " san pham ", " nghiep vu ")):
                preferred_fee_col = c_lbl
                break
        chosen_fee_col = preferred_fee_col
        if not chosen_fee_col:
            chosen_fee_col = fallback_fee_col
        if chosen_fee_col:
            group_by.append(chosen_fee_col)

    if not group_by:
        return None
    excel_spec = {"table_id": table_id, "filters": filters, "group_by": group_by}
    if as_of is not None:
        excel_spec["as_of"] = as_of
    res = lookup_excel_rows(table_chunks, excel_spec, scope=scope, source=src, limit=1, offset=0)
    if res.get("status") != "success" or res.get("excel_match_count", 0) <= 0:
        return None
    sum_doc = None
    for doc_item in res.get("documents", []):
        if str(doc_item.get("chunk_id", "")).endswith(":summary"):
            sum_doc = doc_item
            break
    if sum_doc is None:
        return None
    return {"chunk": sum_doc, "context_for": anchor["chunk_id"], "relevance_score": 0.99}


def _pin_intra_document_targets(ranked, ordered, seen, clean_query, related, related_limit):
    """Pin high-signal same-document tables, diagrams, or app steps missed by top_k."""
    if not ranked or len(related) >= related_limit:
        return
    query_key = " " + search_key(clean_query) + " "
    q_tokens = []
    for tok in query_key.split():
        if (len(tok) >= 3 or tok.isdigit()) and tok not in _PIN_STOPWORDS:
            if tok not in q_tokens:
                q_tokens.append(tok)
    if not q_tokens:
        return

    wants_fee_table = any(cue in query_key for cue in _FEE_TABLE_CUES)
    wants_diagram = any(cue in query_key for cue in _DIAGRAM_CUES)
    fee_terms = set()
    if wants_fee_table:
        words = query_key.split()
        fee_terms = {" ".join(words[index:index + 3]) for index, word in enumerate(words[:-2])
                     if word == "phi" and all(part not in _PIN_STOPWORDS for part in words[index:index + 3])}
    time_prefix = str(_RAG_CFG["guide_header"].get("excluded_terms", ["09:41"])[0])

    # Collect immediate linked siblings sharing the exact same section_title or table_id as ranked[:2]
    sibling_keys = {}
    anchor_table_ids = {}
    primary_sources = []
    seen_sources = set()
    for anchor in ranked[:2]:
        src = anchor.get("source")
        if not src or src not in ordered:
            continue
        if src not in seen_sources:
            seen_sources.add(src)
            primary_sources.append((src, anchor))
        anchor_td = anchor.get("table_data")
        if isinstance(anchor_td, dict) and anchor_td.get("table_id"):
            anchor_table_ids.setdefault(src, set()).add(str(anchor_td["table_id"]))
        passages, positions = ordered[src]
        pos = positions.get(payload_index_key(anchor))
        anchor_sec = str(anchor.get("section_title") or "").strip()
        if pos is not None and anchor_sec:
            src_sibs = sibling_keys.setdefault(src, set())
            for delta in (-1, 1):
                sib_idx = pos + delta
                if 0 <= sib_idx < len(passages):
                    sib = passages[sib_idx]
                    if delta == -1:
                        linked = (
                            sib.get("next_chunk_id") == anchor.get("chunk_id")
                            and anchor.get("prev_chunk_id") == sib.get("chunk_id")
                        )
                    else:
                        linked = (
                            anchor.get("next_chunk_id") == sib.get("chunk_id")
                            and sib.get("prev_chunk_id") == anchor.get("chunk_id")
                        )
                    if linked and str(sib.get("section_title") or "").strip() == anchor_sec:
                        src_sibs.add(payload_index_key(sib))

    threshold = 4
    if wants_fee_table or wants_diagram:
        threshold = 3

    table_slice_counts = {}
    ref_targets = set()
    for _, anchor in primary_sources:
        passages_ref, _ = ordered[anchor.get("source")]
        for pch in passages_ref:
            ptd = pch.get("table_data")
            if isinstance(ptd, dict) and ptd.get("table_id"):
                tid = str(ptd["table_id"])
                table_slice_counts[tid] = table_slice_counts.get(tid, 0) + 1
        ref_targets.update(_extract_structural_targets(search_key(str(anchor.get("content") or ""))))

    note_cues = (" luu y ", " ghi chu ", " mien phi ", " toi thieu ", " toi da ")
    pinned = []
    for src, anchor in primary_sources:
        passages, _ = ordered[src]
        anchor_scopes = set(anchor.get("business_metadata", {}).get("publication_scopes", []))
        wants_app_subflow = _app_guide(anchor)
        src_sibs = sibling_keys.get(src, set())
        src_tbl_ids = anchor_table_ids.get(src, set())
        for idx, cand in enumerate(passages):
            cand_key_id = payload_index_key(cand)
            ident = (cand.get("source"), cand_key_id)
            if ident in seen:
                continue
            cand_scopes = set(cand.get("business_metadata", {}).get("publication_scopes", []))
            if (anchor_scopes or cand_scopes) and not anchor_scopes.intersection(cand_scopes):
                continue
            if wants_app_subflow and (_guide_logo(cand) or _guide_header(cand)):
                continue
            sec_str = str(cand.get("section_title") or "") + " " + str(cand.get("breadcrumb") or "")
            sec_key = " " + search_key(sec_str) + " "
            content_str = str(cand.get("content") or "")
            table_str = str(cand.get("table_data") or "")
            cand_key = " " + search_key(content_str + " " + sec_str + " " + table_str) + " "
            cand_tokens = set(cand_key.split())

            token_hits = 0
            for tok in q_tokens:
                if tok in cand_tokens:
                    token_hits += 1
            sec_targets = _extract_structural_targets(sec_key.strip())
            matches_intra_ref = bool(ref_targets and sec_targets and ref_targets.intersection(sec_targets))
            if token_hits == 0 and not matches_intra_ref:
                continue

            score = token_hits
            cand_td = cand.get("table_data")
            cand_tbl_id = ""
            if isinstance(cand_td, dict):
                cand_tbl_id = str(cand_td.get("table_id") or "")
            same_table = bool(cand_tbl_id and cand_tbl_id in src_tbl_ids)
            is_table_chunk = (
                cand.get("chunk_type") == "table"
                or bool(cand_td)
                or " phu luc " in sec_key
                or " bieu phi " in sec_key
            )
            if wants_fee_table and is_table_chunk:
                score += 3
                body_key = " " + search_key(content_str) + " "
                for fee_term in fee_terms:
                    if f" {fee_term} " in body_key:
                        score += 4
                        if extract_canonical_numeric_tokens(content_str, max_tokens=1):
                            score += 2
                if anchor is ranked[0]:
                    score += 1
            if same_table:
                slice_total = table_slice_counts.get(cand_tbl_id, 0)
                if 2 <= slice_total <= 5 and (token_hits >= 1 or any(nc in cand_key for nc in note_cues)):
                    score += 3
                elif token_hits >= 2:
                    score += 2
            if matches_intra_ref:
                score += 4
            has_diagram_cue = any(marker in cand_key for marker in _DIAGRAM_CHUNK_MARKERS)
            if wants_diagram and has_diagram_cue:
                score += 3
            if wants_app_subflow:
                instr_lines = []
                for line in content_str.splitlines():
                    if line.strip().startswith(time_prefix):
                        break
                    instr_lines.append(line)
                instr_tokens = set(search_key(" ".join(instr_lines)).split())
                instr_hits = 0
                for tok in q_tokens:
                    if tok in instr_tokens:
                        instr_hits += 1
                if instr_hits >= 2:
                    score += 2
            if cand_key_id in src_sibs and (token_hits >= 1 or is_table_chunk or has_diagram_cue):
                score += 2

            if score >= threshold:
                pinned.append((score, token_hits, idx, ident, cand, anchor["chunk_id"]))

    pinned.sort(key=lambda item: (-item[0], -item[1], item[2]))
    for _, _, _, ident, cand, anchor_id in pinned:
        if len(related) >= related_limit:
            break
        if ident in seen:
            continue
        related.append({"chunk": cand, "context_for": anchor_id})
        seen.add(ident)


class BankingRAGTool:
    """
    Tool for retrieving banking knowledge and regulatory policies.
    """
    name: str = str(_RAG_CFG["name"])
    description: str = RAG_TOOL_DESCRIPTION

    def __init__(self, pipeline_instance=None, config=None, use_memory: bool = False):
        if pipeline_instance is not None:
            self.pipeline = pipeline_instance
        else:
            from agent_banking.pipeline import BankingRAGPipeline
            self.pipeline = BankingRAGPipeline(config=config, use_memory=use_memory)

    def _source_chunks(self, source, scope):
        store = getattr(self.pipeline, "vector_store", None)
        grouped = None
        if isinstance(store, QdrantBankingStore):
            grouped = store.source_groups(scope)
        corpus = getattr(store, "_bm25_corpus", [])
        source = source.replace("\\", "/").casefold()
        path_requested = "/" in source
        matches = []
        entries = grouped
        if entries is None:
            entries = {}
            for chunk in corpus:
                name = chunk.get("source", "")
                if name and (not scope or scope in chunk.get("business_metadata", {}).get("publication_scopes", [])):
                    entries.setdefault(name, []).append(chunk)
        for raw_name, chunks in sorted(entries.items()):
            name = raw_name.replace("\\", "/").casefold()
            basename = name.rsplit("/", 1)[-1]
            if path_requested:
                matches_name = name == source or name.endswith("/" + source)
            else:
                matches_name = name == source or basename == source or _source_stem(basename) == source
            if not matches_name:
                continue
            matches.extend(chunks)
        min_key_len = int(_SOURCE_CFG["min_key_length"])
        min_non_digit_len = int(_SOURCE_CFG["min_non_digit_key_length"])
        if not matches and not path_requested:
            src_key = search_key(_source_stem(source))
            if src_key:
                sources = entries
                labels = {name: search_key(_source_stem(name)) for name in sources}
                matched_names = [name for name, label in labels.items() if label == src_key or label.startswith(src_key + " ")]
                source_code = _parse_source_code(src_key)
                if not matched_names and source_code is not None:
                    for name, label in labels.items():
                        label_code = _parse_source_code(label)
                        if label_code is not None and label_code == source_code:
                            matched_names.append(name)
                if not matched_names and len(src_key) >= min_key_len and (
                    any(ch.isdigit() for ch in src_key) or len(src_key) >= min_non_digit_len
                ):
                    matched_names = [name for name, label in labels.items() if f" {src_key} " in f" {label} "]
                    terms = set(src_key.split())
                    if not matched_names and len(terms) >= 3:
                        matched_names = [name for name, label in labels.items() if terms <= set(label.split())]
                matches = [chunk for name in matched_names for chunk in sources[name]]
        matched_sources = {chunk["source"] for chunk in matches}
        if len(matched_sources) > 1:
            raise AmbiguousSource(matched_sources)
        by_id = {chunk["chunk_id"]: chunk for chunk in matches}
        ordered = []
        seen = set()
        starts = [chunk for chunk in matches if chunk.get("prev_chunk_id") not in by_id]
        for start in starts + matches:
            chunk = start
            while chunk and chunk["chunk_id"] not in seen:
                ordered.append(chunk)
                seen.add(chunk["chunk_id"])
                chunk = by_id.get(chunk.get("next_chunk_id"))
        return ordered

    def execute(self, query: str, top_k: int | None = None, exact: str = "", candidate_k: int | None = None,
                source: str = "", offset: int | None = None, excel: dict | None = None,
                page: int | None = None, as_of: str | None = None, expand_graph: bool = True,
                *, include_candidates: bool = False) -> dict:
        """
        Execute banking knowledge retrieval via Hybrid Retrieval & Reranker.
        """
        if top_k is None:
            top_k = int(_RAG_CFG["default_top_k"])
        start_t = time.perf_counter()
        clean_query = query.strip()
        vector_store_ref = getattr(self.pipeline, "vector_store", None)
        if isinstance(vector_store_ref, QdrantBankingStore):
            vector_store_ref.refresh_if_changed()
        if excel is not None:
            excel_date = excel.get("as_of")
            if as_of is not None and excel_date is not None and as_of != excel_date:
                return {"status": "excel_conflicting_dates", "documents": [], "sources": [], "citations": [],
                        "latency_ms": (time.perf_counter() - start_t) * 1000}
            if not excel_date:
                try:
                    requested_date, _ = policy_query_date(clean_query, as_of)
                except ValueError as error:
                    return {"status": str(error), "documents": [], "sources": [], "citations": [],
                            "latency_ms": (time.perf_counter() - start_t) * 1000}
                if requested_date is not None:
                    excel = {**excel, "as_of": requested_date.isoformat()}
            scope = getattr(self.pipeline.config, "KNOWLEDGE_PUBLICATION_SCOPE", "")
            corpus = getattr(self.pipeline.vector_store, "_bm25_corpus", ())
            result = lookup_excel_rows(corpus, excel, scope=scope, source=source, limit=top_k, offset=offset or 0)
            result["latency_ms"] = (time.perf_counter() - start_t) * 1000
            return result
        if not clean_query:
            return {
                "status": "empty_query",
                "documents": [],
                "sources": [],
                "citations": [],
                "latency_ms": 0.0
            }
        if page is not None and (not source or type(page) is not int or page < 1):
            raise ValueError("Reading a page requires a source and a positive page number")
        reading = offset is not None or page is not None
        if offset is not None and (not source or type(offset) is not int or offset < 0):
            raise ValueError("Reading a document requires source and a nonnegative offset")
        try:
            requested_date, today = policy_query_date(clean_query, as_of)
        except ValueError as error:
            # A literal page read can contain other dates in its topic text.
            if not reading or as_of is not None:
                status = str(error)
                return {"status": status, "documents": [], "sources": [], "citations": [],
                        "latency_ms": (time.perf_counter() - start_t) * 1000,
                        "retrieval_quality": {"temporal": {"status": status}}}
            requested_date, today = policy_query_date("")

        # 1. Parse Query & Acronym Normalization
        search_text = clean_query
        exact = exact.strip()
        if exact and not search_key(exact):
            raise ValueError("Exact search requires a product name or service code")
        if exact and search_key(exact) not in search_key(clean_query):
            search_text += " " + exact
        with trace_span("query_parser", run_type="parser", raw_query=search_text) as qp_span:
            parsed_q = self.pipeline.parse_query(search_text)
            search_query = parsed_q.normalized_query
            qp_span.update(
                normalized_query=search_query,
                keywords=list(getattr(parsed_q, "keywords", []) or []),
                legal_references=list(getattr(parsed_q, "legal_references", []) or []),
                detected_acronyms=list(getattr(parsed_q, "detected_acronyms", []) or []),
                financial_amounts=list(getattr(parsed_q, "financial_amounts", []) or []),
                banks_mentioned=list(getattr(parsed_q, "banks_mentioned", []) or []),
                parsed_intent=str(getattr(parsed_q, "intent", "") or ""),
                parsed_sub_intent=str(getattr(parsed_q, "sub_intent", "") or ""),
                parsed_domain=str(getattr(parsed_q, "domain", "") or ""),
                expanded_query=str(getattr(parsed_q, "expanded_query", "") or ""),
                suggested_filters=dict(getattr(parsed_q, "suggested_filters", {}) or {}),
                exact_filter=exact,
                source_filter=source,
            )

        # 2. Resolve policy catalog and explicit source/folder scope before query planning
        candidate_count = candidate_k or self.pipeline.config.HYBRID_CANDIDATE_K
        top_candidates = max(candidate_count, top_k)
        exact_min_k = int(_RAG_CFG["exact_candidate_min_k"])
        if exact:
            candidate_count = max(exact_min_k, candidate_count)
            top_candidates = candidate_count
        scope = getattr(self.pipeline.config, "KNOWLEDGE_PUBLICATION_SCOPE", "")
        corpus = getattr(vector_store_ref, "_bm25_corpus", ())
        graph_path = getattr(self.pipeline.config, "KNOWLEDGE_GRAPH_PATH", None)
        storage = getattr(self.pipeline.config, "STORAGE_DIR", None)
        if graph_path is None and storage:
            graph_path = Path(storage) / GRAPH_FILENAME
        if isinstance(vector_store_ref, QdrantBankingStore):
            policy = vector_store_ref.policy_catalog(scope, requested_date, today, graph_path)
        else:
            policy = PolicyVersions(corpus, scope, requested_date, today, graph_path=graph_path)
        filters = {}
        if scope:
            filters = {"business_metadata.publication_scopes": scope}
        source_chunks = []
        source_status = ""
        source_candidates = []
        if source:
            try:
                source_chunks = self._source_chunks(source, scope)
            except AmbiguousSource as error:
                eligible = {name for name, _ in policy.labels} & set(error.sources)
                families = {chunk.get("business_metadata", {}).get("policy_revision", {}).get("policy_id")
                            for chunk in policy.corpus if chunk["source"] in error.sources}
                if len(eligible) == 1 and len(families) == 1 and None not in families:
                    source_chunks = self._source_chunks(next(iter(eligible)), scope)
                else:
                    source_status = "source_ambiguous"
                    terms = set(search_key(search_query).split()) - _PIN_STOPWORDS
                    source_candidates = sorted(
                        error.sources,
                        key=lambda name: (
                            -len(terms.intersection(search_key(_source_stem(name)).split())),
                            name not in eligible,
                            name,
                        ),
                    )[:8]
            if not source_chunks and not source_status:
                folder = source.strip().replace("\\", "/").rstrip("/").casefold()
                if folder:
                    corpus = getattr(getattr(self.pipeline, "vector_store", None), "_bm25_corpus", ())
                    source_chunks = [
                        chunk for chunk in corpus
                        if str(chunk.get("source", "")).replace("\\", "/").casefold().startswith(folder + "/")
                        and (not scope or scope in chunk.get("business_metadata", {}).get("publication_scopes", ()))
                    ]
                    if source_chunks and offset == 0 and page is None:
                        offset = None
                        reading = False
                    elif source_chunks and reading:
                        source_status = "source_ambiguous"
                        source_candidates = sorted({chunk["source"] for chunk in source_chunks})[:8]
                        source_chunks = []
            if source_chunks:
                names = sorted({chunk["source"] for chunk in source_chunks})
                if len(names) == 1:
                    filters["source"] = names[0]
                else:
                    filters["source"] = names
            elif not source_status:
                source_status = "source_not_found"
        labels = policy.labels
        if source and source_chunks and policy.enabled:
            if len({chunk["source"] for chunk in source_chunks}) == 1:
                labels = policy.with_sources(source_chunks)
            source_chunks = [chunk for chunk in source_chunks if policy.allows(chunk, labels)]
            selected_names = {chunk["source"] for chunk in source_chunks}
            labels = {edition: status for edition, status in labels.items() if edition[0] in selected_names}
        filters = policy.search_filters(filters, labels, policy.corpus)

        single_file_source = bool(source) and len({chunk.get("source") for chunk in source_chunks}) == 1
        plan_fn = getattr(self.pipeline, "plan_query", None)
        llm_plan = None
        if not reading and not (source and (single_file_source or not source_chunks)) and callable(plan_fn):
            try:
                llm_plan = plan_fn(clean_query)
            except Exception:
                llm_plan = None
        if isinstance(llm_plan, dict) and llm_plan.get("method") == "llm_model":
            rewritten = str(llm_plan.get("rewritten_query") or "").strip()
            sem_terms = [str(t).strip() for t in (llm_plan.get("semantic_terms") or ()) if str(t).strip()]
            exp_base = str(getattr(parsed_q, "expanded_query", "") or clean_query)
            extra_dense = []
            if rewritten and rewritten.casefold() not in exp_base.casefold():
                extra_dense.append(rewritten)
            if sem_terms:
                extra_dense.append(" ".join(sem_terms))
            if extra_dense:
                parsed_q.expanded_query = (exp_base + " " + " ".join(extra_dense)).strip()
            bm25_additions = []
            if rewritten:
                bm25_additions.append(rewritten)
            if sem_terms:
                bm25_additions.append(" ".join(sem_terms))
            if bm25_additions:
                extra_kws = self.pipeline.parse_query(" ".join(bm25_additions)).keywords
                cur_kws = list(getattr(parsed_q, "keywords", []) or [])
                for ek in extra_kws:
                    if ek not in cur_kws:
                        cur_kws.append(ek)
                parsed_q.keywords = cur_kws
                existing_kw_str = str(getattr(parsed_q, "keywords_str", "") or "").strip()
                add_tokens = [ek for ek in extra_kws if f" {ek} " not in f" {existing_kw_str} "]
                if add_tokens:
                    parsed_q.keywords_str = (existing_kw_str + " " + " ".join(add_tokens)).strip()
        reading_chunks = source_chunks
        start_offset = offset or 0
        if page is None and source_chunks and offset is not None:
            pages = _extract_query_pages(clean_query)
            if len(pages) == 1:
                page = int(pages.pop())
        if page is not None and source_chunks:
            reading_chunks = [chunk for chunk in source_chunks if page in _extract_item_pages(chunk)]
            if not reading_chunks:
                source_status = "page_not_found"
        if reading and exact:
            reading_chunks = [chunk for chunk in reading_chunks if matches_exact(chunk, exact)]
        allowed = {(chunk["source"], payload_index_key(chunk)) for chunk in source_chunks}
        retriever = self.pipeline.retriever
        retrieval_errors = []
        query_embeddings = {}

        def _retrieve_with_weight(query, eff_top, eff_cand, fdict, d_weight=None):
            if policy.enabled and not labels:
                return []
            options = {}
            if policy.enabled:
                options["strict_filters"] = True
            if isinstance(retriever, HybridRetriever):
                options.update(source_restricted=bool(source), prefer_query_filters=not source and not exact,
                               query_embeddings=query_embeddings)
            if d_weight is not None:
                options["dense_weight"] = d_weight
            try:
                items = retriever.retrieve(query=query, top_k=eff_top, candidate_k=eff_cand, filter_dict=fdict, **options)
            except RetrievalUnavailable as error:
                retrieval_errors.append(error)
                return []
            if policy.enabled:
                return [item for item in items if policy.allows(item.get("chunk", item), labels)]
            return items

        def search(query):
            d_weight = None
            if isinstance(llm_plan, dict) and llm_plan.get("method") == "llm_model":
                d_weight = llm_plan.get("dense_weight")
            items = _retrieve_with_weight(query, top_candidates, candidate_count, filters, d_weight=d_weight)
            if not source:
                return items
            selected = []
            for item in items:
                raw = item.get("chunk", item)
                if (raw.get("source"), payload_index_key(raw)) not in allowed:
                    continue
                if scope and scope not in raw.get("business_metadata", {}).get("publication_scopes", []):
                    continue
                selected.append(item)
            return selected

        with trace_span("top_k_hybrid_retrieval", run_type="retriever", query=search_query, top_k=top_candidates) as ret_span:
            if reading:
                candidates = [{"chunk": chunk} for chunk in reading_chunks[start_offset:start_offset + top_k]]
            elif source and not source_chunks:
                candidates = []
            else:
                candidates = search(parsed_q)
                if not source and not exact:
                    sub_queries = []
                    if isinstance(llm_plan, dict) and llm_plan.get("method") == "llm_model":
                        sub_queries = list(llm_plan.get("sub_queries") or [])
                    if not sub_queries:
                        sub_queries = decompose_subqueries(clean_query)
                    sub_batches = []
                    if sub_queries:
                        sub_k = min(24, candidate_count)
                        for sq in sub_queries:
                            sq_parsed = self.pipeline.parse_query(sq)
                            sq_items = _retrieve_with_weight(sq_parsed, sub_k, sub_k, filters)
                            if sq_items:
                                sub_batches.append(sq_items)
                    if isinstance(llm_plan, dict) and llm_plan.get("method") == "llm_model":
                        track_domains = []
                        if llm_plan.get("needs_app_guide"):
                            track_domains.append(("ngan_hang_so", 12))
                        if llm_plan.get("needs_fee_schedule"):
                            track_domains.append(("bieu_phi_dich_vu", 10))
                        if llm_plan.get("needs_network_lookup"):
                            track_domains.append(("mang_luoi_hoat_dong", 10))
                        for dom_name, dom_k in track_domains:
                            dom_filter = dict(filters)
                            dom_filter["business_metadata.domain"] = dom_name
                            dom_items = _retrieve_with_weight(parsed_q, dom_k, dom_k, dom_filter)
                            if dom_items:
                                sub_batches.append((dom_items[:8], 0.35))
                    if sub_batches:
                        candidates = _merge_subquery_candidates(candidates, sub_batches, top_candidates)
            relaxed = any(item.get("retrieval_meta", {}).get("fallback_unfiltered") for item in candidates)
            if exact:
                candidates = [item for item in candidates if matches_exact(item, exact)]
            source_fallback = bool((source_chunks or exact and not source) and not reading and not candidates)
            if source_fallback:
                terms = set(search_key(search_query).split()) - _PIN_STOPWORDS
                scored = []
                for chunk in source_chunks or policy.corpus:
                    if not policy.allows(chunk, labels):
                        continue
                    if exact and not matches_exact(chunk, exact):
                        continue
                    passage = search_key(str(chunk.get("section_title") or "") + " " + str(chunk.get("content") or ""))
                    overlap = len(terms.intersection(passage.split()))
                    if overlap or exact:
                        scored.append((overlap, chunk))
                scored.sort(key=lambda pair: pair[0], reverse=True)
                candidates = [{"chunk": chunk} for _, chunk in scored[:top_candidates]]
            ret_span.update(
                candidate_count=len(candidates),
                candidate_k=candidate_count,
                filters=dict(filters),
                offset=offset,
                page=page,
                relaxed_filters=relaxed,
                source_fallback=source_fallback,
                candidates_preview=_summarize_rag_items(candidates, limit=15),
            )

        # Expand original search seeds only, within the configured retrieval scope.
        expansions = []
        graph_annotations = {}
        local_names = {edition[0] for edition in labels}
        all_labels = {edition: status for edition, status in policy.labels.items() if edition[0] not in local_names}
        all_labels.update(labels)
        # 3. Cross-Encoder Reranker
        table_paths_map = getattr(vector_store_ref, "_bm25_table_paths", None)
        table_notes_map = getattr(vector_store_ref, "_bm25_table_notes", None)
        if not reading and (table_paths_map or table_notes_map):
            candidates = _attach_table_hierarchy_view(candidates, table_paths_map, table_notes_map)
        rerank_query = parsed_q.normalized_query
        if not rerank_query:
            rerank_query = clean_query
        if isinstance(llm_plan, dict) and llm_plan.get("method") == "llm_model":
            planned_rw = str(llm_plan.get("rewritten_query") or "").strip()
            if planned_rw:
                rerank_query = planned_rw
        with trace_span("top_k_cross_encoder_reranker", run_type="retriever", query=clean_query, rerank_query=rerank_query, input_count=len(candidates), top_k=top_k) as rr_span:
            if reading:
                reranked = candidates
            else:
                reranked = self.pipeline.reranker.rerank(query=rerank_query, candidates=candidates, top_k=top_k)
                if not exact:
                    reranked = [item for item in reranked if _relevant_result(item)]
            rr_span.update(
                reranked_count=len(reranked),
                reranked_preview=_summarize_rag_items(reranked, limit=15),
            )
        if expand_graph and not reading and reranked:
            expansions = expand_links(graph_path, reranked, policy, all_labels, clean_query)
            if exact:
                expansions = [expansion for expansion in expansions
                              if all(matches_exact(chunk, exact) for chunk in expansion[:3])]
            for _, target, evidence, details in expansions:
                for raw in (target, evidence):
                    graph_annotations.setdefault((raw["source"], payload_index_key(raw)), details)

        # 4. Extract documents and source references
        docs = []
        sources = set()
        citations = []
        primary_count = len(reranked)
        ordered = {}
        related = []
        if page is not None and source_chunks and start_offset == 0:
            topic = " ".join(_strip_query_page_mentions(search_text).split())
            topic_words = {word for word in search_key(topic).split() if word.isalpha()} - _PIN_STOPWORDS
            if len(topic_words) >= 3:
                with trace_span("source_page_alternatives", run_type="retriever", page=page) as span:
                    off_page = []
                    parsed_topic = self.pipeline.parse_query(topic)
                    for item in search(parsed_topic):
                        pages = _extract_item_pages(item.get("chunk", item))
                        if pages and page not in pages and (not exact or matches_exact(item, exact)):
                            off_page.append(item)
                    if off_page:
                        related = self.pipeline.reranker.rerank(query=parsed_topic.normalized_query, candidates=off_page, top_k=3)
                        positions = {payload_index_key(chunk): i for i, chunk in enumerate(source_chunks)}
                        included = {payload_index_key(item.get("chunk", item)) for item in related}
                        for item in list(related):
                            anchor = item.get("chunk", item)
                            if anchor.get("chunk_type") == "table" or anchor.get("table_data"):
                                continue
                            for _, _, neighbor in _source_neighbors(anchor, source_chunks, positions, procedure=False):
                                key = payload_index_key(neighbor)
                                if (neighbor.get("chunk_type") != "table" or key in included
                                        or not set(_extract_item_pages(anchor)).intersection(_extract_item_pages(neighbor))
                                        or (exact and not matches_exact(neighbor, exact))):
                                    continue
                                related.append({"chunk": neighbor, "context_for": anchor["chunk_id"]})
                                included.add(key)
                                break
                    span.update(candidate_count=len(off_page), alternatives=_summarize_rag_items(related, limit=6))
        ranked = [item.get("chunk", item) for item in reranked if isinstance(item, dict)]
        ranked = [chunk for chunk in ranked if isinstance(chunk, dict)]
        seen = {(chunk.get("source"), payload_index_key(chunk)) for chunk in ranked}
        ranked_sources = {chunk.get("source") for chunk in ranked}
        graph_related = []
        for anchor, target, evidence, details in expansions:
            if not {anchor["source"], target["source"]} & ranked_sources:
                continue
            for raw in (target, evidence, anchor):
                identity = (raw["source"], payload_index_key(raw))
                if identity not in seen:
                    graph_related.append({"chunk": raw})
                    graph_annotations.setdefault(identity, details)
                    seen.add(identity)
        query_key = " " + search_key(clean_query) + " "
        procedure = is_procedure_query(clean_query)
        contact_phrases = tuple(_RAG_CFG["contact_phrases"])
        contact = any(phrase in query_key for phrase in contact_phrases)
        for chunk in [*ranked, *(item.get("chunk", item) for item in related)]:
            name = chunk.get("source")
            if name and name not in ordered:
                passages = self._source_chunks(name, scope)
                if policy.enabled:
                    passages = [passage for passage in passages if policy.allows(passage, all_labels)]
                ordered[name] = (passages, {payload_index_key(p): i for i, p in enumerate(passages)})
        if not reading:
            related_limit = int(_RAG_CFG["related_limit"])
            if procedure and ranked and _app_guide(ranked[0]):
                related_limit = int(_RAG_CFG["app_procedure_related_limit"])
            if top_k >= 3 and related_limit > 1:
                fee_rows = _exact_excel_fee_evidence(candidates, query_key, policy, all_labels)
                related.extend(row for row in fee_rows if not exact or matches_exact(row, exact))
            if top_k >= 3 and related_limit > 1 and any(cue in query_key for cue in _FEE_TABLE_CUES):
                _pin_intra_document_targets(ranked, ordered, seen, clean_query, related, len(related) + 1)
            reserved = []
            if top_k >= 3 and ranked:
                reserved = _reserve_query_targets(candidates, ranked, seen, clean_query)[:related_limit - len(related)]
                related_limit -= len(reserved)
            header_scan_limit = int(_RAG_CFG["guide_header_scan_limit"])
            anchor_limit = int(_RAG_CFG["neighbor_anchor_limit"])
            if related_limit and contact and ranked:
                anchor = ranked[0]
                name = anchor.get("source")
                if name in ordered and _app_guide(anchor):
                    passages, positions = ordered[name]
                    if payload_index_key(anchor) in positions:
                        anchor_scopes = set(anchor.get("business_metadata", {}).get("publication_scopes", []))
                        for header in passages[:header_scan_limit]:
                            if not _guide_header(header):
                                continue
                            header_scopes = set(header.get("business_metadata", {}).get("publication_scopes", []))
                            if (anchor_scopes or header_scopes) and not anchor_scopes.intersection(header_scopes):
                                continue
                            identity = (name, payload_index_key(header))
                            if identity not in seen:
                                related.append({"chunk": header, "context_for": anchor["chunk_id"],
                                                "document_contact": True})
                                seen.add(identity)
                            break
            neighbor_cap = related_limit
            if top_k >= 3 and not procedure and related_limit > 2:
                neighbor_cap = related_limit - 2
            deferred_neighbors = []
            for anchor in ranked[:anchor_limit]:
                is_table = anchor.get("chunk_type") == "table" or bool(anchor.get("table_data"))
                if not (procedure or is_table) or anchor.get("source") not in ordered:
                    continue
                passages, positions = ordered[anchor["source"]]
                for _, _, chunk in _source_neighbors(anchor, passages, positions, procedure):
                    identity = (chunk.get("source"), payload_index_key(chunk))
                    if identity in seen:
                        continue
                    if len(related) < neighbor_cap:
                        related.append({"chunk": chunk, "context_for": anchor["chunk_id"]})
                        seen.add(identity)
                    else:
                        deferred_neighbors.append((identity, chunk, anchor["chunk_id"]))
                    if len(related) >= related_limit and not (top_k >= 3 and not procedure):
                        break
                if len(related) >= related_limit and not (top_k >= 3 and not procedure):
                    break
            if top_k >= 3 and len(related) < related_limit:
                _pin_intra_document_targets(ranked, ordered, seen, clean_query, related, related_limit)
            for identity, chunk, anchor_id in deferred_neighbors:
                if len(related) >= related_limit:
                    break
                if identity in seen:
                    continue
                related.append({"chunk": chunk, "context_for": anchor_id})
                seen.add(identity)
            related.extend(reserved)
            related = graph_related + related

        # Attach hierarchical table path/notes for any returned docs missing them & add auto-Excel summary
        for name, (passages, _) in ordered.items():
            src_paths, src_notes = compute_corpus_table_paths(passages)
            if src_paths or src_notes:
                reranked = _attach_table_hierarchy_view(reranked, src_paths, src_notes)
                related = _attach_table_hierarchy_view(related, src_paths, src_notes)
        if not reading and ranked:
            corpus_ref = getattr(getattr(self.pipeline, "vector_store", None), "_bm25_corpus", ())
            auto_sum = None
            if not exact:
                excel_date = None
                if policy.dated:
                    excel_date = policy.day
                auto_sum = _maybe_auto_excel_summary(clean_query, ranked, ordered, corpus_ref, scope, as_of=excel_date)
            if auto_sum is not None:
                related.insert(0, auto_sum)

        default_source = TOOL_TEMPLATES["rag_default_source_label"]
        if exact:
            related = [item for item in related if matches_exact(item, exact)]
        score_precision = int(_RAG_CFG["score_precision"])
        for idx, item in enumerate([*reranked, *related]):
            if isinstance(item, dict):
                raw_chunk = item.get("chunk", item)
                score = float(item.get("relevance_score", item.get("rerank_score", item.get("score", 0.0))))
                rerank_score = float(item.get("rerank_score", score))
                rerank_rank = item.get("rerank_rank", idx + 1)
                rrf_score = float(item.get("rrf_score", 0.0))
            else:
                raw_chunk = item
                score = 0.0
                rerank_score = 0.0
                rerank_rank = idx + 1
                rrf_score = 0.0

            if not isinstance(raw_chunk, dict):
                chunk_id = f"chunk_{idx}"
                parent_id = ""
                source = default_source
                chunk_type = "text"
                breadcrumb = ""
                section_title = ""
                content = str(raw_chunk)
                parent_content = ""
                business_metadata = {}
                prev_chunk_id = ""
                next_chunk_id = ""
                table_data = None
                images = []
            else:
                chunk_id = str(raw_chunk.get("chunk_id") or f"chunk_{idx}")
                parent_id = str(raw_chunk.get("parent_id") or "")
                source = str(raw_chunk.get("source") or default_source)
                chunk_type = str(raw_chunk.get("chunk_type") or "text")
                breadcrumb = str(raw_chunk.get("breadcrumb") or "")
                section_title = str(raw_chunk.get("section_title") or "")
                content = str(raw_chunk.get("content") or "")
                parent_content = str(raw_chunk.get("parent_content") or "")
                bm = raw_chunk.get("business_metadata")
                if isinstance(bm, dict):
                    business_metadata = policy.annotate(raw_chunk, all_labels)
                else:
                    business_metadata = {}
                prev_chunk_id = str(raw_chunk.get("prev_chunk_id") or "")
                next_chunk_id = str(raw_chunk.get("next_chunk_id") or "")
                table_data = raw_chunk.get("table_data")
                raw_images = raw_chunk.get("images")
                if isinstance(raw_images, list):
                    images = raw_images
                else:
                    images = []

            doc_name = str(business_metadata.get("document_name") or source or "")
            if source:
                sources.add(source)
            graph_link = None
            if isinstance(raw_chunk, dict):
                graph_link = graph_annotations.get((source, payload_index_key(raw_chunk)))
            if graph_link:
                business_metadata = {**business_metadata, "graph_link": graph_link}

            excel_citation = business_metadata.get("citation", {})
            excel_location = {}
            if isinstance(excel_citation, dict) and excel_citation.get("sheet_name"):
                excel_location = {"sheet_name": excel_citation["sheet_name"],
                                  "cell_ranges": excel_citation.get("cell_ranges", [])}
                ranges = ", ".join(excel_location["cell_ranges"])
                if ranges:
                    label = excel_location["sheet_name"] + "!" + ranges
                else:
                    label = excel_location["sheet_name"]
                cite_label = f"[{doc_name} · {label}]"
            elif breadcrumb:
                cite_label = f"[{breadcrumb}]"
            elif section_title:
                cite_label = f"[{section_title}]"
            else:
                cite_label = f"[{source}]"

            citation = {
                "index": idx + 1,
                "label": cite_label,
                "source": source,
                "document_name": doc_name,
                "breadcrumb": breadcrumb,
                "section_title": section_title,
                "chunk_id": chunk_id,
                "chunk_type": chunk_type,
                "relevance_score": round(score, score_precision),
                **excel_location,
            }
            if isinstance(raw_chunk, dict) and (cited_pages := _extract_item_pages(raw_chunk)):
                citation["page_numbers"] = cited_pages
            if "temporal" in business_metadata:
                citation["temporal"] = business_metadata["temporal"]
            if graph_link:
                citation["graph_link"] = graph_link
            citations.append(citation)

            docs.append({
                "rank": idx + 1,
                "chunk_id": chunk_id,
                "parent_id": parent_id,
                "source": source,
                "chunk_type": chunk_type,
                "breadcrumb": breadcrumb,
                "section_title": section_title,
                "content": content,
                "parent_content": parent_content,
                "business_metadata": business_metadata,
                "prev_chunk_id": prev_chunk_id,
                "next_chunk_id": next_chunk_id,
                "table_data": table_data,
                "images": images,
                "relevance_score": score,
                "rerank_score": rerank_score,
                "rerank_rank": rerank_rank,
                "rrf_score": rrf_score
            })
            if source in ordered and isinstance(raw_chunk, dict):
                _, positions = ordered[source]
                position = positions.get(payload_index_key(raw_chunk))
                if position is not None:
                    docs[-1]["source_offset"] = position
            if isinstance(item, dict) and item.get("context_for"):
                docs[-1]["context_for"] = item["context_for"]
                if item.get("document_contact"):
                    docs[-1]["document_contact"] = True

        latency_ms = round((time.perf_counter() - start_t) * 1000.0, 2)
        inventory = []
        for name in sorted(ordered):
            count = len(ordered[name][0])
            if count:
                inventory.append({"source": name, "chunks": count})
        method = "none"
        if reading:
            if page is not None:
                method = "page_order"
            else:
                method = "source_order"
        elif reranked:
            method = reranked[0].get("rerank_method", "unknown")

        retrieval_quality = {}
        if source_status:
            retrieval_quality["source_status"] = source_status
            retrieval_quality["source_candidates"] = source_candidates
        retrieval_quality["relaxed_filters"] = relaxed
        if retrieval_errors or any(item.get("retrieval_meta", {}).get("dense_degraded") for item in candidates):
            retrieval_quality["dense_degraded"] = True
        retrieval_quality["reranker"] = method
        retrieval_quality["exact_match_missing"] = bool(exact and not docs)
        retrieval_quality["searched_candidates"] = len(candidates)
        retrieval_quality["top_scores"] = [round(item["relevance_score"], score_precision) for item in docs[:primary_count]]
        retrieval_quality["source_chunk_count"] = len(source_chunks)
        if page is not None:
            retrieval_quality["page"] = page
            retrieval_quality["page_chunk_count"] = len(reading_chunks)
            retrieval_quality["other_page_matches"] = len(related)
        if reading:
            read_start = start_offset
        else:
            read_start = None
        retrieval_quality["next_offset"] = self._next_offset(read_start, top_k, len(reading_chunks))
        retrieval_quality["source_inventory"] = inventory
        temporal_statuses = sorted({item["business_metadata"]["temporal"]["status"] for item in docs
                                    if "temporal" in item["business_metadata"]})
        if policy.dated or temporal_statuses:
            retrieval_quality["temporal"] = {"as_of": policy.day, "statuses": temporal_statuses}
            if policy.reference_only and "effective" not in temporal_statuses:
                retrieval_quality["temporal"]["status"] = "unverified_as_of"
        if graph_related or any("graph_link" in item for item in citations[:primary_count]):
            retrieval_quality["graph_related_count"] = len(graph_related)
        excel_tables = []
        seen_excel_ids = set()
        for doc_item in docs[:primary_count]:
            bm_item = doc_item.get("business_metadata")
            if not isinstance(bm_item, dict):
                continue
            tbl_id = str(bm_item.get("excel_table_id") or "")
            if not tbl_id or tbl_id in seen_excel_ids:
                continue
            seen_excel_ids.add(tbl_id)
            td_item = doc_item.get("table_data")
            cols_item = {}
            sheet_item = ""
            if isinstance(td_item, dict) and isinstance(td_item.get("excel"), dict):
                cols_item = td_item["excel"].get("columns") or {}
                sheet_item = str(td_item["excel"].get("sheet") or "")
            if not sheet_item and isinstance(bm_item.get("citation"), dict):
                sheet_item = str(bm_item["citation"].get("sheet_name") or "")
            entry = {
                "source": doc_item.get("source", ""),
                "excel_table_id": tbl_id,
                "columns": cols_item,
            }
            if sheet_item:
                entry["sheet_name"] = sheet_item
            excel_tables.append(entry)
        if excel_tables:
            retrieval_quality["excel_tables_available"] = excel_tables

        status = "success"
        if not docs:
            status = "no_results"
            if retrieval_errors:
                status = "retrieval_unavailable"
        result = {
            "status": status,
            "query": clean_query,
            "normalized_query": search_query,
            "total_found": primary_count,
            "documents": docs[:primary_count],
            "related_documents": docs[primary_count:],
            "sources": sorted(list(sources)),
            "citations": citations[:primary_count] + [
                item for item in citations[primary_count:]
                if "graph_link" in item or str(item.get("chunk_id") or "").endswith(":summary")
            ],
            "latency_ms": latency_ms,
            "retrieval_quality": retrieval_quality,
        }
        if include_candidates:
            result["candidate_documents"] = [item.get("chunk", item) for item in candidates]
        return result

    @staticmethod
    def _next_offset(offset, count, total):
        if offset is not None and offset + count < total:
            return offset + count
        return None
