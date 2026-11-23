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

import time

from agent_banking.chunking.universal.planning.schema import payload_index_key
from agent_banking.core.search_text import is_procedure_query, search_key
from agent_banking.tools.config.prompts import RAG_TOOL_DESCRIPTION, TOOL_TEMPLATES
from agent_banking.tools.config.settings import get_tool_runtime_config
from agent_banking.tools.excel_rows import lookup_excel_rows
from agent_banking.tracing.tracer import trace_span

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


def matches_exact(item: dict, exact: str) -> bool:
    raw = item.get("chunk", item)
    meta = raw.get("business_metadata") or {}
    fields = [raw.get(field) for field in ("source", "section_title", "breadcrumb", "content")]
    fields += [meta.get(field) for field in ("document_name", "product_name", "service_code")]
    needle = " " + search_key(exact) + " "
    return any(needle in " " + search_key(value) + " " for value in fields if isinstance(value, str))


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
            if app_steps and (_guide_logo(chunk) or current_section == section and _guide_header(chunk)):
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
    time_prefix = str(_RAG_CFG["guide_header"].get("excluded_terms", ["09:41"])[0])

    # Collect immediate linked siblings sharing the exact same section_title as ranked[:2]
    sibling_keys = {}
    primary_sources = []
    seen_sources = set()
    for anchor in ranked[:2]:
        src = anchor.get("source")
        if not src or src not in ordered:
            continue
        if src not in seen_sources:
            seen_sources.add(src)
            primary_sources.append((src, anchor))
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

    pinned = []
    for src, anchor in primary_sources:
        passages, _ = ordered[src]
        anchor_scopes = set(anchor.get("business_metadata", {}).get("publication_scopes", []))
        wants_app_subflow = _app_guide(anchor)
        src_sibs = sibling_keys.get(src, set())
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
            if token_hits == 0:
                continue

            score = token_hits
            is_table_chunk = (
                cand.get("chunk_type") == "table"
                or bool(cand.get("table_data"))
                or " phu luc " in sec_key
                or " bieu phi " in sec_key
            )
            if wants_fee_table and is_table_chunk:
                score += 3
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
        corpus = getattr(getattr(self.pipeline, "vector_store", None), "_bm25_corpus", [])
        source = source.replace("\\", "/").casefold()
        matches = []
        for chunk in corpus:
            name = str(chunk.get("source", "")).replace("\\", "/").casefold()
            basename = name.rsplit("/", 1)[-1]
            if name != source and basename != source and basename.rsplit(".", 1)[0] != source:
                continue
            if scope and scope not in chunk.get("business_metadata", {}).get("publication_scopes", []):
                continue
            matches.append(chunk)
        min_key_len = int(_SOURCE_CFG["min_key_length"])
        min_non_digit_len = int(_SOURCE_CFG["min_non_digit_key_length"])
        if not matches:
            tail = source.rsplit("/", 1)[-1]
            if "." in tail:
                stem = tail.rsplit(".", 1)[0]
            else:
                stem = tail
            src_key = search_key(stem)
            if src_key:
                prefix_matches = []
                for chunk in corpus:
                    if scope and scope not in chunk.get("business_metadata", {}).get("publication_scopes", []):
                        continue
                    basename = str(chunk.get("source", "")).replace("\\", "/").casefold().rsplit("/", 1)[-1]
                    if "." in basename:
                        base_stem = basename.rsplit(".", 1)[0]
                    else:
                        base_stem = basename
                    base_key = search_key(base_stem)
                    if base_key == src_key or base_key.startswith(src_key + " "):
                        prefix_matches.append(chunk)
                if prefix_matches:
                    matches = prefix_matches
                elif len(src_key) >= min_key_len:
                    padded_src = " " + src_key + " "
                    has_digit = any(ch.isdigit() for ch in src_key)
                    if has_digit or len(src_key) >= min_non_digit_len:
                        sub_matches = []
                        for chunk in corpus:
                            if scope and scope not in chunk.get("business_metadata", {}).get("publication_scopes", []):
                                continue
                            basename = str(chunk.get("source", "")).replace("\\", "/").casefold().rsplit("/", 1)[-1]
                            if "." in basename:
                                base_stem = basename.rsplit(".", 1)[0]
                            else:
                                base_stem = basename
                            padded_base = " " + search_key(base_stem) + " "
                            if padded_src in padded_base:
                                sub_matches.append(chunk)
                        if len({chunk["source"] for chunk in sub_matches}) == 1:
                            matches = sub_matches
        if len({chunk["source"] for chunk in matches}) > 1:
            raise ValueError("Source basename is ambiguous; use the complete source label")
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
                source: str = "", offset: int | None = None, excel: dict | None = None) -> dict:
        """
        Execute banking knowledge retrieval via Hybrid Retrieval & Reranker.
        """
        if top_k is None:
            top_k = int(_RAG_CFG["default_top_k"])
        start_t = time.perf_counter()
        if excel is not None:
            scope = getattr(self.pipeline.config, "KNOWLEDGE_PUBLICATION_SCOPE", "")
            corpus = getattr(self.pipeline.vector_store, "_bm25_corpus", ())
            result = lookup_excel_rows(corpus, excel, scope=scope, source=source, limit=top_k, offset=offset or 0)
            result["latency_ms"] = (time.perf_counter() - start_t) * 1000
            return result
        clean_query = query.strip()

        if not clean_query:
            return {
                "status": "empty_query",
                "documents": [],
                "sources": [],
                "citations": [],
                "latency_ms": 0.0
            }

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

        # 2. Hybrid Retrieval (Dense Vector + BM25 Okapi)
        candidate_count = candidate_k or self.pipeline.config.HYBRID_CANDIDATE_K
        top_candidates = max(candidate_count, top_k)
        exact_min_k = int(_RAG_CFG["exact_candidate_min_k"])
        if exact:
            candidate_count = max(exact_min_k, candidate_count)
            top_candidates = candidate_count
        scope = getattr(self.pipeline.config, "KNOWLEDGE_PUBLICATION_SCOPE", "")
        filters = {}
        if scope:
            filters = {"business_metadata.publication_scopes": scope}
        source_chunks = []
        if source:
            source_chunks = self._source_chunks(source, scope)
            if source_chunks:
                filters["source"] = source_chunks[0]["source"]
        with trace_span("top_k_hybrid_retrieval", run_type="retriever", query=search_query, top_k=top_candidates) as ret_span:
            if offset is not None:
                if not source or type(offset) is not int or offset < 0:
                    raise ValueError("Reading a document requires source and a nonnegative offset")
                candidates = [{"chunk": chunk} for chunk in source_chunks[offset:offset + top_k]]
            elif source and not source_chunks:
                candidates = []
            else:
                candidates = self.pipeline.retriever.retrieve(
                    query=parsed_q, top_k=top_candidates, candidate_k=candidate_count, filter_dict=filters,
                )
                if source:
                    allowed = {(chunk["source"], payload_index_key(chunk)) for chunk in source_chunks}
                    selected = []
                    for item in candidates:
                        raw = item.get("chunk", item)
                        if (raw.get("source"), payload_index_key(raw)) not in allowed:
                            continue
                        if scope and scope not in raw.get("business_metadata", {}).get("publication_scopes", []):
                            continue
                        selected.append(item)
                    candidates = selected
            relaxed = any(item.get("retrieval_meta", {}).get("fallback_unfiltered") for item in candidates)
            if exact:
                candidates = [item for item in candidates if matches_exact(item, exact)]
            ret_span.update(
                candidate_count=len(candidates),
                candidate_k=candidate_count,
                filters=dict(filters),
                offset=offset,
                relaxed_filters=relaxed,
                candidates_preview=_summarize_rag_items(candidates, limit=15),
            )

        # 3. Cross-Encoder Reranker
        with trace_span("top_k_cross_encoder_reranker", run_type="retriever", query=clean_query, input_count=len(candidates), top_k=top_k) as rr_span:
            if offset is not None:
                reranked = candidates
            else:
                reranked = self.pipeline.reranker.rerank(query=clean_query, candidates=candidates, top_k=top_k)
            rr_span.update(
                reranked_count=len(reranked),
                reranked_preview=_summarize_rag_items(reranked, limit=15),
            )

        # 4. Extract documents and source references
        docs = []
        sources = set()
        citations = []
        primary_count = len(reranked)
        ordered = {}
        related = []
        ranked = [item.get("chunk", item) for item in reranked if isinstance(item, dict)]
        ranked = [chunk for chunk in ranked if isinstance(chunk, dict)]
        seen = {(chunk.get("source"), payload_index_key(chunk)) for chunk in ranked}
        query_key = " " + search_key(clean_query) + " "
        procedure = is_procedure_query(clean_query)
        contact_phrases = tuple(_RAG_CFG["contact_phrases"])
        contact = any(phrase in query_key for phrase in contact_phrases)
        for chunk in ranked:
            name = chunk.get("source")
            if name and name not in ordered:
                passages = self._source_chunks(name, scope)
                ordered[name] = (passages, {payload_index_key(p): i for i, p in enumerate(passages)})
        if offset is None:
            related_limit = int(_RAG_CFG["related_limit"])
            if procedure and ranked and _app_guide(ranked[0]):
                related_limit = int(_RAG_CFG["app_procedure_related_limit"])
            header_scan_limit = int(_RAG_CFG["guide_header_scan_limit"])
            anchor_limit = int(_RAG_CFG["neighbor_anchor_limit"])
            if contact and ranked:
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
            for anchor in ranked[:anchor_limit]:
                is_table = anchor.get("chunk_type") == "table" or bool(anchor.get("table_data"))
                if not (procedure or is_table) or anchor.get("source") not in ordered:
                    continue
                passages, positions = ordered[anchor["source"]]
                for _, _, chunk in _source_neighbors(anchor, passages, positions, procedure):
                    identity = (chunk.get("source"), payload_index_key(chunk))
                    if identity in seen:
                        continue
                    related.append({"chunk": chunk, "context_for": anchor["chunk_id"]})
                    seen.add(identity)
                    if len(related) == related_limit:
                        break
                if len(related) == related_limit:
                    break
            if top_k >= 3 and len(related) < related_limit:
                _pin_intra_document_targets(ranked, ordered, seen, clean_query, related, related_limit)

        default_source = TOOL_TEMPLATES["rag_default_source_label"]
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
                    business_metadata = bm
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

            excel_citation = business_metadata.get("citation", {})
            excel_location = {}
            if isinstance(excel_citation, dict) and excel_citation.get("sheet_name"):
                excel_location = {"sheet_name": excel_citation["sheet_name"],
                                  "cell_ranges": excel_citation.get("cell_ranges", [])}
                ranges = ", ".join(excel_location["cell_ranges"])
                label = excel_location["sheet_name"] + ("!" + ranges if ranges else "")
                cite_label = f"[{doc_name} · {label}]"
            elif breadcrumb:
                cite_label = f"[{breadcrumb}]"
            elif section_title:
                cite_label = f"[{section_title}]"
            else:
                cite_label = f"[{source}]"

            citations.append({
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
            })

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
        if offset is not None:
            method = "source_order"
        elif reranked:
            method = reranked[0].get("rerank_method", "unknown")

        return {
            "status": "success",
            "query": clean_query,
            "normalized_query": search_query,
            "total_found": primary_count,
            "documents": docs[:primary_count],
            "related_documents": docs[primary_count:],
            "sources": sorted(list(sources)),
            "citations": citations[:primary_count],
            "latency_ms": latency_ms,
            "retrieval_quality": {
                "relaxed_filters": relaxed,
                "reranker": method,
                "exact_match_missing": bool(exact and not docs),
                "searched_candidates": len(candidates),
                "top_scores": [round(item["relevance_score"], score_precision) for item in docs[:primary_count]],
                "source_chunk_count": len(source_chunks),
                "next_offset": self._next_offset(offset, top_k, len(source_chunks)),
                "source_inventory": inventory,
            }
        }

    @staticmethod
    def _next_offset(offset, count, total):
        if offset is not None and offset + count < total:
            return offset + count
        return None
