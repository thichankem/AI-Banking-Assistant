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

_RAG_CFG = get_tool_runtime_config()["rag_tool"]
_HEADER_CFG = _RAG_CFG["guide_header"]
_LOGO_CFG = _RAG_CFG["guide_logo"]
_NEIGHBOR_CFG = _RAG_CFG["neighbors"]
_SOURCE_CFG = _RAG_CFG["source_matching"]


def matches_exact(item: dict, exact: str) -> bool:
    raw = item.get("chunk", item)
    meta = raw.get("business_metadata") or {}
    fields = [raw.get(field) for field in ("source", "section_title", "breadcrumb", "content")]
    fields += [meta.get(field) for field in ("document_name", "product_name", "service_code")]
    needle = " " + search_key(exact) + " "
    return any(needle in " " + search_key(value) + " " for value in fields if isinstance(value, str))


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
            logical_distance += 1
            found.append((logical_distance, index, chunk))
            if logical_distance == step_limit:
                break
    return sorted(found, key=lambda item: (item[0], item[1]))


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
                source: str = "", offset: int | None = None) -> dict:
        """
        Execute banking knowledge retrieval via Hybrid Retrieval & Reranker.
        """
        if top_k is None:
            top_k = int(_RAG_CFG["default_top_k"])
        start_t = time.perf_counter()
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
        parsed_q = self.pipeline.parse_query(search_text)
        search_query = parsed_q.normalized_query

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

        # 3. Cross-Encoder Reranker
        if offset is not None:
            reranked = candidates
        else:
            reranked = self.pipeline.reranker.rerank(query=clean_query, candidates=candidates, top_k=top_k)

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

            if breadcrumb:
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
                "relevance_score": round(score, score_precision)
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
