"""
Banking Document Reranker Module.
======================================================================
Core Responsibilities:
1. Receives Top-K candidates from Hybrid Search (Dense + BM25).
2. Calls Cross-Encoder API to compute deep semantic relevance scores between Query and Context.
3. Refines candidates to the most accurate Top-N chunks for LLM context generation.
4. Falls back to lexical Jaccard matching with breadcrumb boosting if cloud API is unavailable.

DESIGN PRINCIPLES (GEMINI.md & AGENTS.md):
1. 0% REGEX LAW: String methods only, no regex imports or functions.
2. ZERO TERNARY: Explicit if/else statements only, no ternary operators.
4. NO __INIT__.PY: Direct module imports.
"""

import os
import json
import urllib.request
import math
from agent_banking.core.models import get_model_chain, model_primary
from agent_banking.core.fallback import run_fallback
from agent_banking.reranking.config.models import MODELS
from agent_banking.reranking.config.settings import RerankingConfig


class BankingReranker:
    """
    Lightweight, resilient Cross-Encoder document reranker with local fallback.
    """

    def __init__(
        self,
        model_name=None,
        api_key=None,
        rerank_url=None,
        base_url=None
    ):
        try:
            primary = model_primary("reranking")
            cfg_model = primary.model
            cfg_key = primary.api_key
            cfg_url = primary.url
        except Exception:
            cfg_model = RerankingConfig.RERANKER_MODEL
            cfg_key = ""
            cfg_url = ""

        self.model_name = model_name
        if self.model_name is None:
            if cfg_model:
                self.model_name = cfg_model
            else:
                self.model_name = RerankingConfig.RERANKER_MODEL

        self.api_key = api_key
        if self.api_key is None:
            self.api_key = cfg_key

        self.rerank_url = rerank_url
        if self.rerank_url is None:
            if base_url:
                rerank_path = ""
                if MODELS:
                    rerank_path = str(MODELS[0].get("path", ""))
                self.rerank_url = base_url.rstrip("/") + rerank_path
            else:
                self.rerank_url = cfg_url

    @staticmethod
    def _normalize_query(query):
        """Extract and normalize query string from diverse input formats."""
        if query is None:
            return ""
        if isinstance(query, str):
            return query.strip()
        if isinstance(query, dict):
            q_val = (
                query.get("clean_query")
                or query.get("query_text")
                or query.get("query")
                or query.get("raw_query")
                or query.get("normalized_query")
                or ""
            )
            if not isinstance(q_val, str):
                q_val = str(q_val)
            return q_val.strip()
        for attr in ("clean_query", "query_text", "raw_query", "normalized_query", "query"):
            val = getattr(query, attr, None)
            if val is not None:
                return str(val).strip()
        return str(query).strip()

    @staticmethod
    def _metadata_score_bonus(query_text, chunk):
        from agent_banking.core.search_text import search_key

        if not isinstance(chunk, dict) or not query_text:
            return 0.0
        bonus_cfg = RerankingConfig.RERANK_METADATA_BONUS
        stopwords = set(bonus_cfg["stopwords"])
        generic_title_words = stopwords | set(bonus_cfg["extra_generic_title_words"])
        min_tok_len = int(bonus_cfg["min_token_length"])
        sec_tok_min_len = int(bonus_cfg["section_token_min_length"])
        doc_code_bonus = float(bonus_cfg["doc_code_bonus"])
        bigram_weight = float(bonus_cfg["bigram_weight"])
        distinct_overlap_weight = float(bonus_cfg["distinct_overlap_weight"])
        max_title_bonus = float(bonus_cfg["max_title_bonus"])
        hdsd_prefix = str(bonus_cfg["hdsd_source_prefix"])
        app_cues = tuple(bonus_cfg["app_cues"])
        app_cue_bonus = float(bonus_cfg["app_cue_bonus"])
        years = tuple(bonus_cfg["year_tokens"])
        year_match_bonus = float(bonus_cfg["year_match_bonus"])
        year_mismatch_penalty = float(bonus_cfg["year_mismatch_penalty"])
        section_fallback_bonus = float(bonus_cfg["section_fallback_bonus"])

        q_key = search_key(query_text)
        q_tokens = []
        for token in q_key.split():
            if len(token) >= min_tok_len and token not in stopwords:
                q_tokens.append(token)
        if not q_tokens:
            return 0.0

        src_name = str(chunk.get("source") or "").replace("\\", "/").rsplit("/", 1)[-1]
        if "." in src_name:
            src_stem = src_name.rsplit(".", 1)[0]
        else:
            src_stem = src_name
        doc_name = str(chunk.get("document_name") or "")
        src_key = search_key(src_stem)
        doc_key = search_key(src_stem + " " + doc_name)
        bc_val = chunk.get("breadcrumb", "")
        if not isinstance(bc_val, str):
            bc_val = str(bc_val)
        title_val = chunk.get("section_title", "")
        if not isinstance(title_val, str):
            title_val = str(title_val)
        sec_key = search_key(bc_val + " " + title_val)

        bonus = 0.0
        padded_q = " " + q_key + " "
        src_tokens = src_key.split()
        if len(src_tokens) >= 2 and src_tokens[1] and src_tokens[1][0].isdigit():
            doc_code = src_tokens[0] + " " + src_tokens[1]
            if (" " + doc_code + " ") in padded_q:
                bonus += doc_code_bonus

        distinctive_q = [tok for tok in q_tokens if tok not in generic_title_words]
        padded_src = " " + src_key + " "
        padded_sec = " " + sec_key + " "
        distinct_src_tokens = {tok for tok in src_tokens if len(tok) >= min_tok_len and tok not in generic_title_words}
        distinct_overlap = 0
        for tok in set(distinctive_q):
            if tok in distinct_src_tokens:
                distinct_overlap += 1
        bigram_src_hits = 0
        seen_bigrams = set()
        for idx in range(len(q_tokens) - 1):
            t1 = q_tokens[idx]
            t2 = q_tokens[idx + 1]
            if t1 in generic_title_words and t2 in generic_title_words:
                continue
            bigram = t1 + " " + t2
            if bigram in seen_bigrams:
                continue
            seen_bigrams.add(bigram)
            if (" " + bigram + " ") in padded_src:
                bigram_src_hits += 1

        title_bonus = min(max_title_bonus, bigram_src_hits * bigram_weight + distinct_overlap * distinct_overlap_weight)
        bonus += title_bonus

        if src_key.startswith(hdsd_prefix) and (distinct_overlap > 0 or bigram_src_hits > 0):
            for cue in app_cues:
                if cue in padded_q:
                    bonus += app_cue_bonus
                    break

        for yr in years:
            if yr in q_tokens:
                if yr in src_tokens:
                    bonus += year_match_bonus
                else:
                    for other_yr in years:
                        if other_yr != yr and other_yr in src_tokens:
                            bonus -= year_mismatch_penalty
                            break
                break

        if title_bonus == 0.0:
            for token in q_key.split():
                if len(token) >= sec_tok_min_len and (" " + token + " ") in padded_sec:
                    bonus += section_fallback_bonus
                    break

        structural_query_markers = (
            " phu luc ", " bieu phi ", " bang phi ",
            " hinh 1 ", " hinh 2 ", " hinh 3 ",
            " so do ", " luu do ", " chu trinh ", " lo trinh ",
        )
        if any(marker in padded_q for marker in structural_query_markers):
            head_key = search_key(str(chunk.get("content") or "")[:320])
            padded_head = " " + head_key + " "
            structural_chunk_markers = (
                " phu luc ", " bieu phi ",
                " hinh 1 ", " hinh 2 ", " hinh 3 ",
                " so do ", " quy trinh chuan ", " ma tran ",
            )
            if (
                any(marker in padded_sec or marker in padded_head for marker in structural_chunk_markers)
                or chunk.get("chunk_type") == "table"
                or bool(chunk.get("table_data"))
            ):
                bonus += 0.045

        q_all_tokens = q_key.split()
        for idx in range(len(q_all_tokens) - 1):
            unit_tok = q_all_tokens[idx]
            num_tok = q_all_tokens[idx + 1]
            if unit_tok in ("dieu", "buoc") and num_tok.isdigit():
                if (" " + unit_tok + " " + num_tok + " ") in padded_sec:
                    bonus += 0.04
                    break

        return bonus

    @staticmethod
    def _extract_text(item):
        """Extract clean text representation from any candidate object."""
        if isinstance(item, str):
            return item
        if isinstance(item, tuple) and len(item) > 0:
            item = item[0]
            if isinstance(item, str):
                return item
        if isinstance(item, dict):
            chunk = item.get("chunk", item)
            if not isinstance(chunk, dict):
                chunk = item
            raw_text = chunk.get("content")
            if raw_text is None or raw_text == "":
                raw_text = chunk.get("original_text")
            if raw_text is None or raw_text == "":
                raw_text = chunk.get("text")
            if raw_text is None:
                raw_text = ""
            if not isinstance(raw_text, str):
                raw_text = str(raw_text)
            text = raw_text.strip()
            if text.startswith(RerankingConfig.RERANK_CONTEXT_PREFIX_MARKER):
                return text
            bc = chunk.get("breadcrumb", "")
            if not isinstance(bc, str):
                bc = str(bc)
            title = chunk.get("section_title", "")
            if not isinstance(title, str):
                title = str(title)
            doc_name = str(chunk.get("document_name") or "").strip()
            src_name = str(chunk.get("source") or "").replace("\\", "/").rsplit("/", 1)[-1]
            if "." in src_name:
                src_name = src_name.rsplit(".", 1)[0]
            header_doc = src_name
            if doc_name and doc_name.casefold() not in src_name.casefold():
                if src_name:
                    header_doc = src_name + " - " + doc_name
                else:
                    header_doc = doc_name
            prefix_parts = []
            if header_doc and header_doc.casefold() not in bc.casefold() and header_doc.casefold() not in title.casefold():
                prefix_parts.append(RerankingConfig.RERANK_DOC_HEADER_PREFIX + header_doc)
            if bc and title:
                prefix_parts.append(bc + " > " + title)
            elif title:
                prefix_parts.append(title)
            elif bc:
                prefix_parts.append(bc)
            meta = chunk.get("business_metadata")
            if isinstance(meta, dict):
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
                            prefix_parts.append("Trang " + ", ".join(str(p) for p in pages))
            parent_text = str(chunk.get("parent_content") or "").strip()
            if len(text) < 320 and len(parent_text) > len(text) + 40:
                text = text + "\n" + parent_text[:700]
            prefix = ""
            if prefix_parts:
                prefix = "[" + " | ".join(prefix_parts) + "]\n"
            return (prefix + text).strip()
        return str(item)

    def _fallback_local_rerank(self, query, candidates, top_k=RerankingConfig.RERANK_TOP_K):
        """
        Local fallback via lexical Jaccard matching with Breadcrumb boosting (0% Regex Law).
        Used when offline, unconfigured, or during network disruptions.
        """
        if not candidates:
            return []
        if top_k is None:
            return []
        if top_k <= 0:
            return []

        def _get_tokens(s):
            if not s:
                return set()
            cleaned = []
            for ch in str(s).lower():
                if ch.isalnum() or ch == "_":
                    cleaned.append(ch)
                else:
                    cleaned.append(" ")
            return set("".join(cleaned).split())

        q_tokens = _get_tokens(query)
        scored = []
        for idx, item in enumerate(candidates):
            text = self._extract_text(item)
            doc_tokens = _get_tokens(text)

            chunk = {}
            if isinstance(item, dict):
                chunk = item.get("chunk", item)
                if not isinstance(chunk, dict):
                    chunk = item
            bc_str = ""
            if isinstance(chunk, dict):
                bc_val = chunk.get("breadcrumb", "")
                if not isinstance(bc_val, str):
                    bc_val = str(bc_val)
                title_val = chunk.get("section_title", "")
                if not isinstance(title_val, str):
                    title_val = str(title_val)
                bc_str = f"{bc_val} {title_val}"
            bc_tokens = _get_tokens(bc_str)
            bc_matches = q_tokens.intersection(bc_tokens)

            intersection = q_tokens.intersection(doc_tokens)
            union = q_tokens.union(doc_tokens)
            if len(union) > 0:
                base_score = len(intersection) / len(union)
            else:
                base_score = 0.0

            # Breadcrumb boost bonus when query matches entity in breadcrumb
            bc_mult = RerankingConfig.RERANK_LEXICAL_BREADCRUMB_MULTIPLIER
            bc_overlap_w = RerankingConfig.RERANK_LEXICAL_BREADCRUMB_OVERLAP_WEIGHT
            if bc_matches:
                score = base_score * bc_mult + (len(bc_matches) / max(len(q_tokens), 1)) * bc_overlap_w
            else:
                score = base_score

            scored.append((item, score, idx))

        scored.sort(key=lambda x: x[1], reverse=True)
        score_decimals = RerankingConfig.RERANK_SCORE_ROUND_DECIMALS
        results = []
        for rank, (item, score, orig_idx) in enumerate(scored[:top_k], start=1):
            if isinstance(item, dict):
                item_copy = dict(item)
                item_copy["relevance_score"] = round(float(score), score_decimals)
                item_copy["rerank_score"] = round(float(score), score_decimals)
                item_copy["rerank_rank"] = rank
                item_copy["rerank_method"] = "local_lexical"
                results.append(item_copy)
            else:
                results.append({
                    "document": item,
                    "relevance_score": round(float(score), score_decimals),
                    "rerank_score": round(float(score), score_decimals),
                    "rerank_rank": rank,
                    "rerank_method": "local_lexical"
                })
        return results

    def rerank(self, query, candidates, top_k=RerankingConfig.RERANK_TOP_K):
        """
        Rerank a candidate list relative to a user query.
        Parameters:
        - query: User question (str, dict, or ParsedQuery)
        - candidates: List of chunks (list of dicts or list of strings)
        - top_k: Number of refined documents to retrieve
        """
        if not candidates:
            return []
        if top_k is None:
            return []
        if top_k <= 0:
            return []

        if query is None:
            query = ""

        # 1. Extract query text supporting str, dict, ParsedQuery
        query_text = self._normalize_query(query)

        if not query_text:
            return self._fallback_local_rerank("", candidates, top_k=top_k)

        # Extract text snippets (bounded to configured max_doc_chars)
        max_doc_chars = RerankingConfig.RERANK_MAX_DOC_CHARS
        docs = [self._extract_text(c)[:max_doc_chars] for c in candidates]

        if not self.rerank_url or not self.api_key or not self.model_name:
            return self._fallback_local_rerank(query_text, candidates, top_k=top_k)

        chain = get_model_chain("reranking", {
            "model": self.model_name, "url": self.rerank_url, "api_key": self.api_key,
        })

        def call(candidate):
            headers = {
                "Authorization": f"Bearer {candidate.api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": candidate.model,
                "query": query_text,
                "documents": docs,
                "top_n": len(candidates)
            }

            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(candidate.url, data=data, headers=headers)
            with urllib.request.urlopen(req, timeout=candidate.timeout) as resp:
                resp_status = getattr(resp, "status", 200)
                if resp_status == 200 or not isinstance(resp_status, int):
                    resp_data = json.loads(resp.read().decode("utf-8"))
                    api_results = resp_data.get("results", [])
                    if not isinstance(api_results, list) or not api_results:
                        raise ValueError("Empty reranker response")
                    seen = set()
                    for result in api_results:
                        index = result.get("index")
                        score = result.get("relevance_score")
                        if not isinstance(index, int) or index in seen or not 0 <= index < len(candidates):
                            raise ValueError("Invalid reranker index")
                        if not isinstance(score, (int, float)) or not math.isfinite(score):
                            raise ValueError("Invalid reranker score")
                        seen.add(index)

                    score_min = RerankingConfig.RERANK_SCORE_MIN
                    score_max = RerankingConfig.RERANK_SCORE_MAX
                    score_decimals = RerankingConfig.RERANK_SCORE_ROUND_DECIMALS
                    reranked = []
                    for res in api_results:
                        idx = res.get("index", 0)
                        score = float(res.get("relevance_score", 0.0))

                        if 0 <= idx < len(candidates):
                            orig_item = candidates[idx]

                            # Breadcrumb Entity Alignment bonus
                            raw_adj_score = score
                            chunk = {}
                            if isinstance(orig_item, dict):
                                chunk = orig_item.get("chunk", orig_item)
                            if isinstance(chunk, dict):
                                meta_bonus = self._metadata_score_bonus(query_text, chunk)
                                raw_adj_score = score + meta_bonus
                                score = max(score_min, min(score_max, raw_adj_score))

                            if isinstance(orig_item, dict):
                                item_copy = dict(orig_item)
                                item_copy["relevance_score"] = round(score, score_decimals)
                                item_copy["rerank_score"] = round(score, score_decimals)
                                reranked.append((raw_adj_score, item_copy))
                            else:
                                reranked.append((raw_adj_score, {
                                    "document": orig_item,
                                    "relevance_score": round(score, score_decimals),
                                    "rerank_score": round(score, score_decimals)
                                }))

                    if reranked:
                        reranked.sort(key=lambda pair: pair[0], reverse=True)
                        ordered = [pair[1] for pair in reranked]
                        snippet_chars = RerankingConfig.RERANK_DEDUP_SNIPPET_CHARS
                        div_min_top_k = RerankingConfig.RERANK_DIVERSITY_MIN_TOP_K
                        div_max_per_src = RerankingConfig.RERANK_DIVERSITY_MAX_PER_SOURCE
                        if len(ordered) > top_k:
                            selected = []
                            deferred = []
                            dup_deferred = []
                            source_counts = {}
                            seen_snippets = set()
                            best_deferred_score = 0.0
                            for item in ordered:
                                raw_chunk = item.get("chunk", item)
                                src_id = ""
                                snippet_key = None
                                if isinstance(raw_chunk, dict):
                                    src_id = str(raw_chunk.get("source") or "").replace("\\", "/").casefold()
                                    snippet = " ".join(str(raw_chunk.get("content") or "").split())[:snippet_chars].casefold()
                                    if src_id and snippet:
                                        snippet_key = (src_id, snippet)
                                if snippet_key is not None and snippet_key in seen_snippets:
                                    dup_deferred.append(item)
                                    continue
                                item_score = float(item.get("relevance_score", 0.0))
                                count = source_counts.get(src_id, 0)
                                if top_k >= div_min_top_k and src_id and count >= div_max_per_src:
                                    if item_score > best_deferred_score:
                                        best_deferred_score = item_score
                                    deferred.append(item)
                                elif top_k >= div_min_top_k and best_deferred_score >= 0.45 and item_score < 0.22:
                                    deferred.append(item)
                                else:
                                    selected.append(item)
                                    if src_id:
                                        source_counts[src_id] = count + 1
                                    if snippet_key is not None:
                                        seen_snippets.add(snippet_key)
                                if len(selected) >= top_k:
                                    break
                            if len(selected) < top_k:
                                deferred.sort(key=lambda d_item: float(d_item.get("relevance_score", 0.0)), reverse=True)
                                selected.extend(deferred[:top_k - len(selected)])
                                if len(selected) < top_k:
                                    selected.extend(dup_deferred[:top_k - len(selected)])
                            chosen = selected
                        else:
                            chosen = ordered[:top_k]
                        final_results = []
                        for rank, item in enumerate(chosen, start=1):
                            item["rerank_rank"] = rank
                            item["rerank_method"] = "cloud_cross_encoder"
                            final_results.append(item)
                        return final_results
            raise ValueError("No valid reranker results")

        try:
            return run_fallback("reranking", chain, call)
        except Exception:
            return self._fallback_local_rerank(query_text, candidates, top_k=top_k)


# Backward compatibility aliases
Reranker = BankingReranker
RAGReranker = BankingReranker
