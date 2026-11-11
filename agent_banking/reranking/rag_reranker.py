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
3. NO TYPING LIBRARY: Native Python types only, no typing module imports.
4. NO __INIT__.PY: Direct module imports.
"""

import os
import json
import urllib.request
import math
from agent_banking.config.models import get_model_chain, model_primary
from agent_banking.config.fallback import run_fallback


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
            cfg_model = os.getenv("RERANKER_MODEL", "")
            cfg_key = os.getenv("API_KEY", "")
            cfg_url = os.getenv("RERANK_URL", "")

        self.model_name = model_name
        if self.model_name is None:
            self.model_name = cfg_model or os.getenv("RERANKER_MODEL", "")

        self.api_key = api_key
        if self.api_key is None:
            self.api_key = cfg_key

        self.rerank_url = rerank_url
        if self.rerank_url is None:
            if base_url:
                self.rerank_url = base_url.rstrip("/") + "/rerank"
            else:
                self.rerank_url = cfg_url or os.getenv("RERANK_URL", "")

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
            if text.startswith("[Ngữ cảnh:"):
                return text
            bc = chunk.get("breadcrumb", "")
            if not isinstance(bc, str):
                bc = str(bc)
            title = chunk.get("section_title", "")
            if not isinstance(title, str):
                title = str(title)
            prefix = ""
            if bc and title:
                prefix = f"[{bc} > {title}]\n"
            elif title:
                prefix = f"[{title}]\n"
            elif bc:
                prefix = f"[{bc}]\n"
            return f"{prefix}{text}".strip()
        return str(item)

    def _fallback_local_rerank(self, query, candidates, top_k=3):
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
            if bc_matches:
                score = base_score * 1.3 + (len(bc_matches) / max(len(q_tokens), 1)) * 0.1
            else:
                score = base_score

            scored.append((item, score, idx))

        scored.sort(key=lambda x: x[1], reverse=True)
        results = []
        for rank, (item, score, orig_idx) in enumerate(scored[:top_k], start=1):
            if isinstance(item, dict):
                item_copy = dict(item)
                item_copy["relevance_score"] = round(float(score), 4)
                item_copy["rerank_score"] = round(float(score), 4)
                item_copy["rerank_rank"] = rank
                results.append(item_copy)
            else:
                results.append({
                    "document": item,
                    "relevance_score": round(float(score), 4),
                    "rerank_score": round(float(score), 4),
                    "rerank_rank": rank
                })
        return results

    def rerank(self, query, candidates, top_k=3):
        """
        Rerank a candidate list relative to a user query.
        Parameters:
        - query: User question (str, dict, or ParsedQuery)
        - candidates: List of chunks (list of dicts or list of strings)
        - top_k: Number of refined documents to retrieve (default 3)
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

        # Extract text snippets (bounded to 2500 characters)
        docs = [self._extract_text(c)[:2500] for c in candidates]

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
                "top_n": min(len(candidates), max(top_k, 5))
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

                    reranked = []
                    for res in api_results:
                        idx = res.get("index", 0)
                        score = float(res.get("relevance_score", 0.0))

                        if 0 <= idx < len(candidates):
                            orig_item = candidates[idx]

                            # Breadcrumb Entity Alignment bonus
                            chunk = {}
                            if isinstance(orig_item, dict):
                                chunk = orig_item.get("chunk", orig_item)
                            if isinstance(chunk, dict):
                                bc_val = chunk.get("breadcrumb", "")
                                if not isinstance(bc_val, str):
                                    bc_val = str(bc_val)
                                title_val = chunk.get("section_title", "")
                                if not isinstance(title_val, str):
                                    title_val = str(title_val)
                                bc_str = f"{bc_val} {title_val}".lower()
                                for token in query_text.lower().split():
                                    if len(token) >= 3 and token in bc_str:
                                        score = min(1.0, score + 0.02)
                                        break

                            if isinstance(orig_item, dict):
                                item_copy = dict(orig_item)
                                item_copy["relevance_score"] = round(score, 4)
                                item_copy["rerank_score"] = round(score, 4)
                                reranked.append(item_copy)
                            else:
                                reranked.append({
                                    "document": orig_item,
                                    "relevance_score": round(score, 4),
                                    "rerank_score": round(score, 4)
                                })

                    if reranked:
                        reranked.sort(key=lambda x: x["relevance_score"], reverse=True)
                        final_results = []
                        for rank, item in enumerate(reranked[:top_k], start=1):
                            item["rerank_rank"] = rank
                            final_results.append(item)
                        return final_results
            raise ValueError("No valid reranker results")

        return run_fallback("reranking", chain, call)


# Backward compatibility aliases
Reranker = BankingReranker
RAGReranker = BankingReranker
