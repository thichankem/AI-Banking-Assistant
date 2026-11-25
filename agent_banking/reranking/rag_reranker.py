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

import hashlib
import os
import json
import sqlite3
import urllib.request
import math
from pathlib import Path
from agent_banking.chunking.universal.enrichment.contextual_enricher import strip_ui_noise_for_context
from agent_banking.core.environment import ROOT_DIR
from agent_banking.core.llm_rag_cache import call_rag_llm_json, rag_llm_available
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
        base_url=None,
        cache_db_path=None,
        cache_enabled=None,
        use_llm_selector=None,
        llm_client=None,
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

        if cache_enabled is not None:
            self.cache_enabled = bool(cache_enabled)
        else:
            self.cache_enabled = RerankingConfig.RERANK_CACHE_ENABLED

        raw_cache_path = cache_db_path
        if not raw_cache_path:
            raw_cache_path = RerankingConfig.RERANK_CACHE_DB_PATH
        cache_p = Path(raw_cache_path)
        if cache_p.is_absolute():
            self.cache_path = cache_p
        else:
            self.cache_path = (ROOT_DIR / cache_p).resolve()
        self._cache_initialized = False
        self.last_cache_hits = 0
        self.last_cache_misses = 0
        self.use_llm_selector = use_llm_selector
        self.llm_client = llm_client

    def _ensure_cache_initialized(self):
        if not self.cache_enabled or self._cache_initialized:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.cache_path, timeout=RerankingConfig.RERANK_SQLITE_TIMEOUT_SEC)
        try:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
            conn.execute(f"PRAGMA busy_timeout={RerankingConfig.RERANK_SQLITE_BUSY_TIMEOUT_MS};")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS rerank_cache (
                    hash_key TEXT PRIMARY KEY,
                    score REAL NOT NULL
                );
                """
            )
            conn.commit()
            self._cache_initialized = True
        finally:
            conn.close()

    @staticmethod
    def _rerank_cache_key(url, model, query_text, doc_text):
        ver = RerankingConfig.RERANK_CACHE_KEY_VERSION
        material = f"{ver}:{url}:{model}:{query_text}\x1f{doc_text}"
        return hashlib.sha256(material.encode("utf-8")).hexdigest()

    def _get_cached_scores_batch(self, hashes):
        if not self.cache_enabled or not hashes:
            return {}
        try:
            self._ensure_cache_initialized()
        except Exception:
            return {}
        unique_hashes = list(set(hashes))
        results = {}
        conn = sqlite3.connect(self.cache_path, timeout=RerankingConfig.RERANK_SQLITE_TIMEOUT_SEC)
        try:
            conn.execute(f"PRAGMA busy_timeout={RerankingConfig.RERANK_SQLITE_BUSY_TIMEOUT_MS};")
            for start in range(0, len(unique_hashes), 256):
                batch = unique_hashes[start:start + 256]
                placeholders = ",".join("?" for _ in batch)
                cursor = conn.execute(
                    f"SELECT hash_key, score FROM rerank_cache WHERE hash_key IN ({placeholders})",
                    batch,
                )
                for h_key, sc_val in cursor.fetchall():
                    if isinstance(sc_val, (int, float)) and math.isfinite(sc_val):
                        results[h_key] = float(sc_val)
        except Exception:
            return {}
        finally:
            conn.close()
        return results

    def _save_cached_scores_batch(self, items):
        if not self.cache_enabled or not items:
            return
        try:
            self._ensure_cache_initialized()
            conn = sqlite3.connect(self.cache_path, timeout=RerankingConfig.RERANK_SQLITE_TIMEOUT_SEC)
            try:
                conn.execute(f"PRAGMA busy_timeout={RerankingConfig.RERANK_SQLITE_BUSY_TIMEOUT_MS};")
                conn.executemany(
                    "INSERT OR REPLACE INTO rerank_cache (hash_key, score) VALUES (?, ?);",
                    items,
                )
                conn.commit()
            finally:
                conn.close()
        except Exception:
            pass

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
        digital_product_cues = tuple(bonus_cfg.get("digital_product_cues", ()))
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
        meta = chunk.get("business_metadata")
        if not isinstance(meta, dict):
            meta = {}
        doc_name = str(meta.get("document_name") or chunk.get("document_name") or "")
        cit = meta.get("citation")
        if not isinstance(cit, dict):
            cit = {}
        sheet_name = str(cit.get("sheet_name") or "")
        src_key = search_key(src_stem)
        doc_key = search_key(src_stem + " " + doc_name)
        bc_val = chunk.get("breadcrumb", "")
        if not isinstance(bc_val, str):
            bc_val = str(bc_val)
        title_val = chunk.get("section_title", "")
        if not isinstance(title_val, str):
            title_val = str(title_val)
        sec_key = search_key(bc_val + " " + title_val + " " + sheet_name)

        bonus = 0.0
        padded_q = " " + q_key + " "
        src_tokens = src_key.split()
        if len(src_tokens) >= 2 and src_tokens[1] and src_tokens[1][0].isdigit():
            doc_code = src_tokens[0] + " " + src_tokens[1]
            if (" " + doc_code + " ") in padded_q:
                bonus += doc_code_bonus

        distinctive_q = [tok for tok in q_tokens if tok not in generic_title_words]
        doc_tokens = doc_key.split()
        padded_src = " " + doc_key + " "
        padded_sec = " " + sec_key + " "
        distinct_src_tokens = {tok for tok in doc_tokens if len(tok) >= min_tok_len and tok not in generic_title_words}
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

        has_app_cue = any(cue in padded_q for cue in app_cues)
        has_digital_cue = any(dc in padded_q for dc in digital_product_cues)
        if src_key.startswith(hdsd_prefix):
            if distinct_overlap > 0 or bigram_src_hits > 0:
                if has_app_cue or has_digital_cue:
                    if bigram_src_hits > 0 or distinct_overlap >= 2:
                        bonus += max(app_cue_bonus, 0.05)
                    else:
                        bonus += app_cue_bonus

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

        alnum_codes = []
        for tok in q_all_tokens:
            if len(tok) >= 3 and any(ch.isdigit() for ch in tok) and any(ch.isalpha() for ch in tok):
                if tok not in alnum_codes:
                    alnum_codes.append(tok)
        if alnum_codes:
            body_key = " " + search_key(str(chunk.get("content") or "")) + " "
            for code_tok in alnum_codes:
                if (" " + code_tok + " ") in body_key or (" " + code_tok + " ") in padded_sec:
                    bonus += 0.06
                    break

        src_path_low = str(chunk.get("source") or "").replace("\\", "/").casefold()
        fee_cues = (
            " bieu phi ", " muc phi ", " ma phi ", " phi mo tai khoan ", " phi dong tai khoan ",
            " phi quan ly tai khoan ", " phi kiem dem ", " phi doi tien ", " phi kieu hoi ",
            " phi nop tien ", " phi rut tien ", " phi bao lanh ", " phi nho thu ", " phi lc ",
            " phi chuyen tien ", " phi thanh toan quoc te ", " phi sao ke ",
            " chiu khoan phi ", " chiu phi ", " thu phi ", " tinh phi ", " chi phi ",
            " phi dich vu ", " phi giu ho ", " giu ho tai san ", " phi sec ", " phi ngan quy ",
            " phi tai san bao dam ", " phi giai chap ", " muc phi giai chap ", " niem phong ",
            " dormant ", " khong hoat dong ",
        )
        explicit_nv_or_tier = any(m in padded_q for m in (" nv ", " goi bach kim ", " goi uu tien ", " goi thinh vuong ", " phan hang "))
        if any(fc in padded_q for fc in fee_cues) and not explicit_nv_or_tier:
            if src_path_low.endswith(".xlsx") and ("8. biểu phí dịch vụ/" in src_path_low or src_key.startswith("bp ")):
                bonus += 0.08

        network_cues = (" chi nhanh ", " phong giao dich ", " atm ", " cdm ", " mang luoi ", " diem giao dich ")
        if any(nc in padded_q for nc in network_cues):
            if src_path_low.endswith(".xlsx") and ("9. mạng lưới hoạt động/" in src_path_low or src_key.startswith("ml ")):
                bonus += 0.08

        eligibility_cues = (
            " dieu kien ", " toi thieu ", " toi da ", " thoi gian nhan luong ",
            " bao nhieu thang ", " han muc ", " lai suat ", " ty le ",
        )
        ui_cues = (
            " buoc ", " thao tac ", " man hinh ", " bam vao ", " chon muc ",
            " nhan nut ", " giao dien ", " tren app ", " hdsd ",
        )
        has_ui_cue = any(uc in padded_q for uc in ui_cues)
        if src_key.startswith(hdsd_prefix) and any(ec in padded_q for ec in eligibility_cues):
            if not has_ui_cue and not has_digital_cue:
                bonus -= 0.05

        return bonus

    @staticmethod
    def _has_digital_or_app_cue(query_text):
        from agent_banking.core.search_text import search_key

        if not query_text:
            return False
        bonus_cfg = RerankingConfig.RERANK_METADATA_BONUS
        app_cues = tuple(bonus_cfg.get("app_cues", ()))
        digital_cues = tuple(bonus_cfg.get("digital_product_cues", ()))
        padded_q = " " + search_key(query_text) + " "
        if any(cue in padded_q for cue in app_cues):
            return True
        if any(dc in padded_q for dc in digital_cues):
            return True
        return False

    @staticmethod
    def _is_hdsd_item(item):
        if not isinstance(item, dict):
            return False
        raw = item.get("chunk", item)
        if not isinstance(raw, dict):
            return False
        name = str(raw.get("source") or "").replace("\\", "/").rsplit("/", 1)[-1].casefold()
        return name.startswith("hdsd ")

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
            if chunk.get("chunk_type") != "table" and not chunk.get("table_data"):
                cleaned_ui = strip_ui_noise_for_context(text)
                if cleaned_ui:
                    text = cleaned_ui
            bc = chunk.get("breadcrumb", "")
            if not isinstance(bc, str):
                bc = str(bc)
            title = chunk.get("section_title", "")
            if not isinstance(title, str):
                title = str(title)
            meta = chunk.get("business_metadata")
            if not isinstance(meta, dict):
                meta = {}
            doc_name = str(chunk.get("document_name") or meta.get("document_name") or "").strip()
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
            td_raw = chunk.get("table_data")
            if isinstance(td_raw, dict):
                tbl_title = str(td_raw.get("title") or "").strip()
                if tbl_title and tbl_title.casefold() not in bc.casefold() and tbl_title.casefold() not in title.casefold():
                    prefix_parts.append("Bảng: " + tbl_title)
            inherited_path = str(chunk.get("_inherited_table_path") or meta.get("table_group_path") or "").strip()
            if inherited_path and inherited_path.casefold() not in bc.casefold() and inherited_path.casefold() not in text.casefold():
                prefix_parts.append("Nhóm mục: " + inherited_path)
            ret_meta = meta.get("retrieval")
            if isinstance(ret_meta, dict):
                sit_ctx = str(ret_meta.get("situated_context") or "").strip()
                if sit_ctx and sit_ctx.casefold() not in text.casefold():
                    prefix_parts.append(sit_ctx[:240])
            cit = meta.get("citation")
            pages = []
            if isinstance(cit, dict):
                sh_name = str(cit.get("sheet_name") or "").strip()
                if sh_name and sh_name.casefold() not in bc.casefold() and sh_name.casefold() not in title.casefold():
                    prefix_parts.append("Sheet " + sh_name)
                raw_c_pages = cit.get("page_numbers")
                if isinstance(raw_c_pages, list):
                    for p_num in raw_c_pages:
                        if type(p_num) is int and p_num > 0 and p_num not in pages:
                            pages.append(p_num)
            if not pages:
                prov = meta.get("provenance")
                if isinstance(prov, dict):
                    locs = prov.get("locations")
                    if isinstance(locs, list):
                        for loc in locs:
                            if isinstance(loc, dict):
                                p_num = loc.get("page_number")
                                if type(p_num) is not int and type(loc.get("page_index")) is int and loc["page_index"] >= 0:
                                    p_num = loc["page_index"] + 1
                                if type(p_num) is int and p_num > 0 and p_num not in pages:
                                    pages.append(p_num)
            if pages:
                pages.sort()
                prefix_parts.append("Trang " + ", ".join(str(p) for p in pages))
            parent_text = str(chunk.get("parent_content") or "").strip()
            if len(text) < 320 and len(parent_text) > len(text) + 40:
                text = text + "\n" + parent_text[:700]
            raw_images = chunk.get("images")
            if isinstance(raw_images, list) and raw_images:
                img_parts = []
                img_seen = set()
                for img in raw_images[:4]:
                    if isinstance(img, dict):
                        for fk in ("caption", "description", "native_text", "ocr_text", "alt_text", "title"):
                            val = str(img.get(fk) or "").strip()
                            if val and val not in img_seen and val.casefold() not in text.casefold():
                                img_seen.add(val)
                                img_parts.append(val)
                        evs = img.get("context_evidence")
                        if isinstance(evs, list):
                            for ev in evs[:3]:
                                if isinstance(ev, dict):
                                    q_val = str(ev.get("quote") or "").strip()
                                    if q_val and q_val not in img_seen and q_val.casefold() not in text.casefold():
                                        img_seen.add(q_val)
                                        img_parts.append(q_val)
                    elif isinstance(img, str):
                        val = img.strip()
                        if val and val not in img_seen and val.casefold() not in text.casefold():
                            img_seen.add(val)
                            img_parts.append(val)
                if img_parts:
                    joined_img = "; ".join(img_parts)[:900].strip()
                    if joined_img:
                        text = text + "\n[Hình ảnh/Sơ đồ]: " + joined_img
            prefix = ""
            if prefix_parts:
                prefix = "[" + " | ".join(prefix_parts) + "]\n"
            return (prefix + text).strip()
        return str(item)

    @staticmethod
    def _build_dedup_key(raw_chunk, src_id, snippet_chars):
        """Build a deduplication key that preserves distinct continuations of multi-chunk tables."""
        if not isinstance(raw_chunk, dict) or not src_id:
            return None
        td = raw_chunk.get("table_data")
        if isinstance(td, dict):
            tbl_id = str(td.get("table_id") or "").strip()
            r_start = td.get("row_start")
            r_end = td.get("row_end")
            if tbl_id and r_start is not None:
                return (src_id, "table_slice", tbl_id, r_start, r_end)
        raw_content = str(raw_chunk.get("content") or "").strip()
        if not raw_content:
            return None
        is_md_table = raw_chunk.get("chunk_type") == "table" or raw_content.startswith("|")
        if is_md_table:
            data_lines = []
            lines = raw_content.splitlines()
            skipped_header = False
            for idx_line, line in enumerate(lines):
                stripped = line.strip()
                if not stripped:
                    continue
                if stripped.startswith("|") and idx_line <= 3:
                    inner = stripped.replace("|", "").replace("-", "").replace(":", "").strip()
                    if not inner:
                        skipped_header = True
                        continue
                    if idx_line == 0 and len(lines) > 1 and "---" in lines[1]:
                        continue
                if skipped_header or not stripped.startswith("|") or idx_line > 1:
                    data_lines.append(stripped)
            if data_lines:
                body_norm = " ".join(" ".join(data_lines).split()).casefold()
                return (src_id, "table_body", body_norm[:snippet_chars], body_norm[-snippet_chars:])
        snippet = " ".join(raw_content.split())[:snippet_chars].casefold()
        if len(raw_content) > snippet_chars * 2:
            tail = " ".join(raw_content.split())[-80:].casefold()
            return (src_id, snippet, tail)
        return (src_id, snippet)

    def _should_use_llm_selector(self) -> bool:
        if self.use_llm_selector is False:
            return False
        if self.llm_client is not None:
            return True
        url_low = str(self.rerank_url or "").casefold()
        if not url_low or ".invalid" in url_low or "://mock" in url_low or "://localhost" in url_low:
            if self.use_llm_selector is not True:
                return False
        return rag_llm_available("chunking.text")

    @staticmethod
    def _extract_selected_indices(parsed: dict, pool_len: int, max_items: int) -> list:
        """Parse integer/string indices or r1..rN keys from LLM listwise response (0% Regex)."""
        if not isinstance(parsed, dict):
            return []
        raw_vals = []
        sel = parsed.get("selected")
        if isinstance(sel, list):
            for entry in sel:
                if isinstance(entry, str) and "," in entry:
                    for sub in entry.split(","):
                        raw_vals.append(sub)
                else:
                    raw_vals.append(entry)
        elif isinstance(sel, str):
            for sub in sel.split(","):
                raw_vals.append(sub)
        elif sel is not None:
            raw_vals.append(sel)
        if not raw_vals:
            for k_idx in range(1, max_items + 2):
                rk = f"r{k_idx}"
                if rk in parsed and parsed[rk] is not None:
                    raw_vals.append(parsed[rk])
        indices = []
        seen = set()
        for val in raw_vals:
            if type(val) is int:
                iv = val
            elif isinstance(val, str):
                s = val.strip(" []()")
                if not s.isdigit():
                    continue
                iv = int(s)
            else:
                continue
            if 0 <= iv < pool_len and iv not in seen:
                seen.add(iv)
                indices.append(iv)
                if len(indices) >= max_items:
                    break
        return indices

    @staticmethod
    def _build_diverse_llm_pool(ordered: list, pool_limit: int = 24, max_per_source: int = 2) -> list:
        """Build a 2-pass source-diverse pool so distinct docs lower in CE rank remain visible to LLM."""
        selected = []
        picked_ids = set()
        src_counts = {}
        scan_slice = ordered[:45]
        # Pass 1: maximize unique documents first (1 chunk per distinct source across CE Top-45)
        for idx, item in enumerate(scan_slice):
            raw_c = item.get("chunk", item)
            src_id = ""
            if isinstance(raw_c, dict):
                src_id = str(raw_c.get("source") or "").replace("\\", "/").casefold()
            if not src_id or src_id not in src_counts:
                selected.append(item)
                picked_ids.add(idx)
                if src_id:
                    src_counts[src_id] = 1
                if len(selected) >= pool_limit:
                    return selected
        # Pass 2: fill up to max_per_source per document
        for idx, item in enumerate(scan_slice):
            if idx in picked_ids:
                continue
            raw_c = item.get("chunk", item)
            src_id = ""
            if isinstance(raw_c, dict):
                src_id = str(raw_c.get("source") or "").replace("\\", "/").casefold()
            cnt = src_counts.get(src_id, 0)
            if not src_id or cnt < max_per_source:
                selected.append(item)
                picked_ids.add(idx)
                if src_id:
                    src_counts[src_id] = cnt + 1
                if len(selected) >= pool_limit:
                    return selected
        # Pass 3: top up from remaining candidates if pool still under limit
        for idx, item in enumerate(scan_slice):
            if idx not in picked_ids:
                selected.append(item)
                if len(selected) >= pool_limit:
                    break
        return selected

    def _llm_listwise_select(self, query_text, ordered, top_k, fallback_chosen):
        """Model-First LLM Listwise selector over source-diverse Cross-Encoder pool (WAL cached)."""
        if not ordered or top_k <= 1 or len(ordered) < 2:
            return fallback_chosen
        if not self._should_use_llm_selector():
            return fallback_chosen

        pool = self._build_diverse_llm_pool(ordered, pool_limit=max(top_k * 4, 24), max_per_source=2)
        candidate_lines = []
        pool_ids = []
        for idx, item in enumerate(pool):
            chunk = item.get("chunk", item)
            if not isinstance(chunk, dict):
                chunk = {"content": str(item)}
            cid = str(chunk.get("chunk_id") or f"p_{idx}")
            pool_ids.append(cid)
            src = str(chunk.get("source") or "").replace("\\", "/").rsplit("/", 1)[-1]
            meta = chunk.get("business_metadata")
            if not isinstance(meta, dict):
                meta = {}
            dom = str(meta.get("domain") or "")
            sec = str(chunk.get("section_title") or "")
            ret = meta.get("retrieval")
            sit = ""
            if isinstance(ret, dict):
                sit = strip_ui_noise_for_context(str(ret.get("situated_context") or "")).strip()
            body = strip_ui_noise_for_context(str(chunk.get("content") or ""))
            if not body:
                body = str(chunk.get("content") or "")
            score_val = round(float(item.get("relevance_score", 0.0)), 3)
            line = (
                f"[{idx}] source={src} | domain={dom} | section={sec[:60]} | ce={score_val}\n"
                f"  context: {sit[:220]}\n"
                f"  text: {body[:460]}"
            )
            candidate_lines.append(line)

        seed_key = f"{top_k}:{query_text}|" + ",".join(pool_ids)
        system_prompt = (
            "Ban la bo tai xep hang danh sach (Listwise Reranker) cho he thong RAG ngan hang A-Bank. "
            "Tu danh sach ung vien [0..N-1], hay chon ra toi da " + str(top_k) + " doan phu hop nhat de tra loi cau hoi. "
            "Nguyen tac: "
            "(1) Uu tien tai lieu chuyen biet khop dung tinh nang/san pham; "
            "(2) Khi cau hoi hoi ve thao tac, tinh nang, loi hoac xu ly su co tren ung dung A-Bank Digital / A-Bank Plus, "
            "neu trong danh sach co tai lieu huong dan app (domain=ngan_hang_so, file HDSD) phu hop voi nghiep vu/tinh nang do thi BAT BUOC chon it nhat 1-2 doan HDSD dung chu de vao top ket qua (ngay ca khi diem ce thap hon tai lieu .docx chung), ket hop voi tai lieu quy dinh/xu ly su co (NV, THE, VH, PL); "
            "(3) Neu hoi bieu phi hoac mang luoi -> uu tien file BP/ML co con so/ma phi cu the; "
            "(4) Khong chon lap nhieu doan chung chung neu da co tai lieu chuyen biet dung trong danh sach. "
            "Tra ve DUY NHAT JSON dang {\"selected\": [i1, i2, ...]} (chi so nguyen 0.." + str(len(pool) - 1) + "), khong giai thich."
        )
        user_prompt = f"Cau hoi: {query_text}\n\nCac ung vien:\n" + "\n".join(candidate_lines)
        parsed = call_rag_llm_json(
            namespace="rag_listwise_reranker_v3",
            seed_key=seed_key,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=800,
            temperature=0.0,
            chat_client=self.llm_client,
            validator=lambda d: bool(BankingReranker._extract_selected_indices(d, len(pool), top_k)),
        )
        chosen_indices = self._extract_selected_indices(parsed, len(pool), top_k)
        if not chosen_indices:
            return fallback_chosen

        result = []
        picked_ids = set()
        for rank_idx, idx_val in enumerate(chosen_indices, start=1):
            item_copy = dict(pool[idx_val])
            item_copy["llm_listwise_rank"] = rank_idx
            result.append(item_copy)
            raw_c = item_copy.get("chunk", item_copy)
            picked_ids.add(str(raw_c.get("chunk_id") or id(raw_c)))
        for fb_item in fallback_chosen:
            if len(result) >= top_k:
                break
            raw_c = fb_item.get("chunk", fb_item)
            cid = str(raw_c.get("chunk_id") or id(raw_c))
            if cid not in picked_ids:
                picked_ids.add(cid)
                result.append(fb_item)
        return result[:top_k]

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
            doc_hashes = [
                self._rerank_cache_key(candidate.url, candidate.model, query_text, d_txt)
                for d_txt in docs
            ]
            cached_scores = self._get_cached_scores_batch(doc_hashes)
            api_results = []
            miss_indices = []
            miss_docs = []
            for doc_idx, (h_key, d_txt) in enumerate(zip(doc_hashes, docs)):
                if h_key in cached_scores:
                    api_results.append({
                        "index": doc_idx,
                        "relevance_score": cached_scores[h_key],
                    })
                else:
                    miss_indices.append(doc_idx)
                    miss_docs.append(d_txt)
            self.last_cache_hits = len(api_results)
            self.last_cache_misses = len(miss_docs)

            max_api_batch = 48
            to_cache = []
            for batch_start in range(0, len(miss_docs), max_api_batch):
                batch_docs = miss_docs[batch_start:batch_start + max_api_batch]
                batch_orig_indices = miss_indices[batch_start:batch_start + max_api_batch]
                payload = {
                    "model": candidate.model,
                    "query": query_text,
                    "documents": batch_docs,
                    "top_n": len(batch_docs)
                }
                data = json.dumps(payload).encode("utf-8")
                req = urllib.request.Request(candidate.url, data=data, headers=headers)
                with urllib.request.urlopen(req, timeout=candidate.timeout) as resp:
                    resp_status = getattr(resp, "status", 200)
                    if resp_status != 200 and isinstance(resp_status, int):
                        raise ValueError("Reranker HTTP status error")
                    resp_data = json.loads(resp.read().decode("utf-8"))
                    sub_results = resp_data.get("results", [])
                    if not isinstance(sub_results, list) or not sub_results:
                        raise ValueError("Empty reranker response")
                    seen_sub = set()
                    for result in sub_results:
                        index = result.get("index")
                        score = result.get("relevance_score")
                        if not isinstance(index, int) or index in seen_sub or not 0 <= index < len(batch_docs):
                            raise ValueError("Invalid reranker index")
                        if not isinstance(score, (int, float)) or not math.isfinite(score):
                            raise ValueError("Invalid reranker score")
                        seen_sub.add(index)
                        orig_idx = batch_orig_indices[index]
                        api_results.append({
                            "index": orig_idx,
                            "relevance_score": float(score)
                        })
                        to_cache.append((doc_hashes[orig_idx], float(score)))
            if to_cache:
                self._save_cached_scores_batch(to_cache)

            if api_results:
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
                                snippet_key = self._build_dedup_key(raw_chunk, src_id, snippet_chars)
                            if snippet_key is not None and snippet_key in seen_snippets:
                                dup_deferred.append(item)
                                continue
                            item_score = float(item.get("relevance_score", 0.0))
                            count = source_counts.get(src_id, 0)
                            if top_k >= div_min_top_k and src_id and count >= div_max_per_src:
                                if item_score > best_deferred_score:
                                    best_deferred_score = item_score
                                deferred.append(item)
                            elif top_k >= div_min_top_k and best_deferred_score >= 0.45 and (
                                item_score < 0.22 or (best_deferred_score >= 0.55 and best_deferred_score - item_score >= 0.35)
                            ):
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

                    chosen = self._llm_listwise_select(query_text, ordered, top_k, fallback_chosen=chosen)
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
