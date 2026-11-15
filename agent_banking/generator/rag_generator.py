"""
RAG Generator & LLM Serving Module with Banking Guardrails.
======================================================================
Core Responsibilities:
1. Receives synthesized Markdown context with legal citations and user queries.
2. Applies strict Banking System Guardrails preventing financial hallucination.
3. Serves inference via configured LLM endpoints.
4. Returns standardized banking answers with transparent citations and confidentiality.

DESIGN PRINCIPLES:
1. 0% REGEX LAW: No regex imports or functions; string methods only.
2. ZERO TERNARY: Explicit if/else statements only.
3. NO TYPING LIBRARY: Native Python types only.
4. NO __INIT__.PY: Direct module imports.
"""

import os
import json
import time
import sqlite3
import hashlib
from pathlib import Path
import unicodedata
from agent_banking.core.chat import FallbackChatClient
from agent_banking.core.models import model_primary
from agent_banking.generator.config.settings import ContextConfig, GeneratorConfig

class ContextBundle:
    def __init__(self, markdown_context="", total_tokens=0, legal_citations=None, sources_used=None, chunks_included=0):
        self.markdown_context = markdown_context
        self.total_tokens = total_tokens
        if legal_citations is not None:
            self.legal_citations = legal_citations
        else:
            self.legal_citations = []
        if sources_used is not None:
            self.sources_used = sources_used
        else:
            self.sources_used = []
        self.chunks_included = chunks_included

    @property
    def citations(self):
        return self.legal_citations

    @property
    def sources(self):
        return self.sources_used


# Backward compatibility alias
SynthesizedContext = ContextBundle

# Banking System Guardrails
BANKING_SYSTEM_GUARDRAILS = """Bạn là Trợ lý AI Banking chuyên viên tư vấn của Ngân hàng.
Nhiệm vụ: Giải đáp câu hỏi của Quý khách hàng dựa trên tài liệu ngân hàng được cung cấp.

QUY TẮC BẮT BUỘC (100% TUÂN THỦ):
1. BẮT ĐẦU CÂU TRẢ LỜI NGAY LẬP TỨC CHO KHÁCH HÀNG. TUYỆT ĐỐI KHÔNG SUY NGHĨ THÀNH LỜI, KHÔNG PHÂN TÍCH NHÁP, KHÔNG ĐỘC THOẠI (như "chúng ta cần...", "câu hỏi của khách hàng...", "có ba trích dẫn...", "có lẽ nên trả lời...").
2. Chỉ trả lời dựa trên thông tin có trong tài liệu. Không bịa đặt số liệu tài chính.
3. Nếu có điều khoản quy định trong tài liệu, trích dẫn kèm [Điều X, Khoản Y].
4. Tuyệt đối không yêu cầu mật khẩu, mã PIN, mã OTP của khách hàng.
5. Ngôn phong lịch thiệp, trang trọng (xưng "Ngân hàng" hoặc "Trợ lý AI", gọi "Quý khách"). Trả lời súc tích, rõ ràng."""

from agent_banking.generator.config.prompt_config import (
    ProfileBundle,
    PERSONA_PROFILES,
    TONE_PROFILES,
    FORMAT_PROFILES,
    SYSTEM_GUARDRAILS,
    PROMPT_LEAK_SIGNATURES,
    load_prompt_profiles,
    build_banking_system_prompt
)

_load_prompt_profiles_from_json = load_prompt_profiles


def _estimate_tokens(text):
    if not text:
        return 0
    return max(int(len(text.split()) * 1.25), int(len(text) / 3.5))


def _unwrap_chunk_item(item):
    if isinstance(item, dict) and "chunk" in item and isinstance(item["chunk"], dict):
        return item["chunk"], item
    return item, item


def _format_table_data_markdown(table_data):
    if not isinstance(table_data, dict) or not table_data:
        return ""
    lines = []
    title = str(table_data.get("title") or "").strip()
    if title:
        lines.append(f"**Bảng:** {title}")

    headers = table_data.get("headers")
    rows = table_data.get("rows")
    if isinstance(headers, list) and headers:
        clean_headers = []
        for h in headers:
            clean_headers.append(str(h).replace("\n", " ").strip())
        lines.append("| " + " | ".join(clean_headers) + " |")
        sep = []
        for _ in clean_headers:
            sep.append("---")
        lines.append("| " + " | ".join(sep) + " |")

    if isinstance(rows, list) and rows:
        for row in rows:
            if isinstance(row, list):
                cells = []
                for cell in row:
                    cells.append(str(cell).replace("\n", " ").strip())
                lines.append("| " + " | ".join(cells) + " |")
            elif isinstance(row, dict):
                vals = []
                for k, v in row.items():
                    vals.append(f"{k}: {v}")
                lines.append("- " + "; ".join(vals))
    elif not lines:
        col_name = str(table_data.get("column") or "").strip()
        row_key = str(table_data.get("row_key") or "").strip()
        cell_span = table_data.get("cell_span")
        detail_parts = []
        if row_key:
            detail_parts.append(f"Hàng: {row_key}")
        if col_name:
            detail_parts.append(f"Cột: {col_name}")
        if isinstance(cell_span, list) and len(cell_span) == 2:
            detail_parts.append(f"Phần: {cell_span[0]}-{cell_span[1]}")
        if detail_parts:
            lines.append("[Bảng dữ liệu] " + " | ".join(detail_parts))

    return "\n".join(lines).strip()


def _format_images_context(images, existing_text=""):
    if not isinstance(images, list) or not images:
        return ""
    lines = []
    norm_existing = existing_text.lower()
    for idx, img in enumerate(images, start=1):
        if not isinstance(img, dict):
            continue
        parts = []
        for field_name, label in [
            ("title", "Tiêu đề"),
            ("caption", "Chú thích"),
            ("description", "Mô tả"),
            ("alt_text", "Alt"),
            ("native_text", "Nội dung văn bản"),
        ]:
            val = str(img.get(field_name) or "").strip()
            if val and val.lower() not in norm_existing:
                parts.append(f"{label}: {val}")
        native_vals = img.get("native_values")
        if native_vals:
            val_text = ""
            if isinstance(native_vals, list):
                cleaned_vals = []
                for nv in native_vals:
                    s = str(nv).strip()
                    if s:
                        cleaned_vals.append(s)
                val_text = ", ".join(cleaned_vals)
            elif isinstance(native_vals, dict):
                kv_list = []
                for k, v in native_vals.items():
                    kv_list.append(f"{k}: {v}")
                val_text = "; ".join(kv_list)
            else:
                val_text = str(native_vals).strip()
            if val_text and val_text.lower() not in norm_existing:
                parts.append(f"Giá trị: {val_text}")
        if parts:
            lines.append(f"- [Hình ảnh {idx}] " + " | ".join(parts))
    return "\n".join(lines).strip()


def _trim_text_to_token_budget(text, max_tokens):
    if not text or max_tokens <= 0:
        return text
    if _estimate_tokens(text) <= max_tokens:
        return text
    max_words = max(20, int(max_tokens / 1.25))
    words = text.split()
    if len(words) <= max_words:
        return text[:max(100, int(max_tokens * 3))] + "\n... [đã rút gọn theo giới hạn ngữ cảnh]"
    return " ".join(words[:max_words]) + "\n... [đã rút gọn theo giới hạn ngữ cảnh]"


def _extract_legal_citations_from_text(text, citations_list=None):
    """
    Extract legal citation tags [Article X, Clause Y] from text.
    Excludes non-legal structural tags such as [Table data], [Diagram/Image], [Context].
    """
    if citations_list is None:
        target_list = []
    else:
        target_list = citations_list

    if not text:
        return target_list

    if not isinstance(text, str):
        text = str(text)

    text = unicodedata.normalize("NFC", text)
    if "[" not in text:
        return target_list

    start = 0
    non_legal_markers = [
        "bảng", "bang", "hình", "hinh", "ảnh", "anh",
        "trích dẫn", "trich dan", "ngữ cảnh", "ngu canh",
        "phụ lục", "phu luc", "link", "xem thêm", "xem them", "trang"
    ]
    legal_markers = [
        "điều", "khoản", "điểm", "thông tư", "nghị định", "luật",
        "quyết định", "quy chế", "hiệp định", "tt", "nđ", "nd", "qđ", "qd",
        "dieu", "khoan", "diem", "thong tu", "nghi dinh", "luat", "quyet dinh"
    ]

    while True:
        o = text.find("[", start)
        if o == -1:
            break
        c = text.find("]", o)
        if c == -1:
            break

        tag = text[o + 1 : c]
        if "[" in tag:
            last_o = text.rfind("[", o, c)
            o = last_o
            tag = text[o + 1 : c]

        tag_clean = tag.strip()
        if len(tag_clean) < 1:
            start = c + 1
            continue
        if len(tag_clean) > 60:
            start = o + 1
            continue
        if "\n" in tag:
            start = o + 1
            continue

        start = c + 1

        tag_lower = tag_clean.lower()
        if tag_clean.isdigit():
            continue

        is_non_legal = False
        for nlm in non_legal_markers:
            if tag_lower.startswith(nlm):
                is_non_legal = True
                break
        if is_non_legal:
            continue

        has_legal = False
        for lm in legal_markers:
            if lm in tag_lower:
                has_legal = True
                break

        if has_legal:
            full_tag = f"[{tag_clean}]"
            if full_tag not in target_list:
                target_list.append(full_tag)

    return target_list


def _extract_citations_from_header(header_text, citations_list=None):
    """
    Extract article citations from section_title or breadcrumb when not bracketed.
    """
    if citations_list is None:
        target_list = []
    else:
        target_list = citations_list

    if not header_text:
        return target_list

    if not isinstance(header_text, str):
        header_text = str(header_text)

    header_text = unicodedata.normalize("NFC", header_text)
    _extract_legal_citations_from_text(header_text, target_list)

    h_lower = header_text.lower()
    target = "điều "
    pos = h_lower.find(target)
    if pos == -1:
        target = "dieu "
        pos = h_lower.find(target)

    if pos != -1:
        sub = header_text[pos:]
        end_idx = len(sub)
        for sep in [".", ";", ":", "-", "\n", "|", ">"]:
            s_idx = sub.find(sep)
            if s_idx != -1:
                if s_idx < end_idx:
                    end_idx = s_idx
        extracted = sub[:end_idx].strip()
        if len(extracted) >= 2:
            if len(extracted) <= 60:
                full_tag = f"[{extracted}]"
                if full_tag not in target_list:
                    target_list.append(full_tag)

    return target_list


class RAGGenerator:
    def __init__(
        self,
        api_key=None,
        base_url=None,
        model=None,
        temperature=None,
        max_tokens=None,
        enable_cache=True,
        cache_db_path=None,
        max_retries=None,
        retry_delays=None,
        config=None
    ):
        try:
            primary = model_primary("generator")
            cfg_model = primary.model
            cfg_key = primary.api_key
            cfg_url = primary.url
            cfg_attempts = primary.attempts
        except Exception:
            cfg_model = os.getenv("MODEL", "")
            cfg_key = os.getenv("API_KEY", "")
            cfg_url = os.getenv("BASE_URL", "")
            cfg_attempts = 1

        if api_key is not None:
            self.api_key = api_key
        elif config is not None and getattr(config, "LLM_API_KEY", None) is not None:
            self.api_key = config.LLM_API_KEY
        else:
            self.api_key = cfg_key

        if base_url is not None and base_url != "":
            self.base_url = base_url
        elif config is not None and getattr(config, "LLM_BASE_URL", None):
            self.base_url = config.LLM_BASE_URL
        else:
            self.base_url = cfg_url

        if model is not None and model != "":
            self.model = model
        elif config is not None and getattr(config, "LLM_MODEL", None):
            self.model = config.LLM_MODEL
        else:
            self.model = cfg_model

        self.temperature = GeneratorConfig.LLM_TEMPERATURE
        if temperature is not None:
            self.temperature = temperature
        elif config is not None and getattr(config, "LLM_TEMPERATURE", None) is not None:
            self.temperature = config.LLM_TEMPERATURE

        self.max_tokens = GeneratorConfig.LLM_MAX_TOKENS
        if max_tokens is not None:
            self.max_tokens = max_tokens
        elif config is not None and getattr(config, "LLM_MAX_TOKENS", None) is not None:
            self.max_tokens = config.LLM_MAX_TOKENS

        self.max_context_tokens = int(getattr(config, "MAX_CONTEXT_TOKENS", ContextConfig.MAX_CONTEXT_TOKENS))
        self.default_persona = str(getattr(config, "DEFAULT_PERSONA", GeneratorConfig.DEFAULT_PERSONA))
        self.default_tone = str(getattr(config, "DEFAULT_TONE", GeneratorConfig.DEFAULT_TONE))
        self.default_format = str(getattr(config, "DEFAULT_FORMAT", GeneratorConfig.DEFAULT_FORMAT))
        self.enable_cache = enable_cache

        if retry_delays is not None:
            self.retry_delays = retry_delays
        else:
            self.retry_delays = [1.0, 2.0, 4.0]

        cfg_retries = getattr(GeneratorConfig, "GENERATOR_MAX_RETRIES", getattr(GeneratorConfig, "MAX_RETRIES", 3))
        if config is not None:
            cfg_retries = getattr(config, "GENERATOR_MAX_RETRIES", getattr(config, "MAX_RETRIES", cfg_retries))

        if max_retries is not None:
            self.max_retries = max_retries
        elif retry_delays is not None:
            self.max_retries = max(len(retry_delays), cfg_attempts)
        else:
            self.max_retries = max(cfg_attempts, cfg_retries, 3)

        if cache_db_path is not None:
            self.cache_db_path = Path(cache_db_path)
        else:
            self.cache_db_path = Path(__file__).resolve().parent / "generator_cache.db"

        if self.enable_cache:
            self._init_cache_db()

        self.client = None
        # An explicitly empty key preserves the intentional offline mode.
        if api_key != "":
            primary_spec = {
                "model": self.model, "url": self.base_url, "api_key": self.api_key,
            }
            if max_retries is not None:
                primary_spec["attempts"] = self.max_retries
            self.client = FallbackChatClient("generator", primary_spec, self.retry_delays)

    def _init_cache_db(self):
        conn = None
        try:
            self.cache_db_path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(str(self.cache_db_path), timeout=10.0)
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS generator_response_cache (
                    cache_key TEXT PRIMARY KEY,
                    query TEXT,
                    answer TEXT,
                    legal_citations TEXT,
                    sources_used TEXT,
                    model_served TEXT,
                    persona TEXT,
                    tone TEXT,
                    format_style TEXT,
                    prompt_tokens INTEGER,
                    completion_tokens INTEGER,
                    total_tokens INTEGER,
                    created_at REAL
                )
            """)
            conn.commit()
        except Exception as e:
            print(f"[RAG Generator] Lỗi khởi tạo SQLite cache: {e}")
        finally:
            if conn is not None:
                conn.close()

    def _get_cached_response(self, cache_key):
        if not self.enable_cache:
            return None
        conn = None
        try:
            conn = sqlite3.connect(str(self.cache_db_path), timeout=5.0)
            cursor = conn.cursor()
            cursor.execute("""
                SELECT query, answer, legal_citations, sources_used, model_served,
                       persona, tone, format_style, prompt_tokens, completion_tokens,
                       total_tokens
                FROM generator_response_cache
                WHERE cache_key = ?
            """, (cache_key,))
            row = cursor.fetchone()
            if row:
                q, ans, leg_c_json, src_json, mdl, per, ton, fmt, p_tok, c_tok, t_tok = row
                return {
                    "query": q,
                    "answer": ans,
                    "legal_citations": json.loads(leg_c_json),
                    "citations": json.loads(leg_c_json),
                    "sources_used": json.loads(src_json),
                    "sources": json.loads(src_json),
                    "model_served": mdl,
                    "guardrails_applied": True,
                    "cached": True,
                    "status": "success",
                    "persona": per,
                    "tone": ton,
                    "format_style": fmt,
                    "usage": {
                        "prompt_tokens": p_tok,
                        "completion_tokens": c_tok,
                        "total_tokens": t_tok
                    }
                }
        except Exception:
            pass
        finally:
            if conn is not None:
                conn.close()
        return None

    def _save_cached_response(self, cache_key, query, answer_text, legal_citations, sources_used,
                              model_used, used_persona, used_tone, used_format,
                              prompt_tokens, completion_tokens, total_tokens):
        if not self.enable_cache:
            return
        conn = None
        try:
            conn = sqlite3.connect(str(self.cache_db_path), timeout=5.0)
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO generator_response_cache (
                    cache_key, query, answer, legal_citations, sources_used,
                    model_served, persona, tone, format_style,
                    prompt_tokens, completion_tokens, total_tokens, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                cache_key,
                query,
                answer_text,
                json.dumps(legal_citations, ensure_ascii=False),
                json.dumps(sources_used, ensure_ascii=False),
                model_used,
                used_persona,
                used_tone,
                used_format,
                prompt_tokens,
                completion_tokens,
                total_tokens,
                time.time()
            ))
            conn.commit()
        except Exception as e:
            print(f"[RAG Generator] Lỗi ghi SQLite cache: {e}")
        finally:
            if conn is not None:
                conn.close()

    _extract_legal_citations_from_text = staticmethod(_extract_legal_citations_from_text)
    _extract_citations_from_header = staticmethod(_extract_citations_from_header)

    @staticmethod
    def format_chunks(chunks, max_tokens=None):
        """
        Assemble 13-field universal chunks (plus wrapped/legacy formats) into clear Markdown text
        with deduplicated parent windows, structured table_data, multimodal image context, and legal citations.
        """
        if not chunks:
            return ContextBundle()

        if max_tokens is None:
            token_budget = int(ContextConfig.MAX_CONTEXT_TOKENS)
        else:
            token_budget = int(max_tokens)

        blocks = []
        sources = []
        citations = []
        seen_parent_windows = set()
        fallback_content_keys = ["content", "document", "text", "page_content"]
        running_tokens = 0
        chunks_included = 0

        for idx, item in enumerate(chunks, start=1):
            chunk, raw_item = _unwrap_chunk_item(item)
            if not isinstance(chunk, dict):
                continue

            bmeta = chunk.get("business_metadata")
            if not isinstance(bmeta, dict):
                bmeta = {}

            legacy_meta = chunk.get("metadata")
            if not isinstance(legacy_meta, dict):
                legacy_meta = {}

            parent_id = str(chunk.get("parent_id") or "").strip()
            chunk_type = str(chunk.get("chunk_type") or "text").strip()

            source = str(chunk.get("source") or "").strip()
            if not source:
                source = str(bmeta.get("document_name") or "").strip()
            if not source:
                source = str(legacy_meta.get("source") or "").strip()
            if not source:
                source = "Tài liệu ngân hàng"

            breadcrumb = str(chunk.get("breadcrumb") or legacy_meta.get("breadcrumb") or "").strip()
            section_title = str(chunk.get("section_title") or legacy_meta.get("section_title") or "").strip()

            content_body = ""
            for k in fallback_content_keys:
                val = chunk.get(k)
                if val:
                    content_body = str(val).strip()
                    break

            parent_body = str(chunk.get("parent_content") or "").strip()
            body = ""
            if parent_body:
                p_key = (parent_id, parent_body)
                if p_key not in seen_parent_windows:
                    seen_parent_windows.add(p_key)
                    body = parent_body
                    if content_body and content_body not in parent_body:
                        body = parent_body + "\n\n" + content_body
                else:
                    if content_body:
                        body = content_body
                    else:
                        body = parent_body
            else:
                body = content_body

            table_data = chunk.get("table_data")
            table_md = _format_table_data_markdown(table_data)
            if table_md:
                # Only append structured Markdown table when not already verbatim in body.
                if table_md not in body and "| --- |" not in body:
                    if body:
                        body = body + "\n\n" + table_md
                    else:
                        body = table_md

            images = chunk.get("images")
            img_ctx = _format_images_context(images, existing_text=body)
            if img_ctx:
                if body:
                    body = body + "\n\n" + img_ctx
                else:
                    body = img_ctx

            retrieval_meta = bmeta.get("retrieval")
            if not isinstance(retrieval_meta, dict):
                retrieval_meta = {}
            situated_ctx = str(retrieval_meta.get("situated_context") or "").strip()

            meta_lines = [f"### [TRÍCH DẪN {idx}] Nguồn tài liệu: {source}"]
            if breadcrumb:
                meta_lines.append(f"- **Phân cấp:** `{breadcrumb}`")
            if section_title:
                meta_lines.append(f"- **Tiêu đề điều khoản:** `{section_title}`")

            domain_name = str(bmeta.get("domain") or "").strip()
            tag_parts = []
            if chunk_type and chunk_type != "text":
                tag_parts.append(f"Loại: {chunk_type}")
            if domain_name:
                tag_parts.append(f"Phân hệ: {domain_name}")
            if tag_parts:
                meta_lines.append("- **Phân loại:** `" + " | ".join(tag_parts) + "`")

            if situated_ctx and situated_ctx not in body:
                meta_lines.append(f"- **Ngữ cảnh:** {situated_ctx}")

            meta_lines.append(f"- **Nội dung xác thực:**\n{body}")
            block_text = "\n".join(meta_lines)

            sep_tokens = 0
            if blocks:
                sep_tokens = 2
            block_tokens = _estimate_tokens(block_text) + sep_tokens

            if token_budget > 0:
                if blocks and running_tokens + block_tokens > token_budget:
                    break
                if not blocks and block_tokens > token_budget:
                    block_text = _trim_text_to_token_budget(block_text, token_budget)
                    block_tokens = _estimate_tokens(block_text)

            blocks.append(block_text)
            running_tokens += block_tokens
            chunks_included += 1

            if source and source not in sources:
                sources.append(source)

            _extract_legal_citations_from_text(body, citations)
            if section_title:
                _extract_citations_from_header(section_title, citations)
            if breadcrumb:
                _extract_citations_from_header(breadcrumb, citations)

        context_text = "\n\n---\n\n".join(blocks)
        total_tokens = _estimate_tokens(context_text)

        return ContextBundle(
            markdown_context=context_text,
            total_tokens=total_tokens,
            legal_citations=citations,
            sources_used=sources,
            chunks_included=chunks_included
        )

    def generate_answer(self, query, synthesized_context, persona_config=None):
        """
        Execute LLM Serving: Inject context with Dynamic Persona and System Guardrails into model.
        Automatically retries with exponential backoff on transient network/API errors.
        """
        # Extract context content and citations list
        if isinstance(synthesized_context, ContextBundle):
            context_text = synthesized_context.markdown_context
            legal_citations = synthesized_context.legal_citations
            sources_used = synthesized_context.sources_used
            tokens_used = synthesized_context.total_tokens
        elif isinstance(synthesized_context, list):
            bundle = self.format_chunks(synthesized_context, max_tokens=self.max_context_tokens)
            context_text = bundle.markdown_context
            legal_citations = bundle.legal_citations
            sources_used = bundle.sources_used
            tokens_used = bundle.total_tokens
        elif isinstance(synthesized_context, str):
            context_text = synthesized_context
            legal_citations = []
            _extract_legal_citations_from_text(context_text, legal_citations)
            sources_used = []
            tokens_used = _estimate_tokens(context_text)
        else:
            context_text = str(synthesized_context)
            legal_citations = []
            _extract_legal_citations_from_text(context_text, legal_citations)
            sources_used = []
            tokens_used = _estimate_tokens(context_text)

        # Construct dynamic System Prompt based on Persona profile
        if isinstance(persona_config, dict) and persona_config:
            p_key = persona_config.get("persona")
            if not p_key:
                p_key = self.default_persona
            t_key = persona_config.get("tone")
            if not t_key:
                t_key = self.default_tone
            f_key = persona_config.get("format_style")
            if not f_key:
                f_key = persona_config.get("format_type")
            if not f_key:
                f_key = self.default_format
            custom_p = persona_config.get("custom_persona", "")
            if not custom_p:
                custom_p = persona_config.get("custom_instructions", "")
            system_instruction = build_banking_system_prompt(p_key, t_key, f_key, custom_p)
            used_persona = p_key
            used_tone = t_key
            used_format = f_key
        else:
            system_instruction = build_banking_system_prompt(self.default_persona, self.default_tone, self.default_format)
            used_persona = self.default_persona
            used_tone = self.default_tone
            used_format = self.default_format

        # Compute deterministic cache key
        cache_str = f"{query}:{context_text}:{used_persona}:{used_tone}:{used_format}"
        if isinstance(self.client, FallbackChatClient):
            cache_str += ":" + self.client.fingerprint + ":" + system_instruction
        cache_str += ":" + str(self.temperature) + ":" + str(self.max_tokens)
        cache_key = hashlib.sha256(cache_str.encode("utf-8")).hexdigest()

        # Check whether client is mock/fake to avoid bypassing calls in tests
        is_mock_client = False
        if self.client is not None:
            c_name = type(self.client).__name__.lower()
            if "fake" in c_name or "mock" in c_name:
                is_mock_client = True

        # Check SQLite Cache (when not using mock/fake client)
        if not is_mock_client:
            cached_result = self._get_cached_response(cache_key)
            if cached_result is not None:
                cached_result["context_tokens"] = tokens_used
                return cached_result

        # Construct User Prompt
        user_prompt = (
            f"TÀI LIỆU CĂN CỨ:\n\n"
            f"{context_text}\n\n"
            f"CÂU HỎI CỦA QUÝ KHÁCH HÀNG:\n"
            f"{query}\n\n"
            f"HÃY TRẢ LỜI TRỰC TIẾP CHO QUÝ KHÁCH (Bắt đầu ngay bằng câu trả lời, tuân thủ đúng vai trò và phong cách được chỉ định):"
        )

        answer_text = ""
        model_used = self.model
        prompt_tokens_est = tokens_used + (len(user_prompt) // 4)
        completion_tokens_est = 0
        status_result = "success"

        if self.client:
            call_success = False
            last_err = None
            max_retries = self.max_retries
            if isinstance(self.client, FallbackChatClient):
                max_retries = 1
            delays = self.retry_delays

            for attempt in range(max_retries):
                try:
                    request_options = {
                        "model": self.model,
                        "messages": [
                            {"role": "system", "content": system_instruction},
                            {"role": "user", "content": user_prompt}
                        ],
                        "temperature": self.temperature,
                        "max_tokens": self.max_tokens,
                    }
                    if isinstance(self.client, FallbackChatClient):
                        request_options["validate"] = self._validate_model_answer
                    response = self.client.chat.completions.create(**request_options)
                    msg = response.choices[0].message
                    ans = msg.content
                    if not ans:
                        ans = getattr(msg, "reasoning_content", None)
                    if not ans:
                        ans = ""
                    answer_text = ans
                    model_used = getattr(response, "model", self.model)

                    usage_obj = getattr(response, "usage", None)
                    if usage_obj is not None:
                        prompt_tokens_est = getattr(usage_obj, "prompt_tokens", prompt_tokens_est)
                        c_tok = getattr(usage_obj, "completion_tokens", 0)
                        if c_tok > 0:
                            completion_tokens_est = c_tok
                        else:
                            completion_tokens_est = max(1, len(answer_text) // 4)
                    else:
                        completion_tokens_est = max(1, len(answer_text) // 4)

                    call_success = True
                    break
                except Exception as e:
                    last_err = e
                    print(f"[RAG Generator] API attempt {attempt + 1}/{max_retries}: {type(e).__name__}")
                    if attempt < max_retries - 1:
                        if is_mock_client:
                            delay = 0.01
                        else:
                            if attempt < len(delays):
                                delay = delays[attempt]
                            else:
                                delay = 4.0
                        time.sleep(delay)

            if not call_success:
                if isinstance(self.client, FallbackChatClient):
                    raise last_err
                status_result = "fallback"
                fallback_lines = [
                    "**[Chế độ Dự phòng Ngữ cảnh Xác thực - Hệ thống đang bảo trì kết nối Cloud AI]**",
                    "",
                    "Dưới đây là căn cứ pháp lý và nội dung chính thức được tìm thấy trong tài liệu ngân hàng:"
                ]
                if legal_citations:
                    fallback_lines.append(f"- Căn cứ pháp lý: {', '.join(legal_citations)}")
                fallback_lines.append("")
                if len(context_text) > 1200:
                    fallback_lines.append(f"{context_text[:1200]}...")
                else:
                    fallback_lines.append(context_text)
                fallback_lines.append("")
                fallback_lines.append("Quý khách vui lòng liên hệ Hotline hỗ trợ để được nhân viên quầy giải đáp trực tiếp.")
                answer_text = "\n".join(fallback_lines)
                completion_tokens_est = len(answer_text) // 4
        else:
            status_result = "fallback"
            answer_text = (
                f"**[Chế độ Trả lời Xác thực Cục bộ - Sẵn sàng nạp API Key]**\n\n"
                f"Hệ thống đã truy xuất và tổng hợp thành công ngữ cảnh pháp lý:\n\n"
                f"{context_text}\n\n"
                f"Để nhận câu trả lời đầy đủ từ LLM, vui lòng cấu hình API_KEY."
            )
            completion_tokens_est = len(answer_text) // 4

        # Clean and strip any internal reasoning/thinking tokens
        answer_text = self._clean_bot_response(answer_text)

        total_tokens = prompt_tokens_est + completion_tokens_est

        # Persist cache entry on success when not a mock client
        if status_result == "success" and not is_mock_client:
            self._save_cached_response(
                cache_key=cache_key,
                query=query,
                answer_text=answer_text,
                legal_citations=legal_citations,
                sources_used=sources_used,
                model_used=model_used,
                used_persona=used_persona,
                used_tone=used_tone,
                used_format=used_format,
                prompt_tokens=prompt_tokens_est,
                completion_tokens=completion_tokens_est,
                total_tokens=total_tokens
            )

        return {
            "query": query,
            "answer": answer_text,
            "legal_citations": legal_citations,
            "citations": legal_citations,
            "sources_used": sources_used,
            "sources": sources_used,
            "context_tokens": tokens_used,
            "model_served": model_used,
            "guardrails_applied": True,
            "cached": False,
            "status": status_result,
            "persona": used_persona,
            "tone": used_tone,
            "format_style": used_format,
            "usage": {
                "prompt_tokens": prompt_tokens_est,
                "completion_tokens": completion_tokens_est,
                "total_tokens": total_tokens
            }
        }

    @staticmethod
    def _validate_model_answer(response):
        if not RAGGenerator._clean_bot_response(response.choices[0].message.content).strip():
            raise ValueError("Model returned no customer-facing answer")

    @staticmethod
    def _clean_bot_response(text):
        """
        Strip internal reasoning/scratchpad/self-critique tokens from model output.
        0% Regex Law: Pure str methods only (find, rfind, split, replace, startswith, in).
        Zero Ternary: Explicit if/else statements only.
        """
        if not text:
            return ""

        # Normalize to Unicode NFC for robust Vietnamese diacritic matching
        text = unicodedata.normalize("NFC", text).strip()

        # 1. Strip XML/HTML thinking tags: think, thought, thinking, scratchpad, reasoning
        thinking_tags = ["think", "thought", "thinking", "scratchpad", "reasoning"]
        official_greetings = [
            "Kính thưa Quý khách",
            "Kính chào Quý khách",
            "Chào Quý khách",
            "Thưa Quý khách",
            "Kính gửi Quý khách",
            "Kính gửi",
            "Theo quy định",
            "Căn cứ theo",
            "Về hạn mức",
            "Hạn mức",
            "Thông tin"
        ]

        for tag in thinking_tags:
            open_tag = f"<{tag}>"
            close_tag = f"</{tag}>"
            while open_tag in text:
                start_idx = text.find(open_tag)
                end_idx = text.find(close_tag, start_idx)
                if end_idx != -1:
                    text = text[:start_idx] + text[end_idx + len(close_tag):]
                else:
                    # Unclosed tag: locate official greeting anchor following opening tag
                    post_open = text[start_idx + len(open_tag):]
                    earliest_idx = -1
                    for g in official_greetings:
                        g_pos = post_open.find(g)
                        if g_pos != -1:
                            if earliest_idx == -1:
                                earliest_idx = g_pos
                            elif g_pos < earliest_idx:
                                earliest_idx = g_pos
                    if earliest_idx != -1:
                        text = text[:start_idx] + post_open[earliest_idx:]
                    else:
                        text = text[:start_idx]
                    break

        text = text.strip()

        # 2. Strip Markdown thought code blocks: ```thought ... ```, ```thinking ... ```
        md_blocks = ["```thought", "```thinking", "```scratchpad", "```reasoning"]
        for mb in md_blocks:
            while mb in text:
                start_idx = text.find(mb)
                end_idx = text.find("```", start_idx + len(mb))
                if end_idx != -1:
                    text = text[:start_idx] + text[end_idx + 3:]
                else:
                    text = text[:start_idx]
                    break
        text = text.strip()

        # 3. Capture conclusive response phrase following internal monologue
        final_anchors = [
            "Vậy câu trả lời cuối cùng:",
            "Câu trả lời cuối cùng:",
            "Phản hồi cuối cùng:",
            "Nội dung phản hồi chính thức:",
            "Soạn:",
            "Soạn thảo:",
            "Có lẽ nên trả lời:",
            "Có thể trả lời:",
            "Nên trả lời:",
            "Có thể nói:",
            "Câu trả lời của bot:",
            "Câu trả lời:",
            "Nội dung phản hồi:",
            "Phản hồi Quý khách:",
            "Phản hồi chính thức:",
            "Answer:",
            "Final Answer:"
        ]
        for fa in final_anchors:
            if fa in text:
                parts = text.split(fa)
                candidate = parts[-1].strip()
                if len(candidate) > 20:
                    text = candidate
                    break

        # Strip enclosing double quotes if present
        if text.startswith('"') or text.startswith("'"):
            text = text[1:].strip()

        # 4. Strip guardrail self-censorship monologue if present
        # Note: Removed risky anchor phrasing
        critique_anchors = [
            '" Kiểm tra:',
            'Kiểm tra:',
            'Kiểm tra lại:',
            "Cần trích dẫn điều khoản",
            "Như vậy có ổn không?",
            "Cần lưu ý guardrail",
            "Hmm, guardrail",
            "Guardrail nói",
            "Liệu như vậy có ổn",
            "Vậy ta sẽ thay",
            "Có sự không nhất quán",
            "Để an toàn, có thể nói",
            "Tốt nhất nên tập trung"
        ]
        for ca in critique_anchors:
            if ca in text:
                c_idx = text.find(ca)
                if c_idx > 40:
                    text = text[:c_idx].strip()
                    break

        if text.endswith('"') or text.endswith("'"):
            text = text[:-1].strip()

        # 5. Remove opening monologue prefixes in English and Vietnamese
        prefixes = (
            "Let's think step by step",
            "Thinking Process:",
            "Phân tích câu hỏi:",
            "Bước 1: Phân tích",
            "Chúng ta cần",
            "Ta cần",
            "Tôi cần",
            "Để trả lời",
            "Cần trả lời",
            "Câu hỏi:",
            "Phân tích:",
            "Dựa trên các trích dẫn",
            "Có ba trích dẫn",
            "The user is asking",
            "Analyzing documents",
            "Drafting response"
        )
        if text.startswith(prefixes):
            earliest_idx = -1
            for anchor in official_greetings:
                idx = text.find(anchor)
                if idx != -1:
                    if earliest_idx == -1:
                        earliest_idx = idx
                    elif idx < earliest_idx:
                        earliest_idx = idx
            if earliest_idx != -1:
                text = text[earliest_idx:].strip()

        # 6. Remove prompt leakage signatures
        for sig in PROMPT_LEAK_SIGNATURES:
            if sig in text:
                text = text.replace(sig + ":", "").replace(sig, "")

        return text.strip()
