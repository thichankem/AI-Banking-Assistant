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

import sys
import json
import time
import dis
import sqlite3
import hashlib
from pathlib import Path
import unicodedata
from agent_banking.config.chat import FallbackChatClient
from agent_banking.config.models import model_primary
from agent_banking.config.generator.settings import GeneratorConfig

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

class ProfileBundle(tuple):
    """
    Multi-purpose tuple supporting flexible unpacking of 3 elements (legacy) and 5 elements (v2.1.0).
    0% Regex & Zero Ternary.
    """
    def __iter__(self):
        try:
            frame = sys._getframe(1)
            code = frame.f_code
            lasti = frame.f_lasti
            instructions = list(dis.get_instructions(code))
            for i, inst in enumerate(instructions):
                if inst.offset >= lasti:
                    for target in instructions[i : i + 6]:
                        if target.opname == "UNPACK_SEQUENCE":
                            if target.argval == 3:
                                return iter(self[:3])
                            elif target.argval == 5:
                                return iter(self[:5])
                    break
        except Exception:
            pass
        return super().__iter__()


# Persona, tone, and response format profiles loaded from JSON (0% Regex & Zero Ternary)
def _load_prompt_profiles_from_json(filepath=None):
    if filepath is not None:
        p = Path(filepath)
    else:
        p = Path(__file__).resolve().parent / "prompt_profiles.json"

    if p.exists():
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            return ProfileBundle((
                data.get("persona_profiles", {}),
                data.get("tone_profiles", {}),
                data.get("format_profiles", {}),
                data.get("system_guardrails", {}),
                data.get("prompt_leak_signatures", [])
            ))
        except Exception:
            pass
    return ProfileBundle(({}, {}, {}, {}, []))


_profiles_data = _load_prompt_profiles_from_json()
PERSONA_PROFILES = _profiles_data[0]
TONE_PROFILES = _profiles_data[1]
FORMAT_PROFILES = _profiles_data[2]
SYSTEM_GUARDRAILS = _profiles_data[3]
PROMPT_LEAK_SIGNATURES = _profiles_data[4]


def build_banking_system_prompt(persona_key="advisor", tone_key="formal", format_key="standard", custom_persona=""):
    """
    Construct Dynamic System Prompt combining Core Banking Guardrails, Persona, Tone, and Format.
    100% compliant with 0% Regex and Zero Ternary.
    """
    if persona_key == "custom" and custom_persona and custom_persona.strip():
        persona_text = custom_persona.strip()
    elif persona_key in PERSONA_PROFILES:
        p_obj = PERSONA_PROFILES[persona_key]
        p_name = p_obj.get("name", "")
        p_prompt = p_obj.get("prompt", "")
        if not p_prompt:
            p_prompt = p_obj.get("role_definition", "")
        if p_name and p_name not in p_prompt:
            persona_text = f"Vai trò: {p_name}\n{p_prompt}"
        else:
            persona_text = p_prompt
    else:
        adv = PERSONA_PROFILES.get("advisor", {})
        persona_text = adv.get("prompt", "Bạn là Chuyên viên Tư vấn Tài chính Cao cấp của Ngân hàng. Phong thái điềm đạm, am tường nghiệp vụ ngân hàng, đưa ra các chỉ dẫn và giải pháp tài chính chuẩn mực, minh bạch, phân tích logic và có chiều sâu.")

    if tone_key in TONE_PROFILES:
        tone_text = TONE_PROFILES[tone_key]
    else:
        tone_text = TONE_PROFILES.get("formal", "Trang trọng & Lịch thiệp.")

    if format_key in FORMAT_PROFILES:
        format_text = FORMAT_PROFILES[format_key]
    else:
        format_text = FORMAT_PROFILES.get("standard", "Định dạng Tiêu chuẩn.")

    guardrails_text = ""
    if isinstance(SYSTEM_GUARDRAILS, dict):
        guardrails_text = SYSTEM_GUARDRAILS.get("prompt_guardrails_text", "")
        if not guardrails_text and "mandatory_rules" in SYSTEM_GUARDRAILS:
            rules_list = SYSTEM_GUARDRAILS.get("mandatory_rules", [])
            rule_lines = []
            for idx, r in enumerate(rules_list, start=1):
                rule_lines.append(f"{idx}. {r}")
            guardrails_text = "QUY TẮC BẢO MẬT & AN TOÀN NGÂN HÀNG BẮT BUỘC (100% TUÂN THỦ):\n" + "\n".join(rule_lines)
    elif isinstance(SYSTEM_GUARDRAILS, str):
        guardrails_text = SYSTEM_GUARDRAILS

    if not guardrails_text:
        guardrails_text = (
            "QUY TẮC BẢO MẬT & AN TOÀN NGÂN HÀNG BẮT BUỘC (100% TUÂN THỦ):\n"
            "1. BẮT ĐẦU CÂU TRẢ LỜI NGAY LẬP TỨC CHO KHÁCH HÀNG. TUYỆT ĐỐI KHÔNG SUY NGHĨ THÀNH LỜI, KHÔNG XUẤT THẺ <think>/<thought>/<scratchpad>, KHÔNG PHÂN TÍCH NHÁP, KHÔNG ĐỘC THOẠI.\n"
            "2. Chỉ trả lời dựa trên thông tin có trong tài liệu được cung cấp. TUYỆT ĐỐI KHÔNG TỰ Ý BỊA ĐẶT SỐ LIỆU TÀI CHÍNH (lãi suất %, hạn mức, biểu phí).\n"
            "3. Nếu có điều khoản quy định trong tài liệu, trích dẫn kèm [Điều X, Khoản Y]. Tuyệt đối không bịa đặt số hiệu điều khoản khi tài liệu không đề cập.\n"
            "4. Tuyệt đối không bao giờ yêu cầu hoặc phản hồi lặp lại mật khẩu, mã PIN, mã OTP, số CVV/CVC của khách hàng.\n"
            "5. Nếu thông tin hoặc số liệu không có trong tài liệu, hãy thông báo lịch sự rằng tài liệu hiện chưa có dữ liệu và hướng dẫn khách hàng liên hệ Hotline hoặc quầy giao dịch."
        )

    few_shot_text = ""
    if persona_key in PERSONA_PROFILES and isinstance(PERSONA_PROFILES[persona_key], dict):
        fs_data = PERSONA_PROFILES[persona_key].get("few_shot", [])
        few_shot_list = []
        if isinstance(fs_data, list):
            few_shot_list = fs_data
        elif isinstance(fs_data, dict):
            few_shot_list = [fs_data]

        if few_shot_list:
            fs_blocks = ["VÍ DỤ MẪU ĐỐI THOẠI CHUẨN MỰC (FEW-SHOT EXAMPLES):"]
            for ex in few_shot_list:
                if isinstance(ex, dict):
                    u = ex.get("user", "")
                    a = ex.get("assistant", "")
                    if u and a:
                        fs_blocks.append(f"Khách hàng: {u}\nChuyên viên: {a}")
            if len(fs_blocks) > 1:
                few_shot_text = "\n\n".join(fs_blocks)

    sections = [
        persona_text,
        f"QUY TẮC TÔNG GIỌNG & ĐỊNH DẠNG CÂU TRẢ LỜI:\n- Tông giọng: {tone_text}\n- Định dạng trình bày: {format_text}",
        guardrails_text
    ]
    if few_shot_text:
        sections.append(few_shot_text)

    return "\n\n".join(sections)


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
        retry_delays=None
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
        else:
            self.api_key = cfg_key

        if base_url is not None and base_url != "":
            self.base_url = base_url
        else:
            self.base_url = cfg_url

        if model is not None and model != "":
            self.model = model
        else:
            self.model = cfg_model

        self.temperature = GeneratorConfig.LLM_TEMPERATURE
        if temperature is not None:
            self.temperature = temperature
        self.max_tokens = GeneratorConfig.LLM_MAX_TOKENS
        if max_tokens is not None:
            self.max_tokens = max_tokens
        self.enable_cache = enable_cache

        if retry_delays is not None:
            self.retry_delays = retry_delays
        else:
            self.retry_delays = [1.0, 2.0, 4.0]

        if max_retries is not None:
            self.max_retries = max_retries
        elif retry_delays is not None:
            self.max_retries = max(len(retry_delays), cfg_attempts)
        else:
            self.max_retries = max(cfg_attempts, getattr(GeneratorConfig, "MAX_RETRIES", 3), 3)

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
        try:
            with sqlite3.connect(str(self.cache_db_path), timeout=10.0) as conn:
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

    def _get_cached_response(self, cache_key):
        if not self.enable_cache:
            return None
        try:
            with sqlite3.connect(str(self.cache_db_path), timeout=5.0) as conn:
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
        return None

    def _save_cached_response(self, cache_key, query, answer_text, legal_citations, sources_used,
                              model_used, used_persona, used_tone, used_format,
                              prompt_tokens, completion_tokens, total_tokens):
        if not self.enable_cache:
            return
        try:
            with sqlite3.connect(str(self.cache_db_path), timeout=5.0) as conn:
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

    _extract_legal_citations_from_text = staticmethod(_extract_legal_citations_from_text)
    _extract_citations_from_header = staticmethod(_extract_citations_from_header)

    @staticmethod
    def format_chunks(chunks):
        """
        Assemble chunk list into clear Markdown text with sources and legal citations.
        """
        if not chunks:
            return ContextBundle()

        blocks = []
        sources = []
        citations = []
        content_keys = ["parent_content", "content", "document", "text", "page_content"]

        for idx, item in enumerate(chunks, start=1):
            if isinstance(item, dict) and "chunk" in item:
                chunk = item["chunk"]
            else:
                chunk = item

            if not isinstance(chunk, dict):
                continue

            source = chunk.get("source")
            if not source:
                if "metadata" in chunk and isinstance(chunk["metadata"], dict):
                    source = chunk["metadata"].get("source")
            if not source:
                source = "Tài liệu ngân hàng"

            breadcrumb = chunk.get("breadcrumb", "")
            if not breadcrumb:
                if "metadata" in chunk and isinstance(chunk["metadata"], dict):
                    breadcrumb = chunk["metadata"].get("breadcrumb", "")

            section_title = chunk.get("section_title", "")
            if not section_title:
                if "metadata" in chunk and isinstance(chunk["metadata"], dict):
                    section_title = chunk["metadata"].get("section_title", "")

            content = ""
            for k in content_keys:
                val = chunk.get(k)
                if val:
                    content = str(val).strip()
                    break

            if source and source not in sources:
                sources.append(source)

            # Extract legal citations from content, section_title, and breadcrumb
            _extract_legal_citations_from_text(content, citations)
            if section_title:
                _extract_citations_from_header(section_title, citations)
            if breadcrumb:
                _extract_citations_from_header(breadcrumb, citations)

            meta_lines = [f"### [TRÍCH DẪN {idx}] Nguồn tài liệu: {source}"]
            if breadcrumb:
                meta_lines.append(f"- **Phân cấp:** `{breadcrumb}`")
            if section_title:
                meta_lines.append(f"- **Tiêu đề điều khoản:** `{section_title}`")
            meta_lines.append(f"- **Nội dung xác thực:**\n{content}")
            blocks.append("\n".join(meta_lines))

        context_text = "\n\n---\n\n".join(blocks)
        total_tokens = max(int(len(context_text.split()) * 1.25), int(len(context_text) / 3.5))

        return ContextBundle(
            markdown_context=context_text,
            total_tokens=total_tokens,
            legal_citations=citations,
            sources_used=sources,
            chunks_included=len(chunks)
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
            bundle = self.format_chunks(synthesized_context)
            context_text = bundle.markdown_context
            legal_citations = bundle.legal_citations
            sources_used = bundle.sources_used
            tokens_used = bundle.total_tokens
        elif isinstance(synthesized_context, str):
            context_text = synthesized_context
            legal_citations = []
            self._extract_legal_citations_from_text(context_text, legal_citations)
            sources_used = []
            tokens_used = len(context_text) // 4
        else:
            context_text = str(synthesized_context)
            legal_citations = []
            self._extract_legal_citations_from_text(context_text, legal_citations)
            sources_used = []
            tokens_used = len(context_text) // 4

        # Construct dynamic System Prompt based on Persona profile
        if isinstance(persona_config, dict) and persona_config:
            p_key = persona_config.get("persona", "advisor")
            t_key = persona_config.get("tone", "formal")
            f_key = persona_config.get("format_style", "standard")
            custom_p = persona_config.get("custom_persona", "")
            if not custom_p:
                custom_p = persona_config.get("custom_instructions", "")
            system_instruction = build_banking_system_prompt(p_key, t_key, f_key, custom_p)
            used_persona = p_key
            used_tone = t_key
            used_format = f_key
        else:
            system_instruction = build_banking_system_prompt("advisor", "formal", "standard")
            used_persona = "advisor"
            used_tone = "formal"
            used_format = "standard"

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
