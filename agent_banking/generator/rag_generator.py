import unicodedata
"""
Module RAG Generator & LLM Serving (Bước 5: LLM Serving với System Guardrails Ngân hàng)
======================================================================
Nhiệm vụ:
1. Tiếp nhận ngữ cảnh Markdown đã tổng hợp (gắn thẻ [Điều X, Khoản Y]) và câu hỏi người dùng.
2. Áp dụng Hệ thống Banking System Guardrails nghiêm ngặt chống bịa đặt số liệu tài chính.
3. Phục vụ suy luận qua DeepSeek-V4-Flash (FPT Cloud / OpenAI API) hoặc Google Gemini.
4. Trả về câu trả lời chuẩn mực ngân hàng, dẫn chứng số liệu minh bạch và bảo mật.

NGUYÊN TẮC THIẾT KẾ:
1. 0% REGEX LAW: Tuyệt đối không sử dụng `import re` hay regex methods.
2. ZERO TERNARY: Ưu tiên if/else tường minh, dễ đọc.
3. NO __INIT__.PY: Import trực tiếp file module.
"""

import os
from openai import OpenAI

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


# Tương thích ngược với các hàm gọi cũ
SynthesizedContext = ContextBundle

# HỆ THỐNG SYSTEM GUARDRAILS NGÂN HÀNG (BANKING GRADE GUARDRAILS)
BANKING_SYSTEM_GUARDRAILS = """Bạn là Trợ lý AI Banking chuyên viên tư vấn của Ngân hàng.
Nhiệm vụ: Giải đáp câu hỏi của Quý khách hàng dựa trên tài liệu ngân hàng được cung cấp.

QUY TẮC BẮT BUỘC (100% TUÂN THỦ):
1. BẮT ĐẦU CÂU TRẢ LỜI NGAY LẬP TỨC CHO KHÁCH HÀNG. TUYỆT ĐỐI KHÔNG SUY NGHĨ THÀNH LỜI, KHÔNG PHÂN TÍCH NHÁP, KHÔNG ĐỘC THOẠI (như "chúng ta cần...", "câu hỏi của khách hàng...", "có ba trích dẫn...", "có lẽ nên trả lời...").
2. Chỉ trả lời dựa trên thông tin có trong tài liệu. Không bịa đặt số liệu tài chính.
3. Nếu có điều khoản quy định trong tài liệu, trích dẫn kèm [Điều X, Khoản Y].
4. Tuyệt đối không yêu cầu mật khẩu, mã PIN, mã OTP của khách hàng.
5. Ngôn phong lịch thiệp, trang trọng (xưng "Ngân hàng" hoặc "Trợ lý AI", gọi "Quý khách"). Trả lời súc tích, rõ ràng."""

# CẤU HÌNH DANH MỤC PERSONA, TÔNG GIỌNG & ĐỊNH DẠNG TRẢ LỜI (0% REGEX & ZERO TERNARY)
PERSONA_PROFILES = {
    "advisor": {
        "name": "Chuyên viên Tư vấn Tài chính Cao cấp",
        "description": "Chuyên nghiệp, phân tích logic, am hiểu sâu nghiệp vụ ngân hàng và điều khoản giao dịch.",
        "prompt": "Bạn là Chuyên viên Tư vấn Tài chính Cao cấp của Ngân hàng. Phong thái điềm đạm, am tường nghiệp vụ ngân hàng, đưa ra các chỉ dẫn và giải pháp tài chính chuẩn mực, minh bạch và có chiều sâu."
    },
    "teller": {
        "name": "Giao dịch viên Thân thiện & Tận tâm",
        "description": "Niềm nở, ân cần, giải thích bình dân dễ hiểu, phù hợp cho khách hàng đại chúng.",
        "prompt": "Bạn là Giao dịch viên Thân thiện tại quầy Ngân hàng. Luôn nở nụ cười, lắng nghe và giải thích các thủ tục, điều kiện tài chính bằng từ ngữ mộc mạc, gần gũi, ân cần và kiên nhẫn hướng dẫn từng bước."
    },
    "compliance": {
        "name": "Cố vấn Pháp chế & Kiểm soát Rủi ro",
        "description": "Chuẩn mực pháp lý cao độ, dẫn chiếu số hiệu điều khoản chính xác, cảnh báo rủi ro minh bạch.",
        "prompt": "Bạn là Cố vấn Pháp chế & Kiểm soát Rủi ro Ngân hàng. Luôn bám sát nguyên văn các quy chế, hiệp định và quy chuẩn. Mọi câu trả lời đều phải viện dẫn chính xác căn cứ điều khoản pháp lý và nêu rõ cảnh báo rủi ro cho khách hàng."
    },
    "quick": {
        "name": "Trợ lý Hỗ trợ Nhanh (Súc tích)",
        "description": "Trực diện, nhanh chóng, đi thẳng vào đáp án và số liệu cốt lõi bằng các gạch đầu dòng.",
        "prompt": "Bạn là Trợ lý AI Hỗ trợ Nhanh. Ưu tiên cao nhất là tính chính xác và súc tích: Đi thẳng vào câu trả lời, nêu rõ số liệu/hạn mức/lãi suất cốt lõi mà không viết dài dòng, lan man."
    },
    "custom": {
        "name": "Tùy chỉnh (Custom Persona)",
        "description": "Tự do định nghĩa vai trò, phong cách theo nhu cầu cụ thể của người dùng.",
        "prompt": ""
    }
}

TONE_PROFILES = {
    "formal": "Trang trọng & Lịch thiệp (Xưng 'Ngân hàng' hoặc 'Trợ lý AI', kính cẩn gọi 'Quý khách', lời văn nhã nhặn, tôn trọng).",
    "warm": "Ấm áp & Thân thiện (Ngôn từ gần gũi, đồng cảm, tạo cảm giác an tâm và đồng hành cùng Quý khách).",
    "direct": "Trực diện & Dứt khoát (Đi thẳng vào trọng tâm câu hỏi, câu văn gãy gọn, không mở bài hay kết bài rườm rà).",
    "detailed": "Giảng giải Chi tiết & Từng bước (Phân tích cặn kẽ, giải thích các thuật ngữ chuyên môn để khách hàng dễ áp dụng)."
}

FORMAT_PROFILES = {
    "standard": "Định dạng Tiêu chuẩn: Đoạn văn mạch lạc, phân đoạn rõ ràng giữa lời giải đáp, căn cứ và kết luận.",
    "bullet_points": "Định dạng Gạch đầu dòng: Sử dụng các bullet points ngắn gọn (- hoặc *) để liệt kê điều kiện, thủ tục, mức phí và quyền lợi.",
    "table_comparison": "Định dạng Bảng biểu & So sánh: Khi có thông tin kỳ hạn, lãi suất, hạn mức hoặc nhiều phương án, BẮT BUỘC trình bày dưới dạng bảng Markdown trực quan.",
    "legal_strict": "Định dạng Căn cứ Pháp lý: Bắt buộc gắn thẻ tham chiếu [Điều X, Khoản Y] ngay sau từng ý phát biểu.",
    "with_disclaimer": "Định dạng Kèm Lưu ý Rủi ro: Bắt buộc bổ sung phần '⚠️ Lưu ý an toàn & Bảo mật' ở cuối câu trả lời để nhắc nhở khách hàng."
}


def build_banking_system_prompt(persona_key="advisor", tone_key="formal", format_key="standard", custom_persona=""):
    """
    Xây dựng Dynamic System Prompt kết hợp giữa Core Banking Guardrails và Persona được cấu hình.
    Tuân thủ 100% Zero Regex & Zero Ternary.
    """
    if persona_key == "custom" and custom_persona and custom_persona.strip():
        persona_text = custom_persona.strip()
    elif persona_key in PERSONA_PROFILES:
        persona_text = PERSONA_PROFILES[persona_key]["prompt"]
    else:
        persona_text = PERSONA_PROFILES["advisor"]["prompt"]

    if tone_key in TONE_PROFILES:
        tone_text = TONE_PROFILES[tone_key]
    else:
        tone_text = TONE_PROFILES["formal"]

    if format_key in FORMAT_PROFILES:
        format_text = FORMAT_PROFILES[format_key]
    else:
        format_text = FORMAT_PROFILES["standard"]

    system_prompt = (
        f"{persona_text}\n\n"
        f"QUY TẮC TÔNG GIỌNG & ĐỊNH DẠNG CÂU TRẢ LỜI:\n"
        f"- Tông giọng: {tone_text}\n"
        f"- Định dạng trình bày: {format_text}\n\n"
        f"QUY TẮC BẢO MẬT & AN TOÀN NGÂN HÀNG BẮT BUỘC (100% TUÂN THỦ):\n"
        f"1. BẮT ĐẦU CÂU TRẢ LỜI NGAY LẬP TỨC CHO KHÁCH HÀNG. TUYỆT ĐỐI KHÔNG SUY NGHĨ THÀNH LỜI, KHÔNG PHÂN TÍCH NHÁP, KHÔNG ĐỘC THOẠI (như 'chúng ta cần...', 'câu hỏi của khách hàng...', 'có ba trích dẫn...').\n"
        f"2. Chỉ trả lời dựa trên thông tin có trong tài liệu được cung cấp. Không tự ý bịa đặt số liệu tài chính.\n"
        f"3. Nếu có điều khoản quy định trong tài liệu, trích dẫn kèm [Điều X, Khoản Y].\n"
        f"4. Tuyệt đối không bao giờ yêu cầu mật khẩu, mã PIN, mã OTP của khách hàng."
    )
    return system_prompt

class RAGGenerator:
    def __init__(
        self,
        api_key=None,
        base_url=None,
        model=None,
        temperature=0.2,
        max_tokens=2500
    ):
        self.api_key = api_key or os.getenv("API_KEY", "")
        self.base_url = base_url or os.getenv("BASE_URL", "")
        self.model = model or os.getenv("MODEL", "")
        self.temperature = temperature
        self.max_tokens = max_tokens

        self.client = None
        if self.api_key:
            try:
                self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            except Exception as e:
                print(f"[RAG Generator] OpenAI Client init error: {e}")

    @staticmethod
    def format_chunks(chunks):
        """
        Đóng gói danh sách chunks thành văn bản Markdown rõ ràng kèm nguồn và trích dẫn pháp lý.
        Hoàn toàn không regex, không phụ thuộc thư viện ngoài.
        """
        if not chunks:
            return ContextBundle()

        blocks = []
        sources = []
        citations = []

        for idx, item in enumerate(chunks, start=1):
            if isinstance(item, dict) and "chunk" in item:
                chunk = item["chunk"]
            else:
                chunk = item

            source = chunk.get("source", "Tài liệu ngân hàng")
            breadcrumb = chunk.get("breadcrumb", "")
            content = chunk.get("content", "")

            if source and source not in sources:
                sources.append(source)

            # Trích xuất nhãn tham chiếu [Điều ...] có sẵn trong text (0% regex)
            if "[" in content:
                start = 0
                while True:
                    o = content.find("[", start)
                    if o == -1:
                        break
                    c = content.find("]", o)
                    if c == -1:
                        break
                    tag = content[o + 1 : c].strip()
                    if 1 <= len(tag) <= 60 and "\n" not in tag:
                        full_tag = f"[{tag}]"
                        if full_tag not in citations:
                            citations.append(full_tag)
                    start = c + 1

            meta_lines = [f"### [TRÍCH DẪN {idx}] Nguồn tài liệu: {source}"]
            if breadcrumb:
                meta_lines.append(f"- **Phân cấp:** `{breadcrumb}`")
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
        Thực thi LLM Serving: Truyền ngữ cảnh kèm Dynamic Persona & System Guardrails vào LLM để sinh câu trả lời.
        """
        # Trích xuất nội dung ngữ cảnh và danh sách trích dẫn
        if isinstance(synthesized_context, ContextBundle):
            context_text = synthesized_context.markdown_context
            legal_citations = synthesized_context.legal_citations
            sources_used = synthesized_context.sources_used
            tokens_used = synthesized_context.total_tokens
        elif isinstance(synthesized_context, list):
            # Nhận danh sách chunks trực tiếp từ Reranker
            bundle = self.format_chunks(synthesized_context)
            context_text = bundle.markdown_context
            legal_citations = bundle.legal_citations
            sources_used = bundle.sources_used
            tokens_used = bundle.total_tokens
        elif isinstance(synthesized_context, str):
            context_text = synthesized_context
            legal_citations = []
            sources_used = []
            tokens_used = len(context_text) // 4
        else:
            context_text = str(synthesized_context)
            legal_citations = []
            sources_used = []
            tokens_used = len(context_text) // 4

        # Xây dựng System Prompt động theo cấu hình Persona
        if isinstance(persona_config, dict) and persona_config:
            p_key = persona_config.get("persona", "advisor")
            t_key = persona_config.get("tone", "formal")
            f_key = persona_config.get("format_style", "standard")
            custom_p = persona_config.get("custom_persona", "") or persona_config.get("custom_instructions", "")
            system_instruction = build_banking_system_prompt(p_key, t_key, f_key, custom_p)
            used_persona = p_key
            used_tone = t_key
            used_format = f_key
        else:
            system_instruction = BANKING_SYSTEM_GUARDRAILS
            used_persona = "advisor"
            used_tone = "formal"
            used_format = "standard"

        # Xây dựng User Prompt
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

        # 1. Gọi LLM DeepSeek qua OpenAI-compatible API
        if self.client:
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_instruction},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
                msg = response.choices[0].message
                answer_text = msg.content or getattr(msg, "reasoning_content", None) or ""

                usage_obj = getattr(response, "usage", None)
                if usage_obj is not None:
                    prompt_tokens_est = getattr(usage_obj, "prompt_tokens", prompt_tokens_est)
                    completion_tokens_est = getattr(usage_obj, "completion_tokens", max(1, len(answer_text) // 4))
                else:
                    completion_tokens_est = max(1, len(answer_text) // 4)
            except Exception as e:
                print(f"[RAG Generator] Lỗi gọi API {self.model}: {e}")
                answer_text = (
                    f"**[Chế độ Dự phòng Ngữ cảnh Xác thực - Không thể kết nối API AI: {e}]**\n\n"
                    f"Dưới đây là căn cứ pháp lý và nội dung chính thức được tìm thấy trong tài liệu ngân hàng:\n\n"
                    f"{context_text[:1200]}...\n\n"
                    f"Quý khách vui lòng liên hệ Hotline hỗ trợ để được nhân viên quầy giải đáp trực tiếp."
                )
                completion_tokens_est = len(answer_text) // 4
        else:
            answer_text = (
                f"**[Chế độ Trả lời Xác thực Cục bộ - Sẵn sàng nạp API Key]**\n\n"
                f"Hệ thống đã truy xuất và tổng hợp thành công ngữ cảnh pháp lý:\n\n"
                f"{context_text}\n\n"
                f"Để nhận câu trả lời đầy đủ từ LLM DeepSeek/Gemini, vui lòng cấu hình API_KEY."
            )
            completion_tokens_est = len(answer_text) // 4

        # Làm sạch và loại bỏ hoàn toàn các đoạn suy nghĩ nháp/thinking nếu có
        answer_text = self._clean_bot_response(answer_text)

        total_tokens = prompt_tokens_est + completion_tokens_est

        return {
            "query": query,
            "answer": answer_text,
            "legal_citations": legal_citations,
            "sources_used": sources_used,
            "context_tokens": tokens_used,
            "model_served": model_used,
            "guardrails_applied": True,
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
    def _clean_bot_response(text):
        """
        Loại bỏ hoàn toàn tư duy nội bộ (thinking / scratchpad / self-critique) khỏi câu trả lời của bot (0% Regex Law).
        """
        if not text:
            return ""

        # Chuẩn hóa Unicode NFC để đảm bảo khớp dấu tiếng Việt (tránh lỗi NFD decomposed accents)
        text = unicodedata.normalize("NFC", text).strip()

        # 1. Bỏ toàn bộ thẻ <think>...</think> thuần túy bằng string search
        while "<think>" in text:
            start_idx = text.find("<think>")
            end_idx = text.find("</think>", start_idx)
            if end_idx != -1:
                text = text[:start_idx] + text[end_idx + 8:]
            else:
                text = text[:start_idx]
                break
        text = text.strip()

        # 2. Bắt cụm từ chốt đáp án sau đoạn độc thoại (như "Soạn:", "Vậy câu trả lời cuối cùng:", ...)
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
            "Phản hồi chính thức:"
        ]
        for fa in final_anchors:
            if fa in text:
                parts = text.split(fa)
                candidate = parts[-1].strip()
                if len(candidate) > 20:
                    text = candidate
                    break

        # Nếu đoạn phản hồi bị bọc trong dấu ngoặc kép ("...")
        if text.startswith('"') or text.startswith("'"):
            text = text[1:].strip()

        # 3. Cắt bỏ đoạn độc thoại tự kiểm duyệt guardrail nếu có
        critique_anchors = [
            '" Kiểm tra:',
            'Kiểm tra:',
            'Kiểm tra lại:',
            "Cần trích dẫn điều khoản",
            "Cần trích dẫn",
            "Như vậy có ổn không?",
            "Cần lưu ý guardrail",
            "Hmm, guardrail",
            "Guardrail nói",
            "Liệu như vậy có ổn",
            "Vậy ta sẽ thay",
            "Có sự không nhất quán",
            "Để an toàn, có thể nói",
            "Tốt nhất nên tập trung",
            "Tuy nhiên, câu hỏi"
        ]
        for ca in critique_anchors:
            if ca in text:
                c_idx = text.find(ca)
                if c_idx > 40:
                    text = text[:c_idx].strip()
                    break

        if text.endswith('"') or text.endswith("'"):
            text = text[:-1].strip()

        # 4. Nếu còn dính "Chúng ta cần trả lời..." hoặc "Tôi cần...", cắt tới lời chào hoặc trích dẫn chính thức
        prefixes = ("Chúng ta cần", "Ta cần", "Tôi cần", "Để trả lời", "Cần trả lời", "Câu hỏi:", "Phân tích:", "Dựa trên các trích dẫn")
        if text.startswith(prefixes):
            official_anchors = [
                "Kính thưa Quý khách",
                "Kính chào Quý khách",
                "Chào Quý khách",
                "Thưa Quý khách",
                "Theo quy định",
                "Về hạn mức",
                "Hạn mức eKYC"
            ]
            earliest_idx = -1
            for anchor in official_anchors:
                idx = text.find(anchor)
                if idx != -1:
                    if earliest_idx == -1 or idx < earliest_idx:
                        earliest_idx = idx
            if earliest_idx != -1:
                text = text[earliest_idx:].strip()

        return text.strip()
