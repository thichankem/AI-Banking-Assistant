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

# HỆ THỐNG SYSTEM GUARDRAILS NGÂN HÀNG (VIETCOMBANK GRADE GUARDRAILS)
BANKING_SYSTEM_GUARDRAILS = """Bạn là Trợ lý AI Chuyên viên Tư vấn Khách hàng Cao cấp của Ngân hàng TMCP Ngoại thương Việt Nam (Vietcombank AI Assistant).
Nhiệm vụ cốt lõi của bạn là giải đáp chính xác, khách quan và chuyên nghiệp mọi thắc mắc của khách hàng dựa trên ngữ cảnh tài liệu nội bộ và văn bản pháp lý được cung cấp.

HỆ THỐNG NGUYÊN TẮC AN TOÀN & BẢO VỆ NGÂN HÀNG (SYSTEM GUARDRAILS) - BẮT BUỘC TUÂN THỦ 100%:

1. NGUYÊN TẮC ZERO-HALLUCINATION & TRUNG THỰC SỐ LIỆU:
   - TUYỆT ĐỐI KHÔNG BỊA ĐẶT, suy đoán hoặc tự ý điều chỉnh bất kỳ số liệu tài chính nào (lãi suất %/năm, hạn mức giao dịch VND, biểu phí dịch vụ, số tiền dư tối thiểu, ngày hiệu lực...).
   - Mọi con số trả lời PHẢI xuất hiện chính xác trong ngữ cảnh tài liệu được cung cấp.

2. NGUYÊN TẮC DẪN CHỨNG PHÁP LÝ BẮT BUỘC [ĐIỀU X, KHOẢN Y]:
   - Khi giải thích quy chế, quyền hạn, nghĩa vụ hoặc điều kiện mở tài khoản, BẮT BUỘC phải trích dẫn kèm mã điều khoản cụ thể theo định dạng: `[Điều X, Khoản Y]` hoặc `[Điều X]` đã được cung cấp trong ngữ cảnh.

3. NGUYÊN TẮC AN TOÀN BẢO MẬT THÔNG TIN:
   - Tuyệt đối không yêu cầu khách hàng cung cấp các thông tin bảo mật nhạy cảm: Mật khẩu VCB Digibank, Mã OTP, Mã PIN thẻ ATM, Mã bảo mật CVV/CVC.
   - Luôn chủ động kèm câu cảnh báo bảo mật khi khách hàng hỏi về các giao dịch trực tuyến hoặc phương thức xác thực Smart OTP.

4. NGUYÊN TẮC MINH BẠCH VỀ GIỚI HẠN THÔNG TIN:
   - Nếu ngữ cảnh tài liệu KHÔNG chứa câu trả lời cho câu hỏi của khách hàng, hãy lịch sự và trung thực thông báo rằng hiện chưa có thông tin quy định trong tài liệu được cung cấp.
   - Hướng dẫn khách hàng liên hệ Tổng đài Trung tâm Hỗ trợ Khách hàng Vietcombank 24/7 qua hotline chính thức: 1900 54 54 13 hoặc đến quầy giao dịch Vietcombank gần nhất.

5. PHONG CÁCH DIỄN ĐẠT:
   - Lịch thiệp, chuẩn mực ngôn phong tài chính ngân hàng, sử dụng đại từ "Vietcombank" và "Quý khách".
   - Trả lời súc tích, trực diện vào nội dung câu hỏi, tránh dài dòng.
   - Sử dụng gạch đầu dòng, bảng Markdown rõ ràng khi liệt kê các bước quy trình, biểu phí hoặc so sánh sản phẩm.

6. QUY CÁCH PHẢN HỒI (OUTPUT DIRECTIVE):
   - Trả lời TRỰC TIẾP đến khách hàng.
   - TUYỆT ĐỐI KHÔNG xuất các bước suy nghĩ nội bộ, phân tích đề bài, lập dàn ý hoặc câu từ nháp như: "Chúng ta cần trả lời...", "Soạn câu trả lời:", "Cần trích dẫn...", "Phải tuân thủ guardrails...".
   - Chỉ xuất nội dung phản hồi hoàn chỉnh duy nhất gửi tới Quý khách hàng.
"""

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

    def generate_answer(self, query, synthesized_context):
        """
        Thực thi LLM Serving: Truyền ngữ cảnh kèm System Guardrails vào LLM để sinh câu trả lời.
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

        # Xây dựng User Prompt
        user_prompt = (
            f"NGỮ CẢNH TÀI LIỆU NỘI BỘ & CĂN CỨ PHÁP LÝ VIETCOMBANK:\n\n"
            f"{context_text}\n\n"
            f"CÂU HỎI CỦA QUÝ KHÁCH HÀNG:\n"
            f"{query}\n\n"
            f"HÃY ÁP DỤNG ĐÚNG SYSTEM GUARDRAILS VÀ TRẢ LỜI CHO KHÁCH HÀNG:"
        )

        answer_text = ""
        model_used = self.model

        # 1. Gọi LLM DeepSeek qua OpenAI-compatible API
        if self.client:
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": BANKING_SYSTEM_GUARDRAILS},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
                msg = response.choices[0].message
                answer_text = msg.content or getattr(msg, "reasoning_content", None) or ""
            except Exception as e:
                print(f"[RAG Generator] Lỗi gọi API {self.model}: {e}")
                answer_text = (
                    f"**[Chế độ Dự phòng Ngữ cảnh Xác thực - Không thể kết nối API AI: {e}]**\n\n"
                    f"Dưới đây là căn cứ pháp lý và nội dung chính thức được tìm thấy trong tài liệu Vietcombank:\n\n"
                    f"{context_text[:1200]}...\n\n"
                    f"Quý khách vui lòng liên hệ Hotline 1900 54 54 13 để được nhân viên quầy giải đáp trực tiếp."
                )
        else:
            answer_text = (
                f"**[Chế độ Trả lời Xác thực Cục bộ - Sẵn sàng nạp API Key]**\n\n"
                f"Hệ thống đã truy xuất và tổng hợp thành công ngữ cảnh pháp lý:\n\n"
                f"{context_text}\n\n"
                f"Để nhận câu trả lời đầy đủ từ LLM DeepSeek/Gemini, vui lòng cấu hình API_KEY."
            )

        # Làm sạch và loại bỏ hoàn toàn các đoạn suy nghĩ nháp/thinking nếu có
        answer_text = self._clean_bot_response(answer_text)

        return {
            "query": query,
            "answer": answer_text,
            "legal_citations": legal_citations,
            "sources_used": sources_used,
            "context_tokens": tokens_used,
            "model_served": model_used,
            "guardrails_applied": True
        }

    @staticmethod
    def _clean_bot_response(text):
        """
        Loại bỏ hoàn toàn tư duy nội bộ (thinking / scratchpad) khỏi câu trả lời của bot (0% Regex Law).
        """
        if not text:
            return ""

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

        # 2. Bỏ nháp suy nghĩ nội bộ nếu bắt đầu bằng các cụm từ phân tích
        meta_phrases = [
            "Viết phản hồi:",
            "Soạn câu trả lời:",
            "Soạn thảo câu trả lời:",
            "Câu trả lời của bot:",
            "Câu trả lời:",
            "Nội dung phản hồi:",
            "Phản hồi Quý khách:",
            "Phản hồi chính thức:"
        ]
        for phrase in meta_phrases:
            if phrase in text:
                parts = text.split(phrase)
                last_part = parts[-1].strip()
                if len(last_part) > 30:
                    text = last_part
                    break

        # 3. Nếu còn dính "Chúng ta cần trả lời..." hoặc "Tôi cần...", cắt tới lời chào hoặc trích dẫn chính thức (0% Regex)
        prefixes = ("Chúng ta cần", "Ta cần", "Tôi cần", "Để trả lời", "Cần trả lời")
        if text.startswith(prefixes):
            official_anchors = [
                "Kính thưa Quý khách",
                "Kính chào Quý khách",
                "Chào Quý khách",
                "Theo quy định",
                "Về hạn mức"
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
