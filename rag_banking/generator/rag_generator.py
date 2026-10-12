"""
RAG Generator & Banking Guardrails (Bước 5: LLM Serving với System Guardrails Ngân hàng)
Nhiệm vụ:
1. Tiếp nhận ngữ cảnh Markdown đã tổng hợp (gắn thẻ [Điều X, Khoản Y]) và câu hỏi người dùng.
2. Áp dụng Hệ thống Banking System Guardrails nghiêm ngặt chống bịa đặt số liệu tài chính.
3. Phục vụ suy luận qua DeepSeek-V4-Flash (FPT Cloud / OpenAI API) hoặc Google Gemini.
4. Trả về câu trả lời chuẩn mực ngân hàng, dẫn chứng số liệu minh bạch và bảo mật.
"""

import os
from typing import Dict, Any, Union
from openai import OpenAI
from ..synthesizer.context_synthesizer import SynthesizedContext

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
   - Sử dụng gạch đầu dòng, bảng Markdown rõ ràng khi liệt kê các bước quy trình, biểu phí hoặc so sánh sản phẩm.
"""

class RAGGenerator:
    def __init__(
        self,
        api_key: str = None,
        base_url: str = None,
        model: str = None,
        temperature: float = 0.2,
        max_tokens: int = 1500
    ):
        self.api_key = api_key or os.getenv("API_KEY", "")
        self.base_url = base_url or os.getenv("BASE_URL", "https://mkp-api.fptcloud.com/v1")
        self.model = model or os.getenv("MODEL", "DeepSeek-V4-Flash")
        self.temperature = temperature
        self.max_tokens = max_tokens

        self.client = None
        if self.api_key:
            try:
                self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            except Exception as e:
                print(f"[RAG Generator] OpenAI Client init error: {e}")

    def generate_answer(
        self,
        query: str,
        synthesized_context: Union[SynthesizedContext, str, list]
    ) -> Dict[str, Any]:
        """
        Thực thi Bước 5: Truyền ngữ cảnh kèm System Guardrails vào LLM để sinh câu trả lời.
        """
        # Trích xuất nội dung ngữ cảnh và danh sách trích dẫn
        if isinstance(synthesized_context, SynthesizedContext):
            context_text = synthesized_context.markdown_context
            legal_citations = synthesized_context.legal_citations
            sources_used = synthesized_context.sources_used
            tokens_used = synthesized_context.total_tokens
        elif isinstance(synthesized_context, str):
            context_text = synthesized_context
            legal_citations = []
            sources_used = []
            tokens_used = len(context_text) // 4
        else:
            # Dạng list chunks thô
            blocks = []
            for item in synthesized_context:
                c = item.get("chunk", item)
                blocks.append(f"[{c.get('source', '')} - {c.get('breadcrumb', '')}]\n{c.get('content', '')}")
            context_text = "\n\n".join(blocks)
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

        return {
            "query": query,
            "answer": answer_text,
            "legal_citations": legal_citations,
            "sources_used": sources_used,
            "context_tokens": tokens_used,
            "model_served": model_used,
            "guardrails_applied": True
        }
