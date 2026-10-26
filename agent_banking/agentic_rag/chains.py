# -*- coding: utf-8 -*-
"""
LLM Chains for Banking Agentic RAG
======================================================================
Khởi tạo các Chains xử lý ngôn ngữ sử dụng LangChain:
- Ghép nối ChatPromptTemplate + LLM (DeepSeek / OpenAI compatible endpoint).
- Ứng dụng `llm.with_structured_output()` với các Pydantic Schemas:
  * Router Chain -> RouteQuery
  * Decomposer Chain -> DecomposeQuery
  * Document Grader Chain -> GradeDocument
  * Hallucination Grader Chain -> GradeHallucination
  * Answer Grader Chain -> GradeAnswer
- Các Chains tạo sinh văn bản (Contextualize, Rewrite, Generation, Direct Chat).

NGUYÊN TẮC:
1. 0% REGEX LAW: Tuyệt đối không dùng thư viện regex.
2. ZERO TERNARY: Ưu tiên if/else tường minh, dễ đọc.
3. NO HARDCODED SECRETS: Đọc toàn bộ cấu hình từ Config hiện có.
"""

from __future__ import annotations
import logging
from typing import Optional, Any

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from agent_banking.config import Config
from agent_banking.agentic_rag.schemas import (
    RouteQuery,
    DecomposeQuery,
    GradeDocument,
    GradeHallucination,
    GradeAnswer
)

logger = logging.getLogger(__name__)


def get_llm(
    temperature: float = 0.0,
    model: Optional[str] = None,
    max_tokens: Optional[int] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None
) -> ChatOpenAI:
    """
    Khởi tạo ChatOpenAI tương thích với DeepSeek / Cloud LLM API từ Config hệ thống.
    """
    cfg = Config()

    if model is not None:
        selected_model = model
    else:
        if cfg.LLM_MODEL:
            selected_model = cfg.LLM_MODEL
        else:
            selected_model = "DeepSeek-V4-Flash"

    if api_key is not None:
        selected_api_key = api_key
    else:
        if cfg.LLM_API_KEY:
            selected_api_key = cfg.LLM_API_KEY
        else:
            selected_api_key = "local-key"

    if base_url is not None:
        selected_base_url = base_url
    else:
        if cfg.LLM_BASE_URL:
            selected_base_url = cfg.LLM_BASE_URL
        else:
            selected_base_url = None

    if max_tokens is not None:
        selected_max_tokens = max_tokens
    else:
        selected_max_tokens = cfg.LLM_MAX_TOKENS

    return ChatOpenAI(
        model=selected_model,
        api_key=selected_api_key,
        base_url=selected_base_url,
        temperature=temperature,
        max_tokens=selected_max_tokens,
        streaming=False
    )


# =====================================================================
# 1. ROUTER CHAIN (ADAPTIVE ROUTING)
# =====================================================================
ROUTER_SYSTEM_PROMPT = """Bạn là Chuyên gia Điều hướng Ý định (Adaptive Router) của Trợ lý AI Ngân hàng Vietcombank.
Hãy phân tích câu hỏi của khách hàng và định tuyến chính xác vào một trong các nguồn dữ liệu:

1. 'vectorstore':
   - Câu hỏi về quy chế, quy trình mở tài khoản eKYC, hạn mức giao dịch chuyển tiền, biểu phí phát hành và sử dụng thẻ (thẻ tín dụng, thẻ ghi nợ), tính năng VCB Digibank, thủ tục tài chính, điều khoản hợp đồng nội bộ Vietcombank.
   - Ví dụ: 'Hạn mức chuyển tiền eKYC tối đa bao nhiêu một ngày?', 'Phí thường niên thẻ visa platinum?', 'Bị nuốt thẻ ở cây ATM Vietcombank thì làm thế nào?'

2. 'web_search':
   - Câu hỏi về tin tức thời sự kinh tế bên ngoài, sự kiện tài chính vĩ mô quốc tế, chính sách ngân hàng trung ương nước ngoài, hoặc thông tin cập nhật không có trong cẩm nang nghiệp vụ nội bộ của Vietcombank.
   - Ví dụ: 'Chỉ số CPI Mỹ tháng vừa rồi tăng bao nhiêu?', 'Tin tức mới nhất về cuộc họp FOMC của Cục Dự trữ Liên bang?', 'Giá vàng thế giới hôm nay?'

3. 'direct_chat':
   - Lời chào hỏi, cảm ơn, hỏi thăm xã giao thông thường, hoặc câu hỏi không liên quan đến dữ liệu tra cứu.
   - Ví dụ: 'Xin chào bạn', 'Cảm ơn bạn rất nhiều', 'Bạn là ai?', 'Thời tiết hôm nay thế nào?'

Hãy đưa ra lý do ngắn gọn và trả về đúng schema được yêu cầu."""

ROUTER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", ROUTER_SYSTEM_PROMPT),
    ("human", "{question}")
])


def get_router_chain(llm: Optional[ChatOpenAI] = None):
    """Tạo chain định tuyến truy vấn với Structured Output (RouteQuery)."""
    if llm is None:
        target_llm = get_llm(temperature=0.0)
    else:
        target_llm = llm
    structured_llm = target_llm.with_structured_output(RouteQuery)
    return ROUTER_PROMPT | structured_llm


# =====================================================================
# 2. QUERY DECOMPOSITION CHAIN
# =====================================================================
DECOMPOSE_SYSTEM_PROMPT = """Bạn là Chuyên gia Phân rã Truy vấn (Query Decomposition) trong lĩnh vực Tài chính - Ngân hàng.
Nhiệm vụ: Phân tích câu hỏi của khách hàng.
- Nếu câu hỏi chứa nhiều ý phức tạp hoặc hỏi về nhiều sản phẩm/chính sách khác nhau (ví dụ vừa hỏi điều kiện mở thẻ vừa hỏi hạn mức chuyển tiền), hãy tách thành tối đa 3 câu hỏi phụ độc lập, rõ ràng để tra cứu tài liệu đầy đủ nhất.
- Nếu câu hỏi chỉ chứa 1 ý đơn giản hoặc đã rất cụ thể, chỉ trả về danh sách chứa duy nhất câu hỏi hiện tại.

Trả về danh sách các câu hỏi phụ theo đúng schema."""

DECOMPOSE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", DECOMPOSE_SYSTEM_PROMPT),
    ("human", "Câu hỏi của khách hàng: {question}")
])


def get_decomposer_chain(llm: Optional[ChatOpenAI] = None):
    """Tạo chain phân rã câu hỏi phức tạp thành danh sách sub_questions."""
    if llm is None:
        target_llm = get_llm(temperature=0.0)
    else:
        target_llm = llm
    structured_llm = target_llm.with_structured_output(DecomposeQuery)
    return DECOMPOSE_PROMPT | structured_llm


# =====================================================================
# 3. DOCUMENT GRADER CHAIN (CORRECTIVE RAG)
# =====================================================================
DOCUMENT_GRADER_SYSTEM_PROMPT = """Bạn là Kiểm soát viên Thẩm định Tài liệu (Document Grader) của Ngân hàng Vietcombank.
Nhiệm vụ: Đánh giá xem đoạn tài liệu trích xuất (retrieved document) có chứa thông tin hữu ích hoặc liên quan đến câu hỏi của khách hàng hay không.

Tiêu chuẩn chấm điểm:
- Nếu đoạn tài liệu có chứa từ khóa, điều khoản, số liệu hoặc nội dung (dù là một phần) giúp trả lời câu hỏi: đánh giá 'yes'.
- Nếu đoạn tài liệu hoàn toàn lạc đề, không chứa thông tin liên quan đến câu hỏi: đánh giá 'no'.

Không suy đoán ngoài nội dung văn bản. Trả về đúng schema GradeDocument."""

DOCUMENT_GRADER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", DOCUMENT_GRADER_SYSTEM_PROMPT),
    ("human", "CÂU HỎI CỦA KHÁCH HÀNG:\n{question}\n\nNỘI DUNG TÀI LIỆU TRÍCH XUẤT:\n{document}")
])


def get_document_grader_chain(llm: Optional[ChatOpenAI] = None):
    """Tạo chain thẩm định tài liệu trích xuất với Structured Output (GradeDocument)."""
    if llm is None:
        target_llm = get_llm(temperature=0.0)
    else:
        target_llm = llm
    structured_llm = target_llm.with_structured_output(GradeDocument)
    return DOCUMENT_GRADER_PROMPT | structured_llm


# =====================================================================
# 4. HALLUCINATION GRADER CHAIN (SELF-RAG GROUNDEDNESS)
# =====================================================================
HALLUCINATION_GRADER_SYSTEM_PROMPT = """Bạn là Chuyên gia Kiểm toán Độc lập (Hallucination Grader) của Ngân hàng Vietcombank.
Nhiệm vụ: Kiểm tra tính xác thực của câu trả lời do mô hình sinh ra so với toàn bộ các tài liệu căn cứ được cung cấp.

Tiêu chuẩn chấm điểm:
- 'yes': Câu trả lời hoàn toàn bám sát vào các sự kiện, số liệu và điều khoản có trong tài liệu căn cứ (grounded/không bịa đặt).
- 'no': Câu trả lời chứa bất kỳ thông tin nào bịa đặt, suy đoán không có trong tài liệu, hoặc mâu thuẫn trực tiếp với tài liệu (hallucinated).

Trả về đúng schema GradeHallucination."""

HALLUCINATION_GRADER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", HALLUCINATION_GRADER_SYSTEM_PROMPT),
    ("human", "TÀI LIỆU CĂN CỨ:\n{documents}\n\nCÂU TRẢ LỜI CẦN KIỂM ĐỊNH:\n{generation}")
])


def get_hallucination_grader_chain(llm: Optional[ChatOpenAI] = None):
    """Tạo chain kiểm tra ảo giác của câu trả lời với Structured Output (GradeHallucination)."""
    if llm is None:
        target_llm = get_llm(temperature=0.0)
    else:
        target_llm = llm
    structured_llm = target_llm.with_structured_output(GradeHallucination)
    return HALLUCINATION_GRADER_PROMPT | structured_llm


# =====================================================================
# 5. ANSWER GRADER CHAIN (SELF-RAG ANSWER UTILITY)
# =====================================================================
ANSWER_GRADER_SYSTEM_PROMPT = """Bạn là Kiểm soát viên Chất lượng Dịch vụ (Answer Utility Grader) của Ngân hàng Vietcombank.
Nhiệm vụ: Đánh giá xem câu trả lời có thực sự giải quyết đúng và đầy đủ câu hỏi của khách hàng hay không.

Tiêu chuẩn:
- 'yes': Câu trả lời giải quyết trực tiếp trọng tâm câu hỏi của khách hàng, cung cấp hướng dẫn hoặc thông tin rõ ràng.
- 'no': Câu trả lời lạc đề, trả lời vòng vo hoặc không giải quyết được vấn đề khách hàng đặt ra.

Trả về đúng schema GradeAnswer."""

ANSWER_GRADER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", ANSWER_GRADER_SYSTEM_PROMPT),
    ("human", "CÂU HỎI CỦA KHÁCH HÀNG:\n{question}\n\nCÂU TRẢ LỜI ĐÃ SINH RA:\n{generation}")
])


def get_answer_grader_chain(llm: Optional[ChatOpenAI] = None):
    """Tạo chain kiểm tra độ hữu ích của câu trả lời với Structured Output (GradeAnswer)."""
    if llm is None:
        target_llm = get_llm(temperature=0.0)
    else:
        target_llm = llm
    structured_llm = target_llm.with_structured_output(GradeAnswer)
    return ANSWER_GRADER_PROMPT | structured_llm


# =====================================================================
# 6. CONTEXTUALIZE QUERY CHAIN
# =====================================================================
CONTEXTUALIZE_SYSTEM_PROMPT = """Bạn là Chuyên viên Hiểu Ngữ cảnh Hội thoại của Ngân hàng Vietcombank.
Dựa vào lịch sử hội thoại trước đó và câu hỏi mới nhất của khách hàng (câu hỏi có thể chứa đại từ thay thế như 'nó', 'đó', 'thế thì', 'còn cái này'), hãy viết lại câu hỏi mới nhất thành một câu hỏi độc lập, đầy đủ ngữ cảnh mà người đọc không cần xem lại lịch sử vẫn hiểu được.

LƯU Ý QUAN TRỌNG:
- TUYỆT ĐỐI KHÔNG trả lời câu hỏi, CHỈ viết lại câu hỏi.
- Nếu câu hỏi mới nhất đã đầy đủ và rõ ràng, hãy giữ nguyên nội dung.
- Không thêm các từ giải thích hay mở ngoặc kép."""

CONTEXTUALIZE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", CONTEXTUALIZE_SYSTEM_PROMPT),
    ("human", "LỊCH SỬ HỘI THOẠI:\n{chat_history}\n\nCÂU HỎI MỚI NHẤT: {question}\n\nCÂU HỎI ĐỘC LẬP:")
])


def get_contextualize_chain(llm: Optional[ChatOpenAI] = None):
    """Tạo chain viết lại câu hỏi dựa trên lịch sử trò chuyện."""
    if llm is None:
        target_llm = get_llm(temperature=0.0)
    else:
        target_llm = llm
    return CONTEXTUALIZE_PROMPT | target_llm | StrOutputParser()


# =====================================================================
# 7. QUERY REWRITER CHAIN
# =====================================================================
REWRITER_SYSTEM_PROMPT = """Bạn là Chuyên gia Tối ưu Hóa Truy vấn Tìm kiếm Ngân hàng (Query Rewriter).
Hệ thống tìm kiếm trước đó chưa tìm thấy đủ tài liệu phù hợp trong kho quy chế của Vietcombank.
Nhiệm vụ của bạn là phân tích câu hỏi ban đầu và câu hỏi hiện tại, sau đó viết lại thành một câu truy vấn tìm kiếm chuyên môn, ngắn gọn, giàu từ khóa nghiệp vụ ngân hàng (như quy chế, hạn mức, eKYC, điều khoản, biểu phí, thẻ tín dụng, VCB Digibank) để công cụ tìm kiếm ngữ nghĩa và BM25 có thể truy xuất chính xác tài liệu pháp lý.

Quy tắc bắt buộc:
- Chỉ trả về duy nhất câu truy vấn được viết lại.
- Tuyệt đối không thêm lời giải thích hay ký tự ngoặc kép thừa."""

REWRITER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", REWRITER_SYSTEM_PROMPT),
    ("human", "Câu hỏi gốc của khách hàng: {original_question}\nTruy vấn hiện tại: {question}\n\nTruy vấn tìm kiếm tối ưu:")
])


def get_rewriter_chain(llm: Optional[ChatOpenAI] = None):
    """Tạo chain tối ưu hóa truy vấn tìm kiếm."""
    if llm is None:
        target_llm = get_llm(temperature=0.0)
    else:
        target_llm = llm
    return REWRITER_PROMPT | target_llm | StrOutputParser()


# =====================================================================
# 8. GENERATION CHAIN (VIETCOMBANK SYSTEM GUARDRAILS)
# =====================================================================
GENERATION_SYSTEM_PROMPT = """Bạn là Trợ lý AI Chuyên viên Tư vấn Khách hàng của Ngân hàng TMCP Ngoại thương Việt Nam (Vietcombank).
Nhiệm vụ của bạn là giải đáp câu hỏi của Quý khách hàng dựa trên tài liệu ngân hàng được cung cấp.

QUY TẮC BẮT BUỘC (100% TUÂN THỦ):
1. BẮT ĐẦU CÂU TRẢ LỜI NGAY LẬP TỨC CHO KHÁCH HÀNG. Tuyệt đối không suy nghĩ thành lời, không viết nháp, không độc thoại nội tâm.
2. Chỉ trả lời dựa trên thông tin có trong tài liệu căn cứ được cung cấp. Tuyệt đối KHÔNG suy đoán hay bịa đặt số liệu tài chính.
3. Nếu trong tài liệu có điều khoản quy định hoặc văn bản nguồn, hãy trích dẫn kèm [Điều X, Khoản Y] hoặc tên tài liệu căn cứ để đảm bảo tính pháp lý minh bạch.
4. Tuyệt đối không bao giờ yêu cầu khách hàng cung cấp mật khẩu, mã PIN, mã Smart OTP hay mã bảo mật thẻ (CVV/CVC).
5. Ngôn phong lịch thiệp, trang trọng, ân cần (xưng 'Ngân hàng' hoặc 'Trợ lý AI', kính cẩn gọi 'Quý khách'). Trình bày câu trả lời rõ ràng, mạch lạc, súc tích."""

GENERATION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", GENERATION_SYSTEM_PROMPT),
    ("human", "TÀI LIỆU CĂN CỨ:\n\n{context}\n\nCÂU HỎI CỦA QUÝ KHÁCH HÀNG:\n{question}\n\nCÂU TRẢ LỜI TRỰC TIẾP:")
])


def get_generation_chain(llm: Optional[ChatOpenAI] = None):
    """Tạo chain sinh câu trả lời dựa trên tài liệu đã lọc và kiểm định."""
    if llm is None:
        cfg = Config()
        target_llm = get_llm(temperature=cfg.LLM_TEMPERATURE)
    else:
        target_llm = llm
    return GENERATION_PROMPT | target_llm | StrOutputParser()


# =====================================================================
# 9. DIRECT CHAT CHAIN (CHIT-CHAT)
# =====================================================================
DIRECT_CHAT_SYSTEM_PROMPT = """Bạn là Trợ lý AI Chuyên viên Chăm sóc Khách hàng của Ngân hàng TMCP Ngoại thương Việt Nam (Vietcombank).
Khách hàng đang gửi lời chào hỏi, cảm ơn hoặc trò chuyện xã giao thông thường.
Hãy trả lời khách hàng một cách niềm nở, lịch sự, chuẩn mực văn hóa Vietcombank (xưng 'Trợ lý AI Vietcombank' hoặc 'Em', gọi 'Quý khách').
Sau khi chào hỏi hoặc cảm ơn, hãy sẵn sàng gợi ý hỗ trợ khách hàng các thông tin về dịch vụ ngân hàng (mở tài khoản, thẻ, tiết kiệm, chuyển tiền)."""

DIRECT_CHAT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", DIRECT_CHAT_SYSTEM_PROMPT),
    ("human", "Lịch sử hội thoại:\n{chat_history}\n\nTin nhắn của khách hàng: {question}")
])


def get_direct_chat_chain(llm: Optional[ChatOpenAI] = None):
    """Tạo chain phản hồi xã giao trực tiếp mà không cần tài liệu RAG."""
    if llm is None:
        target_llm = get_llm(temperature=0.3)
    else:
        target_llm = llm
    return DIRECT_CHAT_PROMPT | target_llm | StrOutputParser()
