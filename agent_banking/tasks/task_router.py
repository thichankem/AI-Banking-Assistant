"""
Bộ định tuyến Tác vụ (Task Router - Node Đầu Tiên).
Thiết kế tối giản, trực quan, dễ hiểu theo chuẩn Chunking:
- Nhận diện Ý định (Intent) của người dùng dựa trên danh mục Kỹ năng (Skills).
- Chọn Task nghiệp vụ chính xác để kích hoạt luồng xử lý tương ứng.

NGUYÊN TẮC BẮT BUỘC:
1. KHÔNG DÙNG __INIT__.PY: Import trực tiếp file module.
2. KHÔNG DÙNG THƯ VIỆN TYPING: Thuần túy dùng kiểu gốc của Python.
3. ZERO TERNARY: Bắt buộc dùng if/else tường minh, không dùng ?: hoặc toán tử 3 ngôi.
4. 0% REGEX LAW: Tuyệt đối không dùng regex.
5. HẠN CHẾ TỐI ĐA THỪA THÃI: Code ngắn gọn, trực diện, dễ hiểu.
"""

from __future__ import annotations
from dataclasses import dataclass
from openai import OpenAI

from agent_banking.skills.skill_registry import SkillRegistry
from agent_banking.config import Config


@dataclass
class RoutingDecision:
    """Quyết định điều hướng trả về từ Intent Router Node."""
    task_id: str
    task_name: str
    target_skills: list[str]
    confidence: float
    reason: str


class TaskRouter:
    """Node đầu tiên: Phân loại ý định và chỉ định Task dựa trên bộ Skills."""

    def __init__(self, skill_registry: SkillRegistry, config: Config | None = None):
        self.registry = skill_registry

        if config is not None:
            self.config = config
        else:
            self.config = Config()

        self.client = None
        if self.config.LLM_API_KEY and self.config.LLM_BASE_URL:
            try:
                self.client = OpenAI(
                    api_key=self.config.LLM_API_KEY,
                    base_url=self.config.LLM_BASE_URL
                )
            except Exception:
                self.client = None

    def route(self, user_query: str) -> RoutingDecision:
        """Phân tích câu hỏi và chọn Task dựa trên Skills."""
        q = user_query.strip().lower()

        # 0. Yêu cầu Suy luận đa bước ReAct
        react_kw = ["react", "suy luận đa bước", "vừa vay vừa", "vừa tính lãi vừa", "vừa chuyển tiền vừa", "kết hợp nhiều", "đa bước"]
        for kw in react_kw:
            if kw in q:
                return RoutingDecision(
                    task_id="TASK_REACT_AGENT",
                    task_name="AI Agent Suy Luận & Hành Động Đa Bước (ReAct)",
                    target_skills=["financial_calculator", "compliance_qd2345_audit", "rag_knowledge_retrieval"],
                    confidence=0.98,
                    reason=f"Phát hiện yêu cầu suy luận đa bước ReAct: '{kw}'"
                )

        # 1. Sự cố khẩn cấp (Ưu tiên số 1)
        incident_kw = ["nuốt thẻ", "bị nuốt thẻ", "mất thẻ", "mất ví", "khóa thẻ khẩn cấp", "lừa đảo", "gian lận", "tra soát"]
        for kw in incident_kw:
            if kw in q:
                return RoutingDecision(
                    task_id="TASK_INCIDENT_SUPPORT",
                    task_name="Xử lý Sự cố Khẩn cấp & Cấp Phiếu Tra Soát",
                    target_skills=["incident_support_ticket"],
                    confidence=0.99,
                    reason=f"Phát hiện sự cố khẩn cấp: '{kw}'"
                )

        # 2. Sinh trắc học & Pháp chế QĐ 2345
        compliance_kw = [
            "sinh trắc học", "khuôn mặt", "facepay", "2345", "thông tư 50",
            "trên 10 triệu", "trên 10tr", "trên 20 triệu", "trên 20tr",
            "cccd gắn chip", "nfc", "thiết bị mới"
        ]
        for kw in compliance_kw:
            if kw in q:
                return RoutingDecision(
                    task_id="TASK_BIOMETRIC_COMPLIANCE",
                    task_name="Thẩm định Sinh trắc học & Pháp chế QĐ 2345",
                    target_skills=["compliance_qd2345_audit", "rag_knowledge_retrieval"],
                    confidence=0.95,
                    reason=f"Kích hoạt thẩm định QĐ 2345: '{kw}'"
                )

        # 3. Tư vấn vay vốn & Tính dư nợ giảm dần
        loan_kw = [
            "vay vốn", "muốn vay", "cần vay", "gói vay", "lãi suất vay", "vay tiền",
            "vay mua nhà", "vay mua xe", "vay ô tô", "vay tiêu dùng", "dư nợ giảm dần",
            "lịch trả nợ", "mỗi tháng trả bao nhiêu", "khấu hao khoản vay", "khoản vay", "vay "
        ]
        for kw in loan_kw:
            if kw in q:
                return RoutingDecision(
                    task_id="TASK_LOAN_ADVISORY",
                    task_name="Tư vấn Gói Vay & Lịch Trả Nợ Dư Nợ Giảm Dần",
                    target_skills=["financial_calculator", "rag_knowledge_retrieval"],
                    confidence=0.95,
                    reason=f"Kích hoạt tính toán vay vốn: '{kw}'"
                )

        # 4. Tư vấn Dịch vụ Thẻ
        card_kw = ["thẻ tín dụng", "thẻ visa", "thẻ mastercard", "thẻ jcb", "thẻ ghi nợ", "mở thẻ", "hạn mức thẻ", "phí thường niên thẻ"]
        for kw in card_kw:
            if kw in q:
                return RoutingDecision(
                    task_id="TASK_CARD_SERVICE",
                    task_name="Tư vấn & Hỗ trợ Dịch vụ Thẻ Vietcombank",
                    target_skills=["rag_knowledge_retrieval"],
                    confidence=0.92,
                    reason=f"Kích hoạt tư vấn thẻ: '{kw}'"
                )

        # 5. Chào hỏi xã giao
        chat_kw = ["xin chào", "chào bạn", "chào em", "chào anh", "chào chị", "hello", "hi", "cảm ơn", "tạm biệt"]
        for kw in chat_kw:
            if q.startswith(kw) or q == kw:
                return RoutingDecision(
                    task_id="TASK_GENERAL_CHAT",
                    task_name="Tư vấn Tổng quan & Chăm sóc Khách hàng",
                    target_skills=[],
                    confidence=0.98,
                    reason="Hội thoại chào hỏi xã giao"
                )

        # 6. Fallback LLM dựa trên Skill Catalog
        return self._route_with_llm(user_query)

    def _route_with_llm(self, query: str) -> RoutingDecision:
        """Fallback LLM phân tích sâu khi câu hỏi tự do."""
        if not self.client:
            return RoutingDecision(
                task_id="TASK_POLICY_INQUIRY",
                task_name="Tra cứu Quy chế & Biểu phí Dịch vụ Ngân hàng",
                target_skills=["rag_knowledge_retrieval", "fact_checking_audit"],
                confidence=0.85,
                reason="Định tuyến mặc định: Quy chế RAG"
            )

        skills_desc = self.registry.get_skills_prompt_description()
        system_prompt = (
            "Bạn là Node Đầu Tiên (Intent Router) của Trợ lý AI Ngân hàng Vietcombank.\n"
            f"Danh bạ Kỹ năng:\n{skills_desc}\n\n"
            "Hãy phân loại câu hỏi vào DUY NHẤT 1 trong các mã sau:\n"
            "- TASK_POLICY_INQUIRY: Tra cứu quy chế, biểu phí, hạn mức mở tài khoản eKYC.\n"
            "- TASK_LOAN_ADVISORY: Vay vốn, lãi suất vay, trả nợ, dư nợ giảm dần.\n"
            "- TASK_BIOMETRIC_COMPLIANCE: Sinh trắc học, QĐ 2345, Facepay, CCCD gắn chip.\n"
            "- TASK_CARD_SERVICE: Mở thẻ tín dụng, quyền lợi thẻ, ưu đãi thẻ.\n"
            "- TASK_INCIDENT_SUPPORT: Sự cố nuốt thẻ, mất thẻ, lừa đảo.\n"
            "- TASK_GENERAL_CHAT: Chào hỏi, cảm ơn, thông tin chung.\n"
            "- TASK_REACT_AGENT: Câu hỏi phức tạp kết hợp nhiều nghiệp vụ hoặc cần suy luận đa bước.\n"
            "Chỉ trả về duy nhất mã task. Không giải thích."
        )

        try:
            resp = self.client.chat.completions.create(
                model=self.config.LLM_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query}
                ],
                temperature=0.1,
                max_tokens=30
            )
            raw = resp.choices[0].message.content or ""
            code = raw.strip().upper()

            if "REACT" in code:
                return RoutingDecision("TASK_REACT_AGENT", "AI Agent Suy Luận & Hành Động Đa Bước (ReAct)", ["rag_knowledge_retrieval", "financial_calculator", "compliance_qd2345_audit"], 0.95, "LLM phân loại ReAct đa bước")
            if "LOAN" in code:
                return RoutingDecision("TASK_LOAN_ADVISORY", "Tư vấn Gói Vay & Lịch Trả Nợ", ["financial_calculator", "rag_knowledge_retrieval"], 0.90, "LLM phân loại vay vốn")
            if "BIOMETRIC" in code or "2345" in code:
                return RoutingDecision("TASK_BIOMETRIC_COMPLIANCE", "Thẩm định Sinh trắc học QĐ 2345", ["compliance_qd2345_audit"], 0.90, "LLM phân loại QĐ 2345")
            if "CARD" in code:
                return RoutingDecision("TASK_CARD_SERVICE", "Tư vấn Dịch vụ Thẻ", ["rag_knowledge_retrieval"], 0.90, "LLM phân loại thẻ")
            if "INCIDENT" in code:
                return RoutingDecision("TASK_INCIDENT_SUPPORT", "Xử lý Sự cố Khẩn cấp", ["incident_support_ticket"], 0.90, "LLM phân loại sự cố")
            if "CHAT" in code:
                return RoutingDecision("TASK_GENERAL_CHAT", "Tư vấn Tổng quan", [], 0.90, "LLM phân loại giao tiếp")

            return RoutingDecision("TASK_POLICY_INQUIRY", "Tra cứu Quy chế & Biểu phí", ["rag_knowledge_retrieval"], 0.85, "LLM phân loại quy chế RAG")
        except Exception:
            return RoutingDecision("TASK_POLICY_INQUIRY", "Tra cứu Quy chế & Biểu phí", ["rag_knowledge_retrieval"], 0.80, "Dự phòng: Quy chế RAG")
