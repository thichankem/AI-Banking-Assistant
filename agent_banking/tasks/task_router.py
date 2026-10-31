"""
Bộ định tuyến Tác vụ (Task Router - Node Đầu Tiên - Kiến trúc JSON-Driven).
Thiết kế tối giản, trực quan, dễ hiểu theo chuẩn Chunking:
- Tự động nạp quy tắc định tuyến ý định từ tệp JSON cấu hình (tasks_manifest.json).
- Nhận diện Ý định (Intent) của người dùng dựa trên danh mục Kỹ năng (Skills) và Task.
- Hỗ trợ Fallback phân loại sâu bằng LLM khi gặp câu hỏi tự do.

NGUYÊN TẮC BẮT BUỘC:
1. KHÔNG DÙNG __INIT__.PY: Import trực tiếp file module.
2. KHÔNG DÙNG THƯ VIỆN TYPING: Thuần túy dùng kiểu gốc của Python.
3. ZERO TERNARY: Bắt buộc dùng if/else tường minh, không dùng ?: hoặc toán tử 3 ngôi.
4. 0% REGEX LAW: Tuyệt đối không dùng regex.
5. HẠN CHẾ TỐI ĐA THỪA THÃI: Code ngắn gọn, trực diện, dễ hiểu.
"""

from __future__ import annotations
import json
from pathlib import Path
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
    """Node đầu tiên: Phân loại ý định và chỉ định Task dựa trên bộ cấu hình JSON."""

    def __init__(self, skill_registry: SkillRegistry, config: Config | None = None, manifest_path: str | Path | None = None):
        self.registry = skill_registry

        if config is not None:
            self.config = config
        else:
            self.config = Config()

        if manifest_path is not None:
            self.manifest_path = Path(manifest_path)
        else:
            self.manifest_path = Path(__file__).resolve().parent / "tasks_manifest.json"

        self.routing_rules: list[dict] = []
        self._load_routing_manifest()

        self.client = None
        if self.config.LLM_API_KEY and self.config.LLM_BASE_URL:
            try:
                self.client = OpenAI(
                    api_key=self.config.LLM_API_KEY,
                    base_url=self.config.LLM_BASE_URL
                )
            except Exception:
                self.client = None

    def _load_routing_manifest(self) -> None:
        """Nạp các quy tắc điều hướng từ JSON manifest."""
        if not self.manifest_path.exists():
            return

        try:
            with open(self.manifest_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            rules = data.get("routing_rules", [])
            # Sắp xếp theo priority tăng dần (0 là ưu tiên cao nhất)
            self.routing_rules = sorted(rules, key=lambda x: x.get("priority", 99))
        except Exception as e:
            print(f"⚠️ Không thể đọc tasks_manifest.json: {e}")
            self.routing_rules = []

    def route(self, user_query: str) -> RoutingDecision:
        """Phân tích câu hỏi và chọn Task dựa trên bộ quy tắc JSON."""
        q = user_query.strip().lower()

        # 1. Duyệt qua từng quy tắc được định nghĩa trong JSON
        for rule in self.routing_rules:
            keywords = rule.get("keywords", [])
            task_id = rule.get("task_id", "")
            task_name = rule.get("task_name", "")
            target_skills = rule.get("target_skills", [])
            confidence = float(rule.get("confidence", 0.95))

            if task_id == "TASK_GENERAL_CHAT":
                for kw in keywords:
                    if q.startswith(kw) or q == kw:
                        return RoutingDecision(
                            task_id=task_id,
                            task_name=task_name,
                            target_skills=target_skills,
                            confidence=confidence,
                            reason=f"Hội thoại chào hỏi xã giao khớp từ khóa: '{kw}'"
                        )
            else:
                for kw in keywords:
                    if kw in q:
                        return RoutingDecision(
                            task_id=task_id,
                            task_name=task_name,
                            target_skills=target_skills,
                            confidence=confidence,
                            reason=f"Khớp từ khóa nghiệp vụ JSON: '{kw}'"
                        )

        # 2. Fallback LLM dựa trên Skill Catalog nếu không khớp từ khóa
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
            raw = resp.choices[0].message.content
            if not raw:
                raw = ""
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
