"""
Danh bạ Kỹ năng Nghiệp vụ Ngân hàng (Skill Registry).
Thiết kế tối giản, trực quan, dễ hiểu theo chuẩn Chunking:
- Đăng ký và quản lý tập trung toàn bộ các Skills.
- Cung cấp mô tả kỹ năng cho Intent Router để tự động chọn Task.

NGUYÊN TẮC BẮT BUỘC:
1. KHÔNG DÙNG __INIT__.PY: Import trực tiếp file module.
2. KHÔNG DÙNG THƯ VIỆN TYPING: Thuần túy dùng kiểu gốc của Python.
3. ZERO TERNARY: Bắt buộc dùng if/else tường minh, không dùng ?: hoặc toán tử 3 ngôi.
4. 0% REGEX LAW: Tuyệt đối không dùng regex.
5. HẠN CHẾ TỐI ĐA THỪA THÃI: Code ngắn gọn, trực diện, dễ hiểu.
"""

from __future__ import annotations

from .base_skill import BaseSkill
from .rag_knowledge_skill import RAGKnowledgeSkill
from .financial_calculator_skill import FinancialCalculatorSkill
from .compliance_qd2345_skill import ComplianceQD2345Skill
from .incident_support_skill import IncidentSupportSkill
from .fact_checking_skill import FactCheckingSkill
from agent_banking.config import Config


class SkillRegistry:
    """Trung tâm quản lý danh mục Kỹ năng nghiệp vụ."""

    def __init__(self, pipeline_instance=None, config: Config | None = None):
        if config is not None:
            self.config = config
        else:
            self.config = Config()

        self.skills: dict[str, BaseSkill] = {}

        # Đăng ký các kỹ năng chuẩn hóa
        self.register(RAGKnowledgeSkill(pipeline_instance=pipeline_instance, config=self.config))
        self.register(FinancialCalculatorSkill(config=self.config))
        self.register(ComplianceQD2345Skill(config=self.config))
        self.register(IncidentSupportSkill(config=self.config))
        self.register(FactCheckingSkill(config=self.config))

    def register(self, skill: BaseSkill) -> None:
        """Đăng ký một skill vào danh bạ."""
        self.skills[skill.metadata.skill_id] = skill

    def get_skill(self, skill_id: str) -> BaseSkill | None:
        """Lấy đối tượng skill theo mã định danh."""
        return self.skills.get(skill_id)

    def list_skills(self) -> list[dict]:
        """Trả về danh sách tóm tắt tất cả các kỹ năng."""
        records = []
        for s in self.skills.values():
            meta = s.metadata
            records.append({
                "skill_id": meta.skill_id,
                "display_name": meta.display_name,
                "description": meta.description,
                "category": meta.category,
                "triggers": meta.intent_triggers
            })
        return records

    def get_skills_prompt_description(self) -> str:
        """Xuất mô tả kỹ năng phục vụ Router phân loại Intent."""
        lines = []
        for s in self.skills.values():
            m = s.metadata
            lines.append(f"- [{m.skill_id}] ({m.display_name}): {m.description}")
        return "\n".join(lines)

    def get_skills_react_schema(self) -> str:
        """Xuất danh mục kỹ năng kèm tham số chi tiết phục vụ vòng lặp ReAct."""
        lines = []
        for s in self.skills.values():
            m = s.metadata
            lines.append(f"Kỹ năng: {m.skill_id} ({m.display_name})")
            lines.append(f"Mô tả: {m.description}")
            if m.skill_id == "rag_knowledge_retrieval":
                lines.append("Tham số JSON: {\"query\": \"từ khóa tra cứu\", \"top_k\": 3}")
            elif m.skill_id == "financial_calculator":
                lines.append("Tham số JSON: {\"operation\": \"loan_amortization\"|\"savings_profit\"|\"fx_rates\", \"principal\": <số tiền>, \"term_months\": <tháng>, \"annual_rate_percent\": <lãi>}")
            elif m.skill_id == "compliance_qd2345_audit":
                lines.append("Tham số JSON: {\"transaction_amount\": <số tiền>, \"daily_accumulated_amount\": <tổng ngày>, \"is_new_device\": false/true}")
            elif m.skill_id == "incident_support_ticket":
                lines.append("Tham số JSON: {\"issue_type\": \"nuot_the_atm\"|\"mat_the_khoa_the\", \"user_contact\": \"<sdt>\", \"details\": \"<chi tiết>\"}")
            elif m.skill_id == "fact_checking_audit":
                lines.append("Tham số JSON: {\"check_type\": \"grade_documents\"|\"audit_answer\", \"query\": \"<câu hỏi>\"}")
            lines.append("")
        return "\n".join(lines)
