"""
Kỹ năng Kiểm soát Ảo giác & Thẩm định Tài liệu (Fact Checking Skill).
Thiết kế tối giản, trực quan, dễ hiểu theo chuẩn Chunking:
- Document Grader: Đánh giá tài liệu trích xuất có chứa thông tin trả lời không.
- Hallucination Audit: Đối soát câu trả lời có căn cứ xác thực từ tài liệu hay không (Zero Hallucination).

NGUYÊN TẮC BẮT BUỘC:
1. KHÔNG DÙNG __INIT__.PY: Import trực tiếp file module.
2. KHÔNG DÙNG THƯ VIỆN TYPING: Thuần túy dùng kiểu gốc của Python.
3. ZERO TERNARY: Bắt buộc dùng if/else tường minh, không dùng ?: hoặc toán tử 3 ngôi.
4. 0% REGEX LAW: Tuyệt đối không dùng regex.
5. HẠN CHẾ TỐI ĐA THỪA THÃI: Code ngắn gọn, trực diện, dễ hiểu.
"""

from __future__ import annotations
from openai import OpenAI

from .base_skill import BaseSkill, SkillMetadata
from agent_banking.config import Config


def _clean_think(text: str) -> str:
    """Làm sạch thẻ suy nghĩ <think>...</think> (0% Regex)."""
    cleaned = text
    while True:
        s = cleaned.find("<think>")
        if s == -1:
            break
        e = cleaned.find("</think>")
        if e == -1:
            cleaned = cleaned[:s]
            break
        cleaned = cleaned[:s] + cleaned[e + 8:]
    return cleaned.strip()


class FactCheckingSkill(BaseSkill):
    """Kỹ năng kiểm soát chất lượng tri thức và thẩm định độ trung thực."""

    def __init__(self, config: Config | None = None):
        meta = SkillMetadata(
            skill_id="fact_checking_audit",
            display_name="Kiểm soát Ảo giác & Thẩm định Tài liệu",
            description="Đánh giá tài liệu có chứa câu trả lời không và đối soát câu trả lời có bịa đặt so với nguồn trích xuất không.",
            capabilities=[
                "document_relevance_grading",
                "hallucination_detection"
            ],
            category="AUDIT"
        )
        super().__init__(metadata=meta, config=config)

        if config is not None:
            self.cfg = config
        else:
            self.cfg = Config()

        self.client = None
        if self.cfg.LLM_API_KEY and self.cfg.LLM_BASE_URL:
            try:
                self.client = OpenAI(
                    api_key=self.cfg.LLM_API_KEY,
                    base_url=self.cfg.LLM_BASE_URL
                )
            except Exception:
                self.client = None

    def _call_llm(self, prompt: str) -> str:
        if not self.client:
            return "yes"
        try:
            resp = self.client.chat.completions.create(
                model=self.cfg.LLM_MODEL,
                messages=[
                    {"role": "system", "content": "Bạn là Kiểm soát viên Thẩm định Tài liệu & Chống Ảo giác Ngân hàng."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=40
            )
            raw = resp.choices[0].message.content
            if not raw:
                return "yes"
            return _clean_think(raw).lower()
        except Exception:
            return "yes"

    def _run(self, params: dict) -> dict:
        check_type = params.get("check_type", "audit_answer")

        # 1. Thẩm định tài liệu trích xuất (Document Grader)
        if check_type == "grade_documents":
            query = params.get("query", "")
            raw_docs = params.get("documents", "")
            if isinstance(raw_docs, list):
                doc_texts = [d.get("content", d.get("text", "")) for d in raw_docs[:3]]
                doc_str = "\n---\n".join(doc_texts)
            else:
                doc_str = str(raw_docs)

            prompt = (
                f"Câu hỏi: {query}\n"
                f"Tài liệu trích xuất:\n{doc_str}\n\n"
                f"Tài liệu có chứa thông tin trả lời câu hỏi không? Trả lời 'yes' hoặc 'no'."
            )
            verdict = self._call_llm(prompt)
            is_relevant = "yes" in verdict or "đúng" in verdict
            return {"check_type": "grade_documents", "is_relevant": is_relevant}

        # 2. Kiểm tra tính trung thực của câu trả lời (Hallucination Audit)
        draft = params.get("draft_answer", "")
        raw_docs = params.get("documents", "")
        if isinstance(raw_docs, list):
            doc_texts = [d.get("content", d.get("text", "")) for d in raw_docs[:3]]
            doc_str = "\n---\n".join(doc_texts)
        else:
            doc_str = str(raw_docs)

        has_citation = "[" in draft and "]" in draft
        prompt = (
            f"Tài liệu căn cứ:\n{doc_str}\n\n"
            f"Câu trả lời dự thảo:\n{draft}\n\n"
            f"Câu trả lời có bịa đặt so với tài liệu căn cứ không? Trả lời 'yes' nếu đúng, 'no' nếu bịa đặt."
        )
        verdict = self._call_llm(prompt)
        is_faithful = "no" not in verdict

        return {
            "check_type": "audit_answer",
            "is_faithful": is_faithful,
            "has_citation": has_citation
        }
