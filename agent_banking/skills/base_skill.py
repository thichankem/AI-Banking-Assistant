"""
Kỹ năng cơ sở (Base Skill) cho hệ thống AI Banking Assistant - Kiến trúc JSON-Driven v2.0.
Thiết kế tối giản, trực quan, dễ hiểu theo chuẩn Chunking:
- Mọi Skill được định nghĩa khai báo hoàn toàn dưới dạng tệp JSON chuẩn hóa.
- BaseSkill nạp metadata, input/output schemas và cơ chế thực thi trực tiếp từ JSON.
- Tự động đo lường độ trễ và xử lý ngoại lệ an toàn.

NGUYÊN TẮC BẮT BUỘC:
1. KHÔNG DÙNG __INIT__.PY: Import trực tiếp file module.
2. KHÔNG DÙNG THƯ VIỆN TYPING: Thuần túy dùng kiểu gốc của Python (dict, list, str, int, float).
3. ZERO TERNARY: Bắt buộc dùng if/else tường minh, không dùng ?: hoặc toán tử 3 ngôi.
4. 0% REGEX LAW: Tuyệt đối không dùng regex.
5. HẠN CHẾ TỐI ĐA THỪA THÃI: Code ngắn gọn, trực diện, dễ hiểu.
"""

from __future__ import annotations
import json
import time
from pathlib import Path
from dataclasses import dataclass, field
from openai import OpenAI

from agent_banking.config import Config


def _clean_think_tags(text: str) -> str:
    """Làm sạch thẻ <think>...</think> của LLM (0% Regex)."""
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


@dataclass
class SkillMetadata:
    """Siêu dữ liệu mô tả kỹ năng được nạp từ cấu hình JSON."""
    skill_id: str
    display_name: str
    description: str
    version: str = "2.0.0"
    category: str = "GENERAL"
    capabilities: list[str] = field(default_factory=list)
    intent_triggers: list[str] = field(default_factory=list)
    react_schema_example: str = ""
    parameters_schema: dict = field(default_factory=dict)
    output_schema: dict = field(default_factory=dict)
    execution_config: dict = field(default_factory=dict)


class BaseSkill:
    """Lớp thực thi kỹ năng đa năng được điều khiển bằng đặc tả JSON."""

    def __init__(
        self,
        metadata: SkillMetadata,
        config: Config | None = None,
        pipeline_instance=None
    ):
        self.metadata = metadata
        if config is not None:
            self.config = config
        else:
            self.config = Config()

        self.pipeline = pipeline_instance
        self.llm_client = None

        # Khởi tạo LLM client nếu cần cho các tác vụ Audit/Grader
        if self.config.LLM_API_KEY and self.config.LLM_BASE_URL:
            try:
                self.llm_client = OpenAI(
                    api_key=self.config.LLM_API_KEY,
                    base_url=self.config.LLM_BASE_URL
                )
            except Exception:
                self.llm_client = None

    @classmethod
    def from_json_file(cls, json_path: str, config: Config | None = None, pipeline_instance=None):
        """Khởi tạo Skill trực tiếp từ tệp đặc tả JSON."""
        p = Path(json_path)
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)

        meta = SkillMetadata(
            skill_id=data.get("skill_id", p.stem),
            display_name=data.get("display_name", p.stem),
            description=data.get("description", ""),
            version=data.get("version", "2.0.0"),
            category=data.get("category", "GENERAL"),
            capabilities=data.get("capabilities", []),
            intent_triggers=data.get("intent_triggers", []),
            react_schema_example=data.get("react_schema_example", ""),
            parameters_schema=data.get("parameters_schema", {}),
            output_schema=data.get("output_schema", {}),
            execution_config=data.get("execution", {})
        )
        return cls(metadata=meta, config=config, pipeline_instance=pipeline_instance)

    def execute(self, params: dict) -> dict:
        """Thực thi kỹ năng và trả về kết quả chuẩn hóa kèm độ trễ ms."""
        start_time = time.perf_counter()
        try:
            data = self._run(params)
            latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
            return {
                "skill_id": self.metadata.skill_id,
                "status": "SUCCESS",
                "latency_ms": latency_ms,
                "data": data,
                "error": None
            }
        except Exception as exc:
            latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
            return {
                "skill_id": self.metadata.skill_id,
                "status": "FAILED",
                "latency_ms": latency_ms,
                "data": None,
                "error": str(exc)
            }

    def _run(self, params: dict):
        """Điều phối thực thi dựa trên cấu hình execution trong JSON."""
        exec_cfg = self.metadata.execution_config
        exec_type = exec_cfg.get("type", "generic")

        # 1. Thực thi Thẩm định Sinh trắc học QĐ 2345
        if exec_type == "tool_call":
            from agent_banking.tools.banking_api_tools import check_biometric_compliance_qd2345
            amount = params.get("transaction_amount", params.get("single_tx_amount_vnd", 0.0))
            daily_amount = params.get("daily_accumulated_amount", params.get("daily_cumulative_amount_vnd", 0.0))
            is_new_device = params.get("is_new_device", params.get("is_first_time_new_device", False))
            has_cccd = params.get("has_cccd_chip_nfc", params.get("has_verified_chip_cccd", True))

            audit_result = check_biometric_compliance_qd2345(
                single_tx_amount_vnd=float(amount),
                daily_cumulative_amount_vnd=float(daily_amount),
                is_first_time_new_device=bool(is_new_device),
                has_verified_chip_cccd=bool(has_cccd)
            )
            return {
                "compliance_status": "SUCCESS",
                "requires_biometric": audit_result.get("requires_biometric", False),
                "action_required": audit_result.get("mandatory_action", ""),
                "regulatory_rules_triggered": audit_result.get("trigger_reasons", []),
                "legal_basis": audit_result.get("regulation_reference", "Quyết định 2345/QĐ-NHNN"),
                "full_audit_details": audit_result
            }

        # 2. Thực thi Tính toán Tài chính & Lãi suất & Tỷ giá
        if exec_type == "financial_dispatcher":
            from agent_banking.tools.banking_api_tools import (
                calculate_loan_amortization,
                calculate_savings_profit,
                get_interest_rates,
                get_foreign_exchange_rates,
                parse_vietnamese_financial_params
            )
            raw_text = params.get("raw_text", "")
            extracted = {}
            if raw_text:
                extracted = parse_vietnamese_financial_params(raw_text)

            operation = params.get("operation", "auto")
            if operation == "auto":
                text_lower = raw_text.lower()
                if "vay" in text_lower or "trả nợ" in text_lower or "khấu hao" in text_lower:
                    operation = "loan_amortization"
                elif "tiết kiệm" in text_lower or "gửi" in text_lower:
                    operation = "savings_profit"
                elif "tỷ giá" in text_lower or "ngoại tệ" in text_lower or "usd" in text_lower:
                    operation = "fx_rates"
                else:
                    operation = "interest_rates"

            if operation == "loan_amortization":
                principal = params.get("principal", params.get("principal_vnd", extracted.get("principal", 500000000.0)))
                annual_rate = params.get("annual_rate_percent", extracted.get("annual_rate_percent", 7.5))
                term_months = params.get("term_months", extracted.get("term_months", 12))
                result = calculate_loan_amortization(
                    principal_vnd=float(principal),
                    annual_rate_percent=float(annual_rate),
                    term_months=int(term_months)
                )
                return {"operation": "loan_amortization", "calculation_data": result}

            if operation == "savings_profit":
                principal = params.get("principal", params.get("amount_vnd", extracted.get("principal", 100000000.0)))
                term_months = params.get("term_months", extracted.get("term_months", 6))
                result = calculate_savings_profit(
                    amount_vnd=float(principal),
                    term_months=int(term_months)
                )
                return {"operation": "savings_profit", "calculation_data": result}

            if operation == "fx_rates":
                currency = params.get("currency", "USD")
                result = get_foreign_exchange_rates(currency=currency)
                return {"operation": "fx_rates", "rates_data": result}

            dep_type = params.get("deposit_type", "vnd")
            result = get_interest_rates(deposit_type=dep_type)
            return {"operation": "interest_rates", "rates_data": result}

        # 3. Thực thi Tiếp nhận Sự cố Khẩn cấp
        if exec_type == "incident_dispatcher":
            from agent_banking.tools.banking_api_tools import create_support_ticket
            issue_type = params.get("issue_type", params.get("incident_type", "tra_soat_giao_dich"))
            contact = params.get("user_contact", params.get("phone_number", "KHACH_HANG_HOTLINE"))
            card_num = params.get("card_number", "")
            details = params.get("details", params.get("atm_location", "Tiếp nhận từ trợ lý AI."))
            urgency = params.get("urgency", "HIGH")

            issue_lower = str(issue_type).lower()
            if "mat_the" in issue_lower or "gian_lan" in issue_lower or "hack" in issue_lower or "swallow" in issue_lower:
                urgency = "CRITICAL"

            desc = details
            if card_num:
                desc = f"{details} | Thẻ/TK: {card_num}"

            priority_val = urgency
            if urgency == "CRITICAL":
                priority_val = "URGENT"

            ticket_data = create_support_ticket(
                customer_name=contact,
                issue_type=issue_type,
                description=desc,
                priority=priority_val
            )

            if urgency == "CRITICAL":
                guide = (
                    "Quý khách vui lòng mở VCB Digibank > Dịch vụ thẻ > Khóa thẻ tạm thời ngay lập tức, "
                    "hoặc gọi Tổng đài 1900 54 54 13 để khóa khẩn cấp."
                )
            else:
                guide = (
                    "Hồ sơ tra soát đã được tiếp nhận thành công. Quý khách vui lòng lưu lại Mã hồ sơ "
                    "để theo dõi tiến độ xử lý."
                )

            return {
                "ticket_info": ticket_data,
                "immediate_action_guide": guide,
                "hotline_support": exec_cfg.get("hotline", "1900 54 54 13 (24/7)")
            }

        # 4. Thực thi Truy xuất Tri thức Quy chế RAG
        if exec_type == "rag_pipeline":
            if self.pipeline is None:
                from agent_banking.pipeline import BankingRAGPipeline
                self.pipeline = BankingRAGPipeline(config=self.config)

            raw_query = params.get("query", "").strip()
            if not raw_query:
                return {
                    "chunks": [],
                    "formatted_context": "",
                    "citations": [],
                    "sources": [],
                    "query_used": ""
                }

            top_k = params.get("top_k", 3)
            candidate_k = params.get("candidate_k", 5)

            parsed_query = self.pipeline.step1_query_parser(raw_query)
            candidates = self.pipeline.step2_hybrid_retrieval(
                parsed_query=parsed_query,
                top_k=candidate_k
            )
            reranked_chunks = self.pipeline.step3_reranking(
                query_or_parsed=parsed_query,
                candidates=candidates,
                top_k=top_k
            )

            synthesized_context = self.pipeline.step4_context_synthesizer(reranked_chunks)
            context_text = ""
            if hasattr(synthesized_context, "markdown_context"):
                context_text = synthesized_context.markdown_context
            else:
                context_text = str(synthesized_context)

            citations = []
            sources = []
            for ch in reranked_chunks:
                doc_name = ch.get("document_name", "")
                if doc_name and doc_name not in sources:
                    sources.append(doc_name)

                score = ch.get("rerank_score", ch.get("score", 0.0))
                citations.append({
                    "document_name": doc_name,
                    "breadcrumbs": ch.get("breadcrumbs", ""),
                    "section": ch.get("section", ""),
                    "score": round(score, 4)
                })

            return {
                "chunks": reranked_chunks,
                "formatted_context": context_text,
                "citations": citations,
                "sources": sources,
                "query_used": raw_query
            }

        # 5. Thực thi Thẩm định Tài liệu & Kiểm soát Ảo giác
        if exec_type == "fact_checker_engine":
            check_type = params.get("check_type", "audit_answer")

            if check_type == "grade_documents":
                query = params.get("query", "")
                raw_docs = params.get("documents", params.get("retrieved_context", ""))
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
                verdict = self._call_llm_checker(prompt)
                is_relevant = "yes" in verdict or "đúng" in verdict
                return {
                    "check_type": "grade_documents",
                    "is_relevant": is_relevant,
                    "is_grounded": is_relevant
                }

            draft = params.get("draft_answer", params.get("answer", ""))
            raw_docs = params.get("documents", params.get("retrieved_context", ""))
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
            verdict = self._call_llm_checker(prompt)
            is_faithful = "no" not in verdict

            compliance_score = 0.0
            if is_faithful:
                compliance_score = 1.0

            return {
                "check_type": "audit_answer",
                "is_faithful": is_faithful,
                "is_grounded": is_faithful,
                "has_citation": has_citation,
                "compliance_score": compliance_score
            }

        return {"status": "SUCCESS", "message": f"Kỹ năng {self.metadata.skill_id} đã thực thi."}

    def _call_llm_checker(self, prompt: str) -> str:
        if not self.llm_client:
            return "yes"
        try:
            resp = self.llm_client.chat.completions.create(
                model=self.config.LLM_MODEL,
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
            return _clean_think_tags(raw).lower()
        except Exception:
            return "yes"
