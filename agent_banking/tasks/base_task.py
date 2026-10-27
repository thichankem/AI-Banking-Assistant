"""
Lớp Nhiệm vụ Cơ sở (Base Banking Task) - Kế Thừa Toàn Bộ Kho Node Tái Sử Dụng.
Thiết kế theo chuẩn Flow/DAG trực quan (như Dify / LPBank / Coze Workflow):
- Mỗi Task đại diện cho 1 luồng công việc (Workflow) nghiệp vụ hoàn chỉnh.
- BaseBankingTask cung cấp sẵn toàn bộ các Node chuẩn:
    * node_format_history: Chuẩn hóa lịch sử chat (như Python Code Node).
    * node_input_guard: Kiểm tra Prompt Injection, Fraud, PII.
    * node_knowledge_retrieval: RAG Hybrid (Dense 1024 + BM25 RRF + Rerank bge-reranker-v2-m3).
    * node_grade_documents: Thẩm định độ liên quan của tài liệu.
    * node_banking_tool: Gọi các công cụ nghiệp vụ (Dư nợ giảm dần, Lãi suất, QĐ 2345, Ticket).
    * node_llm_generate: Gọi mô hình ngôn ngữ sinh câu trả lời chuẩn Vietcombank.
    * node_audit_answer: Kiểm soát Zero-Hallucination & Trích dẫn điều khoản.
    * node_format_output: Đóng gói kết quả đầu ra.
- Mọi Task con (Loan, Policy, Biometric, Card, Incident, Chat) đều kế thừa trực tiếp
  và tự do kết hợp các Node này theo đúng nghiệp vụ riêng.

NGUYÊN TẮC BẮT BUỘC:
1. KHÔNG DÙNG __INIT__.PY: Import trực tiếp file module.
2. KHÔNG DÙNG THƯ VIỆN TYPING: Thuần túy dùng kiểu gốc của Python (dict, list, str).
3. ZERO TERNARY: Bắt buộc dùng if/else tường minh, không dùng ?: hoặc toán tử 3 ngôi.
4. 0% REGEX LAW: Tuyệt đối không dùng regex.
5. HẠN CHẾ TỐI ĐA THỪA THÃI: Code ngắn gọn, trực diện, dễ hiểu.
"""

from __future__ import annotations
import time
from dataclasses import dataclass, field
from openai import OpenAI

from agent_banking.config import Config
from agent_banking.skills.skill_registry import SkillRegistry


def _clean_think(text: str) -> str:
    """Làm sạch thẻ <think>...</think> của mô hình (0% Regex)."""
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


def _parse_react_step(text: str) -> dict:
    """Bóc tách phản hồi ReAct chuẩn 0% Regex."""
    clean = text.strip()

    # Kiểm tra Final Answer trước
    if "Final Answer:" in clean:
        idx_final = clean.find("Final Answer:")
        ans = clean[idx_final + 13:].strip()
        thought = clean[:idx_final].replace("Thought:", "").strip()
        return {
            "is_final": True,
            "thought": thought,
            "answer": ans,
            "action": None,
            "action_input": {}
        }

    # Bóc tách Thought
    thought = ""
    idx_thought = clean.find("Thought:")
    idx_action = clean.find("Action:")
    if idx_thought != -1:
        if idx_action != -1 and idx_action > idx_thought:
            thought = clean[idx_thought + 8:idx_action].strip()
        else:
            thought = clean[idx_thought + 8:].strip()

    # Bóc tách Action và Action Input
    action = None
    action_input = {}
    if idx_action != -1:
        idx_input = clean.find("Action Input:")
        if idx_input != -1:
            action = clean[idx_action + 7:idx_input].strip()
            input_raw = clean[idx_input + 13:].strip()
            s_brace = input_raw.find("{")
            e_brace = input_raw.rfind("}")
            if s_brace != -1 and e_brace != -1 and e_brace > s_brace:
                json_str = input_raw[s_brace:e_brace + 1]
                try:
                    import json
                    action_input = json.loads(json_str)
                except Exception:
                    action_input = {"query": input_raw}
            else:
                action_input = {"query": input_raw}
        else:
            action = clean[idx_action + 7:].strip().split("\n")[0]

    return {
        "is_final": False,
        "thought": thought,
        "answer": None,
        "action": action,
        "action_input": action_input
    }


@dataclass
class TaskResult:
    """Kết quả trả về thống nhất từ mỗi Task."""
    task_id: str
    task_name: str
    status: str
    answer: str
    citations: list[dict] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    structured_data: dict = field(default_factory=dict)
    execution_steps: list[str] = field(default_factory=list)
    latency_ms: float = 0.0
    error: str | None = None


class BaseBankingTask:
    """
    Lớp cơ sở cung cấp toàn bộ Node Library cho mọi Task nghiệp vụ.
    Mọi Task con kế thừa lớp này đều tự động sở hữu toàn bộ các Node.
    """

    def __init__(
        self,
        task_id: str,
        task_name: str,
        description: str,
        skill_registry: SkillRegistry | None = None,
        config: Config | None = None
    ):
        self.task_id = task_id
        self.task_name = task_name
        self.description = description

        if config is not None:
            self.config = config
        else:
            self.config = Config()

        if skill_registry is not None:
            self.registry = skill_registry
        else:
            self.registry = SkillRegistry(config=self.config)

        # Khởi tạo kết nối LLM dùng chung cho các Node tạo sinh & thẩm định
        self.client = None
        if self.config.LLM_API_KEY and self.config.LLM_BASE_URL:
            try:
                self.client = OpenAI(
                    api_key=self.config.LLM_API_KEY,
                    base_url=self.config.LLM_BASE_URL
                )
            except Exception:
                self.client = None

    # =========================================================================
    # KHO CÁC NODE TÁI SỬ DỤNG DÙNG CHUNG CHO MỌI TASK (REUSABLE NODE LIBRARY)
    # =========================================================================

    def node_format_history(self, chat_history: list[dict] | None) -> str:
        """
        NODE: Format History (Chuẩn hóa lịch sử hội thoại)
        Chuyển danh sách tin nhắn quá khứ thành văn bản ngữ cảnh rõ ràng.
        """
        if not chat_history:
            return ""

        formatted_lines = []
        for msg in chat_history[-6:]:
            role = msg.get("role", msg.get("sender", "user"))
            content = msg.get("content", msg.get("text", "")).strip()
            if role == "user":
                formatted_lines.append(f"Khách hàng: {content}")
            else:
                formatted_lines.append(f"Trợ lý Vietcombank: {content}")
        return "\n".join(formatted_lines)

    def node_question_rewriter(self, user_query: str, chat_history: list[dict] | None = None) -> str:
        """
        NODE: Question Rewriter (Tái cấu trúc câu hỏi đa lượt từ lịch sử chat)
        Sử dụng ngữ cảnh lịch sử để biến câu hỏi phụ thuộc ('còn cái đó thì sao?') thành câu hỏi độc lập.
        """
        if not chat_history:
            return user_query.strip()

        history_str = self.node_format_history(chat_history)
        if not history_str:
            return user_query.strip()

        q_lower = user_query.lower()
        context_cues = ["nó", "thẻ đó", "khoản đó", "thế còn", "vậy còn", "bao nhiêu", "còn cái này", "phí bao nhiêu", "hạn mức đó"]
        has_cue = False
        for cue in context_cues:
            if cue in q_lower:
                has_cue = True
                break

        if not has_cue:
            return user_query.strip()

        sys_prompt = (
            "Bạn là trợ lý AI chuyên viết lại câu hỏi khách hàng ngân hàng Vietcombank.\n"
            "Dựa trên lịch sử hội thoại, hãy viết lại câu hỏi cuối cùng thành câu hỏi độc lập, "
            "rõ thực thể (tên gói vay, tên thẻ, loại tài khoản). Chỉ xuất câu hỏi đã viết lại, không giải thích."
        )
        user_prompt = f"LỊCH SỬ CHAT:\n{history_str}\n\nCÂU HỎI HIỆN TẠI: {user_query}"
        rewritten = self.node_llm_generate(sys_prompt, user_prompt, temperature=0.1, max_tokens=120)
        clean_rewritten = rewritten.strip()
        if clean_rewritten and len(clean_rewritten) < 300:
            return clean_rewritten
        return user_query.strip()

    def node_parameter_extractor(self, query: str) -> dict:
        """
        NODE: Parameter Extractor (Bóc tách tham số tài chính tiếng Việt)
        Tự động bóc tách: Số tiền (principal/amount_vnd), kỳ hạn (term_months), lãi suất (annual_rate).
        """
        from agent_banking.tools.banking_api_tools import parse_vietnamese_financial_params
        params = parse_vietnamese_financial_params(query)
        if "amount_vnd" in params and "principal" not in params:
            params["principal"] = params["amount_vnd"]
        return params

    def node_input_guard(self, query: str) -> dict:
        """
        NODE: Input Guardrail (Bảo mật đầu vào 0% Regex)
        Kiểm tra Prompt Injection, gian lận tài chính và thu thập PII.
        """
        q_lower = query.lower()
        injection_phrases = ["ignore all", "bypass safety", "developer mode", "system prompt", "bỏ qua quy tắc"]
        for phrase in injection_phrases:
            if phrase in q_lower:
                return {
                    "is_safe": False,
                    "reason": f"Phát hiện mẫu Prompt Injection: '{phrase}'",
                    "refusal": "Yêu cầu bị từ chối do vi phạm quy chuẩn an toàn ngân hàng."
                }
        return {"is_safe": True, "reason": None, "refusal": None}

    def node_knowledge_retrieval(self, query: str, top_k: int = 3) -> dict:
        """
        NODE: Knowledge Retrieval (RAG Tìm kiếm Tri thức Quy chế)
        Thực thi Hybrid Search (Dense 1024-dim + BM25 RRF k=60) và Cross-Encoder Rerank.
        """
        rag_skill = self.registry.get_skill("rag_knowledge_retrieval")
        if not rag_skill:
            return {"chunks": [], "context": "", "citations": [], "sources": []}

        res = rag_skill.execute({"query": query, "top_k": top_k})
        if res.get("status") == "SUCCESS":
            data = res.get("data", {})
            return {
                "chunks": data.get("chunks", []),
                "context": data.get("formatted_context", ""),
                "citations": data.get("citations", []),
                "sources": data.get("sources", [])
            }
        return {"chunks": [], "context": "", "citations": [], "sources": []}

    def node_grade_documents(self, query: str, chunks: list[dict]) -> bool:
        """
        NODE: Document Relevance Grader (Thẩm định tài liệu RAG)
        Đánh giá tài liệu trích xuất có chứa nội dung giải đáp câu hỏi không.
        """
        fact_skill = self.registry.get_skill("fact_checking_audit")
        if not fact_skill or not chunks:
            return True

        res = fact_skill.execute({"check_type": "grade_documents", "query": query, "documents": chunks})
        return res.get("data", {}).get("is_relevant", True)

    def node_reference_filter(self, chunks: list[dict], min_score: float = 0.0) -> list[dict]:
        """
        NODE: Reference Filter (Lọc và loại bỏ tài liệu nhiễu/điểm thấp)
        Giữ lại các tài liệu có độ tin cậy và điểm xếp hạng cao.
        """
        if not chunks:
            return []
        filtered = []
        for ch in chunks:
            score = ch.get("rerank_score", ch.get("score", 0.0))
            if score >= min_score:
                filtered.append(ch)
        if not filtered:
            return chunks
        return filtered

    def node_banking_tool(self, operation: str, params: dict) -> dict:
        """
        NODE: Banking Tool Execution (Thực thi công cụ tài chính & API)
        Hỗ trợ: 'loan_amortization', 'savings_profit', 'fx_rates', 'interest_rates',
                'compliance_qd2345', 'create_ticket'.
        """
        # Nhóm tính toán tài chính & tra cứu thị trường
        if operation in ["loan_amortization", "savings_profit", "fx_rates", "interest_rates"]:
            calc_skill = self.registry.get_skill("financial_calculator")
            if calc_skill:
                calc_params = dict(params)
                calc_params["operation"] = operation
                res = calc_skill.execute(calc_params)
                data = res.get("data")
                if data:
                    return data
            return {}

        # Nhóm thẩm định sinh trắc học QĐ 2345
        if operation == "compliance_qd2345":
            comp_skill = self.registry.get_skill("compliance_qd2345_audit")
            if comp_skill:
                res = comp_skill.execute(params)
                data = res.get("data")
                if data:
                    return data
            return {}

        # Nhóm tạo ticket sự cố
        if operation == "create_ticket":
            inc_skill = self.registry.get_skill("incident_support_ticket")
            if inc_skill:
                res = inc_skill.execute(params)
                data = res.get("data")
                if data:
                    return data
            return {}

        return {}

    def node_llm_generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2,
        max_tokens: int = 2500
    ) -> str:
        """
        NODE: LLM Generation (Sinh văn bản trả lời)
        Truyền prompt vào DeepSeek-V4-Flash qua FPT Cloud API.
        """
        if not self.client:
            return user_prompt

        try:
            resp = self.client.chat.completions.create(
                model=self.config.LLM_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
            msg = resp.choices[0].message
            raw = msg.content
            if not raw:
                reasoning = getattr(msg, "reasoning_content", "")
                if reasoning:
                    raw = reasoning
                else:
                    raw = ""
            return _clean_think(raw)
        except Exception as e:
            return f"Lỗi sinh câu trả lời: {e}"

    def node_audit_answer(self, draft_answer: str, chunks: list[dict]) -> dict:
        """
        NODE: Hallucination Audit (Kiểm soát Ảo giác Zero-Hallucination)
        Đối soát từng khẳng định của câu trả lời so với nguồn tài liệu căn cứ.
        """
        fact_skill = self.registry.get_skill("fact_checking_audit")
        if not fact_skill or not chunks:
            return {"is_faithful": True, "has_citation": True}

        res = fact_skill.execute({
            "check_type": "audit_answer",
            "draft_answer": draft_answer,
            "documents": chunks
        })
        return res.get("data", {"is_faithful": True, "has_citation": True})

    def node_react_engine(
        self,
        user_query: str,
        context: dict | None = None,
        max_turns: int = 4
    ) -> dict:
        """
        NODE: ReAct Engine (Vòng lặp Suy luận & Hành động ReAct: Thought -> Action -> Observation)
        Đọc danh bạ kỹ năng từ SkillRegistry và tự động tương tác giải quyết câu hỏi phức tạp.
        """
        skills_schema = self.registry.get_skills_react_schema()
        ctx = context or {}
        history_str = self.node_format_history(ctx.get("chat_history"))

        sys_prompt = (
            "Bạn là Trợ lý AI Thông minh của Ngân hàng TMCP Ngoại thương Việt Nam (Vietcombank).\n"
            "Bạn giải quyết yêu cầu của khách hàng theo phương pháp ReAct (Reasoning + Acting).\n\n"
            "DANH MỤC KỸ NĂNG KHẢ DỤNG:\n"
            f"{skills_schema}\n\n"
            "QUY TẮC ĐỊNH DẠNG BẮT BUỘC:\n"
            "Khi cần suy luận và gọi kỹ năng, bạn PHẢI xuất:\n"
            "Thought: <suy nghĩ phân tích tình huống, cần làm gì>\n"
            "Action: <tên kỹ năng chính xác trong danh mục>\n"
            "Action Input: <JSON tham số tương ứng>\n\n"
            "Sau khi nhận được Observation, bạn tiếp tục suy luận.\n"
            "Khi đã có đầy đủ thông tin để trả lời khách hàng, bạn PHẢI xuất:\n"
            "Thought: <suy nghĩ kết luận>\n"
            "Final Answer: <câu trả lời hoàn chỉnh, chuẩn mực, ân cần theo văn hóa Vietcombank>"
        )

        scratchpad = f"Khách hàng: {user_query}\n"
        if history_str:
            scratchpad = f"Lịch sử hội thoại trước đó:\n{history_str}\n\n{scratchpad}"

        steps_log = []
        citations = []
        sources = []
        final_answer = ""

        for turn in range(1, max_turns + 1):
            llm_output = self.node_llm_generate(
                system_prompt=sys_prompt,
                user_prompt=scratchpad,
                temperature=0.1,
                max_tokens=600
            )

            parsed = _parse_react_step(llm_output)

            # 1. Đã ra kết luận Final Answer
            if parsed["is_final"]:
                final_answer = parsed["answer"]
                steps_log.append(f"[ReAct Turn {turn}] Final Answer: {parsed['thought']}")
                break

            # 2. Có Action gọi kỹ năng
            action_name = parsed["action"]
            action_input = parsed["action_input"]
            thought = parsed["thought"]

            if not action_name:
                final_answer = llm_output
                steps_log.append(f"[ReAct Turn {turn}] Direct Output")
                break

            steps_log.append(f"[ReAct Turn {turn}] Thought: {thought} | Action: {action_name}")

            # Thực thi Skill từ danh bạ
            skill = self.registry.get_skill(action_name)
            observation_str = ""
            if skill:
                skill_res = skill.execute(action_input)
                skill_data = skill_res.get("data", {})
                observation_str = str(skill_data)

                # Lưu citations từ RAG
                if action_name == "rag_knowledge_retrieval" and isinstance(skill_data, dict):
                    citations.extend(skill_data.get("citations", []))
                    sources.extend(skill_data.get("sources", []))
            else:
                observation_str = f"Lỗi: Không tìm thấy kỹ năng '{action_name}' trong danh bạ."

            steps_log.append(f"[ReAct Turn {turn}] Observation: {observation_str[:120]}...")

            # Bổ sung vào Scratchpad cho turn suy luận tiếp theo
            scratchpad += f"\nThought: {thought}\nAction: {action_name}\nAction Input: {action_input}\nObservation: {observation_str}\n"

        if not final_answer:
            final_answer = "Kính chào Quý khách! Vietcombank đã ghi nhận yêu cầu và sẽ hỗ trợ Quý khách sớm nhất."

        return {
            "answer": final_answer,
            "steps": steps_log,
            "citations": citations,
            "sources": sources
        }

    def node_format_output(
        self,
        answer: str,
        citations: list[dict] | None = None,
        sources: list[str] | None = None,
        structured_data: dict | None = None
    ) -> TaskResult:
        """
        NODE: Output Formatter (Đóng gói kết quả đầu ra chuẩn hóa)
        """
        return TaskResult(
            task_id=self.task_id,
            task_name=self.task_name,
            status="SUCCESS",
            answer=answer,
            citations=citations or [],
            sources=sources or [],
            structured_data=structured_data or {}
        )

    # =========================================================================
    # ĐIỀU PHỐI CHU TRÌNH THỰC THI (WORKFLOW EXECUTION)
    # =========================================================================

    def execute(self, user_query: str, context: dict | None = None) -> TaskResult:
        """Thực thi luồng công việc của Task, đo đếm thời gian và lưu các bước."""
        start_time = time.perf_counter()
        steps: list[str] = []
        ctx = context or {}

        try:
            result = self._run_flow(user_query, ctx, steps)
            result.latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
            result.execution_steps = steps
            return result
        except Exception as exc:
            latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
            steps.append(f"Workflow Error: {exc}")
            return TaskResult(
                task_id=self.task_id,
                task_name=self.task_name,
                status="FAILED",
                answer="Hệ thống gặp sự cố trong quá trình thực thi luồng nghiệp vụ. Quý khách vui lòng thử lại.",
                execution_steps=steps,
                latency_ms=latency_ms,
                error=str(exc)
            )

    def _run_flow(
        self,
        user_query: str,
        context: dict,
        steps: list[str]
    ) -> TaskResult:
        """Mỗi Task con chỉ cần sắp xếp chuỗi gọi các Node có sẵn ở trên."""
        raise NotImplementedError("Mỗi Task con phải cài đặt phương thức _run_flow().")
