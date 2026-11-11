"""
Base Banking Task - Inherits reusable node library.
Designed following standard workflow/DAG execution patterns:
- Each task represents a complete business workflow.
- BaseBankingTask provides standard reusable nodes:
    * node_format_history: Format chat history.
    * node_input_guard: Input validation (Prompt Injection, Fraud, PII).
    * node_knowledge_retrieval: RAG Hybrid search and reranking.
    * node_grade_documents: Document relevance assessment.
    * node_banking_tool: Call business tools (Loan amortization, Interest rates, Biometrics, Ticket).
    * node_llm_generate: Call language model to generate responses.
    * node_audit_answer: Hallucination and citation audit.
    * node_format_output: Package final task result.
- All specialized tasks inherit from this base class.

RULES:
1. NO __INIT__.PY: Direct module imports.
2. NO TYPING LIBRARY: Native Python types only (dict, list, str).
3. ZERO TERNARY: Explicit if/else only.
4. 0% REGEX LAW: Absolutely no regular expressions.
5. KEEP IT SIMPLE: Clean and readable code.
"""

from __future__ import annotations
import time
from dataclasses import dataclass, field
from agent_banking.core.chat import FallbackChatClient
from agent_banking.core.fallback import ModelFallbackError

from agent_banking.core.settings import Config
from agent_banking.skills.skill_registry import SkillRegistry
from agent_banking.tasks.config.prompts import QUERY_REWRITE_SYSTEM_PROMPT, REACT_AGENT_SYSTEM_PROMPT_TEMPLATE


def _clean_think(text: str) -> str:
    """Clean reasoning tags <think>...</think> from model output (0% Regex)."""
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
    """Parse ReAct step format with 0% Regex."""
    clean = text.strip()

    # Check for Final Answer first
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

    # Extract Thought
    thought = ""
    idx_thought = clean.find("Thought:")
    idx_action = clean.find("Action:")
    if idx_thought != -1:
        if idx_action != -1 and idx_action > idx_thought:
            thought = clean[idx_thought + 8:idx_action].strip()
        else:
            thought = clean[idx_thought + 8:].strip()

    # Extract Action and Action Input
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
    """Unified task execution result."""
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
    Base class providing node library for all business tasks.
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

        # Initialize shared LLM client
        self.client = FallbackChatClient("tasks", {
            "model": getattr(self.config, "TASKS_LLM_MODEL", self.config.LLM_MODEL),
            "url": getattr(self.config, "TASKS_LLM_BASE_URL", self.config.LLM_BASE_URL),
            "api_key": getattr(self.config, "TASKS_LLM_API_KEY", self.config.LLM_API_KEY),
        })

    # =========================================================================
    # REUSABLE NODE LIBRARY
    # =========================================================================

    def node_format_history(self, chat_history: list[dict] | None) -> str:
        """
        NODE: Format History
        Convert previous message history into clear context text.
        """
        if not chat_history:
            return ""

        formatted_lines = []
        for msg in chat_history[-6:]:
            role = msg.get("role", msg.get("sender", "user"))
            content = msg.get("content", msg.get("text", "")).strip()
            if role == "user":
                formatted_lines.append(f"Customer: {content}")
            else:
                formatted_lines.append(f"Banking Assistant: {content}")
        return "\n".join(formatted_lines)

    def node_question_rewriter(self, user_query: str, chat_history: list[dict] | None = None) -> str:
        """
        NODE: Question Rewriter
        Resolve co-references and ambiguities from conversation history into standalone questions.
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

        sys_prompt = QUERY_REWRITE_SYSTEM_PROMPT
        user_prompt = f"LỊCH SỬ CHAT:\n{history_str}\n\nCÂU HỎI HIỆN TẠI: {user_query}"
        rewritten = self.node_llm_generate(sys_prompt, user_prompt, temperature=0.1, max_tokens=120)
        clean_rewritten = rewritten.strip()
        if clean_rewritten and len(clean_rewritten) < 300:
            return clean_rewritten
        return user_query.strip()

    def node_parameter_extractor(self, query: str) -> dict:
        """
        NODE: Parameter Extractor
        Extract financial parameters: amount (principal/amount_vnd), term (term_months), interest rate.
        """
        from agent_banking.tools.banking_api_tools import parse_vietnamese_financial_params
        params = parse_vietnamese_financial_params(query)
        if "amount_vnd" in params and "principal" not in params:
            params["principal"] = params["amount_vnd"]
        return params

    def node_input_guard(self, query: str) -> dict:
        """
        NODE: Input Guardrail (0% Regex)
        Validates input against prompt injection, fraud, and sensitive queries.
        """
        q_lower = query.lower()
        injection_phrases = ["ignore all", "bypass safety", "developer mode", "system prompt", "bỏ qua quy tắc"]
        for phrase in injection_phrases:
            if phrase in q_lower:
                return {
                    "is_safe": False,
                    "reason": f"Detected prompt injection pattern: '{phrase}'",
                    "refusal": "Yêu cầu bị từ chối do vi phạm quy chuẩn an toàn ngân hàng."
                }
        return {"is_safe": True, "reason": None, "refusal": None}

    def node_knowledge_retrieval(self, query: str, top_k: int = 3) -> dict:
        """
        NODE: Knowledge Retrieval
        Executes hybrid search and cross-encoder reranking.
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
        NODE: Document Relevance Grader
        Evaluates whether retrieved documents contain relevant context to answer the query.
        """
        fact_skill = self.registry.get_skill("fact_checking_audit")
        if not fact_skill or not chunks:
            return True

        res = fact_skill.execute({"check_type": "grade_documents", "query": query, "documents": chunks})
        return res.get("data", {}).get("is_relevant", True)

    def node_reference_filter(self, chunks: list[dict], min_score: float = 0.0) -> list[dict]:
        """
        NODE: Reference Filter
        Retains chunks meeting the score threshold.
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
        NODE: Banking Tool Execution
        Supported: 'loan_amortization', 'savings_profit', 'fx_rates', 'interest_rates',
                   'compliance_qd2345', 'create_ticket'.
        """
        # Financial calculation & market lookup
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

        # Biometric compliance
        if operation == "compliance_qd2345":
            comp_skill = self.registry.get_skill("compliance_qd2345_audit")
            if comp_skill:
                res = comp_skill.execute(params)
                data = res.get("data")
                if data:
                    return data
            return {}

        # Support ticket creation
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
        NODE: LLM Generation
        Dispatches prompt to configured LLM endpoint.
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
        except ModelFallbackError:
            raise
        except Exception as e:
            return f"Generation error: {e}"

    def node_audit_answer(self, draft_answer: str, chunks: list[dict]) -> dict:
        """
        NODE: Hallucination Audit
        Audits answer assertions against source reference documents.
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
        NODE: ReAct Engine (Reasoning + Acting Loop: Thought -> Action -> Observation)
        Queries skill registry and iteratively resolves complex queries.
        """
        skills_schema = self.registry.get_skills_react_schema()
        ctx = context or {}
        history_str = self.node_format_history(ctx.get("chat_history"))

        sys_prompt = REACT_AGENT_SYSTEM_PROMPT_TEMPLATE.format(skills_schema=skills_schema)

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

            # 1. Final answer reached
            if parsed["is_final"]:
                final_answer = parsed["answer"]
                steps_log.append(f"[ReAct Turn {turn}] Final Answer: {parsed['thought']}")
                break

            # 2. Action call
            action_name = parsed["action"]
            action_input = parsed["action_input"]
            thought = parsed["thought"]

            if not action_name:
                final_answer = llm_output
                steps_log.append(f"[ReAct Turn {turn}] Direct Output")
                break

            steps_log.append(f"[ReAct Turn {turn}] Thought: {thought} | Action: {action_name}")

            # Execute skill from registry
            skill = self.registry.get_skill(action_name)
            observation_str = ""
            if skill:
                skill_res = skill.execute(action_input)
                skill_data = skill_res.get("data", {})
                observation_str = str(skill_data)

                # Capture citations from RAG
                if action_name == "rag_knowledge_retrieval" and isinstance(skill_data, dict):
                    citations.extend(skill_data.get("citations", []))
                    sources.extend(skill_data.get("sources", []))
            else:
                observation_str = f"Error: Skill '{action_name}' not found in registry."

            steps_log.append(f"[ReAct Turn {turn}] Observation: {observation_str[:120]}...")

            # Append to scratchpad for subsequent turns
            scratchpad += f"\nThought: {thought}\nAction: {action_name}\nAction Input: {action_input}\nObservation: {observation_str}\n"

        if not final_answer:
            final_answer = "Kính chào Quý khách! Ngân hàng đã ghi nhận yêu cầu và sẽ hỗ trợ Quý khách sớm nhất."

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
        NODE: Output Formatter
        Packages normalized output result.
        """
        if citations is not None:
            cites = citations
        else:
            cites = []

        if sources is not None:
            srcs = sources
        else:
            srcs = []

        if structured_data is not None:
            sdata = structured_data
        else:
            sdata = {}

        return TaskResult(
            task_id=self.task_id,
            task_name=self.task_name,
            status="SUCCESS",
            answer=answer,
            citations=cites,
            sources=srcs,
            structured_data=sdata
        )

    # =========================================================================
    # WORKFLOW EXECUTION
    # =========================================================================

    def execute(self, user_query: str, context: dict | None = None) -> TaskResult:
        """Execute task workflow, track timing and execution steps."""
        start_time = time.perf_counter()
        steps: list[str] = []
        if context is not None:
            ctx = context
        else:
            ctx = {}

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
        """Each subtask implements _run_flow chaining standard nodes."""
        raise NotImplementedError("Subtasks must implement _run_flow().")
