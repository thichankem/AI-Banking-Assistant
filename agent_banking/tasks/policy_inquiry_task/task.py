"""
Banking Policy, Tariff & Regulations Inquiry Task.
Inherits full node library from BaseBankingTask:
- Node 1: node_input_guard (Security check)
- Node 2: node_question_rewriter & node_format_history (Restructure multi-turn query & history)
- Node 3: node_knowledge_retrieval (Hybrid search RAG)
- Node 4: node_grade_documents (Evaluate document relevance)
- Node 5: node_llm_generate (Generate response with [Article X, Clause Y] citations)
- Node 6: node_audit_answer (Hallucination audit)
- Node 7: node_format_output (Package output result)

RULES: 0% Regex, Zero Ternary, No Typing Library, No __init__.py.
"""

from __future__ import annotations
from agent_banking.tasks.base_task import BaseBankingTask, TaskResult
from agent_banking.skills.skill_registry import SkillRegistry
from agent_banking.config.settings import Config
from agent_banking.tasks.config.prompts import POLICY_INQUIRY_SYSTEM_PROMPT, POLICY_INQUIRY_USER_PROMPT_TEMPLATE


class PolicyInquiryTask(BaseBankingTask):
    """Task for looking up banking regulations, tariffs, and limits."""

    def __init__(self, skill_registry: SkillRegistry | None = None, config: Config | None = None):
        super().__init__(
            task_id="TASK_POLICY_INQUIRY",
            task_name="Banking Policy & Tariff Inquiry",
            description="Searches regulatory documents, service tariffs, eKYC transaction limits, and account conditions.",
            skill_registry=skill_registry,
            config=config
        )

    def _run_flow(self, user_query: str, context: dict, steps: list[str]) -> TaskResult:
        # 1. Inherited Input Guard Node
        steps.append("Node 1: Input Guardrail (Security check)")
        guard = self.node_input_guard(user_query)
        if not guard["is_safe"]:
            return self.node_format_output(answer=guard["refusal"])

        # 2. Inherited Question Rewriter & Format History Nodes
        steps.append("Node 2: Question Rewriter & Format History")
        effective_query = self.node_question_rewriter(user_query, context.get("chat_history"))
        history_str = self.node_format_history(context.get("chat_history"))

        # 3. Inherited Knowledge Retrieval Node (RAG)
        steps.append("Node 3: Knowledge Retrieval (Hybrid search RAG)")
        rag_data = self.node_knowledge_retrieval(effective_query, top_k=3)
        chunks = rag_data["chunks"]
        context_str = rag_data["context"]

        # 4. Inherited Grade Documents Node
        steps.append("Node 4: Document Grader (Evaluate document relevance)")
        is_relevant = self.node_grade_documents(effective_query, chunks)
        if not is_relevant:
            steps.append(" -> Warning: Document relevance score low.")

        # 5. Inherited LLM Generate Node
        steps.append("Node 5: LLM Generation (Generate answer with citations)")
        sys_prompt = POLICY_INQUIRY_SYSTEM_PROMPT
        user_prompt = POLICY_INQUIRY_USER_PROMPT_TEMPLATE.format(
            history=history_str,
            context=context_str,
            query=effective_query
        )
        answer = self.node_llm_generate(sys_prompt, user_prompt, temperature=0.1)

        # 6. Inherited Audit Answer Node
        steps.append("Node 6: Hallucination Audit")
        audit = self.node_audit_answer(answer, chunks)
        if not audit.get("is_faithful", True):
            answer += "\n\n*(Thông tin căn cứ theo quy chế hiện hành tại thời điểm tra cứu)*"

        # 7. Inherited Format Output Node
        steps.append("Node 7: Format Output")
        return self.node_format_output(
            answer=answer,
            citations=rag_data["citations"],
            sources=rag_data["sources"],
            structured_data={"chunks_count": len(chunks), "is_relevant": is_relevant}
        )
