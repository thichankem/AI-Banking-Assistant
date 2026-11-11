# -*- coding: utf-8 -*-
"""
AgentState Definition for Banking LangGraph Architecture v4.0
======================================================================
Defines the state schema passed between nodes in the StateGraph.
Adheres strictly to:
- 0% REGEX LAW: Absolutely no regular expressions.
- ZERO TERNARY LAW: Explicit if/else only.
- NO TYPING LIBRARY: Native Python dict for LangGraph StateGraph channels.
- INFINITE LOOP PROTECTION: retry_count and hallucination_retries bounds.
- MULTI-AGENT SUPERVISOR & SELF-AUDITING RAG v4.0.

MAIN FIELDS IN AGENTSTATE:
1. Query & Session Information:
   - session_id (str): Conversation session identifier.
   - user_query / query / original_query / rewritten_query (str): User query strings.
   - messages (list[dict]): Message history in [{"role": ..., "content": ...}] format.
   - chat_history (list[dict]): Historical turns stored in session memory.

2. Input Security (Node 0: input_guardrail):
   - is_safe (bool): Safety state after scanning Prompt Injection, Jailbreak, Domain.
   - guardrail_reason (str): Violation reason if triggered.
   - input_guard (dict): Detailed input audit results.

3. Multi-Agent Supervisor & Intent Routing (Node 1: supervisor_router):
   - intent (str): "CHIT_CHAT" | "BANKING_ACTION" | "RAG_SEARCH" | "REFUSAL".
   - sub_agent_role (str): "GENERAL_ADVISOR" | "CREDIT_CARD_AGENT" | "LOAN_MORTGAGE_AGENT" | "COMPLIANCE_AUDIT_AGENT".
   - target_tool (str | None): Target tool name if invoked.

4. Retrieval & Document Grading (Node 2, 3, 4: retrieve, grade_documents, rewrite_query):
   - retrieved_chunks / retrieved_docs (list[dict]): Retrieved document chunks.
   - doc_quality / document_quality (str): "RELEVANT" | "IRRELEVANT".
   - retry_count (int): Number of search rewrite attempts (maximum 2).
   - max_retries (int): Maximum retry limit.

5. Banking Tools (Node 5: banking_tool):
   - tool_name (str): Executed tool name.
   - tool_output (dict): Primary result payload.
   - tool_outputs / tool_calls (list[dict]): List of executed tool outputs.

6. Generation & Hallucination Audit (Node 6, 7: generate, audit_generation):
   - draft_answer (str): Initial draft answer before audit.
   - final_answer (str): Validated answer after audit.
   - audit_status (str): "PASS" | "FAIL".
   - audit_feedback (str): Specific audit feedback for regeneration.
   - hallucination_retries (int): Retry counter for hallucination fixes (maximum 1).

7. Output Guardrail & Legal Citations (Node 8: output_guardrail):
   - output_guard (dict): Output safety scan results.
   - citations (list[dict]): Citations list [Article X, Clause Y].
   - sources (list[str]): Referenced document sources.

8. Observability & Steps:
   - execution_steps (list[str]): Trace of executed graph nodes.
"""

from __future__ import annotations

# Native Python dictionary state schema for LangGraph StateGraph
AgentState = dict
