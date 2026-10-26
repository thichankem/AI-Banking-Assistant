# -*- coding: utf-8 -*-
"""
AI Banking Agent Pipeline (LangGraph Production Entrypoint v4.0)
======================================================================
Điểm khởi chạy chính (Entrypoint) cho hệ thống Agentic RAG Ngân hàng Vietcombank:
- Tích hợp toàn diện LangGraph StateGraph v4.0 (Multi-Agent Supervisor + Self-Auditing RAG).
- Duy trì bộ nhớ phiên hội thoại bền vững (Persistent SQLite Checkpointer với Thread ID).
- Bổ sung Node 7 Thẩm định ảo giác và bắt buộc trích dẫn văn bản pháp lý [Điều X, Khoản Y].
- Mở rộng bộ công cụ nghiệp vụ: Khấu hao dư nợ giảm dần & Thẩm định sinh trắc học QĐ 2345/QĐ-NHNN.
- Tương thích 100% với giao diện người dùng và API Web hiện tại.

NGUYÊN TẮC:
- 0% REGEX LAW: Tuyệt đối không dùng regex.
- ZERO TERNARY LAW: Tuyệt đối không dùng toán tử 3 ngôi (x if c else y).
"""

from __future__ import annotations
import sys
import time
from typing import Any, Dict
from pathlib import Path

# Đảm bảo UTF-8 an toàn trên Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent_banking.config import Config
from agent_banking.pipeline import BankingRAGPipeline
from agent_banking.agents.graph import build_banking_agent_graph
from agent_banking.tracing.tracer import BankingTracer
from agent_banking.regression_test.eval_benchmark import BankingRAGEvaluator


class BankingAgentPipeline:
    """
    Bộ điều phối toàn trình AI Banking Agent ứng dụng LangGraph StateGraph.
    """

    def __init__(
        self,
        config: Config | None = None,
        use_memory: bool = False,
        sqlite_db_path: str = "agent_banking/tracing/agent_memory.db"
    ):
        if config is not None:
            self.config = config
        else:
            self.config = Config()
        self.use_memory = use_memory
        self.sqlite_db_path = sqlite_db_path

        print("=" * 70)
        print("  🚀 KHỞI TẠO LANGGRAPH AI BANKING AGENT (VIETCOMBANK PRODUCTION)")
        print("=" * 70)

        # 1. Khởi tạo RAG Pipeline nền tảng (Dual Index Qdrant + BM25)
        self.rag_pipeline = BankingRAGPipeline(config=self.config, use_memory=self.use_memory)

        # 2. Khởi tạo LangGraph StateGraph (Self-Reflective RAG & Banking Tools)
        self.graph = build_banking_agent_graph(
            pipeline_instance=self.rag_pipeline,
            config=self.config,
            sqlite_db_path=self.sqlite_db_path
        )
        print("✅ [LangGraph Engine] Đồ thị State Machine đã sẵn sàng phục vụ!")

        # 3. Khởi tạo Tracing Manager & Chấm điểm 3 chỉ số vàng (Arize Phoenix & Benchmark)
        self.tracer = BankingTracer(config=self.config)
        self.evaluator = BankingRAGEvaluator()

    def chat(
        self,
        query: str,
        session_id: str = "default_session",
        persona: str = "standard",
        tone: str = "professional",
        format_type: str = "standard"
    ) -> Dict[str, Any]:
        """
        Thực thi phiên hội thoại với LangGraph Agent kèm Session Memory (Thread ID).
        """
        start_time = time.perf_counter()
        clean_query = query.strip()

        if clean_query == "":
            return {
                "answer": "Kính chào Quý khách! Vui lòng nhập nội dung câu hỏi hoặc yêu cầu cần hỗ trợ.",
                "citations": [],
                "sources": [],
                "intent": "CHIT_CHAT",
                "sub_agent_role": "GENERAL_ADVISOR",
                "retry_count": 0,
                "audit_status": "PASS",
                "evaluation_metrics": {
                    "context_precision": 1.0,
                    "faithfulness": 1.0,
                    "answer_relevance": 1.0,
                    "overall_score": 1.0
                },
                "session_id": session_id,
                "latency_ms": 0.0,
                "execution_steps": [],
                "nodes": {}
            }

        # Thiết lập Thread ID cho Checkpointer (Quản lý bộ nhớ phiên)
        thread_config = {"configurable": {"thread_id": session_id}}

        # Trạng thái khởi tạo đưa vào StateGraph (Làm mới toàn bộ biến tạm của turn mới)
        initial_state = {
            "user_query": clean_query,
            "query": clean_query,
            "original_query": clean_query,
            "session_id": session_id,
            "intent": "ROUTING",
            "sub_agent_role": "GENERAL_ADVISOR",
            "input_guard": {"is_safe": True, "sanitized_query": clean_query},
            "output_guard": {},
            "retry_count": 0,
            "max_retries": self.config.MAX_RETRIES,
            "hallucination_retries": 0,
            "retrieved_chunks": [],
            "retrieved_docs": [],
            "tool_calls": [],
            "tool_outputs": [],
            "tool_name": "",
            "tool_output": {},
            "citations": [],
            "sources": [],
            "draft_answer": "",
            "final_answer": "",
            "audit_status": "PENDING",
            "audit_feedback": "",
            "execution_steps": []
        }

        # Kích hoạt thực thi qua LangGraph StateGraph
        try:
            final_state = self.graph.invoke(initial_state, config=thread_config)
        except Exception as e:
            print(f"❌ [LangGraph Execution Error]: {e}")
            latency_err = round((time.perf_counter() - start_time) * 1000.0, 2)
            return {
                "answer": (
                    "Hệ thống đang bảo trì một số tính năng nâng cao. "
                    "Quý khách vui lòng liên hệ hotline Vietcombank 1900 54 54 13 để được trợ giúp tức thì."
                ),
                "citations": [],
                "sources": [],
                "intent": "ERROR",
                "sub_agent_role": "GENERAL_ADVISOR",
                "retry_count": 0,
                "audit_status": "ERROR",
                "evaluation_metrics": {
                    "context_precision": 0.0,
                    "faithfulness": 0.0,
                    "answer_relevance": 0.0,
                    "overall_score": 0.0
                },
                "session_id": session_id,
                "latency_ms": latency_err,
                "execution_steps": [],
                "nodes": {}
            }

        total_latency_ms = round((time.perf_counter() - start_time) * 1000.0, 2)

        # Trích xuất dữ liệu trả về cho Frontend UI Inspector
        answer = final_state.get("final_answer")
        if answer is None or answer == "":
            answer = final_state.get("draft_answer", "")

        citations = final_state.get("citations", [])
        sources = final_state.get("sources", [])
        intent = final_state.get("intent", "RAG_SEARCH")
        sub_agent = final_state.get("sub_agent_role", "GENERAL_ADVISOR")
        retries = int(final_state.get("retry_count", 0))
        audit_stat = final_state.get("audit_status", "PASS")
        tool_outputs = final_state.get("tool_outputs", [])
        steps = final_state.get("execution_steps", [])

        docs = final_state.get("retrieved_chunks", [])
        if len(docs) == 0:
            docs = final_state.get("retrieved_docs", [])

        is_input_safe = final_state.get("is_safe", True)
        if final_state.get("input_guard", {}).get("is_safe", True) is False:
            is_input_safe = False

        status_guardrail = "blocked"
        if is_input_safe is True:
            status_guardrail = "passed"

        # Tính toán 3 chỉ số vàng (Context Precision, Faithfulness, Answer Relevance)
        eval_metrics = {
            "context_precision": 1.0,
            "faithfulness": 1.0,
            "answer_relevance": 1.0,
            "overall_score": 1.0
        }

        if is_input_safe is False:
            eval_metrics = {
                "context_precision": 0.0,
                "faithfulness": 1.0,
                "answer_relevance": 1.0,
                "overall_score": 0.6667
            }
        elif intent == "RAG_SEARCH":
            cleaned_q = clean_query.lower()
            punctuation_chars = [".", ",", ";", ":", "!", "?", "(", ")", "[", "]", "\"", "'", "\n", "\t", "*", "-", "_", "/", "\\", "|"]
            for ch in punctuation_chars:
                cleaned_q = cleaned_q.replace(ch, " ")
            raw_tokens = cleaned_q.split(" ")
            stopwords = {"cho", "tôi", "hỏi", "được", "không", "như", "thế", "nào", "vietcombank", "với", "các", "những", "có", "là", "và", "của", "tại"}
            query_keywords = []
            for t in raw_tokens:
                stripped_t = t.strip()
                if len(stripped_t) > 2 and stripped_t not in stopwords:
                    query_keywords.append(stripped_t)
            if len(query_keywords) == 0:
                query_keywords.append(clean_query.lower())

            eval_metrics = self.evaluator.evaluate_single_case(
                question=clean_query,
                retrieved_chunks=docs,
                generated_answer=answer,
                expected_keywords=query_keywords
            )

        # Ghi nhận Telemetry vết phiên vào BankingTracer & Arize Phoenix
        with self.tracer.trace_query(query=clean_query) as trace:
            for s in steps:
                trace.spans.append({
                    "name": str(s),
                    "duration_ms": 0.0,
                    "details": {
                        "sub_agent_role": sub_agent,
                        "intent": intent,
                        "audit_status": audit_stat
                    }
                })
            trace.finish(
                answer=answer,
                citations=citations,
                sources=sources,
                is_blocked=(not is_input_safe),
                guardrail_trigger=None
            )

        # Đóng gói Node Inspector Data chi tiết
        nodes_detail = {
            "00_guardrail": {
                "name": "Node 0: Banking Input Guardrail (State Reset)",
                "is_safe": is_input_safe,
                "status": status_guardrail
            },
            "01_router": {
                "name": "Node 1: Multi-Agent Supervisor Router",
                "detected_intent": intent,
                "sub_agent_role": sub_agent,
                "target_tool": final_state.get("target_tool")
            },
            "02_retrieval": {
                "name": "Node 2: Knowledge Retrieval (RAG Hybrid Search)",
                "total_documents_found": len(docs),
                "document_quality": final_state.get("doc_quality", final_state.get("document_quality", "N/A")),
                "retry_count": retries
            },
            "03_tools": {
                "name": "Node 5: Banking Action Tools",
                "executed_tools": [to.get("tool") for to in tool_outputs]
            },
            "04_audit": {
                "name": "Node 7: Post-Generation Hallucination & Citation Audit",
                "audit_status": audit_stat,
                "hallucination_retries": final_state.get("hallucination_retries", 0),
                "audit_feedback": final_state.get("audit_feedback", "")
            },
            "05_output_guard": {
                "name": "Node 8: Output Guardrail & Verification",
                "is_safe": final_state.get("output_guard", {}).get("is_safe", True),
                "flags": final_state.get("output_guard", {}).get("flags", [])
            }
        }

        return {
            "answer": answer,
            "citations": citations,
            "sources": sources,
            "intent": intent,
            "sub_agent_role": sub_agent,
            "retry_count": retries,
            "audit_status": audit_stat,
            "evaluation_metrics": eval_metrics,
            "tool_calls": tool_outputs,
            "session_id": session_id,
            "latency_ms": total_latency_ms,
            "execution_steps": steps,
            "nodes": nodes_detail
        }

    def query(self, query: str) -> Dict[str, Any]:
        """
        Hàm giao tiếp tương thích ngược 100% với pipeline.py cũ.
        """
        return self.chat(query=query)

    async def astream_chat(
        self,
        query: str,
        session_id: str = "default_session"
    ):
        """
        Stream toàn bộ sự kiện Super-steps từ Native Pregel Engine.
        """
        clean_query = query.strip()
        initial_state = {
            "user_query": clean_query,
            "query": clean_query,
            "original_query": clean_query,
            "session_id": session_id,
            "intent": "ROUTING",
            "sub_agent_role": "GENERAL_ADVISOR",
            "input_guard": {"is_safe": True, "sanitized_query": clean_query},
            "output_guard": {},
            "retry_count": 0,
            "max_retries": self.config.MAX_RETRIES,
            "hallucination_retries": 0,
            "retrieved_chunks": [],
            "retrieved_docs": [],
            "tool_calls": [],
            "tool_outputs": [],
            "tool_name": "",
            "tool_output": {},
            "citations": [],
            "sources": [],
            "draft_answer": "",
            "final_answer": "",
            "audit_status": "PENDING",
            "audit_feedback": "",
            "execution_steps": []
        }
        thread_config = {"configurable": {"thread_id": session_id}}
        async for event in self.graph.astream(initial_state, config=thread_config):
            yield event
