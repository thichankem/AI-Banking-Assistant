# -*- coding: utf-8 -*-
"""Banking ReAct chat entrypoint for synchronous callers and SSE."""

import sys
import time
from threading import Lock
from pathlib import Path

# Ensure UTF-8 on Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent_banking.core.settings import Config
from agent_banking.pipeline import BankingRAGPipeline
from agent_banking.langgraph_agent.graph import build_banking_agent_graph


class BankingAgentPipeline:
    """
    Enterprise AI Banking Agent orchestrator using LangGraph StateGraph.
    """

    def __init__(
        self,
        config: Config | None = None,
        use_memory: bool = False,
        sqlite_db_path: str | None = None,
        rag_pipeline: BankingRAGPipeline | None = None
    ):
        if config is not None:
            self.config = config
        else:
            self.config = Config()
        self.use_memory = use_memory
        if sqlite_db_path is not None:
            self.sqlite_db_path = sqlite_db_path
        else:
            self.sqlite_db_path = str(Path(__file__).resolve().parent / "tracing" / "agent_memory.db")

        print("=" * 70)
        print("  INITIALIZING BANKING REACT AGENT")
        print("=" * 70)

        # 1. Initialize foundation RAG Pipeline (Dual Index Vector + BM25)
        self.rag_pipeline = rag_pipeline
        if self.rag_pipeline is None:
            self.rag_pipeline = BankingRAGPipeline(config=self.config, use_memory=self.use_memory)
        self._session_locks = [Lock() for _ in range(64)]

        # 2. Initialize the ReAct graph and session checkpoint
        self.graph = build_banking_agent_graph(
            pipeline_instance=self.rag_pipeline,
            config=self.config,
            sqlite_db_path=self.sqlite_db_path
        )
        print("[LangGraph Engine] State Machine graph initialized successfully.")

        self.tracer = self.rag_pipeline.tracer

    def build_initial_state(
        self,
        query: str,
        session_id: str = "default_session",
        persona: str = "advisor",
        tone: str = "formal",
        format_type: str = "standard",
        custom_persona: str = ""
    ) -> dict:
        """Build the initial StateGraph payload shared by sync and streaming chat entrypoints."""
        clean_query = str(query).strip()
        return {
            "user_query": clean_query,
            "session_id": session_id,
            "persona": persona,
            "tone": tone,
            "format_type": format_type,
            "custom_persona": custom_persona,
        }

    def _session_lock(self, session_id: str):
        return self._session_locks[hash(session_id) % len(self._session_locks)]

    def _restore_history(self, state: dict, thread_config: dict) -> None:
        # The graph uses a root dict channel, which is replaced by each new request.
        previous = self.graph.get_state(thread_config).values
        state["chat_history"] = previous.get("chat_history", [])

    def chat(
        self,
        query: str,
        session_id: str = "default_session",
        persona: str = "advisor",
        tone: str = "formal",
        format_type: str = "standard",
        custom_persona: str = ""
    ) -> dict:
        """
        Execute chat turn with LangGraph Agent and persistent session memory (Thread ID).
        """
        start_time = time.perf_counter()
        clean_query = query.strip()

        if clean_query == "":
            return {
                "answer": "Welcome! Please enter your question or inquiry to get started.",
                "citations": [],
                "sources": [],
                "intent": "CHIT_CHAT",
                "retry_count": 0,
                "audit_status": "PASS",
                "evaluation_metrics": None,
                "session_id": session_id,
                "latency_ms": 0.0,
                "execution_steps": [],
                "nodes": {}
            }

        # Configure Thread ID for Checkpointer (session memory)
        thread_config = {"configurable": {"thread_id": session_id}}

        # Initial state payload for StateGraph
        initial_state = self.build_initial_state(
            query=clean_query,
            session_id=session_id,
            persona=persona,
            tone=tone,
            format_type=format_type,
            custom_persona=custom_persona
        )

        # Execute through LangGraph StateGraph
        try:
            with self._session_lock(session_id):
                self._restore_history(initial_state, thread_config)
                final_state = self.graph.invoke(initial_state, config=thread_config)
        except Exception as e:
            print(f"[LangGraph Execution Error]: {e}")
            latency_err = round((time.perf_counter() - start_time) * 1000.0, 2)
            return {
                "answer": (
                    "The system is currently undergoing scheduled maintenance on some advanced features. "
                    "Please contact customer service for immediate assistance."
                ),
                "citations": [],
                "sources": [],
                "intent": "ERROR",
                "retry_count": 0,
                "audit_status": "ERROR",
                "evaluation_metrics": None,
                "session_id": session_id,
                "latency_ms": latency_err,
                "execution_steps": [],
                "nodes": {}
            }

        total_latency_ms = round((time.perf_counter() - start_time) * 1000.0, 2)

        # Extract output fields for Frontend UI Inspector
        answer = final_state.get("final_answer", "")

        citations = final_state.get("citations", [])
        sources = final_state.get("sources", [])
        intent = final_state.get("intent", "RAG_SEARCH")
        retries = int(final_state.get("retry_count", 0))
        audit_stat = final_state.get("audit_status", "PASS")
        tool_outputs = final_state.get("tool_outputs", [])
        steps = final_state.get("execution_steps", [])

        docs = final_state.get("retrieved_chunks", [])

        is_input_safe = final_state.get("is_safe", True)
        if final_state.get("input_guard", {}).get("is_safe", True) is False:
            is_input_safe = False

        status_guardrail = "blocked"
        if is_input_safe is True:
            status_guardrail = "passed"

        trace_query = "[blocked]"
        if is_input_safe:
            trace_query = final_state["user_query"]
        with self.tracer.trace_query(query=trace_query) as trace:
            for s in steps:
                trace.spans.append({
                    "name": str(s),
                    "duration_ms": 0.0,
                    "details": {
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

        # Package node inspection details
        nodes_detail = {
            "00_guardrail": {
                "name": "Node 0: Banking Input Guardrail (State Reset)",
                "is_safe": is_input_safe,
                "status": status_guardrail
            },
            "01_react": {
                "name": "ReAct Decisions",
                "detected_intent": intent,
                "last_action": final_state.get("last_action")
            },
            "02_retrieval": {
                "name": "Knowledge Retrieval (Hybrid RAG)",
                "total_documents_found": len(docs),
                "search_count": final_state.get("search_count", 0),
                "retry_count": retries
            },
            "03_tools": {
                "name": "Banking Tools and Skills",
                "executed_tools": [to.get("tool") for to in tool_outputs]
            },
            "04_audit": {
                "name": "Citation Audit",
                "audit_status": audit_stat,
                "hallucination_retries": final_state.get("hallucination_retries", 0),
                "audit_feedback": final_state.get("audit_feedback", "")
            },
            "05_output_guard": {
                "name": "Output Guardrail",
                "is_safe": final_state.get("output_guard", {}).get("is_safe", True),
                "flags": final_state.get("output_guard", {}).get("flags", [])
            }
        }

        return {
            "answer": answer,
            "citations": citations,
            "sources": sources,
            "intent": intent,
            "retry_count": retries,
            "audit_status": audit_stat,
            "evaluation_metrics": None,
            "tool_calls": tool_outputs,
            "session_id": session_id,
            "latency_ms": total_latency_ms,
            "execution_steps": steps,
            "nodes": nodes_detail
        }

    def query(self, query: str) -> dict:
        """
        Backward-compatible query method.
        """
        return self.chat(query=query)

    async def astream_chat(
        self,
        query: str,
        session_id: str = "default_session",
        persona: str = "advisor",
        tone: str = "formal",
        format_type: str = "standard",
        custom_persona: str = ""
    ):
        """
        Stream all super-step events from graph execution without blocking the event loop,
        staying compatible with synchronous checkpointers such as SqliteSaver.
        """
        import asyncio
        import contextlib
        import threading

        initial_state = self.build_initial_state(
            query=query,
            session_id=session_id,
            persona=persona,
            tone=tone,
            format_type=format_type,
            custom_persona=custom_persona
        )
        thread_config = {"configurable": {"thread_id": session_id}}

        loop = asyncio.get_running_loop()
        queue = asyncio.Queue()
        stop_event = threading.Event()
        sentinel = object()

        def _worker():
            try:
                with self._session_lock(session_id):
                    if stop_event.is_set():
                        return
                    self._restore_history(initial_state, thread_config)
                    with contextlib.closing(self.graph.stream(initial_state, config=thread_config)) as events:
                        while not stop_event.is_set():
                            event = next(events, None)
                            if event is None:
                                break
                            progress = {}
                            for node, node_state in event.items():
                                progress[node] = {}
                                if node == "react_action":
                                    progress[node]["last_action"] = node_state["last_action"]
                                if node == "output_guardrail":
                                    for key in ("final_answer", "audit_status", "citations", "sources"):
                                        progress[node][key] = node_state[key]
                            loop.call_soon_threadsafe(queue.put_nowait, ("event", progress))
            except Exception as exc:
                loop.call_soon_threadsafe(queue.put_nowait, ("error", exc))
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, ("done", sentinel))

        worker = threading.Thread(target=_worker, daemon=True)
        worker.start()

        try:
            while True:
                kind, payload = await queue.get()
                if kind == "event":
                    yield payload
                elif kind == "error":
                    raise payload
                else:
                    break
        finally:
            stop_event.set()
