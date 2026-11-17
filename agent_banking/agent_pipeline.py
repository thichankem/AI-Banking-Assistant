# -*- coding: utf-8 -*-
"""Banking deep agent entrypoint for synchronous chat and SSE."""

import sys
import time
from threading import Lock
from pathlib import Path
from langgraph.types import Command

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
from agent_banking.agent.graph import build_banking_agent_graph
from agent_banking.agent.deep import BankingDeepAgent
from agent_banking.agent.react import FALLBACK_ANSWER


class InteractionError(ValueError):
    def __init__(self, status_code: int, message: str):
        super().__init__(message)
        self.status_code = status_code


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
        print("  INITIALIZING BANKING DEEP AGENT")
        print("=" * 70)

        # 1. Initialize foundation RAG Pipeline (Dual Index Vector + BM25)
        self.rag_pipeline = rag_pipeline
        if self.rag_pipeline is None:
            self.rag_pipeline = BankingRAGPipeline(config=self.config, use_memory=self.use_memory)
        self._session_locks = [Lock() for _ in range(64)]

        # 2. Initialize the ReAct graph and session checkpoint
        self.runtime = BankingDeepAgent(self.rag_pipeline, self.config)
        self.graph = build_banking_agent_graph(nodes=self.runtime, sqlite_db_path=self.sqlite_db_path)
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
        screened = self.runtime.screen_input(clean_query)
        return {
            "user_query": screened["query"],
            "screened_input": screened,
            "session_id": session_id,
            "persona": persona,
            "tone": tone,
            "format_type": format_type,
            "custom_persona": self.runtime.guard_artifact(str(custom_persona or "")[:1000]),
        }

    def _session_lock(self, session_id: str):
        return self._session_locks[hash(session_id) % len(self._session_locks)]

    @staticmethod
    def _thread_config(session_id: str) -> dict:
        return {"configurable": {"thread_id": session_id}, "recursion_limit": 64}

    @staticmethod
    def _waiting(snapshot, session_id: str) -> dict:
        return {
            "status": "awaiting_input",
            "session_id": session_id,
            "answer": "",
            "interaction": snapshot.interrupts[0].value,
            "plan": snapshot.values.get("plan", []),
            "plan_revision": snapshot.values.get("plan_revision", 0),
        }

    def status(self, session_id: str) -> dict:
        with self._session_lock(session_id):
            snapshot = self.graph.get_state(self._thread_config(session_id))
            if snapshot.interrupts:
                return self._waiting(snapshot, session_id)
            if snapshot.values.get("accepted_interaction_id") and not snapshot.next:
                return self._format_result(snapshot.values, session_id, 0.0)
        return {"status": "idle", "session_id": session_id}

    def resume(self, session_id: str, interaction_id: str, choice_id: str, text: str | None = None) -> dict:
        started = time.perf_counter()
        config = self._thread_config(session_id)
        with self._session_lock(session_id):
            snapshot = self.graph.get_state(config)
            if not snapshot.interrupts or snapshot.interrupts[0].value["id"] != interaction_id:
                raise InteractionError(409, "Câu hỏi này không còn chờ phản hồi trong phiên hiện tại.")
            pending = snapshot.interrupts[0]
            if not any(option["id"] == choice_id for option in pending.value["options"]):
                raise InteractionError(422, "Lựa chọn không hợp lệ.")
            reply = {"interaction_id": interaction_id, "choice_id": choice_id}
            if choice_id == "other":
                if not text or not text.strip() or len(text) > 500:
                    raise InteractionError(422, "Vui lòng nhập câu trả lời từ 1 đến 500 ký tự.")
                screened = self.runtime.screen_input(text)
                if not screened["is_safe"]:
                    raise InteractionError(422, "Không thể sử dụng câu trả lời này. Vui lòng chỉnh sửa.")
                guarded = self.runtime.guard_artifact(screened["query"])
                if not guarded:
                    raise InteractionError(422, "Không thể sử dụng câu trả lời này. Vui lòng chỉnh sửa.")
                reply["text"] = guarded
            elif text is not None:
                raise InteractionError(422, "Chỉ lựa chọn Khác nhận câu trả lời tự do.")
            self.graph.invoke(Command(resume={pending.id: reply}), config=config, durability="sync")
            snapshot = self.graph.get_state(config)
            if snapshot.interrupts:
                return self._waiting(snapshot, session_id)
        return self._format_result(snapshot.values, session_id, round((time.perf_counter() - started) * 1000, 2),
                                   trace_result=True)

    def _restore_history(self, state: dict, previous: dict) -> None:
        # The graph uses a root dict channel, which is replaced by each new request.
        history = previous.get("chat_history", [])
        memory = previous.get("session_memory")
        if memory is None:
            # Import the latest history from pre-harness checkpoints once.
            history = [
                {"role": item["role"], "content": self.runtime.guard_artifact(item["content"])}
                for item in history if item.get("role") in ("user", "assistant")
            ]
            history = [item for item in history if item["content"]]
        else:
            state["session_memory"] = memory
        state["chat_history"] = history

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
            reply = self.runtime.guardrails.guard_output("Quý khách vui lòng nhập câu hỏi.")
            return {
                "answer": reply.sanitized_answer,
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
        thread_config = self._thread_config(session_id)

        try:
            with self._session_lock(session_id):
                snapshot = self.graph.get_state(thread_config)
                if snapshot.interrupts:
                    return self._waiting(snapshot, session_id)
                initial_state = self.build_initial_state(
                    query=query, session_id=session_id, persona=persona, tone=tone,
                    format_type=format_type, custom_persona=custom_persona
                )
                self._restore_history(initial_state, snapshot.values)
                final_state = self.graph.invoke(initial_state, config=thread_config, durability="sync")
                if "__interrupt__" in final_state:
                    return self._waiting(self.graph.get_state(thread_config), session_id)
        except Exception as e:
            print(f"[LangGraph Execution Error]: {type(e).__name__}")
            latency_err = round((time.perf_counter() - start_time) * 1000.0, 2)
            reply = self.runtime.guardrails.guard_output(FALLBACK_ANSWER)
            return {
                "answer": reply.sanitized_answer,
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
        return self._format_result(final_state, session_id, total_latency_ms, trace_result=True)

    def _format_result(self, final_state: dict, session_id: str, total_latency_ms: float,
                       *, trace_result: bool = False) -> dict:
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
        if trace_result:
            with self.tracer.trace_query(query=trace_query) as trace:
                for step in steps:
                    trace.spans.append({
                        "name": str(step), "duration_ms": 0.0,
                        "details": {"intent": intent, "audit_status": audit_stat}
                    })
                trace.finish(answer=answer, citations=citations, sources=sources,
                             is_blocked=not is_input_safe, guardrail_trigger=None)

        # Package node inspection details
        nodes_detail = {
            "00_guardrail": {
                "name": "Node 0: Banking Input Guardrail (State Reset)",
                "is_safe": is_input_safe,
                "status": status_guardrail
            },
            "01_react": {
                "name": "Deep agent decisions",
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
                "name": "Independent answer review",
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

        status = "completed"
        if audit_stat == "CANCELED":
            status = "canceled"
        return {
            "status": status,
            "accepted_interaction_id": final_state.get("accepted_interaction_id"),
            "answer": answer,
            "citations": citations,
            "sources": sources,
            "intent": intent,
            "retry_count": retries,
            "audit_status": audit_stat,
            "error_category": final_state.get("error_category"),
            "evaluation_metrics": None,
            "tool_calls": tool_outputs,
            "session_id": session_id,
            "latency_ms": total_latency_ms,
            "execution_steps": steps,
            "plan": final_state.get("plan", []),
            "plan_revision": final_state.get("plan_revision", 0),
            "delegations": final_state.get("delegations", []),
            "review_outcomes": final_state.get("reviews", []),
            "verification": final_state.get("verification", {}),
            "memory": final_state.get("memory_status", {}),
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

        thread_config = self._thread_config(session_id)

        loop = asyncio.get_running_loop()
        queue = asyncio.Queue()
        stop_event = threading.Event()
        sentinel = object()

        def _worker():
            try:
                with self._session_lock(session_id):
                    if stop_event.is_set():
                        return
                    snapshot = self.graph.get_state(thread_config)
                    if snapshot.interrupts:
                        loop.call_soon_threadsafe(queue.put_nowait, ("event", {
                            "awaiting_input": self._waiting(snapshot, session_id)
                        }))
                        return
                    initial_state = self.build_initial_state(
                        query=query, session_id=session_id, persona=persona, tone=tone,
                        format_type=format_type, custom_persona=custom_persona
                    )
                    self._restore_history(initial_state, snapshot.values)
                    paused = False
                    with contextlib.closing(self.graph.stream(initial_state, config=thread_config, durability="sync")) as events:
                        while not stop_event.is_set():
                            event = next(events, None)
                            if event is None:
                                break
                            if "__interrupt__" in event:
                                paused = True
                                continue
                            progress = {}
                            for node, node_state in event.items():
                                public = {}
                                if node in ("react_action", "plan_update", "subagent", "memory_update", "context_prepare", "skill_inspect"):
                                    if node_state.get("last_action"):
                                        public["last_action"] = node_state["last_action"]
                                if node == "plan_update":
                                    public["plan"] = node_state["plan"]
                                if node == "answer_review":
                                    public["audit_status"] = node_state["audit_status"]
                                if node == "output_guardrail":
                                    for key in ("final_answer", "audit_status", "citations", "sources", "plan", "delegations"):
                                        public[key] = node_state[key]
                                    public["review_outcomes"] = node_state["reviews"]
                                    public["verification"] = node_state.get("verification", {})
                                    public["memory"] = node_state["memory_status"]
                                progress[node] = public
                            loop.call_soon_threadsafe(queue.put_nowait, ("event", progress))
                    if paused and not stop_event.is_set():
                        snapshot = self.graph.get_state(thread_config)
                        loop.call_soon_threadsafe(queue.put_nowait, ("event", {
                            "awaiting_input": self._waiting(snapshot, session_id)
                        }))
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
