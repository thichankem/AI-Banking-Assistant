"""
Task Router (First Node - JSON-Driven Architecture).
Architecture:
- Dynamically loads intent routing rules from configuration JSON (tasks_manifest.json).
- Identifies user Intent based on skill catalog and business tasks.
- Supports Decision Model and LLM fallback for open-ended queries.

RULES:
1. NO __INIT__.PY: Direct module imports.
2. NO TYPING LIBRARY: Native Python types only.
3. ZERO TERNARY: Explicit if/else only.
4. 0% REGEX LAW: Absolutely no regular expressions.
5. KEEP IT SIMPLE: Clear, direct, readable code.
"""

from __future__ import annotations
import json
from pathlib import Path
from dataclasses import dataclass
from agent_banking.config.chat import FallbackChatClient
from agent_banking.config.fallback import ModelFallbackError

from agent_banking.skills.skill_registry import SkillRegistry
from agent_banking.config.settings import Config
from agent_banking.decision.decision_client import DecisionClient
from agent_banking.tasks.config.prompts import TASK_ROUTER_SYSTEM_PROMPT_TEMPLATE


@dataclass
class RoutingDecision:
    """Routing decision returned by Intent Router Node."""
    task_id: str
    task_name: str
    target_skills: list[str]
    confidence: float
    reason: str


class TaskRouter:
    """First node: Classify intent and assign Task based on JSON rules and decision engine."""

    def __init__(self, skill_registry: SkillRegistry, config: Config | None = None, manifest_path: str | Path | None = None):
        self.registry = skill_registry

        if config is not None:
            self.config = config
        else:
            self.config = Config()

        if manifest_path is not None:
            self.manifest_path = Path(manifest_path)
        else:
            self.manifest_path = Path(__file__).resolve().parent / "tasks_manifest.json"

        self.routing_rules: list[dict] = []
        self._load_routing_manifest()

        self.decision_client = DecisionClient(config=self.config)

        self.client = FallbackChatClient("tasks", {
            "model": getattr(self.config, "TASKS_LLM_MODEL", self.config.LLM_MODEL),
            "url": getattr(self.config, "TASKS_LLM_BASE_URL", self.config.LLM_BASE_URL),
            "api_key": getattr(self.config, "TASKS_LLM_API_KEY", self.config.LLM_API_KEY),
        })

    def _load_routing_manifest(self) -> None:
        """Load routing rules from JSON manifest."""
        if not self.manifest_path.exists():
            return

        try:
            with open(self.manifest_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            rules = data.get("routing_rules", [])
            # Sort by priority ascending (0 is highest priority)
            self.routing_rules = sorted(rules, key=lambda x: x.get("priority", 99))
        except Exception as e:
            print(f"⚠️ Failed to load tasks_manifest.json: {e}")
            self.routing_rules = []

    def route(self, user_query: str) -> RoutingDecision:
        """Analyze query and select Task based on JSON rules."""
        q = user_query.strip().lower()

        # 1. Iterate over JSON defined rules
        for rule in self.routing_rules:
            keywords = rule.get("keywords", [])
            task_id = rule.get("task_id", "")
            task_name = rule.get("task_name", "")
            target_skills = rule.get("target_skills", [])
            confidence = float(rule.get("confidence", 0.95))

            if task_id == "TASK_GENERAL_CHAT":
                for kw in keywords:
                    if q.startswith(kw) or q == kw:
                        return RoutingDecision(
                            task_id=task_id,
                            task_name=task_name,
                            target_skills=target_skills,
                            confidence=confidence,
                            reason=f"Matched greeting keyword: '{kw}'"
                        )
            else:
                for kw in keywords:
                    if kw in q:
                        return RoutingDecision(
                            task_id=task_id,
                            task_name=task_name,
                            target_skills=target_skills,
                            confidence=confidence,
                            reason=f"Matched business keyword: '{kw}'"
                        )

        # 2. Fallback to decision engine / LLM classification
        return self._route_with_llm(user_query)

    def _route_with_llm(self, query: str) -> RoutingDecision:
        """Deep classification for open-ended queries: decision engine first, then LLM."""
        # 1. Fast classification via Decision Model
        if self.decision_client.is_available():
            decision_task = self.decision_client.route_task(query)
            if decision_task is not None:
                t_conf = float(decision_task.get("confidence", 0.0))
                if t_conf >= 0.70:
                    t_id = decision_task.get("task_id", "")
                    if t_id == "TASK_REACT_AGENT":
                        return RoutingDecision("TASK_REACT_AGENT", "Multi-Step ReAct Agent", ["rag_knowledge_retrieval", "financial_calculator", "compliance_qd2345_audit"], t_conf, "Decision Engine: Multi-step ReAct")
                    if t_id == "TASK_LOAN_ADVISORY":
                        return RoutingDecision("TASK_LOAN_ADVISORY", "Loan Advisory & Amortization", ["financial_calculator", "rag_knowledge_retrieval"], t_conf, "Decision Engine: Loan Advisory")
                    if t_id == "TASK_BIOMETRIC_COMPLIANCE":
                        return RoutingDecision("TASK_BIOMETRIC_COMPLIANCE", "Biometric Compliance Evaluation", ["compliance_qd2345_audit"], t_conf, "Decision Engine: Biometrics")
                    if t_id == "TASK_CARD_SERVICE":
                        return RoutingDecision("TASK_CARD_SERVICE", "Card Services Advisory", ["rag_knowledge_retrieval"], t_conf, "Decision Engine: Card Services")
                    if t_id == "TASK_INCIDENT_SUPPORT":
                        return RoutingDecision("TASK_INCIDENT_SUPPORT", "Urgent Incident Support", ["incident_support_ticket"], t_conf, "Decision Engine: Incident Support")
                    if t_id == "TASK_GENERAL_CHAT":
                        return RoutingDecision("TASK_GENERAL_CHAT", "General Banking Chat", [], t_conf, "Decision Engine: General Chat")
                    if t_id == "TASK_POLICY_INQUIRY":
                        return RoutingDecision("TASK_POLICY_INQUIRY", "Banking Policy & Tariff Inquiry", ["rag_knowledge_retrieval", "fact_checking_audit"], t_conf, "Decision Engine: Policy RAG")

        if not self.client:
            return RoutingDecision(
                task_id="TASK_POLICY_INQUIRY",
                task_name="Banking Policy & Tariff Inquiry",
                target_skills=["rag_knowledge_retrieval", "fact_checking_audit"],
                confidence=0.85,
                reason="Default routing: Policy RAG"
            )

        skills_desc = self.registry.get_skills_prompt_description()
        system_prompt = TASK_ROUTER_SYSTEM_PROMPT_TEMPLATE.format(skills_desc=skills_desc)

        try:
            resp = self.client.chat.completions.create(
                model=self.config.LLM_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query}
                ],
                temperature=0.1,
                max_tokens=30
            )
            raw = resp.choices[0].message.content
            if not raw:
                raw = ""
            code = raw.strip().upper()

            if "REACT" in code:
                return RoutingDecision("TASK_REACT_AGENT", "Multi-Step ReAct Agent", ["rag_knowledge_retrieval", "financial_calculator", "compliance_qd2345_audit"], 0.95, "LLM routed: Multi-step ReAct")
            if "LOAN" in code:
                return RoutingDecision("TASK_LOAN_ADVISORY", "Loan Advisory & Amortization", ["financial_calculator", "rag_knowledge_retrieval"], 0.90, "LLM routed: Loan Advisory")
            if "BIOMETRIC" in code or "2345" in code:
                return RoutingDecision("TASK_BIOMETRIC_COMPLIANCE", "Biometric Compliance Evaluation", ["compliance_qd2345_audit"], 0.90, "LLM routed: Biometrics")
            if "CARD" in code:
                return RoutingDecision("TASK_CARD_SERVICE", "Card Services Advisory", ["rag_knowledge_retrieval"], 0.90, "LLM routed: Card Services")
            if "INCIDENT" in code:
                return RoutingDecision("TASK_INCIDENT_SUPPORT", "Urgent Incident Support", ["incident_support_ticket"], 0.90, "LLM routed: Incident Support")
            if "CHAT" in code:
                return RoutingDecision("TASK_GENERAL_CHAT", "General Banking Chat", [], 0.90, "LLM routed: General Chat")

            return RoutingDecision("TASK_POLICY_INQUIRY", "Banking Policy & Tariff Inquiry", ["rag_knowledge_retrieval"], 0.85, "LLM routed: Policy RAG")
        except ModelFallbackError:
            raise
        except Exception:
            return RoutingDecision("TASK_POLICY_INQUIRY", "Banking Policy & Tariff Inquiry", ["rag_knowledge_retrieval"], 0.80, "Fallback: Policy RAG")
