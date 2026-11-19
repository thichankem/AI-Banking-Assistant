"""Banking agent harness with planning, isolated workers and independent review."""

import json
from unicodedata import normalize
from uuid import uuid4

from langgraph.types import interrupt

from agent_banking.agent.config.response_profiles import (
    FORMAT_PROFILES,
    PERSONA_PROFILES,
    TONE_PROFILES,
    get_messages,
    get_prompt,
)
from agent_banking.agent.config.settings import (
    get_int_setting,
    get_setting,
    get_workers,
)
from agent_banking.agent.memory import (
    MAX_COMPACT_PER_TURN,
    RECENT_MESSAGES,
    change_note,
    compact_memory,
    memory_view,
    public_memory,
    should_compact,
)
from agent_banking.agent.model_protocol import REVIEW_SCHEMA, StructuredModel
from agent_banking.agent.react import BankingReActNodes, FALLBACK_ANSWER, MAX_EVIDENCE
from agent_banking.core.fallback import ModelFallbackError, is_temporary_error
from agent_banking.core.search_text import search_key
from agent_banking.skills.skill_registry import resolve_skill
from agent_banking.tools.extensions.banking_query import parse_banking_query
from agent_banking.tracing.agent_trace import annotate_span


MAX_ACTIONS = get_int_setting("max_actions")
MAX_DELEGATIONS = get_int_setting("max_delegations")
MAX_WORKER_ACTIONS = get_int_setting("max_worker_actions")
MAX_REVISIONS = get_int_setting("max_revisions")
MAX_INTERACTIONS = get_int_setting("max_interactions")
MAX_ACTIVE_SKILL_BYTES = get_int_setting("max_active_skill_bytes")
RESTRICTED_CARD_FIELDS = tuple(get_setting("restricted_card_fields", ()))
WORKERS = get_workers()

BASE_PROMPT = get_prompt("BASE_PROMPT")
COORDINATOR_PROMPT = get_prompt("COORDINATOR_PROMPT")
WORKER_PROMPT = get_prompt("WORKER_PROMPT")
REVIEW_PROMPT = get_prompt("REVIEW_PROMPT")
PARTIAL_REVIEW_PROMPT = get_prompt("PARTIAL_REVIEW_PROMPT")

_RECOVER_PROMPT = get_prompt("RECOVER_PROMPT")
_PARTIAL_RECOVERY_PROMPT = get_prompt("PARTIAL_RECOVERY_PROMPT")
_COORDINATOR_SKILL_HINT = get_prompt("COORDINATOR_SKILL_HINT")
_REVISION_INSTRUCTION = get_prompt("REVISION_INSTRUCTION")
_REVIEW_REPAIR_INSTRUCTION = get_prompt("REVIEW_REPAIR_INSTRUCTION")
_DEEP_MESSAGES = get_messages("deep")

_REVIEW_NUMERIC_MARKERS = tuple(get_setting("review_numeric_markers", ()))
_COMPACT_OBSERVATION_MAX_BYTES = get_int_setting("compact_observation_max_bytes")
_MAX_SOURCE_INVENTORY_ITEMS = get_int_setting("max_source_inventory_items")
_PRESERVED_RETRIEVAL_KEYS = tuple(get_setting("preserved_retrieval_keys", ()))
_SUPPORT_HISTORY_LIMIT = get_int_setting("support_history_messages_limit")
_SUPPORT_NOTES_LIMIT = get_int_setting("support_notes_limit")
_SUPPORT_MESSAGE_MAX_CHARS = get_int_setting("support_message_max_chars")
_SUPPORT_USER_TEXT_MAX_CHARS = get_int_setting("support_user_text_max_chars")
_SUPPORT_TOOL_JSON_MAX_CHARS = get_int_setting("support_tool_json_max_chars")
_QUERY_HINTS_MAX_QUERY_CHARS = get_int_setting("query_hints_max_query_chars")
_QUERY_HINTS_MAX_TEXT_CHARS = get_int_setting("query_hints_max_text_chars")
_QUERY_HINTS_MAX_KEYWORDS = get_int_setting("query_hints_max_keywords")
_QUERY_HINTS_MAX_ACRONYMS = get_int_setting("query_hints_max_acronyms")
_HISTORY_MESSAGE_MAX_CHARS = get_int_setting("history_message_max_chars")
_REVIEW_UNLINKED_EVIDENCE_LIMIT = get_int_setting("review_unlinked_evidence_limit")
_PARTIAL_RECOVERY_MAX_SOURCES = get_int_setting("partial_recovery_max_sources")
_DECIDE_MAX_TOKENS = get_int_setting("decide_max_tokens")
_RECOVER_MAX_TOKENS = get_int_setting("recover_max_tokens")
_WORKER_MAX_TOKENS = get_int_setting("worker_max_tokens")
_PARTIAL_RECOVERY_MAX_TOKENS = get_int_setting("partial_recovery_max_tokens")
_REVIEW_MAX_TOKENS = get_int_setting("review_max_tokens")
_PARTIAL_REVIEW_MAX_TOKENS = get_int_setting("partial_review_max_tokens")
_DEFAULT_PERSONA = str(get_setting("default_persona", ""))
_DEFAULT_TONE = str(get_setting("default_tone", ""))
_DEFAULT_FORMAT_TYPE = str(get_setting("default_format_type", ""))


def _label(index: int) -> str:
    prefix = _DEEP_MESSAGES.get("source_label_prefix", "[Nguồn ")
    suffix = _DEEP_MESSAGES.get("source_label_suffix", "]")
    return f"{prefix}{index}{suffix}"


def _alnum_or_space(text: str) -> str:
    chars = []
    for ch in text:
        if ch.isalnum():
            chars.append(ch)
        else:
            chars.append(" ")
    return " " + " ".join("".join(chars).split()) + " "


def _evidence_view(evidence: list[dict]) -> list[dict]:
    result = []
    for item in evidence:
        chunk = item["chunk"]
        record = {"label": _label(item["index"]), "text": item["context"], "source": item["source"],
                  "version": (chunk.get("business_metadata") or {}).get("document_version", "")}
        if type(chunk.get("source_offset")) is int:
            record["source_offset"] = chunk["source_offset"]
        if chunk.get("context_for"):
            record["context_neighbor"] = True
        result.append(record)
    return result


def _support_view(state: dict) -> dict:
    turn = state["react_turn"]
    user_text = [
        item["content"] for item in state.get("chat_history", [])[-_SUPPORT_HISTORY_LIMIT:]
        if item["role"] == "user"
    ]
    user_text += [state["user_query"]] + [reply["response"] for reply in state["human_replies"]]
    notes = [item["value"] for item in memory_view(state)["notes"].values()]
    user_text = notes[:_SUPPORT_NOTES_LIMIT] + user_text
    user_label = _DEEP_MESSAGES.get("user_support_label", "[Khách]")
    tool_prefix = _DEEP_MESSAGES.get("tool_support_prefix", "[Tool ")
    result = {user_label: "\n".join(text[:_SUPPORT_MESSAGE_MAX_CHARS] for text in user_text)[-_SUPPORT_USER_TEXT_MAX_CHARS:]}

    def add(observation):
        if observation["status"] == "SUCCESS" and not observation.get("retrieval") and observation["kind"] in ("tool", "skill"):
            value = {"result": observation.get("data", {}), "simulation": observation.get("simulation", False)}
            result[f"{tool_prefix}{len(result)}]"] = json.dumps(value, ensure_ascii=False)[:_SUPPORT_TOOL_JSON_MAX_CHARS]

    for observation in turn["observations"]:
        add(observation)
        for item in observation.get("data", {}).get("worker_observations", []):
            add(item)
    return result


def _fold_quote(value: str) -> str:
    canonical = normalize("NFC", value).casefold()
    # Formatting may differ between Markdown, plain text, and a quoted table cell.
    return _alnum_or_space(canonical)


def validate_review(verdict: dict, draft: str, citations: list, evidence: list, support: dict) -> None:
    sources = {item["label"]: item["text"] for item in evidence}
    sources.update(support)
    if any(item["status"] == "missing" for item in verdict["coverage"]):
        raise ValueError("Missing requested item")
    claimed = set()
    draft_text = _fold_quote(draft)
    plain = draft.casefold()
    for citation in citations:
        plain = plain.replace(citation["label"].casefold(), "")
    if not verdict["claims"] and any(char.isdigit() for char in plain) and any(
        marker in plain for marker in _REVIEW_NUMERIC_MARKERS
    ):
        raise ValueError("Unsupported financial figure or contact")
    source_labels = {_fold_quote(label): label for label in sources}
    for item in verdict["claims"]:
        source = source_labels.get(_fold_quote(item["source"]))
        excerpt = item.get("claim", "")
        if "draft_line" in item:
            draft_lines = draft.splitlines()
            pointer = item["draft_line"]
            if type(pointer) is not int or not 1 <= pointer <= len(draft_lines):
                raise ValueError("Invalid draft evidence line")
            excerpt = draft_lines[pointer - 1]
        claim = _fold_quote(excerpt)
        if source is None:
            raise ValueError("Invalid claim source label")
        if not claim.strip() or claim not in draft_text:
            raise ValueError("Invalid claim excerpt from draft")
        source_text = sources[source]
        if "line" in item:
            lines = source_text.splitlines()
            line = item["line"]
            if source in support or not 1 <= line <= len(lines) or not _fold_quote(lines[line - 1]).strip():
                raise ValueError("Invalid claim evidence line")
            source_text = lines[line - 1]
        quote = item.get("quote")
        if quote is not None and (not _fold_quote(quote).strip() or _fold_quote(quote) not in _fold_quote(source_text)):
            raise ValueError("Invalid claim evidence excerpt")
        claimed.add(source)
    if verdict.get("verdict", "approve") == "approve":
        missing = [item["label"] for item in citations if item["label"] not in claimed]
        if missing and not claimed:
            raise ValueError("Citation missing claim grounding: " + ", ".join(missing))


def _compact_observations(observations: list[dict]) -> list[dict]:
    bounded = []
    for observation in observations:
        if observation["kind"] == "inspect_skill":
            bounded.append({key: value for key, value in observation.items() if key != "data"})
            continue
        data = observation.get("data")
        if not isinstance(data, dict):
            bounded.append(observation)
            continue
        if observation.get("retrieval"):
            compact_docs = []
            for item in data.get("documents", []):
                if isinstance(item, dict) and "label" in item:
                    compact_docs.append(item["label"])
                else:
                    compact_docs.append(item)
            compact_related = []
            for item in data.get("related_documents", []):
                if isinstance(item, dict) and "label" in item:
                    compact_related.append(item["label"])
                else:
                    compact_related.append(item)
            data = {**data, "documents": compact_docs, "related_documents": compact_related}
        encoded = json.dumps(data, ensure_ascii=False)
        if len(encoded) > _COMPACT_OBSERVATION_MAX_BYTES:
            if observation.get("retrieval"):
                preserved = {}
                for keep_key in _PRESERVED_RETRIEVAL_KEYS:
                    if keep_key in data:
                        preserved[keep_key] = data[keep_key]
                preserved["truncated"] = True
                if len(json.dumps(preserved, ensure_ascii=False)) <= _COMPACT_OBSERVATION_MAX_BYTES:
                    data = preserved
                else:
                    if isinstance(preserved.get("retrieval_quality"), dict):
                        rq = dict(preserved["retrieval_quality"])
                        if isinstance(rq.get("source_inventory"), list) and len(rq["source_inventory"]) > _MAX_SOURCE_INVENTORY_ITEMS:
                            rq["source_inventory"] = rq["source_inventory"][:_MAX_SOURCE_INVENTORY_ITEMS]
                        preserved["retrieval_quality"] = rq
                    if len(json.dumps(preserved, ensure_ascii=False)) <= _COMPACT_OBSERVATION_MAX_BYTES:
                        data = preserved
                    else:
                        data = {"preview": encoded[:_COMPACT_OBSERVATION_MAX_BYTES], "truncated": True}
            else:
                data = {"preview": encoded[:_COMPACT_OBSERVATION_MAX_BYTES], "truncated": True}
        bounded.append({**observation, "data": data})
    return bounded


def _requests_restricted_field(text: str) -> bool:
    normalized = _alnum_or_space(text.casefold().replace(".", ""))
    return any(f" {field} " in normalized for field in RESTRICTED_CARD_FIELDS)


class BankingDeepAgent(BankingReActNodes):
    def __init__(self, pipeline, config):
        super().__init__(pipeline, config)
        review_client = None
        if self.client.routed:
            from agent_banking.core.chat import FallbackChatClient
            review_client = FallbackChatClient("agent_review", routed=True)
        self.model = StructuredModel(self.client, config.AGENT_LLM_MODEL, review_client=review_client)

    def input_guardrail(self, state: dict) -> dict:
        guarded = super().input_guardrail(state)
        guarded.update(plan=[], plan_revision=0, delegations=[], reviews=[], compactions_this_turn=0,
                       active_skills={})
        guarded.update(interaction=None, human_replies=[], accepted_interaction_id=None,
                       approved_scope=None, confirmation_required=False)
        guarded["session_memory"] = memory_view(guarded)
        guarded["memory_at_start"] = guarded["session_memory"]
        guarded["history_at_start"] = guarded.get("chat_history", [])
        return guarded

    def prepare_context(self, state: dict) -> dict:
        parsed = parse_banking_query(state["user_query"][:_QUERY_HINTS_MAX_QUERY_CHARS])
        state = {**state, "query_hints": {"normalized": parsed.normalized_query[:_QUERY_HINTS_MAX_TEXT_CHARS],
                                        "accent_folded": search_key(parsed.raw_query)[:_QUERY_HINTS_MAX_TEXT_CHARS],
                                        "keywords": parsed.keywords[:_QUERY_HINTS_MAX_KEYWORDS],
                                        "abbreviations": parsed.detected_acronyms[:_QUERY_HINTS_MAX_ACRONYMS]}}
        if not should_compact(state.get("chat_history", [])):
            return state
        result, compacted = compact_memory(state, self.model, self.guard_artifact)
        if compacted:
            result["compactions_this_turn"] += 1
            result["execution_steps"] = state["execution_steps"] + ["memory_compact: automatic"]
            result["last_action"] = {"kind": "memory", "name": "compact", "status": "SUCCESS"}
        return result

    def _decide_input(self, state: dict) -> dict:
        turn = state["react_turn"]
        evidence = _evidence_view(turn["evidence"])
        # A revised search must expose its new results even when older sources
        # have filled the context budget. Labels themselves stay stable.
        recent = []
        for observation in reversed(turn["observations"]):
            if observation.get("retrieval"):
                for document in observation.get("data", {}).get("documents", []):
                    label = document.get("label")
                    if label and label not in recent:
                        recent.append(label)
        positions = {label: index for index, label in enumerate(recent)}
        evidence.sort(key=lambda item: positions.get(item["label"], len(positions)))
        return {
            "request": state["user_query"],
            "query_hints": state.get("query_hints", {}),
            "active_skills": state.get("active_skills", {}),
            "history": state.get("chat_history", []),
            "memory": memory_view(state),
            "plan": state["plan"],
            "observations": _compact_observations(turn["observations"]),
            "available_evidence": evidence,
            "remaining_actions": max(0, MAX_ACTIONS - len(turn["observations"])),
            "remaining_delegations": max(0, MAX_DELEGATIONS - len(state["delegations"])),
            "remaining_interactions": max(0, MAX_INTERACTIONS - len(state["human_replies"])),
            "human_replies": state["human_replies"],
            "approved_scope": state["approved_scope"],
            "confirmation_required": state["confirmation_required"],
        }

    def decide(self, state: dict) -> dict:
        data = self._decide_input(state)
        allowed = ["final"]
        system = BASE_PROMPT
        if data["remaining_actions"]:
            allowed += ["plan", "remember", "forget"]
            if not state["confirmation_required"]:
                allowed.append("tool")
                if state["capabilities"]["skill"]:
                    allowed += ["skill", "inspect_skill"]
                if data["remaining_delegations"]:
                    allowed.append("delegate")
            if data["remaining_interactions"]:
                allowed += ["clarify", "confirm"]
            if state["compactions_this_turn"] < MAX_COMPACT_PER_TURN and len(data["history"]) > RECENT_MESSAGES:
                allowed.append("compact")
            system += "\n" + COORDINATOR_PROMPT + _DEEP_MESSAGES.get("catalog_header", "\n") + json.dumps(
                self.catalog_view(state["capabilities"]), ensure_ascii=False
            )
            system += _DEEP_MESSAGES.get("subagents_header", "\n") + json.dumps(
                {name: role["description"] for name, role in WORKERS.items()}, ensure_ascii=False
            )
            if _COORDINATOR_SKILL_HINT:
                system += "\n" + _COORDINATOR_SKILL_HINT
        persona = PERSONA_PROFILES.get(state.get("persona"), PERSONA_PROFILES.get(_DEFAULT_PERSONA, {}))
        tone = TONE_PROFILES.get(state.get("tone"), TONE_PROFILES.get(_DEFAULT_TONE, ""))
        format_style = FORMAT_PROFILES.get(state.get("format_type"), FORMAT_PROFILES.get(_DEFAULT_FORMAT_TYPE, ""))
        system += _DEEP_MESSAGES.get("style_header", "\n") + "\n".join(
            [persona.get("role_definition") or persona.get("description", ""), tone, format_style]
        )
        system += _DEEP_MESSAGES.get("allowed_kinds_header", "\n") + ", ".join(allowed)
        if state.get("persona") == "custom":
            data["style_preference"] = state.get("custom_persona", "")
        if state.get("audit_feedback"):
            data["review_feedback"] = state["audit_feedback"]
            data["previous_draft"] = state.get("draft_answer", "")
            if _REVISION_INSTRUCTION:
                system += "\n" + _REVISION_INSTRUCTION

        deep_reason = None
        if state.get("audit_feedback"):
            deep_reason = "review_revision"
        elif any(item["status"] == "SUCCESS" for item in state["delegations"]):
            deep_reason = "worker_evidence"
        elif len({item["chunk"].get("source") for item in state["react_turn"]["evidence"]
                  if item["chunk"].get("source")}) >= 2:
            deep_reason = "multiple_sources"
        try:
            decision = self.model.decide(system, data, allowed, max_tokens=_DECIDE_MAX_TOKENS, deep_reason=deep_reason)
        except ModelFallbackError as error:
            category = getattr(error, "category", "model_unavailable")
            if isinstance(getattr(error, "last_error", None), (ValueError, TypeError)):
                category = "model_contract"
            elif is_temporary_error(error):
                category = "provider_temporary"
            decision = None
            if not self.client.routed:
                decision = self.recover_decision(data, category, state["confirmation_required"])
            if decision is None:
                return {
                    **state, "audit_status": "ERROR", "final_answer": FALLBACK_ANSWER,
                    "error_category": category,
                    "execution_steps": state["execution_steps"] + ["deep_decide: model_error"]
                }
            state = {**state, "execution_steps": state["execution_steps"] + ["deep_decide: recovered_contract"]}
        if state["confirmation_required"] and decision["kind"] == "final":
            return {
                **state,
                "audit_status": "NEEDS_INPUT",
                "final_answer": _DEEP_MESSAGES.get("confirmation_required_answer", ""),
            }
        if decision["kind"] in ("clarify", "confirm"):
            try:
                decision = self.guard_payload(decision)
            except ValueError:
                return self._unsafe_interaction(state)
            texts = [decision.get("question", decision.get("proposal", "")), *decision.get("options", [])]
            if any(not text or "REDACTED]" in text.upper() or _requests_restricted_field(text) for text in texts):
                return self._unsafe_interaction(state)
        return {
            **state, "react_turn": {**state["react_turn"], "decision": decision},
            "execution_steps": state["execution_steps"] + ["deep_decide: " + decision["kind"]]
        }

    def recover_decision(self, data: dict, category: str, confirmation_required: bool) -> dict | None:
        if category != "model_contract" or confirmation_required:
            return None
        if not any(item["kind"] in ("tool", "skill", "delegate") and item["status"] == "SUCCESS"
                   and "data" in item for item in data["observations"]):
            return None
        instruction = BASE_PROMPT + "\n" + _RECOVER_PROMPT
        try:
            return self.model.decide(instruction, data, ["final"], max_tokens=_RECOVER_MAX_TOKENS)
        except ModelFallbackError:
            return None

    def _record(self, state: dict, observation: dict) -> dict:
        turn = state["react_turn"]
        note = {key: observation[key] for key in ("kind", "name", "status")}
        return {
            **state,
            "react_turn": {**turn, "observations": turn["observations"] + [observation]},
            "last_action": note,
            "execution_steps": state["execution_steps"] + [
                f"{note['kind']}: {note['name']} ({note['status']})"
            ],
        }

    def _available_skills(self, state: dict) -> dict:
        definitions = state["capabilities"]["skill"]
        if state.get("worker_role"):
            definitions = self._worker_snapshot(state["worker_role"], state["capabilities"])["skill"]
        return definitions

    def act(self, state: dict) -> dict:
        decision = state["react_turn"]["decision"]
        if decision["kind"] == "skill":
            spec = resolve_skill(self._available_skills(state), decision["name"])
            if spec is None and state.get("worker_role"):
                return self._record(state, {"kind": "skill", "name": "restricted", "status": "ERROR",
                                            "error": _DEEP_MESSAGES.get("skill_restricted_worker", "")})
            if spec and "revision" in spec and spec["skill_id"] not in state.get("active_skills", {}):
                return self.inspect_skill(state)
        return super().act(state)

    def inspect_skill(self, state: dict) -> dict:
        decision = state["react_turn"]["decision"]
        spec = resolve_skill(self._available_skills(state), decision["name"])
        if spec is None:
            return self._record(state, {"kind": "inspect_skill", "name": "unknown", "status": "ERROR",
                                        "error": _DEEP_MESSAGES.get("skill_not_found", "")})
        name = spec["skill_id"]
        active = state.get("active_skills", {})
        reference = decision.get("reference")
        event = {"kind": "inspect_skill", "name": name, "status": "SUCCESS"}
        try:
            if reference is None:
                if name in active:
                    return self._record(state, {**event, "status": "ALREADY_ACTIVE"})
                loaded = self.skills.instructions(spec)
            else:
                event["reference"] = reference
                if name not in active:
                    raise ValueError("Activate the skill before reading its references")
                current = active[name]
                resources = current.get("loaded_references", {})
                if reference in resources:
                    return self._record(state, {**event, "status": "ALREADY_ACTIVE"})
                text = self.skills.reference(spec, reference, current["references"])
                loaded = {**current, "loaded_references": {**resources, reference: text}}
            updated = {**active, name: loaded}
            if len(json.dumps(updated, ensure_ascii=False).encode("utf-8")) > MAX_ACTIVE_SKILL_BYTES:
                raise ValueError("Active skill context budget exceeded")
        except (OSError, ValueError, RecursionError):
            return self._record(state, {**event, "status": "ERROR",
                                        "error": _DEEP_MESSAGES.get("skill_load_error", "")})
        return self._record({**state, "active_skills": updated}, event)

    def update_plan(self, state: dict) -> dict:
        decision = state["react_turn"]["decision"]
        items = decision["items"]
        try:
            guarded = self.guard_payload(items)
            if len({item["id"] for item in guarded}) != len(guarded):
                raise ValueError("Repeated plan ids")
            if any(not item["task"] for item in guarded):
                raise ValueError("Empty plan task")
            if guarded == state["plan"]:
                return self._record(state, {
                    "kind": "plan", "name": "unchanged", "status": "ERROR",
                    "error": _DEEP_MESSAGES.get("plan_unchanged", "")
                })
        except ValueError:
            return self._record(state, {
                "kind": "plan", "name": "update", "status": "ERROR",
                "error": _DEEP_MESSAGES.get("plan_invalid", "")
            })
        requires_confirmation = state["confirmation_required"]
        if state["approved_scope"] is not None:
            requires_confirmation = self.plan_scope(guarded) != state["approved_scope"]["plan"]
        return self._record({**state, "plan": guarded, "plan_revision": state["plan_revision"] + 1,
                             "confirmation_required": requires_confirmation},
                            {"kind": "plan", "name": "update", "status": "SUCCESS"})

    @staticmethod
    def plan_scope(plan: list) -> list:
        return [{"id": item["id"], "task": item["task"]} for item in plan]

    def prepare_interaction(self, state: dict, *, durable: bool) -> dict:
        decision = state["react_turn"]["decision"]
        mode = decision["kind"]
        field = "question"
        if mode == "confirm":
            field = "proposal"
        message = decision[field]
        labels = decision.get("options", [])
        normalized = [" ".join(label.casefold().split()) for label in labels]
        reserved_labels = set(_DEEP_MESSAGES.get("clarify_reserved_labels", ()))
        if len(set(normalized)) != len(labels) or reserved_labels.intersection(normalized):
            return self._interaction_stop(state, _DEEP_MESSAGES.get("interaction_invalid_options", ""))
        scope = []
        if mode == "confirm":
            scope = self.plan_scope(state["plan"])
        signature = [mode, " ".join(message.casefold().split()), scope]
        if len(state["human_replies"]) >= MAX_INTERACTIONS or any(
            reply["signature"] == signature for reply in state["human_replies"]
        ):
            return self._interaction_stop(state, _DEEP_MESSAGES.get("interaction_exhausted_prefix", "") + message)
        if mode == "confirm":
            options = [
                {"id": "approve", "label": _DEEP_MESSAGES.get("confirm_approve_label", "")},
                {"id": "reject", "label": _DEEP_MESSAGES.get("confirm_reject_label", "")},
            ]
        else:
            options = [{"id": f"option_{i}", "label": label} for i, label in enumerate(labels, 1)]
            options += [
                {"id": "other", "label": _DEEP_MESSAGES.get("clarify_other_label", "")},
                {"id": "cancel", "label": _DEEP_MESSAGES.get("clarify_cancel_label", "")},
            ]
        if not durable:
            answer = " ".join([message, " / ".join(labels),
                               _DEEP_MESSAGES.get("non_durable_interaction_suffix", "")])
            return self._interaction_stop(state, answer)
        return {**state, "interaction": {"id": uuid4().hex, "kind": mode, "message": message,
                                         "options": options, "scope": scope},
                "interaction_signature": signature,
                "execution_steps": state["execution_steps"] + ["interaction: awaiting_input"]}

    def await_interaction(self, state: dict) -> dict:
        card = state["interaction"]
        reply = interrupt(card)
        if not isinstance(reply, dict) or reply.get("interaction_id") != card["id"]:
            return self._interaction_stop(state, _DEEP_MESSAGES.get("interaction_mismatch_id", ""))
        selection = next((option for option in card["options"] if option["id"] == reply.get("choice_id")), None)
        if selection is None:
            return self._interaction_stop(state, _DEEP_MESSAGES.get("interaction_mismatch_choice", ""))
        state = {**state, "interaction": None, "accepted_interaction_id": card["id"]}
        if selection["id"] in ("cancel", "reject"):
            return self._interaction_stop(state, _DEEP_MESSAGES.get("interaction_canceled", ""), status="CANCELED")
        response = selection["label"]
        if selection["id"] == "other":
            response = self.guard_artifact(reply.get("text") or "")
            if not response:
                return self._interaction_stop(state, _DEEP_MESSAGES.get("interaction_unusable_reply", ""))
        if card["kind"] == "confirm":
            state["approved_scope"] = {"proposal": card["message"], "plan": card["scope"]}
            state["confirmation_required"] = False
        entry = {"kind": card["kind"], "question": card["message"], "response": response,
                 "signature": state["interaction_signature"]}
        state["human_replies"] = state["human_replies"] + [entry]
        return self._record(state, {"kind": card["kind"], "name": "human_input", "status": "SUCCESS"})

    @staticmethod
    def _interaction_stop(state: dict, answer: str, status: str = "NEEDS_INPUT") -> dict:
        return {**state, "interaction": None, "final_answer": answer, "audit_status": status,
                "execution_steps": state["execution_steps"] + ["interaction: " + status.lower()]}

    def _unsafe_interaction(self, state: dict) -> dict:
        return self._interaction_stop(state, _DEEP_MESSAGES.get("interaction_unsafe", ""))

    def manage_memory(self, state: dict) -> dict:
        decision = state["react_turn"]["decision"]
        kind = decision["kind"]
        error = ""
        if kind == "compact":
            state, compacted = compact_memory(state, self.model, self.guard_artifact)
            if compacted:
                state["compactions_this_turn"] += 1
            else:
                error = _DEEP_MESSAGES.get("memory_already_compact", "")
        else:
            user_messages = [state["user_query"]] + [
                item["content"] for item in state.get("chat_history", []) if item["role"] == "user"
            ]
            memory, error = change_note(memory_view(state), decision, user_messages, self.guard_artifact)
            state = {**state, "session_memory": memory}
        status = "SUCCESS"
        if error:
            status = "ERROR"
        return self._record(state, {
            "kind": "memory", "name": kind, "status": status, "error": error
        })

    def _worker_snapshot(self, role: str, snapshot: dict) -> dict:
        tools = {}
        for name, spec in snapshot["tool"].items():
            if "policy" in spec:
                allowed_tool = role in spec["policy"].get("workers", [])
            else:
                allowed_tool = name in WORKERS[role]["tool"]
            if allowed_tool:
                tools[name] = spec
        skills = {}
        for name, spec in snapshot["skill"].items():
            execution = spec["execution"]
            if execution["type"] == "registered_tool":
                if execution["tool"] in tools:
                    skills[name] = spec
            elif name.replace("-", "_") in WORKERS[role]["skill"]:
                skills[name] = spec

        # A skill may guard against a registered tool without exposing that tool to the worker.
        return {"tool": tools, "skill": skills}

    def _run_worker(self, role: str, task: str, snapshot: dict) -> dict:
        catalog = self.catalog_view(self._worker_snapshot(role, snapshot))
        system = (
            BASE_PROMPT + "\n" + WORKER_PROMPT
            + _DEEP_MESSAGES.get("worker_role_header", "\n") + WORKERS[role]["description"]
            + _DEEP_MESSAGES.get("worker_catalog_header", "\n") + json.dumps(catalog, ensure_ascii=False)
        )
        worker = self.input_guardrail({"user_query": task, "capabilities": snapshot})
        if not worker["is_safe"]:
            return {"error": _DEEP_MESSAGES.get("worker_guardrail_rejected", "")}
        worker["worker"] = True
        worker["worker_role"] = role
        for step in range(MAX_WORKER_ACTIONS + 1):
            turn = worker["react_turn"]
            allowed = ["final"]
            if step < MAX_WORKER_ACTIONS:
                allowed.append("tool")
                if catalog["skill"]:
                    allowed += ["skill", "inspect_skill"]
            request = {
                "task": worker["user_query"], "allowed_kinds": allowed,
                "remaining_actions": MAX_WORKER_ACTIONS - step,
                "observations": _compact_observations(turn["observations"]),
                "active_skills": worker.get("active_skills", {}),
                "available_evidence": _evidence_view(turn["evidence"])
            }
            deep_reason = None
            if len({item["chunk"].get("source") for item in turn["evidence"]
                    if item["chunk"].get("source")}) >= 2:
                deep_reason = "multiple_sources"
            decision = self.model.decide(system, request, allowed, max_tokens=_WORKER_MAX_TOKENS, deep_reason=deep_reason)
            kind = decision["kind"]
            if kind == "final":
                raw = self.guard_artifact(decision["answer"])
                screened = self.guardrails.guard_output(raw)
                if not screened.is_safe or not screened.sanitized_answer or screened.flags:
                    return {"error": _DEEP_MESSAGES.get("worker_report_guardrail_rejected", "")}
                report = screened.sanitized_answer
                self.validate_citations({**decision, "answer": report}, turn["evidence"])
                return {"report": report, "evidence": turn["evidence"], "calls": worker["tool_outputs"],
                        "observations": turn["observations"], "searches": worker["search_count"]}
            name = decision["name"]
            if kind == "inspect_skill":
                group = "skill"
            else:
                group = kind
            allowed_member = name in catalog[group]
            if group == "skill":
                allowed_member = resolve_skill(catalog[group], name) is not None
            if not allowed_member:
                worker = self._record(worker, {"kind": kind, "name": "restricted", "status": "ERROR",
                                               "error": _DEEP_MESSAGES.get("worker_restricted_tool", "")})
                continue
            worker["react_turn"] = {**turn, "decision": decision}
            if kind == "inspect_skill":
                worker = self.inspect_skill(worker)
            else:
                worker = self.act(worker)
        return {"error": _DEEP_MESSAGES.get("worker_exhausted", "")}

    def _merge_worker_evidence(self, parent: list, child: list) -> tuple[list, dict]:
        merged = list(parent)
        known = {tuple(item["identity"]): item["index"] for item in merged}
        mapping = {}
        for item in child:
            key = tuple(item["identity"])
            if key not in known:
                if len(merged) >= MAX_EVIDENCE:
                    raise ValueError("Turn evidence limit reached")
                known[key] = len(merged) + 1
                merged.append({**item, "index": known[key]})
            mapping[item["index"]] = known[key]
        return merged, mapping

    def delegate(self, state: dict) -> dict:
        turn = state["react_turn"]
        decision = turn["decision"]
        role = decision["role"]
        task = self.guard_artifact(decision["task"])
        status = "ERROR"
        data = {}
        error = _DEEP_MESSAGES.get("delegate_unregistered_or_unsafe", "")
        result = {}
        if role not in WORKERS:
            role = "unknown"
        elif task:
            repeated = any(item["role"] == role and item["task"] == task for item in state["delegations"])
            if repeated:
                error = _DEEP_MESSAGES.get("delegate_repeated", "")
            else:
                try:
                    result = self._run_worker(role, task, state["capabilities"])
                    error = result.get("error", "")
                    if not error:
                        merged, mapping = self._merge_worker_evidence(turn["evidence"], result["evidence"])
                        parts = result["report"].split("[Nguồn ")
                        report = parts[0]
                        for part in parts[1:]:
                            number, _, tail = part.partition("]")
                            report += _label(mapping[int(number)]) + tail
                        docs = [merged[index - 1] for index in dict.fromkeys(mapping.values())]
                        work = [
                            item for item in result["observations"] if not item.get("retrieval")
                        ]
                        searches_used = [
                            {
                                "query_used": item["query_used"],
                                "status": item["status"],
                                "documents_found": len(item.get("data", {}).get("documents", []))
                            }
                            for item in result["observations"] if "query_used" in item
                        ]
                        data = {
                            "report": report, "documents": _evidence_view(docs),
                            "worker_observations": work, "search_attempts": searches_used
                        }
                        turn = {**turn, "evidence": merged}
                        status = "SUCCESS"
                except (ModelFallbackError, ValueError):
                    error = _DEEP_MESSAGES.get("delegate_unverified", "")
        calls = state["tool_outputs"]
        searches = state["search_count"]
        intent = state["intent"]
        if status == "SUCCESS":
            calls = calls + [{**item, "worker": role} for item in result["calls"]]
            searches += result["searches"]
            if searches:
                intent = "RAG_SEARCH"
            elif calls:
                intent = "BANKING_ACTION"
        state = {
            **state, "intent": intent, "react_turn": turn, "tool_outputs": calls,
            "retrieved_chunks": [item["chunk"] for item in turn["evidence"]],
            "search_count": searches, "retry_count": max(0, searches - 1),
            "delegations": state["delegations"] + [{"role": role, "task": task, "status": status}]
        }
        return self._record(state, {
            "kind": "delegate", "name": role, "status": status, "data": data, "error": error,
            "retrieval": "documents" in data
        })

    def _retry_review(self, state: dict, draft: str, feedback: str, outcome: str) -> dict:
        retries = state["hallucination_retries"]
        revised = {
            **state, "audit_feedback": feedback, "draft_answer": draft,
            "reviews": state["reviews"] + [outcome],
            "execution_steps": state["execution_steps"] + ["review: " + outcome]
        }
        if retries < MAX_REVISIONS:
            return {**revised, "audit_status": "RETRY", "hallucination_retries": retries + 1}
        return {**revised, "audit_status": "FAIL", "final_answer": FALLBACK_ANSWER}

    def _review_feedback(self, verdict, draft, evidence):
        feedback = self.guard_artifact(verdict["feedback"])
        labels = {item["label"] for item in evidence}
        line_count = len(draft.splitlines())
        items = []
        if feedback:
            items.append(feedback)
        for issue in verdict.get("issues", []):
            description = self.guard_artifact(issue["item"] + ": " + issue["problem"])
            if not description:
                continue
            pointer = issue.get("draft_line")
            if type(pointer) is int and 1 <= pointer <= line_count:
                description = f"D{pointer} — " + description
            if issue.get("source") in labels:
                description += " " + issue["source"]
            items.append(description)
        return "\n".join(items) or _DEEP_MESSAGES.get("review_default_feedback", "")

    def review(self, state: dict) -> dict:
        result = self._review_draft(state)
        if (result["audit_status"] != "FAIL" or state.get("confirmation_required")
                or result["reviews"][-1] not in ("revise", "claim_failed", "citation_failed")
                or not state["react_turn"]["evidence"] or not result.get("draft_answer")):
            return result

        # A single final-only attempt after ordinary revisions are used up.
        # It cannot call tools, and its new answer must pass the normal reviewer.
        data = self._decide_input(result)
        data["review_feedback"] = result["audit_feedback"]
        data["previous_draft"] = result["draft_answer"]
        instruction = BASE_PROMPT + "\n" + _PARTIAL_RECOVERY_PROMPT
        try:
            decision = self.model.decide(instruction, data, ["final"], max_tokens=_PARTIAL_RECOVERY_MAX_TOKENS,
                                         deep_reason="review_revision")
        except ModelFallbackError:
            return result
        decision = {**decision, "insufficient_evidence": True}
        turn = {**result["react_turn"], "decision": decision}
        limited = self._review_draft({**result, "react_turn": turn}, partial=True)
        if limited["audit_status"] == "PASS":
            return limited
        return {**result, "reviews": limited["reviews"], "execution_steps": limited["execution_steps"]}

    def _review_draft(self, state: dict, *, partial=False) -> dict:
        turn = state["react_turn"]
        decision = turn["decision"]
        raw = self.guard_artifact(decision["answer"])
        output = self.guardrails.guard_output(raw)
        draft = output.sanitized_answer
        if not output.is_safe or not draft or output.flags:
            return self._retry_review(state, "", _DEEP_MESSAGES.get("review_draft_blocked", ""), "blocked")
        try:
            citations = self.validate_citations({**decision, "answer": draft}, turn["evidence"])
            if partial and not citations:
                raise ValueError("Partial answer requires grounded citations")
            if partial and len({item["source"] for item in citations}) > _PARTIAL_RECOVERY_MAX_SOURCES:
                raise ValueError("Partial answer must focus on at most three documents")
        except ValueError as error:
            annotate_span(citation_validation_error=str(error))
            return self._retry_review(
                state, draft, _DEEP_MESSAGES.get("review_citation_failed_prefix", "") + str(error), "citation_failed"
            )

        evidence = _evidence_view(turn["evidence"])
        cited = {item["label"] for item in citations}
        evidence.sort(key=lambda item: item["label"] not in cited)
        if cited:
            # Every cited source must reach the reviewer intact. Unrelated search
            # results and repeated retrieval observations must not crowd it out.
            evidence = [item for item in evidence if item["label"] in cited]
        else:
            evidence = evidence[:_REVIEW_UNLINKED_EVIDENCE_LIMIT]
        support = _support_view(state)
        review_input = {
            "request": state["user_query"],
            "history": state.get("chat_history", []),
            "user_memory": memory_view(state),
            "plan": state["plan"],
            "human_replies": state["human_replies"],
            "approved_scope": state["approved_scope"],
            "draft": draft,
            "draft_lines": "\n".join(f"D{index} {line}" for index, line in enumerate(draft.splitlines(), 1)),
            "evidence": [{**item, "text": "\n".join(
                f"L{index} {line}" for index, line in enumerate(item["text"].splitlines(), 1)
            )} for item in evidence],
            "support": support,
            "observations": _compact_observations(turn["observations"])
        }
        review_prompt = REVIEW_PROMPT
        review_tokens = _REVIEW_MAX_TOKENS
        if partial:
            review_prompt = PARTIAL_REVIEW_PROMPT
            review_tokens = _PARTIAL_REVIEW_MAX_TOKENS
        try:
            deep_reason = None
            if state["hallucination_retries"]:
                deep_reason = "review_revision"
            elif any(key.startswith(_DEEP_MESSAGES.get("tool_support_prefix", "[Tool ")) for key in support) and any(
                char.isdecimal() for char in draft
            ):
                deep_reason = "verify_numbers"
            elif len({item["source"] for item in citations}) >= 2:
                deep_reason = "multiple_sources"
            verdict = self.model.request(review_prompt, review_input, [REVIEW_SCHEMA], max_tokens=review_tokens,
                                         concise=True, role="agent_review", deep_reason=deep_reason)
        except ModelFallbackError as error:
            reason = type(getattr(error, "last_error", None)).__name__
            if isinstance(getattr(error, "last_error", None), ValueError):
                reason += ": " + str(error.last_error)[:160]
            annotate_span(review_failure=reason)
            return self._retry_review(state, draft, _DEEP_MESSAGES.get("review_unavailable", ""), "unavailable")
        if verdict["verdict"] == "revise":
            feedback = self._review_feedback(verdict, draft, evidence)
            return self._retry_review(state, draft, feedback, "revise")
        try:
            validate_review(verdict, draft, citations, evidence, support)
        except ValueError as error:
            # Repair the review record before discarding a potentially sound answer.
            try:
                repair_suffix = _REVIEW_REPAIR_INSTRUCTION.replace("{error}", str(error))
                repaired = self.model.request(
                    review_prompt + "\n" + repair_suffix,
                    review_input, [REVIEW_SCHEMA], max_tokens=review_tokens, concise=True,
                    role="agent_review", deep_reason="review_repair",
                )
                if repaired["verdict"] != "approve":
                    feedback = self._review_feedback(repaired, draft, evidence)
                    return self._retry_review(state, draft, feedback, "revise")
                validate_review(repaired, draft, citations, evidence, support)
                verdict = repaired
            except (ModelFallbackError, ValueError) as repair_error:
                reason = str(error)
                if isinstance(repair_error, ValueError):
                    reason = str(repair_error)
                return self._retry_review(
                    state,
                    draft,
                    _DEEP_MESSAGES.get("review_claim_failed_prefix", "")
                    + reason
                    + _DEEP_MESSAGES.get("review_claim_failed_suffix", ""),
                    "claim_failed",
                )

        if partial and not any(item["status"] == "answered" for item in verdict["coverage"]):
            return self._retry_review(state, draft, _DEEP_MESSAGES.get("review_partial_empty", ""), "revise")
        intent = state["intent"]
        if intent == "ROUTING":
            intent = "CHIT_CHAT"
        return {
            **state, "intent": intent, "audit_status": "PASS", "final_answer": draft, "audit_feedback": "",
            "citations": citations, "sources": list(dict.fromkeys(item["source"] for item in citations)),
            "verification": {"checked_claims": len(verdict["claims"]),
                             "covered_items": len(verdict["coverage"]),
                             "limited_items": sum(item["status"] == "limited" for item in verdict["coverage"]),
                             "partial_recovery": partial},
            "reviews": state["reviews"] + ["approve"], "execution_steps": state["execution_steps"] + ["review: approve"]
        }

    def output_guardrail(self, state: dict) -> dict:
        result = self.guardrails.guard_output(state["final_answer"])
        answer = result.sanitized_answer
        status = state["audit_status"]
        if not result.is_safe or not answer or result.flags:
            answer = self.guardrails.guard_output(FALLBACK_ANSWER).sanitized_answer
            status = "BLOCKED"
        history = state["history_at_start"]
        memory = state["memory_at_start"]
        if state["is_safe"] and status == "PASS":
            history = state.get("chat_history", []) + [{"role": "user", "content": state["user_query"][:_HISTORY_MESSAGE_MAX_CHARS]}]
            for reply in state["human_replies"]:
                history.extend([{"role": "assistant", "content": reply["question"]},
                                {"role": "user", "content": reply["response"]}])
            history.append({"role": "assistant", "content": answer[:_HISTORY_MESSAGE_MAX_CHARS]})
            memory = state["session_memory"]
        citations = [item for item in state["citations"] if item["label"] in answer]
        result_state = {
            **state, "react_turn": {}, "audit_feedback": "", "draft_answer": "", "chat_history": history,
            "session_memory": memory, "audit_status": status, "final_answer": answer, "citations": citations,
            "sources": list(dict.fromkeys(item["source"] for item in citations)),
            "output_guard": {"is_safe": result.is_safe, "flags": result.flags},
            "execution_steps": state["execution_steps"] + ["output_guardrail"]
        }
        result_state["memory_status"] = public_memory(result_state)
        result_state.pop("history_at_start")
        result_state.pop("memory_at_start")
        result_state.pop("capabilities", None)
        result_state.pop("interaction_signature", None)
        result_state.pop("human_replies", None)
        result_state.pop("approved_scope", None)
        result_state.pop("active_skills", None)
        return result_state
