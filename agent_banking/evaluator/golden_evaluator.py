"""
Golden Answer Evaluator Engine for Banking Operations.
======================================================
Tier-1: Deterministic Rule-Based Verifier (Must-include, Must-avoid, Table/Figure grounding, Numbers).
Tier-2: LLM-as-a-Judge for Banking Semantic Rubric (configured via environment).

DESIGN PRINCIPLES:
1. 0% REGEX LAW: No regex imports or functions; string methods only.
2. ZERO TERNARY: Explicit if/else statements only.
4. NO __INIT__.PY: Direct module imports.
"""

import os
import sys
import json

# Ensure repository root is in sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(os.path.dirname(current_dir))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from agent_banking.evaluator.config.settings import (
    EvaluatorConfig,
    get_evaluator_messages,
    get_evaluator_prompt,
    render_prompt,
)

try:
    from agent_banking.core.chat import FallbackChatClient
except Exception:
    FallbackChatClient = None

_GE_CFG = EvaluatorConfig.GOLDEN_EVALUATOR
_FEEDBACK = get_evaluator_messages("feedback")
_JUDGE_PROMPT = get_evaluator_prompt("GOLDEN_JUDGE_PROMPT", "golden_judge.md")


class GoldenAnswerEvaluator:
    def __init__(self, use_llm_judge=True):
        self.use_llm_judge = use_llm_judge
        self.chat_client = None
        if use_llm_judge:
            if FallbackChatClient is not None:
                self.chat_client = FallbackChatClient(module=str(_GE_CFG["judge_model_role"]))

    def evaluate_case(self, prediction_text, case_data, retrieved_sources=None):
        """
        Evaluate a single predicted answer against a golden case.
        """
        pred_clean = prediction_text.strip()
        pred_lower = pred_clean.lower()

        rubric = case_data.get("rubric", {})
        must_include = rubric.get("must_include", [])
        must_avoid = rubric.get("must_avoid", [])

        # 1. Tier-1 Rule Checks: Must-Include
        included_hits = []
        included_misses = []
        for item in must_include:
            item_lower = item.strip().lower()
            if item_lower in pred_lower:
                included_hits.append(item)
            else:
                included_misses.append(item)

        if len(must_include) > 0:
            must_include_ratio = float(len(included_hits)) / float(len(must_include))
        else:
            must_include_ratio = 1.0

        # 2. Tier-1 Rule Checks: Must-Avoid
        avoid_violations = []
        for item in must_avoid:
            item_lower = item.strip().lower()
            if item_lower in pred_lower:
                avoid_violations.append(item)

        has_safety_violation = False
        if len(avoid_violations) > 0:
            has_safety_violation = True

        # 3. Tier-1 Grounding Check
        evidence_list = case_data.get("evidence", [])
        grounding_hits = []

        sources_to_check = []
        if retrieved_sources is not None:
            for s in retrieved_sources:
                sources_to_check.append(str(s).lower())
        sources_to_check.append(pred_lower)

        combined_source_text = " ".join(sources_to_check)

        for ev in evidence_list:
            tgt = ev.get("target_artifact", "").lower()
            src = ev.get("source_path", "").lower()
            sec = ev.get("section", "").lower()

            # Check if artifact name or section is cited/retrieved
            if (tgt and tgt in combined_source_text) or (sec and sec in combined_source_text) or (src and src in combined_source_text):
                grounding_hits.append(ev)

        if len(evidence_list) > 0:
            grounding_ratio = float(len(grounding_hits)) / float(len(evidence_list))
        else:
            grounding_ratio = 1.0

        # Calculate Rule Score (0.0 to 100.0)
        rule_score = (
            must_include_ratio * float(_GE_CFG["must_include_weight"])
            + grounding_ratio * float(_GE_CFG["grounding_weight"])
        )
        if has_safety_violation:
            # Harsh penalty for violating safety or stating forbidden errors
            rule_score = rule_score * float(_GE_CFG["safety_violation_multiplier"])

        # 4. Tier-2 LLM Judge (if enabled and client available)
        llm_score = None
        llm_feedback = ""

        if self.use_llm_judge:
            if self.chat_client is not None:
                llm_res = self._call_llm_judge(pred_clean, case_data)
                llm_score = llm_res.get("score", 0.0)
                llm_feedback = llm_res.get("feedback", "")

        # 5. Composite Final Score
        if llm_score is not None:
            final_score = (
                (rule_score * float(_GE_CFG["rule_score_weight"]))
                + (llm_score * float(_GE_CFG["llm_score_weight"]))
            )
        else:
            final_score = rule_score

        passed = False
        if final_score >= float(_GE_CFG["pass_threshold"]):
            if not has_safety_violation:
                passed = True

        return {
            "case_id": case_data.get("id", ""),
            "topic_id": case_data.get("topic_id", ""),
            "test_criteria": case_data.get("test_criteria", ""),
            "passed": passed,
            "final_score": round(final_score, 2),
            "rule_score": round(rule_score, 2),
            "llm_score": llm_score,
            "must_include_ratio": round(must_include_ratio, 2),
            "included_hits": included_hits,
            "included_misses": included_misses,
            "avoid_violations": avoid_violations,
            "has_safety_violation": has_safety_violation,
            "grounding_ratio": round(grounding_ratio, 2),
            "llm_feedback": llm_feedback
        }

    def _call_llm_judge(self, prediction_text, case_data):
        """
        Call LLM-as-a-Judge via configured API as an impartial banking judge.
        """
        prompt = render_prompt(
            _JUDGE_PROMPT,
            {
                "question": case_data.get("question", ""),
                "reference_answer": case_data.get("reference_answer", ""),
                "must_include": ", ".join(case_data.get("rubric", {}).get("must_include", [])),
                "must_avoid": ", ".join(case_data.get("rubric", {}).get("must_avoid", [])),
                "prediction_text": prediction_text,
            },
        )

        try:
            messages = [{"role": "user", "content": prompt}]
            resp = self.chat_client.create(
                messages=messages,
                temperature=float(_GE_CFG["judge_temperature"]),
            )

            # parse response JSON cleanly without regex
            content = resp.choices[0].message.content.strip()
            start_idx = content.find("{")
            end_idx = content.rfind("}")
            if start_idx != -1:
                if end_idx != -1:
                    json_str = content[start_idx:end_idx + 1]
                    data = json.loads(json_str)
                    return {
                        "score": float(data.get("score", float(_GE_CFG["default_llm_score"]))),
                        "feedback": str(data.get("feedback", _FEEDBACK["golden_default_feedback"])),
                    }
        except Exception:
            pass

        return {
            "score": float(_GE_CFG["fallback_llm_score"]),
            "feedback": _FEEDBACK["golden_fallback_feedback"],
        }
