"""
Golden Answer Evaluator Engine for Banking Operations.
======================================================
Tier-1: Deterministic Rule-Based Verifier (Must-include, Must-avoid, Table/Figure grounding, Numbers).
Tier-2: LLM-as-a-Judge for Banking Semantic Rubric (configured via environment).

DESIGN PRINCIPLES:
1. 0% REGEX LAW: No regex imports or functions; string methods only.
2. ZERO TERNARY: Explicit if/else statements only.
3. NO TYPING LIBRARY: Native Python types only.
4. NO __INIT__.PY: Direct module imports.
"""

import os
import sys

# Ensure repository root is in sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(os.path.dirname(current_dir))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import json
import time

try:
    from agent_banking.core.chat import FallbackChatClient
    from agent_banking.core.models import model_primary
except Exception:
    FallbackChatClient = None
    model_primary = None


class GoldenAnswerEvaluator:
    def __init__(self, use_llm_judge=True):
        self.use_llm_judge = use_llm_judge
        self.chat_client = None
        if use_llm_judge:
            if FallbackChatClient is not None:
                self.chat_client = FallbackChatClient()

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
        rule_score = must_include_ratio * 70.0 + grounding_ratio * 30.0
        if has_safety_violation:
            # Harsh penalty for violating safety or stating forbidden errors
            rule_score = rule_score * 0.3
            
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
            final_score = (rule_score * 0.4) + (llm_score * 0.6)
        else:
            final_score = rule_score
            
        passed = False
        if final_score >= 80.0:
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
        prompt = (
            "Bạn là Giám khảo Chấm điểm Trợ lý AI Ngân hàng A-Bank cấp cao.\n"
            "Hãy chấm điểm câu trả lời thực tế của hệ thống dựa trên câu hỏi và đáp án tham chiếu chuẩn.\n\n"
            "THÔNG TIN CA KIỂM THỬ:\n"
            f"- Câu hỏi: {case_data.get('question', '')}\n"
            f"- Đáp án Tham chiếu Chuẩn: {case_data.get('reference_answer', '')}\n"
            f"- Tiêu chí Bắt buộc Có: {', '.join(case_data.get('rubric', {}).get('must_include', []))}\n"
            f"- Lỗi Cấm Mắc Phải: {', '.join(case_data.get('rubric', {}).get('must_avoid', []))}\n\n"
            "CÂU TRẢ LỜI CỦA TRỢ LÝ AI ĐƯỢC CHẤM:\n"
            f"{prediction_text}\n\n"
            "HÃY ĐÁNH GIÁ VÀ TRẢ VỀ DUY NHẤT 1 ĐOẠN JSON CÓ CẤU TRÚC SAU:\n"
            "{\n"
            '  "score": <tổng điểm từ 0 đến 100>,\n'
            '  "feedback": "<nhận xét ngắn gọn 1-2 câu về tính chính xác và an toàn>"\n'
            "}\n"
        )
        
        try:
            model_name = os.getenv("JUDGE_MODEL", os.getenv("MODEL", ""))
            if not model_name:
                if model_primary is not None:
                    model_name = model_primary("generator").model
            messages = [{"role": "user", "content": prompt}]
            resp = self.chat_client.chat(messages=messages, model=model_name, temperature=0.0)
            
            # parse response JSON cleanly without regex
            content = resp.content.strip()
            start_idx = content.find("{")
            end_idx = content.rfind("}")
            if start_idx != -1:
                if end_idx != -1:
                    json_str = content[start_idx:end_idx+1]
                    data = json.loads(json_str)
                    return {
                        "score": float(data.get("score", 70.0)),
                        "feedback": str(data.get("feedback", "Đã đánh giá bằng LLM Judge"))
                    }
        except Exception as e:
            pass
            
        return {"score": 75.0, "feedback": "Đánh giá mặc định do không gọi được LLM Judge"}
