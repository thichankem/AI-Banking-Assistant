"""
Kiểm thử Đơn vị (Unit Test) từng Node kế thừa trong Policy Inquiry Task.
"""

from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent_banking.tasks.policy_inquiry_task.task import PolicyInquiryTask


def test_individual_nodes():
    task = PolicyInquiryTask()

    # 1. Test Node Input Guard
    safe_res = task.node_input_guard("Quy định mở tài khoản")
    assert safe_res["is_safe"] is True

    # 2. Test Node Format History
    history = [
        {"role": "user", "content": "Câu hỏi 1"},
        {"role": "assistant", "content": "Trả lời 1"}
    ]
    history_str = task.node_format_history(history)
    assert "Khách hàng: Câu hỏi 1" in history_str

    # 3. Test Node Reference Filter
    sample_chunks = [
        {"text": "Chunk tốt", "rerank_score": 0.85},
        {"text": "Chunk kém", "rerank_score": 0.1}
    ]
    filtered = task.node_reference_filter(sample_chunks, min_score=0.5)
    assert len(filtered) == 1
    assert filtered[0]["text"] == "Chunk tốt"

    print("✅ [UNIT TEST PASS] Toàn bộ các Node kế thừa của PolicyInquiryTask hoạt động chính xác!")


if __name__ == "__main__":
    test_individual_nodes()
