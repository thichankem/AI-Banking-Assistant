"""
Kiểm thử Luồng Tích hợp (Integration Test) cho Policy Inquiry Task.
"""

from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent_banking.tasks.policy_inquiry_task.task import PolicyInquiryTask


def test_policy_inquiry_flow():
    task = PolicyInquiryTask()
    res = task.execute("Hạn mức giao dịch chuyển tiền khi mở tài khoản trực tuyến eKYC là bao nhiêu?")

    assert res.task_id == "TASK_POLICY_INQUIRY"
    assert res.status == "SUCCESS"
    assert len(res.execution_steps) >= 5
    assert len(res.answer) > 50
    print("✅ [TEST FLOW PASS] PolicyInquiryTask thực thi thành công!")


if __name__ == "__main__":
    test_policy_inquiry_flow()
