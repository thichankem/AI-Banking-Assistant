"""
Kiểm thử Luồng Tích hợp (Integration Test) cho Loan Advisory Task.
"""

from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent_banking.tasks.loan_advisory_task.task import LoanAdvisoryTask


def test_loan_advisory_flow():
    task = LoanAdvisoryTask()
    res = task.execute("Tôi muốn vay 300 triệu trong 12 tháng lãi suất 8.5%/năm")

    assert res.task_id == "TASK_LOAN_ADVISORY"
    assert res.status == "SUCCESS"
    assert len(res.execution_steps) >= 4
    assert res.structured_data.get("principal") == 300000000.0
    print("✅ [TEST FLOW PASS] LoanAdvisoryTask thực thi thành công!")


if __name__ == "__main__":
    test_loan_advisory_flow()
