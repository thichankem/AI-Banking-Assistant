"""
Kiểm thử Luồng Tích hợp (Integration Test) cho Card Service Task.
"""

from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent_banking.tasks.card_service_task.task import CardServiceTask


def test_card_service_flow():
    task = CardServiceTask()
    res = task.execute("Hạn mức và quyền lợi của thẻ tín dụng Vietcombank Visa Platinum là gì?")

    assert res.task_id == "TASK_CARD_SERVICE"
    assert res.status == "SUCCESS"
    assert len(res.execution_steps) >= 4
    print("✅ [TEST FLOW PASS] CardServiceTask thực thi thành công!")


if __name__ == "__main__":
    test_card_service_flow()
