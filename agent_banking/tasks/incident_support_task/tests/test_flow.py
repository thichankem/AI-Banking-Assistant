"""
Kiểm thử Luồng Tích hợp (Integration Test) cho Incident Support Task.
"""

from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent_banking.tasks.incident_support_task.task import IncidentSupportTask


def test_incident_support_flow():
    task = IncidentSupportTask()
    res = task.execute("Tôi bị nuốt thẻ ở cây ATM số 123 đường Hai Bà Trưng lúc 10h sáng")

    assert res.task_id == "TASK_INCIDENT_SUPPORT"
    assert res.status == "SUCCESS"
    assert len(res.execution_steps) >= 3
    assert "ticket_id" in res.structured_data
    print("✅ [TEST FLOW PASS] IncidentSupportTask thực thi thành công!")


if __name__ == "__main__":
    test_incident_support_flow()
