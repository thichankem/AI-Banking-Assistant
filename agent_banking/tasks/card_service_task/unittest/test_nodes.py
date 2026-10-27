"""
Kiểm thử Đơn vị (Unit Test) từng Node kế thừa trong Card Service Task.
"""

from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent_banking.tasks.card_service_task.task import CardServiceTask


def test_individual_nodes():
    task = CardServiceTask()

    # 1. Test Node Input Guard
    safe_res = task.node_input_guard("Hạn mức thẻ visa platinum")
    assert safe_res["is_safe"] is True

    # 2. Test Node Output Format
    out_res = task.node_format_output(answer="Chi tiết biểu phí", structured_data={"type": "CREDIT"})
    assert out_res.status == "SUCCESS"
    assert out_res.structured_data.get("type") == "CREDIT"

    print("✅ [UNIT TEST PASS] Toàn bộ các Node kế thừa của CardServiceTask hoạt động chính xác!")


if __name__ == "__main__":
    test_individual_nodes()
