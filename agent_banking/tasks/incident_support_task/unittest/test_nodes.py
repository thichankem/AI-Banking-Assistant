"""
Kiểm thử Đơn vị (Unit Test) từng Node kế thừa trong Incident Support Task.
"""

from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent_banking.tasks.incident_support_task.task import IncidentSupportTask


def test_individual_nodes():
    task = IncidentSupportTask()

    # 1. Test Node Banking Tool (Create Ticket)
    ticket_res = task.node_banking_tool("create_ticket", {
        "issue_type": "nuot_the_atm",
        "user_contact": "0912345678",
        "card_number": "1234",
        "details": "Nuốt thẻ ATM",
        "urgency": "HIGH"
    })
    ticket_data = ticket_res.get("ticket_info", {})
    assert "ticket_id" in ticket_data

    # 2. Test Node Output Format
    out_res = task.node_format_output(answer="Phiếu đã tạo", structured_data=ticket_data)
    assert out_res.status == "SUCCESS"

    print("✅ [UNIT TEST PASS] Toàn bộ các Node kế thừa của IncidentSupportTask hoạt động chính xác!")


if __name__ == "__main__":
    test_individual_nodes()
